/*
 * AllReduce - two-shot push (direct reduce-scatter push, then direct all-gather push), TensorRT-LLM "two-shot" style
 *
 * 1. Op: every CTA ends with the element-wise sum of N M-element vectors.
 *
 * 2. Algorithm: chunks of C = M/N, CTA j owns chunk j. Shot 1 (reduce-scatter): CTA b pushes chunk j
 *    into slot[b] of CTA j for every j != b, then reduces the N slots of its own chunk. Shot 2
 *    (all-gather): CTA b pushes its reduced chunk into chunk position b of every peer. Both shots are
 *    a single round of N-1 simultaneous messages (no forwarding), unlike ring/recursive algorithms.
 *
 * 3. Pattern: fully connected, 2 rounds. Message size: C*sizeof(T); N-1 messages per CTA per round.
 *    Total per CTA: 2(N-1)/N * M*sizeof(T) (bandwidth-optimal) with only two mbarrier waits and no
 *    cluster barrier between the shots: a CTA may overwrite peer chunk b only after the peer's shot-1
 *    push of that chunk completed, which is implied by the CTA's own shot-1 wait.
 *    Primitive: N-1 cp.async.bulk pushes per shot, each completing tx bytes on ONE receiver mbarrier
 *    armed with the aggregate (N-1)*C*sizeof(T); parity wait; exit barrier.cluster for DSMEM lifetime.
 *
 * Source: TensorRT-LLM customAllReduce two-shot (reduce-scatter + all-gather over NVLink), ported to DSMEM.
 */
#include "dsmem_primitives.cuh"
using namespace dsmem;

template <int N, typename T, int M, class Op>
__device__ void allreduce_two_shot_push(T* buf, T (*slots)[M / N], uint32_t bar_rs, uint32_t bar_ag, Op op, Stamps st) {
    static_assert(M % N == 0, "M must split into N chunks");
    constexpr int C = M / N;
    static_assert((C * sizeof(T)) % 16 == 0, "chunk must be a multiple of 16 bytes");
    constexpr uint32_t bytes = C * sizeof(T);
    const uint32_t rank = cluster_rank();
    const uint32_t buf_a = smem_u32(buf), slots_a = smem_u32(&slots[0][0]);
    // shot 1: scatter my chunks j to their owners
    if (threadIdx.x == 0) {
        mbar_arrive_expect_tx(bar_rs, (N - 1) * bytes);        // N-1 pushes will land in my slots
        for (uint32_t j = 1; j < N; ++j) {
            const uint32_t dst = (rank + j) % N;               // rotate start to spread link usage
            bulk_push(dst, slots_a + rank * bytes, buf_a + dst * bytes, bytes, bar_rs);   // chunk dst -> slot[rank] of CTA dst
        }
    }
    mbar_wait_parity(bar_rs, 0);                               // all peers' partials of my chunk arrived
    for (int i = threadIdx.x; i < C; i += blockDim.x) {        // reduce my chunk: own copy + N-1 slots
        T acc = buf[rank * C + i];
        #pragma unroll
        for (int r = 0; r < N; ++r) if (r != (int)rank) acc = op(acc, slots[r][i]);
        buf[rank * C + i] = acc;
    }
    fence_proxy_async_smem();
    __syncthreads();
    st.round(0);
    // shot 2: broadcast my reduced chunk into position rank of every peer
    if (threadIdx.x == 0) {
        mbar_arrive_expect_tx(bar_ag, (N - 1) * bytes);
        for (uint32_t j = 1; j < N; ++j) {
            const uint32_t dst = (rank + j) % N;
            bulk_push(dst, buf_a + rank * bytes, buf_a + rank * bytes, bytes, bar_ag);   // reduced chunk -> same slot in CTA dst
        }
    }
    mbar_wait_parity(bar_ag, 0);                               // all N-1 reduced chunks arrived
    st.round(1);
}

// Shared buffers of one CTA, taken from the dynamic shared-memory window (dsmem_primitives.cuh, dyn_smem);
// the launcher passes sizeof(CollectiveSmem) as the dynamic smem size (up to 227 KiB per CTA).
struct CollectiveSmem {
    static constexpr int N = CLUSTER_N, M = MSG_ELEMS;
    __align__(16) ELEM_T buf[M];
    __align__(16) ELEM_T slots[N][M / N];
    __align__(8) uint64_t bars[2];
};

__global__ void __cluster_dims__(CLUSTER_N, 1, 1)
collective_kernel(const ELEM_T* in, ELEM_T* out, unsigned long long* ts, unsigned* smid_out) {
    constexpr int N = CLUSTER_N, M = MSG_ELEMS;
    CollectiveSmem& sm = dyn_smem<CollectiveSmem>();
    auto& buf = sm.buf; auto& slots = sm.slots; auto& bars = sm.bars;
    const uint32_t rank = cluster_rank();
    Stamps st{ts + rank * kStamps};

    load_block<ELEM_T, M>(buf, in + rank * M);
    fence_proxy_async_smem();
    if (threadIdx.x == 0) { mbar_init(smem_u32(&bars[0]), 1); mbar_init(smem_u32(&bars[1]), 1); fence_mbar_init(); smid_out[rank] = smid(); }
    cluster_sync();
    st.mark(0);
    allreduce_two_shot_push<N, ELEM_T, M>(buf, slots, smem_u32(&bars[0]), smem_u32(&bars[1]), Sum{}, st);
    st.mark(1);
    store_block<ELEM_T, M>(out + rank * M, buf);
    cluster_sync();
}
