/*
 * AllReduce - shift-and-accumulate bulk push (ClusterFusion shipped code, dsm.cuh cluster_reduce)
 *
 * 1. Op: every CTA ends with the element-wise sum of the N M-element vectors.
 *
 * 2. Algorithm (as shipped, not Alg. 1 of the paper): rounds i = 1 .. N-2. In round i CTA b
 *    pushes its *running* buffer to CTA (b + i) mod N and receives from (b - i) mod N, then
 *    adds the received buffer into its own. Because the running buffer is forwarded, round 1
 *    delivers {b-1}, round 2 delivers {b-2, b-3}: with N = 4 the two rounds cover all ranks.
 *    The loop bound `i < N-1` is only correct for N == 4 (N = 2 runs zero rounds, N >= 8
 *    double-counts), which matches the hard-coded CLUSTER_SIZE 4; enforced by static_assert.
 *
 * 3. Pattern: ring with shift i (one sender and one receiver per CTA per round).
 *    Rounds: N-2 (= 2). Message per round: full M*sizeof(T) bytes. Total per CTA: 2*M*sizeof(T).
 *    Primitive: thread 0 does mbarrier.arrive.expect_tx on its own barrier, maps destination
 *    buffer + barrier with mapa, issues cp.async.bulk shared::cta -> shared::cluster with
 *    complete_tx; all threads spin on mbarrier parity (i ^ 1); local add; barrier.cluster per round.
 *    Deviation: the origin adds the received buffer into `src` in place although its own outgoing
 *    bulk copy may still be reading `src`; here the sum goes to an alternate buffer (ping-pong),
 *    same message/barrier count.
 *
 * Source: third_party/ClusterFusion/include/dsm.cuh (Stage::LINEAR / ATTN variants), used
 * for partial QKV (3*128 halves) and partial attention output (128 halves) in the Llama kernel.
 */
#include "dsmem_primitives.cuh"
using namespace dsmem;

// Returns the buffer (buf or alt) that holds the result.
template <int N, typename T, int M>
__device__ T* allreduce_shift_accumulate(T* buf, T* alt, T* scratch, uint32_t bar, Stamps st) {
    static_assert(N == 4, "shipped loop bound (i < N-1) only reduces correctly for N == 4");
    static_assert((M * sizeof(T)) % 16 == 0, "bulk copy needs a multiple of 16 bytes");
    constexpr uint32_t bytes = M * sizeof(T);
    const uint32_t rank = cluster_rank();
    const uint32_t scr_a = smem_u32(scratch);
    T* cur = buf; T* other = alt;
    for (int i = 1; i < N - 1; ++i) {
        if (threadIdx.x == 0) {
            mbar_arrive_expect_tx(bar, bytes);                // arm my barrier for `bytes` incoming
            const uint32_t dst_cta = (rank + i) % N;          // shift partner
            bulk_push(dst_cta, scr_a, smem_u32(cur), bytes, bar);   // mapa(dst)+mapa(bar), cp.async.bulk ... complete_tx
        }
        mbar_wait_parity(bar, (i ^ 1) & 1);                   // origin waits parity i^1: round 1 -> 0, round 2 -> 1
        for (int e = threadIdx.x; e < M; e += blockDim.x) other[e] = cur[e] + scratch[e];   // local reduce-add (ping-pong)
        fence_proxy_async_smem();                             // (added) order generic writes before next bulk read
        cluster_sync();                                       // origin: cluster.sync() ends every round
        T* t = cur; cur = other; other = t;
        st.round(i - 1);
    }
    return cur;
}

// Shared buffers of one CTA, taken from the dynamic shared-memory window (dsmem_primitives.cuh, dyn_smem);
// the launcher passes sizeof(CollectiveSmem) as the dynamic smem size (up to 227 KiB per CTA).
struct CollectiveSmem {
    static constexpr int N = CLUSTER_N, M = MSG_ELEMS;
    __align__(16) ELEM_T buf[M];
    __align__(16) ELEM_T alt[M];
    __align__(16) ELEM_T scratch[M];
    __align__(8) uint64_t bar;
};

__global__ void __cluster_dims__(CLUSTER_N, 1, 1)
collective_kernel(const ELEM_T* in, ELEM_T* out, unsigned long long* ts, unsigned* smid_out) {
    constexpr int N = CLUSTER_N, M = MSG_ELEMS;
    CollectiveSmem& sm = dyn_smem<CollectiveSmem>();
    auto& buf = sm.buf; auto& alt = sm.alt; auto& scratch = sm.scratch; auto& bar = sm.bar;
    const uint32_t rank = cluster_rank();
    Stamps st{ts + rank * kStamps};

    load_block<ELEM_T, M>(buf, in + rank * M);
    fence_proxy_async_smem();
    if (threadIdx.x == 0) { mbar_init(smem_u32(&bar), 1); fence_mbar_init(); smid_out[rank] = smid(); }   // origin: init once, count 1
    cluster_sync();
    st.mark(0);
    ELEM_T* res = allreduce_shift_accumulate<N, ELEM_T, M>(buf, alt, scratch, smem_u32(&bar), st);
    st.mark(1);
    store_block<ELEM_T, M>(out + rank * M, res);
    cluster_sync();
}
