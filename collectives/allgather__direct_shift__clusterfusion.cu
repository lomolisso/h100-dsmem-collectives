/*
 * AllGather - direct push with rotating shift (ClusterFusion shipped code, Stage::QUK_DEEPSEEK)
 *
 * 1. Op: every CTA ends with the concatenation of the N M-element vectors in rank order.
 *
 * 2. Algorithm: buffer D_b of N*M elements with own data already in segment b. N-1 rounds,
 *    round i = 1..N-1: CTA b pushes its own segment D_b[b*M : (b+1)*M] into the *same* segment
 *    index of CTA (b + i) mod N. No forwarding: each CTA only ever sends its own data, so the
 *    buffer is in rank order at the end without any un-rotation.
 *
 * 3. Pattern: ring with shift i (one send, one receive per CTA per round). Rounds: N-1.
 *    Message per round: M*sizeof(T) bytes (constant). Total per CTA: (N-1)*M*sizeof(T) - the
 *    same bytes as recursive doubling but N-1 rounds instead of log2(N).
 *    Primitive (as shipped): every round re-initialises the mbarrier (init, arrive.expect_tx),
 *    barrier.cluster, one-thread cp.async.bulk push with complete_tx on the peer's barrier,
 *    wait on parity 0, barrier.cluster. Two cluster barriers per round.
 *
 * Source: third_party/ClusterFusion/include/dsm.cuh (QUK_DEEPSEEK branch), used to gather
 * the 128-element q @ W_UK slices of the 4 CTAs in the DeepSeek MLA kernel.
 */
#include "dsmem_primitives.cuh"
using namespace dsmem;

template <int N, typename T, int M>
__device__ void allgather_direct_shift(T* D, uint32_t bar, Stamps st) {
    static_assert((M * sizeof(T)) % 16 == 0, "bulk copy needs a multiple of 16 bytes");
    constexpr uint32_t bytes = M * sizeof(T);
    const uint32_t rank = cluster_rank();
    const uint32_t seg_a = smem_u32(D + rank * M);             // my own segment (also the destination offset in the peer)
    for (int i = 1; i < N; ++i) {
        if (threadIdx.x == 0) {
            mbar_init(bar, 1);                                 // origin re-inits the barrier each round
            fence_mbar_init();
            mbar_arrive_expect_tx(bar, bytes);                 // expect one segment from (b - i)
        }
        cluster_sync();                                        // barrier (re)initialised everywhere before anyone pushes
        if (threadIdx.x == 0)
            bulk_push((rank + i) % N, seg_a, seg_a, bytes, bar);   // my segment -> slot[b] of CTA (b+i), signal its barrier
        mbar_wait_parity(bar, 0);                              // fresh barrier: phase 0
        cluster_sync();                                        // round complete cluster-wide
        st.round(i - 1);
    }
}

// Shared buffers of one CTA, taken from the dynamic shared-memory window (dsmem_primitives.cuh, dyn_smem);
// the launcher passes sizeof(CollectiveSmem) as the dynamic smem size (up to 227 KiB per CTA).
struct CollectiveSmem {
    static constexpr int N = CLUSTER_N, M = MSG_ELEMS;
    __align__(16) ELEM_T D[N * M];
    __align__(8) uint64_t bar;
};

__global__ void __cluster_dims__(CLUSTER_N, 1, 1)
collective_kernel(const ELEM_T* in, ELEM_T* out, unsigned long long* ts, unsigned* smid_out) {
    constexpr int N = CLUSTER_N, M = MSG_ELEMS;
    CollectiveSmem& sm = dyn_smem<CollectiveSmem>();
    auto& D = sm.D; auto& bar = sm.bar;
    const uint32_t rank = cluster_rank();
    Stamps st{ts + rank * kStamps};

    load_block<ELEM_T, M>(D + rank * M, in + rank * M);        // own data in segment `rank`
    fence_proxy_async_smem();
    if (threadIdx.x == 0) smid_out[rank] = smid();
    cluster_sync();
    st.mark(0);
    allgather_direct_shift<N, ELEM_T, M>(D, smem_u32(&bar), st);
    st.mark(1);
    store_block<ELEM_T, N * M>(out + rank * N * M, D);
    cluster_sync();
}
