/*
 * AlltoAll - pairwise exchange with XOR partners (MPICH long-message algorithm)
 *
 * 1. Op: CTA b holds N chunks of M elements, chunk j destined to CTA j; afterwards CTA b holds
 *    chunk b of every CTA in source-rank order.
 *
 * 2. Algorithm (N power of two): N-1 rounds, round i = 1..N-1 pairs CTA b with p = b XOR i; the two
 *    exchange exactly one chunk each (b's chunk p for p's chunk b). Every round is a perfect
 *    matching of the cluster, so each CTA sends and receives one message per round. Generalises the
 *    FlashAttention-4 2-CTA exchange (N = 2, one round).
 *
 * 3. Pattern: hypercube matchings. Rounds: N-1. Message: M*sizeof(T). Total per CTA:
 *    (N-1)*M*sizeof(T). Primitive: cp.async.bulk push into the partner's recv[b] + mbarrier
 *    complete_tx; parity wait; barrier.cluster per round (shared mbarrier, no run-ahead).
 */
#include "dsmem_primitives.cuh"
using namespace dsmem;

template <int N, typename T, int M>
__device__ void alltoall_pairwise_xor(const T* chunks, T* recv, uint32_t bar, Stamps st) {
    static_assert((N & (N - 1)) == 0, "XOR pairing needs a power-of-two cluster size");
    static_assert((M * sizeof(T)) % 16 == 0, "bulk copy needs a multiple of 16 bytes");
    constexpr uint32_t bytes = M * sizeof(T);
    const uint32_t rank = cluster_rank();
    const uint32_t src_a = smem_u32(chunks), recv_a = smem_u32(recv);
    uint32_t phase = 0;
    for (int i = threadIdx.x; i < M; i += blockDim.x) recv[rank * M + i] = chunks[rank * M + i];
    for (int i = 1; i < N; ++i) {
        const uint32_t p = rank ^ i;                           // this round's partner
        if (threadIdx.x == 0) {
            mbar_arrive_expect_tx(bar, bytes);
            bulk_push(p, recv_a + rank * bytes, src_a + p * bytes, bytes, bar);   // my chunk for p -> p's recv[rank]
        }
        mbar_wait_parity(bar, phase); phase ^= 1;              // p's chunk for me landed in recv[p]
        cluster_sync();
        st.round(i - 1);
    }
}

// Shared buffers of one CTA, taken from the dynamic shared-memory window (dsmem_primitives.cuh, dyn_smem);
// the launcher passes sizeof(CollectiveSmem) as the dynamic smem size (up to 227 KiB per CTA).
struct CollectiveSmem {
    static constexpr int N = CLUSTER_N, M = MSG_ELEMS;
    __align__(16) ELEM_T chunks[N * M];
    __align__(16) ELEM_T recv[N * M];
    __align__(8) uint64_t bar;
};

__global__ void __cluster_dims__(CLUSTER_N, 1, 1)
collective_kernel(const ELEM_T* in, ELEM_T* out, unsigned long long* ts, unsigned* smid_out) {
    constexpr int N = CLUSTER_N, M = MSG_ELEMS;
    CollectiveSmem& sm = dyn_smem<CollectiveSmem>();
    auto& chunks = sm.chunks; auto& recv = sm.recv; auto& bar = sm.bar;
    const uint32_t rank = cluster_rank();
    Stamps st{ts + rank * kStamps};

    load_block<ELEM_T, N * M>(chunks, in + rank * N * M);
    fence_proxy_async_smem();
    if (threadIdx.x == 0) { mbar_init(smem_u32(&bar), 1); fence_mbar_init(); smid_out[rank] = smid(); }
    cluster_sync();
    st.mark(0);
    alltoall_pairwise_xor<N, ELEM_T, M>(chunks, recv, smem_u32(&bar), st);
    st.mark(1);
    store_block<ELEM_T, N * M>(out + rank * N * M, recv);
    cluster_sync();
}
