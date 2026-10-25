/*
 * AlltoAll - direct push (one round, every CTA scatters its chunks to their destinations)
 *
 * 1. Op: CTA b holds N chunks of M elements, chunk j destined to CTA j; afterwards CTA b holds
 *    chunk b of every CTA, in source-rank order (a transpose of the N x N chunk matrix).
 *
 * 2. Algorithm: CTA b pushes chunk j into recv[b] of CTA j for all j != b and copies chunk b locally;
 *    one mbarrier per CTA armed with the aggregate (N-1)*M*sizeof(T) completes when all arrived.
 *
 * 3. Pattern: fully connected push, 1 round, N-1 messages of M*sizeof(T) per CTA; total
 *    (N-1)*M*sizeof(T) per CTA (optimal); N-1 concurrent flows into every SM.
 *    Primitive: cp.async.bulk pushes + one mbarrier complete_tx target; parity wait; exit barrier.cluster.
 */
#include "dsmem_primitives.cuh"
using namespace dsmem;

template <int N, typename T, int M>
__device__ void alltoall_direct_push(const T* chunks /*[N*M]*/, T* recv /*[N*M]*/, uint32_t bar, Stamps st) {
    static_assert((M * sizeof(T)) % 16 == 0, "bulk copy needs a multiple of 16 bytes");
    constexpr uint32_t bytes = M * sizeof(T);
    const uint32_t rank = cluster_rank();
    const uint32_t src_a = smem_u32(chunks), recv_a = smem_u32(recv);
    if (threadIdx.x == 0) {
        mbar_arrive_expect_tx(bar, (N - 1) * bytes);
        for (uint32_t j = 1; j < N; ++j) {
            const uint32_t dst = (rank + j) % N;
            bulk_push(dst, recv_a + rank * bytes, src_a + dst * bytes, bytes, bar);   // chunk dst -> recv[rank] of CTA dst
        }
    }
    for (int i = threadIdx.x; i < M; i += blockDim.x) recv[rank * M + i] = chunks[rank * M + i];   // own chunk
    mbar_wait_parity(bar, 0);
    st.round(0);
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
    alltoall_direct_push<N, ELEM_T, M>(chunks, recv, smem_u32(&bar), st);
    st.mark(1);
    store_block<ELEM_T, N * M>(out + rank * N * M, recv);
    cluster_sync();
}
