/*
 * AlltoAll - direct pull (every CTA reads its chunk out of every peer's send buffer)
 *
 * 1. Op: CTA b holds N chunks of M elements, chunk j destined to CTA j; afterwards CTA b holds
 *    chunk b of every CTA in source-rank order.
 *
 * 2. Algorithm: after a cluster barrier publishes the send buffers, CTA b loads chunk b from every
 *    peer r (same smem offset in each CTA, mapped with mapa) into recv[r]. No pushes, no mbarriers.
 *
 * 3. Pattern: fully connected pull, 1 round. Remote read bytes per CTA: (N-1)*M*sizeof(T).
 *    Barriers: 2 barrier.cluster. Primitive: mapa + ld.shared::cluster.v4.
 */
#include "dsmem_primitives.cuh"
using namespace dsmem;

template <int N, typename T, int M>
__device__ void alltoall_direct_pull(const T* chunks, T* recv, Stamps st) {
    static_assert((M * sizeof(T)) % 16 == 0, "vectorised 16-byte remote loads");
    constexpr int V16 = M * sizeof(T) / 16;
    const uint32_t rank = cluster_rank();
    const uint32_t my_chunk_a = smem_u32(chunks + rank * M);   // chunk `rank` lives at this offset in every peer
    cluster_sync();                                            // send buffers published
    for (int idx = threadIdx.x; idx < N * V16; idx += blockDim.x) {
        const int r = idx / V16, u = idx % V16;
        const uint4 v = (r == (int)rank) ? reinterpret_cast<const uint4*>(chunks + rank * M)[u]
                                         : ld_cluster_v4(mapa(my_chunk_a + u * 16, r));   // chunk `rank` of CTA r
        reinterpret_cast<uint4*>(recv + r * M)[u] = v;
    }
    st.round(0);
    cluster_sync();                                            // all peers finished reading my send buffer
}

// Shared buffers of one CTA, taken from the dynamic shared-memory window (dsmem_primitives.cuh, dyn_smem);
// the launcher passes sizeof(CollectiveSmem) as the dynamic smem size (up to 227 KiB per CTA).
struct CollectiveSmem {
    static constexpr int N = CLUSTER_N, M = MSG_ELEMS;
    __align__(16) ELEM_T chunks[N * M];
    __align__(16) ELEM_T recv[N * M];
};

__global__ void __cluster_dims__(CLUSTER_N, 1, 1)
collective_kernel(const ELEM_T* in, ELEM_T* out, unsigned long long* ts, unsigned* smid_out) {
    constexpr int N = CLUSTER_N, M = MSG_ELEMS;
    CollectiveSmem& sm = dyn_smem<CollectiveSmem>();
    auto& chunks = sm.chunks; auto& recv = sm.recv;
    const uint32_t rank = cluster_rank();
    Stamps st{ts + rank * kStamps};

    load_block<ELEM_T, N * M>(chunks, in + rank * N * M);
    if (threadIdx.x == 0) smid_out[rank] = smid();
    cluster_sync();
    st.mark(0);
    alltoall_direct_pull<N, ELEM_T, M>(chunks, recv, st);
    st.mark(1);
    store_block<ELEM_T, N * M>(out + rank * N * M, recv);
    cluster_sync();
}
