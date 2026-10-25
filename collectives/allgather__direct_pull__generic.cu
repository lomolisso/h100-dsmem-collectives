/*
 * AllGather - direct pull (every CTA reads every peer's vector)
 *
 * 1. Op: every CTA ends with the rank-ordered concatenation of the N M-element vectors.
 *
 * 2. Algorithm: after one cluster barrier publishes the inputs, CTA b copies slot r of its gathered
 *    buffer directly out of CTA r's input buffer with DSMEM loads, for every r != b. No sender
 *    involvement, no mbarriers; the receiver controls the traffic (dual of the replicated push).
 *
 * 3. Pattern: fully connected pull, 1 round. Remote read bytes per CTA: (N-1)*M*sizeof(T).
 *    Barriers: 2 barrier.cluster (publish inputs; all reads done before anyone exits/reuses).
 *    Primitive: mapa + ld.shared::cluster.v4 (16 B per load); all threads issue loads in parallel.
 */
#include "dsmem_primitives.cuh"
using namespace dsmem;

template <int N, typename T, int M>
__device__ void allgather_direct_pull(const T* mine, T* D, Stamps st) {
    static_assert((M * sizeof(T)) % 16 == 0, "vectorised 16-byte remote loads");
    constexpr int V16 = M * sizeof(T) / 16;                    // 16-byte units per vector
    const uint32_t rank = cluster_rank();
    const uint32_t mine_a = smem_u32(mine);
    cluster_sync();                                            // inputs of all peers are ready
    for (int idx = threadIdx.x; idx < N * V16; idx += blockDim.x) {
        const int r = idx / V16, u = idx % V16;
        if (r == (int)rank) continue;                          // own slot is copied locally below
        const uint4 v = ld_cluster_v4(mapa(mine_a + u * 16, r));   // 16 B of rank r's vector (DSMEM load)
        reinterpret_cast<uint4*>(D + r * M)[u] = v;
    }
    for (int i = threadIdx.x; i < M; i += blockDim.x) D[rank * M + i] = mine[i];
    st.round(0);
    cluster_sync();                                            // every peer finished reading my input
}

__global__ void __cluster_dims__(CLUSTER_N, 1, 1)
collective_kernel(const ELEM_T* in, ELEM_T* out, unsigned long long* ts, unsigned* smid_out) {
    constexpr int N = CLUSTER_N, M = MSG_ELEMS;
    __shared__ __align__(16) ELEM_T mine[M];
    __shared__ __align__(16) ELEM_T D[N * M];
    const uint32_t rank = cluster_rank();
    Stamps st{ts + rank * kStamps};

    load_block<ELEM_T, M>(mine, in + rank * M);
    if (threadIdx.x == 0) smid_out[rank] = smid();
    cluster_sync();
    st.mark(0);
    allgather_direct_pull<N, ELEM_T, M>(mine, D, st);
    st.mark(1);
    store_block<ELEM_T, N * M>(out + rank * N * M, D);
    cluster_sync();
}
