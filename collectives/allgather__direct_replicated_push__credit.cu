/*
 * AllGather - direct replicated push (CREDIT cluster_reduce_pair, gather half)
 *
 * 1. Op: every CTA ends with slots[r] = vector of rank r, for all r (rank-ordered concatenation).
 *
 * 2. Algorithm: CTA b stores its M elements into slot[b] of every CTA with plain remote
 *    stores, then one cluster barrier. That is the full all-gather; CREDIT then reduces the
 *    slots locally (see allreduce__direct_replicated_push__credit.cu). Kept as its own file
 *    because the gather alone is a one-round, N-1-message pattern that no other workload uses.
 *
 * 3. Pattern: fully connected, 1 round, N-1 remote stores of M*sizeof(T) bytes per CTA,
 *    total (N-1)*M*sizeof(T) write bytes per CTA, 1 barrier.cluster (plus exit barrier).
 *    Primitive: map_shared_rank() generic pointer + plain store, cluster.sync().
 *
 * Source: third_party/CREDIT/src/workloads/*.cu (cluster_reduce_pair lines "push").
 */
#include "dsmem_primitives.cuh"
using namespace dsmem;

template <int N, typename T, int M>
__device__ void allgather_direct_replicated_push(const T* mine, T (*slots)[M], Stamps st) {
    const uint32_t rank = cluster_rank();
    for (int idx = threadIdx.x; idx < N * M; idx += blockDim.x) {
        const int p = idx / M, e = idx % M;
        *map_ptr(&slots[rank][e], p) = mine[e];                // my element e -> slot[rank] of CTA p (remote store)
    }
    cluster_sync();                                            // all slots of all CTAs filled
    st.round(0);
}

__global__ void __cluster_dims__(CLUSTER_N, 1, 1)
collective_kernel(const ELEM_T* in, ELEM_T* out, unsigned long long* ts, unsigned* smid_out) {
    constexpr int N = CLUSTER_N, M = MSG_ELEMS;
    __shared__ ELEM_T mine[M];
    __shared__ ELEM_T slots[N][M];
    const uint32_t rank = cluster_rank();
    Stamps st{ts + rank * kStamps};

    load_block<ELEM_T, M>(mine, in + rank * M);
    if (threadIdx.x == 0) smid_out[rank] = smid();
    cluster_sync();
    st.mark(0);
    allgather_direct_replicated_push<N, ELEM_T, M>(mine, slots, st);
    st.mark(1);
    store_block<ELEM_T, N * M>(out + rank * N * M, &slots[0][0]);
    cluster_sync();
}
