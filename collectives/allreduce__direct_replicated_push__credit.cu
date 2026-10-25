/*
 * AllReduce - direct replicated push, "one-shot" (CREDIT cluster_reduce_pair; also hpc-ops rmsNormLamport)
 *
 * 1. Op: every CTA ends with the element-wise sum of N M-element vectors (origin: M = 2 floats,
 *    the per-stage statistics of a row-wise reduction-reuse kernel).
 *
 * 2. Algorithm: every CTA owns a slot array slots[N][M]. CTA b writes its vector into slot[b]
 *    of *every* CTA (including itself) with plain remote stores, one barrier, then every CTA
 *    reduces its N local slots independently. No root, no forwarding, no second round of
 *    remote reads ("replicated all-gather + local reduce").
 *
 * 3. Pattern: fully connected, single round. Messages: N-1 remote stores of M*sizeof(T) bytes
 *    per CTA; total DSMEM write bytes per CTA (N-1)*M*sizeof(T), remote read bytes 0
 *    (paper: D_w = s*P*(P-1)*q, D_r = 0, K = 2 barriers). Barriers: 2 barrier.cluster
 *    (one exposes the pushed slots, one protects slot reuse).
 *    Primitive: cooperative-groups map_shared_rank() generic pointer + ordinary store;
 *    cluster.sync() provides release/acquire visibility.
 *
 * Source: third_party/CREDIT/src/workloads/*.cu (cluster_reduce_pair, all six workloads);
 * same pattern in third_party/hpc-ops/src/allreduce/fuse_allreduce_rmsnorm_low_latency.cu
 * (rmsNormLamport, one float, N <= 8, barrier.cluster arrive/wait).
 */
#include "dsmem_primitives.cuh"
using namespace dsmem;

template <int N, typename T, int M, class Op>
__device__ void allreduce_direct_replicated_push(const T* mine, T (*slots)[M], T* result, Op op, Stamps st) {
    const uint32_t rank = cluster_rank();
    // push: element e of my vector into slot[rank][e] of peer p, for every p
    for (int idx = threadIdx.x; idx < N * M; idx += blockDim.x) {
        const int p = idx / M, e = idx % M;
        T* remote = map_ptr(&slots[rank][e], p);               // generic DSMEM pointer into CTA p
        *remote = mine[e];                                     // plain remote store (origin: *remote_a = block_value.a)
    }
    cluster_sync();                                            // first barrier: all pushed partials visible
    st.round(0);
    for (int e = threadIdx.x; e < M; e += blockDim.x) {        // reduce my N local slots
        T acc = slots[0][e];
        #pragma unroll
        for (int r = 1; r < N; ++r) acc = op(acc, slots[r][e]);
        result[e] = acc;
    }
    cluster_sync();                                            // second barrier: slots may be reused by the next stage
}

// Shared buffers of one CTA, taken from the dynamic shared-memory window (dsmem_primitives.cuh, dyn_smem);
// the launcher passes sizeof(CollectiveSmem) as the dynamic smem size (up to 227 KiB per CTA).
struct CollectiveSmem {
    static constexpr int N = CLUSTER_N, M = MSG_ELEMS;
    ELEM_T mine[M];
    ELEM_T slots[N][M];
    ELEM_T result[M];
};

__global__ void __cluster_dims__(CLUSTER_N, 1, 1)
collective_kernel(const ELEM_T* in, ELEM_T* out, unsigned long long* ts, unsigned* smid_out) {
    constexpr int N = CLUSTER_N, M = MSG_ELEMS;
    CollectiveSmem& sm = dyn_smem<CollectiveSmem>();
    auto& mine = sm.mine; auto& slots = sm.slots; auto& result = sm.result;
    const uint32_t rank = cluster_rank();
    Stamps st{ts + rank * kStamps};

    load_block<ELEM_T, M>(mine, in + rank * M);
    if (threadIdx.x == 0) smid_out[rank] = smid();
    cluster_sync();
    st.mark(0);
    allreduce_direct_replicated_push<N, ELEM_T, M>(mine, slots, result, Sum{}, st);
    st.mark(1);
    store_block<ELEM_T, M>(out + rank * M, result);
    cluster_sync();
}
