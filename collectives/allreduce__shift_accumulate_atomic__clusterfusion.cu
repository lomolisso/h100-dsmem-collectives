/*
 * AllReduce - shift-and-accumulate with remote atomics (ClusterFusion shipped code, scalar path)
 *
 * 1. Op: every CTA ends with the sum (or max) of N values per element. Origin uses M = 1
 *    (RMSNorm sum of squares, softmax running max / running sum); here thread e owns element e.
 *
 * 2. Algorithm: same shift schedule as the bulk variant. Round i (i = 1 .. N-2) CTA b adds its
 *    *running* value into the value held by CTA (b + i) mod N. Rank b's value after round 1 is
 *    x_b + x_{b-1}; after round 2 also + (x_{b-2} + x_{b-3}) - complete for N = 4. The shipped
 *    RMSNorm kernel (CLUSTER_SIZE 2) uses a single shift-by-1 round, handled here for N == 2.
 *
 * 3. Pattern: ring with shift i, one scalar message per element per round, N-2 rounds (1 for
 *    N = 2). Message size: sizeof(T) per element. Total per CTA: M*sizeof(T)*(N-2).
 *    Primitive: cooperative-groups cluster.map_shared_rank() gives a generic pointer into the
 *    peer's smem; the update is a remote atomicAdd (sum) or a plain remote read-modify-write
 *    store (max: *p = fmaxf(*p, v), race-free because each slot has exactly one writer per
 *    round). Two barrier.cluster per round: one after snapshotting the running value (so the
 *    atomic does not race with the snapshot read), one after the update.
 *
 * Source: ClusterFusion include/H100/llama/kernel.cuh ("ClusterReduce" loops) and norm/kernel.cuh.
 */
#include "dsmem_primitives.cuh"
using namespace dsmem;

template <int N, typename T, int M>
__device__ void allreduce_shift_accumulate_atomic(T* val, Stamps st) {
    static_assert(N == 2 || N == 4, "shipped schedule: 1 round for N=2 (norm kernel), 2 rounds for N=4 (llama kernel)");
    constexpr int rounds = (N == 2) ? 1 : N - 2;
    const uint32_t rank = cluster_rank();
    for (int i = 1; i <= rounds; ++i) {
        T snapshot = T(0);
        T* dst = nullptr;
        if (threadIdx.x < M) {
            snapshot = val[threadIdx.x];                                   // running value (includes earlier rounds)
            dst = map_ptr(&val[threadIdx.x], (rank + i) % N);              // generic pointer into peer (b+i)'s smem
        }
        cluster_sync();                                                    // everyone snapshotted before any remote add lands
        if (threadIdx.x < M) atomicAdd(dst, snapshot);                     // remote atomic add through DSMEM
        cluster_sync();                                                    // remote adds complete before next snapshot
        st.round(i - 1);
    }
}

__global__ void __cluster_dims__(CLUSTER_N, 1, 1)
collective_kernel(const ELEM_T* in, ELEM_T* out, unsigned long long* ts, unsigned* smid_out) {
    constexpr int N = CLUSTER_N, M = MSG_ELEMS;
    static_assert(M <= 1024, "one element per thread in this variant");
    __shared__ ELEM_T val[M];
    const uint32_t rank = cluster_rank();
    Stamps st{ts + rank * kStamps};

    load_block<ELEM_T, M>(val, in + rank * M);
    if (threadIdx.x == 0) smid_out[rank] = smid();
    cluster_sync();
    st.mark(0);
    allreduce_shift_accumulate_atomic<N, ELEM_T, M>(val, st);
    st.mark(1);
    store_block<ELEM_T, M>(out + rank * M, val);
    cluster_sync();
}
