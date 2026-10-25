/*
 * AllReduce - shift-and-accumulate with remote atomics (ClusterFusion shipped code, scalar path)
 *
 * 1. Op: every CTA ends with the sum (or max) of N values per element. Origin uses M = 1
 *    (RMSNorm sum of squares, softmax running max / running sum); here the threads of the CTA stride
 *    over the M elements (e = tid, tid + blockDim, ...), so any blockDim and any M work.
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
 *    round). Two barrier.cluster per round: one after snapshotting the running value into the
 *    local `snap` buffer (so no peer's atomic lands in `val` before every element was snapshotted),
 *    one after the update. Shared memory: val[M] + snap[M].
 *
 * Source: ClusterFusion include/H100/llama/kernel.cuh ("ClusterReduce" loops) and norm/kernel.cuh.
 */
#include "dsmem_primitives.cuh"
using namespace dsmem;

template <int N, typename T, int M>
__device__ void allreduce_shift_accumulate_atomic(T* val, T* snap, Stamps st) {
    static_assert(N == 2 || N == 4, "shipped schedule: 1 round for N=2 (norm kernel), 2 rounds for N=4 (llama kernel)");
    constexpr int rounds = (N == 2) ? 1 : N - 2;
    const uint32_t rank = cluster_rank();
    for (int i = 1; i <= rounds; ++i) {
        for (int e = threadIdx.x; e < M; e += blockDim.x) snap[e] = val[e];   // running value (includes earlier rounds)
        T* peer = map_ptr(val, (rank + i) % N);                            // generic pointer to peer (b+i)'s val
        cluster_sync();                                                    // everyone snapshotted before any remote add lands
        for (int e = threadIdx.x; e < M; e += blockDim.x) atomicAdd(&peer[e], snap[e]);   // remote atomic add through DSMEM
        cluster_sync();                                                    // remote adds complete before next snapshot
        st.round(i - 1);
    }
}

// Shared buffers of one CTA, taken from the dynamic shared-memory window (dsmem_primitives.cuh, dyn_smem);
// the launcher passes sizeof(CollectiveSmem) as the dynamic smem size (up to 227 KiB per CTA).
struct CollectiveSmem {
    static constexpr int N = CLUSTER_N, M = MSG_ELEMS;
    ELEM_T val[M];
    ELEM_T snap[M];   // per-round snapshot of the running value
};

__global__ void __cluster_dims__(CLUSTER_N, 1, 1)
collective_kernel(const ELEM_T* in, ELEM_T* out, unsigned long long* ts, unsigned* smid_out) {
    constexpr int N = CLUSTER_N, M = MSG_ELEMS;
    CollectiveSmem& sm = dyn_smem<CollectiveSmem>();
    auto& val = sm.val; auto& snap = sm.snap;
    const uint32_t rank = cluster_rank();
    Stamps st{ts + rank * kStamps};

    load_block<ELEM_T, M>(val, in + rank * M);
    if (threadIdx.x == 0) smid_out[rank] = smid();
    cluster_sync();
    st.mark(0);
    allreduce_shift_accumulate_atomic<N, ELEM_T, M>(val, snap, st);
    st.mark(1);
    store_block<ELEM_T, M>(out + rank * M, val);
    cluster_sync();
}
