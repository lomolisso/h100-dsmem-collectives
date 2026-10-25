/*
 * AllReduce - pull reduce-scatter + push all-gather (hpc-ops topk_filtered cluster_allreduce_histogram)
 *
 * 1. Op: every CTA ends with the element-wise sum of N M-element vectors (origin: a 2048-bin
 *    int32 radix histogram replicated in each of the 8 CTAs that scan one row of logits).
 *
 * 2. Algorithm: two phases, no forwarding. Rank b owns chunk b of C = M/N elements.
 *    Phase 1 (reduce-scatter by pulling): for each owned element, N consecutive lanes each
 *    *load* that element from a distinct peer's buffer through DSMEM, then reduce the N
 *    values inside the lane group with shuffles. Phase 2 (all-gather by pushing): the same
 *    lanes *store* the reduced value back into the same element of every peer. One cluster
 *    barrier before (inputs stable) and one after (results visible).
 *
 * 3. Pattern: fully connected in both phases, 1 communication round per phase.
 *    Remote read bytes per CTA: (N-1)*C*sizeof(T); remote write bytes: (N-1)*C*sizeof(T);
 *    total (N-1)*2*M/N*sizeof(T) - the reduce-scatter/all-gather lower bound.
 *    Barriers: 2 barrier.cluster. Requires N | 32 (shuffle groups) and N | M.
 *    Primitive: map_shared_rank() generic pointer, ordinary ld then st to the same remote
 *    address, __shfl_xor_sync over the N-lane group, cluster.sync().
 *
 * Source: third_party/hpc-ops/src/topk/topk_filtered.cu, cluster_allreduce_histogram
 * (lines ~2034-2063), N = 8, RADIX = 2048, 1024 threads.
 */
#include "dsmem_primitives.cuh"
using namespace dsmem;

template <int N, typename T, int M, class Op>
__device__ void allreduce_pull_rs_push_ag(T* buf, Op op, Stamps st) {
    static_assert(32 % N == 0, "lane groups of N lanes must tile a warp");
    static_assert(M % N == 0, "every rank owns M/N elements");
    constexpr int C = M / N;
    const uint32_t rank = cluster_rank();
    cluster_sync();                                            // all peers' buffers are complete before anyone reads them
    // all threads run the same trip count so every lane takes part in the shuffles
    for (int base = 0; base < C * N; base += blockDim.x) {
        const int linear = base + threadIdx.x;
        const bool active = linear < C * N;
        const int e = rank * C + (active ? linear : 0) / N;    // owned element
        const int peer = linear % N;                           // each lane of the group targets one peer
        T* addr = active ? map_ptr(&buf[e], peer) : nullptr;   // DSMEM pointer into peer's copy of element e
        T v = active ? *addr : Op::template identity<T>();     // phase 1: pull peer's partial
        #pragma unroll
        for (int off = N / 2; off > 0; off >>= 1)              // reduce across the N-lane group
            v = op(v, __shfl_xor_sync(0xffffffffu, v, off, N));
        if (active) *addr = v;                                 // phase 2: push the reduced value back to the peer
    }
    st.round(0);
    cluster_sync();                                            // every element of every copy now holds the total
}

// Shared buffers of one CTA, taken from the dynamic shared-memory window (dsmem_primitives.cuh, dyn_smem);
// the launcher passes sizeof(CollectiveSmem) as the dynamic smem size (up to 227 KiB per CTA).
struct CollectiveSmem {
    static constexpr int N = CLUSTER_N, M = MSG_ELEMS;
    ELEM_T buf[M];
};

__global__ void __cluster_dims__(CLUSTER_N, 1, 1)
collective_kernel(const ELEM_T* in, ELEM_T* out, unsigned long long* ts, unsigned* smid_out) {
    constexpr int N = CLUSTER_N, M = MSG_ELEMS;
    CollectiveSmem& sm = dyn_smem<CollectiveSmem>();
    auto& buf = sm.buf;
    const uint32_t rank = cluster_rank();
    Stamps st{ts + rank * kStamps};

    load_block<ELEM_T, M>(buf, in + rank * M);
    if (threadIdx.x == 0) smid_out[rank] = smid();
    cluster_sync();
    st.mark(0);
    allreduce_pull_rs_push_ag<N, ELEM_T, M>(buf, Sum{}, st);
    st.mark(1);
    store_block<ELEM_T, M>(out + rank * M, buf);
    cluster_sync();
}
