/*
 * ReduceScatter - direct pull (every CTA reads its chunk from every peer and reduces locally)
 *
 * 1. Op: CTA b ends with chunk b (M elements) of the element-wise sum of the N (N*M)-element inputs.
 *
 * 2. Algorithm: after a cluster barrier, CTA b loads chunk b of every peer's input through DSMEM,
 *    accumulating in registers; nothing is pushed. Same traffic as the direct push but the receiver
 *    paces the transfer and no mbarrier is needed (hpc-ops' histogram reduce uses this phase).
 *
 * 3. Pattern: fully connected pull, 1 round. Remote read bytes per CTA: (N-1)*M*sizeof(T).
 *    Barriers: 2 barrier.cluster. Primitive: mapa + ld.shared::cluster.v4.
 */
#include "dsmem_primitives.cuh"
using namespace dsmem;

template <int N, typename T, int M, class Op>
__device__ void reducescatter_direct_pull(const T* buf /*[N*M]*/, T* result, Op op, Stamps st) {
    static_assert((M * sizeof(T)) % 16 == 0, "vectorised 16-byte remote loads");
    constexpr int V = 16 / sizeof(T);
    const uint32_t rank = cluster_rank();
    const uint32_t chunk_a = smem_u32(buf + rank * M);         // my chunk's offset is the same in every CTA
    cluster_sync();                                            // all inputs published
    for (int v = threadIdx.x; v < M / V; v += blockDim.x) {
        T acc[V];
        #pragma unroll
        for (int k = 0; k < V; ++k) acc[k] = Op::template identity<T>();
        #pragma unroll 1
        for (int r = 0; r < N; ++r) {
            const uint4 raw = ld_cluster_v4(mapa(chunk_a + v * 16, r));   // chunk `rank` of CTA r (DSMEM load)
            const T* vals = reinterpret_cast<const T*>(&raw);
            #pragma unroll
            for (int k = 0; k < V; ++k) acc[k] = op(acc[k], vals[k]);
        }
        #pragma unroll
        for (int k = 0; k < V; ++k) result[v * V + k] = acc[k];
    }
    st.round(0);
    cluster_sync();                                            // all remote reads of my input done
}

// Shared buffers of one CTA, taken from the dynamic shared-memory window (dsmem_primitives.cuh, dyn_smem);
// the launcher passes sizeof(CollectiveSmem) as the dynamic smem size (up to 227 KiB per CTA).
struct CollectiveSmem {
    static constexpr int N = CLUSTER_N, M = MSG_ELEMS;
    __align__(16) ELEM_T buf[N * M];
    __align__(16) ELEM_T result[M];
};

__global__ void __cluster_dims__(CLUSTER_N, 1, 1)
collective_kernel(const ELEM_T* in, ELEM_T* out, unsigned long long* ts, unsigned* smid_out) {
    constexpr int N = CLUSTER_N, M = MSG_ELEMS;
    CollectiveSmem& sm = dyn_smem<CollectiveSmem>();
    auto& buf = sm.buf; auto& result = sm.result;
    const uint32_t rank = cluster_rank();
    Stamps st{ts + rank * kStamps};

    load_block<ELEM_T, N * M>(buf, in + rank * N * M);
    if (threadIdx.x == 0) smid_out[rank] = smid();
    cluster_sync();
    st.mark(0);
    reducescatter_direct_pull<N, ELEM_T, M>(buf, result, Sum{}, st);
    st.mark(1);
    store_block<ELEM_T, M>(out + rank * M, result);
    cluster_sync();
}
