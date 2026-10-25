/*
 * AllReduce - one-shot pull (every CTA reads every peer's vector and reduces locally), TensorRT-LLM "one-shot" style
 *
 * 1. Op: every CTA ends with the element-wise sum of N M-element vectors.
 *
 * 2. Algorithm: no data is sent. After a cluster barrier publishes the inputs, each CTA loads
 *    element e from all N peers' buffers through DSMEM and accumulates in registers, writing the
 *    total into its own result buffer. Reading is fully parallel across threads and peers; nothing
 *    has to be forwarded and no CTA is a bottleneck. Mirrors TensorRT-LLM's one-shot NVLink
 *    all-reduce (each rank reads all ranks' inputs) with DSMEM loads instead of NVLink loads.
 *
 * 3. Pattern: fully connected pull, 1 round. Remote read bytes per CTA: (N-1)*M*sizeof(T)
 *    (N times the data of a reduce-scatter based algorithm, but only 2 barriers and no mbarriers).
 *    Primitive: mapa + ld.shared::cluster (vectorised 16 B here) - or equivalently a generic load
 *    through cluster.map_shared_rank(); barrier.cluster before (inputs ready) and after (reads done).
 *
 * Source: TensorRT-LLM customAllReduce one-shot; also the "pull" dual of CREDIT's replicated push.
 */
#include "dsmem_primitives.cuh"
using namespace dsmem;

template <int N, typename T, int M, class Op>
__device__ void allreduce_one_shot_pull(const T* buf, T* result, Op op, Stamps st) {
    static_assert((M * sizeof(T)) % 16 == 0, "vectorised 16-byte remote loads");
    constexpr int V = 16 / sizeof(T);                          // elements per 16-byte load
    const uint32_t buf_a = smem_u32(buf);
    cluster_sync();                                            // every peer's buffer is complete and visible
    for (int v = threadIdx.x; v < M / V; v += blockDim.x) {
        T acc[V];
        #pragma unroll
        for (int k = 0; k < V; ++k) acc[k] = Op::template identity<T>();
        #pragma unroll 1
        for (int r = 0; r < N; ++r) {
            const uint4 raw = ld_cluster_v4(mapa(buf_a + v * 16, r));   // 16 B of element block v from CTA r (DSMEM load)
            const T* vals = reinterpret_cast<const T*>(&raw);
            #pragma unroll
            for (int k = 0; k < V; ++k) acc[k] = op(acc[k], vals[k]);
        }
        #pragma unroll
        for (int k = 0; k < V; ++k) result[v * V + k] = acc[k];
    }
    st.round(0);
    cluster_sync();                                            // nobody still reads my buffer
}

__global__ void __cluster_dims__(CLUSTER_N, 1, 1)
collective_kernel(const ELEM_T* in, ELEM_T* out, unsigned long long* ts, unsigned* smid_out) {
    constexpr int N = CLUSTER_N, M = MSG_ELEMS;
    __shared__ __align__(16) ELEM_T buf[M];
    __shared__ __align__(16) ELEM_T result[M];
    const uint32_t rank = cluster_rank();
    Stamps st{ts + rank * kStamps};

    load_block<ELEM_T, M>(buf, in + rank * M);
    if (threadIdx.x == 0) smid_out[rank] = smid();
    cluster_sync();
    st.mark(0);
    allreduce_one_shot_pull<N, ELEM_T, M>(buf, result, Sum{}, st);
    st.mark(1);
    store_block<ELEM_T, M>(out + rank * M, result);
    cluster_sync();
}
