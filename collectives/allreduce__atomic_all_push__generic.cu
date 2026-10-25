/*
 * AllReduce - remote-atomic all-push (every CTA adds its vector into every peer's accumulator)
 *
 * 1. Op: every CTA ends with the element-wise sum of N M-element vectors (sum only: relies on
 *    hardware atomics; summation order is non-deterministic across runs).
 *
 * 2. Algorithm: each CTA's accumulator starts as its own input. After a barrier, every thread issues
 *    one remote atomic add per (peer, element) into the peer's accumulator; a second barrier
 *    completes the operation. There are no slots, no local reduction pass and no mbarriers: the
 *    reduction happens inside the destination SM's shared-memory atomic units.
 *
 * 3. Pattern: fully connected, 1 round, (N-1)*M remote atomics of sizeof(T) per CTA, total
 *    (N-1)*M*sizeof(T) write traffic per CTA; 2 barrier.cluster. Contention: every accumulator word
 *    receives N-1 atomics.
 *    Primitive: mapa + red.shared::cluster.add.f32 (fire-and-forget reduction) for float, generic
 *    atomicAdd through cluster.map_shared_rank() otherwise. This generalises ClusterFusion's scalar
 *    remote atomicAdd to a single-round, all-to-all schedule.
 *
 * Source: generic (same mechanism as ClusterFusion kernel.cuh scalar ClusterReduce, one-shot schedule).
 */
#include "dsmem_primitives.cuh"
#include <type_traits>
using namespace dsmem;

template <int N, typename T, int M>
__device__ void allreduce_atomic_all_push(const T* mine, T* acc, Stamps st) {
    const uint32_t rank = cluster_rank();
    const uint32_t acc_a = smem_u32(acc);
    cluster_sync();                                            // all accumulators initialised before any remote add
    for (int idx = threadIdx.x; idx < (N - 1) * M; idx += blockDim.x) {
        const uint32_t peer = (rank + 1 + idx / M) % N;        // skip self
        const int e = idx % M;
        if constexpr (std::is_same<T, float>::value) {
            red_cluster_add_f32(mapa(acc_a + e * sizeof(T), peer), mine[e]);   // red.shared::cluster.add.f32 into peer's acc[e]
        } else {
            atomicAdd(map_ptr(&acc[e], peer), mine[e]);        // generic remote atomic through a DSMEM pointer
        }
    }
    st.round(0);
    cluster_sync();                                            // all remote adds performed and visible
}

__global__ void __cluster_dims__(CLUSTER_N, 1, 1)
collective_kernel(const ELEM_T* in, ELEM_T* out, unsigned long long* ts, unsigned* smid_out) {
    constexpr int N = CLUSTER_N, M = MSG_ELEMS;
    __shared__ ELEM_T mine[M];
    __shared__ ELEM_T acc[M];
    const uint32_t rank = cluster_rank();
    Stamps st{ts + rank * kStamps};

    load_block<ELEM_T, M>(mine, in + rank * M);
    for (int i = threadIdx.x; i < M; i += blockDim.x) acc[i] = mine[i];
    if (threadIdx.x == 0) smid_out[rank] = smid();
    cluster_sync();
    st.mark(0);
    allreduce_atomic_all_push<N, ELEM_T, M>(mine, acc, st);
    st.mark(1);
    store_block<ELEM_T, M>(out + rank * M, acc);
    cluster_sync();
}
