/*
 * AllReduce - shift-and-accumulate bulk push (ClusterFusion shipped code, dsm.cuh cluster_reduce)
 *
 * 1. Op: every CTA ends with the element-wise sum of the N M-element vectors.
 *
 * 2. Algorithm (as shipped, not Alg. 1 of the paper): rounds i = 1 .. N-2. In round i CTA b
 *    pushes its *running* buffer to CTA (b + i) mod N and receives from (b - i) mod N, then
 *    adds the received buffer into its own. Because the running buffer is forwarded, round 1
 *    delivers {b-1}, round 2 delivers {b-2, b-3}: with N = 4 the two rounds cover all ranks.
 *    The loop bound `i < N-1` is only correct for N == 4 (N = 2 runs zero rounds, N >= 8
 *    double-counts), which matches the hard-coded CLUSTER_SIZE 4; enforced by static_assert.
 *
 * 3. Pattern: ring with shift i (one sender and one receiver per CTA per round).
 *    Rounds: N-2 (= 2). Message per round: full M*sizeof(T) bytes. Total per CTA: 2*M*sizeof(T).
 *    Primitive: thread 0 does mbarrier.arrive.expect_tx on its own barrier, maps destination
 *    buffer + barrier with mapa, issues cp.async.bulk shared::cta -> shared::cluster with
 *    complete_tx; all threads spin on mbarrier parity (i ^ 1); local add; barrier.cluster per round.
 *
 * Source: third_party/ClusterFusion/include/dsm.cuh (Stage::LINEAR / ATTN variants), used
 * for partial QKV (3*128 halves) and partial attention output (128 halves) in the Llama kernel.
 */
#include "dsmem_primitives.cuh"
using namespace dsmem;

template <int N, typename T, int M>
__device__ void allreduce_shift_accumulate(T* buf, T* scratch, uint32_t bar, Stamps st) {
    static_assert(N == 4, "shipped loop bound (i < N-1) only reduces correctly for N == 4");
    static_assert((M * sizeof(T)) % 16 == 0, "bulk copy needs a multiple of 16 bytes");
    constexpr uint32_t bytes = M * sizeof(T);
    const uint32_t rank = cluster_rank();
    const uint32_t buf_a = smem_u32(buf), scr_a = smem_u32(scratch);
    for (int i = 1; i < N - 1; ++i) {
        if (threadIdx.x == 0) {
            mbar_arrive_expect_tx(bar, bytes);                // arm my barrier for `bytes` incoming
            const uint32_t dst_cta = (rank + i) % N;          // shift partner
            bulk_push(dst_cta, scr_a, buf_a, bytes, bar);     // mapa(dst)+mapa(bar), cp.async.bulk ... complete_tx
        }
        mbar_wait_parity(bar, (i ^ 1) & 1);                   // origin waits parity i^1: round 1 -> 0, round 2 -> 1
        for (int e = threadIdx.x; e < M; e += blockDim.x) buf[e] += scratch[e];   // local reduce-add
        fence_proxy_async_smem();                             // (added) order generic writes before next bulk read
        cluster_sync();                                       // origin: cluster.sync() ends every round
        st.round(i - 1);
    }
}

__global__ void __cluster_dims__(CLUSTER_N, 1, 1)
collective_kernel(const ELEM_T* in, ELEM_T* out, unsigned long long* ts, unsigned* smid_out) {
    constexpr int N = CLUSTER_N, M = MSG_ELEMS;
    __shared__ __align__(16) ELEM_T buf[M];
    __shared__ __align__(16) ELEM_T scratch[M];
    __shared__ __align__(8) uint64_t bar;
    const uint32_t rank = cluster_rank();
    Stamps st{ts + rank * kStamps};

    load_block<ELEM_T, M>(buf, in + rank * M);
    fence_proxy_async_smem();
    if (threadIdx.x == 0) { mbar_init(smem_u32(&bar), 1); fence_mbar_init(); smid_out[rank] = smid(); }   // origin: init once, count 1
    cluster_sync();
    st.mark(0);
    allreduce_shift_accumulate<N, ELEM_T, M>(buf, scratch, smem_u32(&bar), st);
    st.mark(1);
    store_block<ELEM_T, M>(out + rank * M, buf);
    cluster_sync();
}
