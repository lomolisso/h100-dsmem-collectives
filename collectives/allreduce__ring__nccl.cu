/*
 * AllReduce - ring (reduce-scatter ring + all-gather ring), NCCL's default algorithm
 *
 * 1. Op: every CTA ends with the element-wise sum of N M-element vectors.
 *
 * 2. Algorithm: split the vector into N chunks of C = M/N. Logical unidirectional ring
 *    b -> (b+1) mod N. Phase 1 (reduce-scatter, N-1 steps): at step s CTA b sends chunk
 *    (b - s) mod N (its running partial) to its successor and adds the chunk (b - s - 1) mod N it
 *    receives from its predecessor into its own copy; after N-1 steps CTA b owns the fully reduced
 *    chunk (b + 1) mod N. Phase 2 (all-gather, N-1 steps): the reduced chunk travels around the
 *    ring, each CTA forwarding what it received in the previous step.
 *
 * 3. Pattern: ring, every CTA has exactly one sender and one receiver (minimal link usage).
 *    Rounds: 2(N-1). Message per round: C*sizeof(T) = M*sizeof(T)/N. Total per CTA:
 *    2(N-1)/N * M*sizeof(T) (bandwidth-optimal). Latency grows linearly with N.
 *    Primitive: one-thread cp.async.bulk push of the chunk into the successor's scratch (phase 1,
 *    receiver adds) or directly into its final slot (phase 2); mbarrier complete_tx + parity wait;
 *    barrier.cluster per step so the single scratch buffer and forwarded chunks are stable.
 *
 * Source: NCCL ring / Patarasuk & Yuan 2009 "bandwidth optimal all-reduce".
 */
#include "dsmem_primitives.cuh"
using namespace dsmem;

template <int N, typename T, int M, class Op>
__device__ void allreduce_ring(T* buf, T* scratch, uint32_t bar, Op op, Stamps st) {
    static_assert(M % N == 0, "M must split into N chunks");
    constexpr int C = M / N;
    static_assert((C * sizeof(T)) % 16 == 0, "chunk must be a multiple of 16 bytes");
    constexpr uint32_t bytes = C * sizeof(T);
    const uint32_t rank = cluster_rank();
    const uint32_t next = (rank + 1) % N;                      // ring successor
    const uint32_t buf_a = smem_u32(buf), scr_a = smem_u32(scratch);
    uint32_t phase = 0;
    int round = 0;
    // phase 1: reduce-scatter
    for (int s = 0; s < N - 1; ++s, ++round) {
        const int send_c = (rank - s + N) % N, recv_c = (rank - s - 1 + N) % N;
        if (threadIdx.x == 0) {
            mbar_arrive_expect_tx(bar, bytes);
            bulk_push(next, scr_a, buf_a + send_c * bytes, bytes, bar);   // my chunk send_c -> successor's scratch
        }
        mbar_wait_parity(bar, phase); phase ^= 1;              // predecessor's chunk recv_c arrived in scratch
        for (int i = threadIdx.x; i < C; i += blockDim.x) buf[recv_c * C + i] = op(buf[recv_c * C + i], scratch[i]);
        fence_proxy_async_smem();
        cluster_sync();                                        // scratch free + updated chunk visible to next push
        st.round(round);
    }
    // phase 2: all-gather (chunk (rank+1) mod N is complete on rank)
    for (int s = 0; s < N - 1; ++s, ++round) {
        const int send_c = (rank + 1 - s + N) % N;             // forward what completed/arrived last step
        if (threadIdx.x == 0) {
            mbar_arrive_expect_tx(bar, bytes);
            bulk_push(next, buf_a + send_c * bytes, buf_a + send_c * bytes, bytes, bar);   // into the successor's same chunk slot
        }
        mbar_wait_parity(bar, phase); phase ^= 1;
        cluster_sync();                                        // received chunk stable before it is forwarded
        st.round(round);
    }
}

__global__ void __cluster_dims__(CLUSTER_N, 1, 1)
collective_kernel(const ELEM_T* in, ELEM_T* out, unsigned long long* ts, unsigned* smid_out) {
    constexpr int N = CLUSTER_N, M = MSG_ELEMS;
    __shared__ __align__(16) ELEM_T buf[M];
    __shared__ __align__(16) ELEM_T scratch[M / N];
    __shared__ __align__(8) uint64_t bar;
    const uint32_t rank = cluster_rank();
    Stamps st{ts + rank * kStamps};

    load_block<ELEM_T, M>(buf, in + rank * M);
    fence_proxy_async_smem();
    if (threadIdx.x == 0) { mbar_init(smem_u32(&bar), 1); fence_mbar_init(); smid_out[rank] = smid(); }
    cluster_sync();
    st.mark(0);
    allreduce_ring<N, ELEM_T, M>(buf, scratch, smem_u32(&bar), Sum{}, st);
    st.mark(1);
    store_block<ELEM_T, M>(out + rank * M, buf);
    cluster_sync();
}
