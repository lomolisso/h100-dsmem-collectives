/*
 * AllReduce - recursive doubling (ClusterFusion paper, Algorithm 1 "ClusterReduce")
 *
 * 1. Op: every CTA ends with op(x_0, ..., x_{N-1}) over M-element vectors (op = sum or max).
 *
 * 2. Algorithm: log2(N) rounds with stride s = 1, 2, 4, ..., N/2. In round k each CTA b
 *    pushes its *current* buffer to CTA (b + s) mod N and receives the current buffer of
 *    CTA (b - s) mod N into a scratch buffer, then reduces scratch into its buffer.
 *    After round k every CTA holds the partial over 2^(k+1) consecutive ranks; after
 *    log2(N) rounds it holds the total. Requires N to be a power of two.
 *
 * 3. Pattern: logical ring with doubling shift (each CTA has exactly one sender and one
 *    receiver per round). Rounds: log2(N). Message per round: the full M*sizeof(T) bytes
 *    (size does not shrink). Total DSMEM bytes per CTA: M*sizeof(T)*log2(N).
 *    Primitive: one-thread cp.async.bulk smem->remote smem push signalling the receiver's
 *    mbarrier (complete_tx); receiver spins on mbarrier parity; one barrier.cluster per round.
 *    The reduced result is written to an alternate buffer (ping-pong) because the CTA's own
 *    outgoing bulk copy may still be reading the current buffer when the incoming data arrives.
 *
 * Source: ClusterFusion (NeurIPS'25) Alg. 1. NOTE: the shipped dsm.cuh implements a
 * different schedule (see allreduce__shift_accumulate_bulk__clusterfusion.cu).
 */
#include "dsmem_primitives.cuh"
using namespace dsmem;

// Returns the buffer (buf or alt) that holds the result.
template <int N, typename T, int M, class Op>
__device__ T* allreduce_recursive_doubling(T* buf, T* alt, T* scratch, uint32_t bar, Op op, Stamps st) {
    static_assert((N & (N - 1)) == 0, "recursive doubling needs a power-of-two cluster size");
    static_assert((M * sizeof(T)) % 16 == 0, "bulk copy needs a multiple of 16 bytes");
    constexpr uint32_t bytes = M * sizeof(T);
    const uint32_t rank = cluster_rank();
    const uint32_t scr_a = smem_u32(scratch);
    T* cur = buf; T* other = alt;
    uint32_t phase = 0;
    int round = 0;
    for (int s = 1; s < N; s <<= 1, ++round) {
        const uint32_t send_to = (rank + s) % N;              // (b + stride) mod N
        if (threadIdx.x == 0) {
            mbar_arrive_expect_tx(bar, bytes);                // expect the peer's bytes on my barrier
            bulk_push(send_to, scr_a, smem_u32(cur), bytes, bar);   // push my buffer into peer's scratch, signal peer's barrier
        }
        mbar_wait_parity(bar, phase);                         // block until (b - stride)'s data has landed
        phase ^= 1;
        for (int i = threadIdx.x; i < M; i += blockDim.x) other[i] = op(cur[i], scratch[i]);   // ping-pong: cur may still be read by my push
        fence_proxy_async_smem();                             // generic writes -> visible to the next bulk copy
        cluster_sync();                                       // all pushes/reads of this round done; scratch reusable
        T* t = cur; cur = other; other = t;
        st.round(round);
    }
    return cur;
}

__global__ void __cluster_dims__(CLUSTER_N, 1, 1)
collective_kernel(const ELEM_T* in, ELEM_T* out, unsigned long long* ts, unsigned* smid_out) {
    constexpr int N = CLUSTER_N, M = MSG_ELEMS;
    __shared__ __align__(16) ELEM_T buf[M];
    __shared__ __align__(16) ELEM_T alt[M];
    __shared__ __align__(16) ELEM_T scratch[M];
    __shared__ __align__(8) uint64_t bar;
    const uint32_t rank = cluster_rank();
    Stamps st{ts + rank * kStamps};

    load_block<ELEM_T, M>(buf, in + rank * M);
    fence_proxy_async_smem();
    if (threadIdx.x == 0) { mbar_init(smem_u32(&bar), 1); fence_mbar_init(); smid_out[rank] = smid(); }
    cluster_sync();                                           // entry: inputs loaded, barriers visible cluster-wide
    st.mark(0);
    ELEM_T* res = allreduce_recursive_doubling<N, ELEM_T, M>(buf, alt, scratch, smem_u32(&bar), Sum{}, st);
    st.mark(1);
    store_block<ELEM_T, M>(out + rank * M, res);
    cluster_sync();                                           // exit: no CTA leaves while peers may still touch its smem
}
