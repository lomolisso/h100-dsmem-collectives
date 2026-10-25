/*
 * AllGather - ring (NCCL)
 *
 * 1. Op: every CTA ends with the rank-ordered concatenation of the N M-element vectors.
 *
 * 2. Algorithm: buffer D of N slots, own data in slot b. N-1 steps around the ring b -> b+1: at
 *    step s CTA b forwards slot (b - s) mod N (its own data at s = 0, afterwards what it received in
 *    the previous step) to its successor, which stores it in the same slot. No un-rotation needed.
 *
 * 3. Pattern: ring, one send + one receive per CTA per step, all links busy every step.
 *    Rounds: N-1. Message: M*sizeof(T). Total per CTA: (N-1)*M*sizeof(T) (bandwidth-optimal,
 *    latency linear in N).
 *    Primitive: cp.async.bulk push into the successor's slot + mbarrier complete_tx; parity wait;
 *    barrier.cluster per step so a CTA cannot run a step ahead on the shared mbarrier.
 */
#include "dsmem_primitives.cuh"
using namespace dsmem;

template <int N, typename T, int M>
__device__ void allgather_ring(T* D, uint32_t bar, Stamps st) {
    static_assert((M * sizeof(T)) % 16 == 0, "bulk copy needs a multiple of 16 bytes");
    constexpr uint32_t bytes = M * sizeof(T);
    const uint32_t rank = cluster_rank(), next = (rank + 1) % N;
    const uint32_t D_a = smem_u32(D);
    uint32_t phase = 0;
    for (int s = 0; s < N - 1; ++s) {
        const int slot = (rank - s + N) % N;                   // forward this slot
        if (threadIdx.x == 0) {
            mbar_arrive_expect_tx(bar, bytes);
            bulk_push(next, D_a + slot * bytes, D_a + slot * bytes, bytes, bar);   // -> successor's same slot
        }
        mbar_wait_parity(bar, phase); phase ^= 1;              // slot (rank - s - 1) arrived from predecessor
        cluster_sync();
        st.round(s);
    }
}

__global__ void __cluster_dims__(CLUSTER_N, 1, 1)
collective_kernel(const ELEM_T* in, ELEM_T* out, unsigned long long* ts, unsigned* smid_out) {
    constexpr int N = CLUSTER_N, M = MSG_ELEMS;
    __shared__ __align__(16) ELEM_T D[N * M];
    __shared__ __align__(8) uint64_t bar;
    const uint32_t rank = cluster_rank();
    Stamps st{ts + rank * kStamps};

    load_block<ELEM_T, M>(D + rank * M, in + rank * M);
    fence_proxy_async_smem();
    if (threadIdx.x == 0) { mbar_init(smem_u32(&bar), 1); fence_mbar_init(); smid_out[rank] = smid(); }
    cluster_sync();
    st.mark(0);
    allgather_ring<N, ELEM_T, M>(D, smem_u32(&bar), st);
    st.mark(1);
    store_block<ELEM_T, N * M>(out + rank * N * M, D);
    cluster_sync();
}
