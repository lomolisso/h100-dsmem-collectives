/*
 * ReduceScatter - ring (NCCL)
 *
 * 1. Op: CTA b ends with chunk b (M elements) of the element-wise sum of the N (N*M)-element inputs.
 *
 * 2. Algorithm: N-1 steps around the ring b -> b+1. At step s CTA b sends its running partial of
 *    chunk (b - s - 1) mod N to its successor and adds the partial of chunk (b - s - 2) mod N it
 *    receives from its predecessor. The schedule is phased so that the chunk completed on CTA b
 *    after N-1 steps is chunk b (NCCL completes chunk b+1; same algorithm, rotated start).
 *
 * 3. Pattern: ring. Rounds: N-1. Message: M*sizeof(T) (one chunk). Total per CTA: (N-1)*M*sizeof(T)
 *    (bandwidth-optimal). Primitive: cp.async.bulk push into the successor's scratch + mbarrier
 *    complete_tx; parity wait; local add; barrier.cluster per step (scratch reuse).
 */
#include "dsmem_primitives.cuh"
using namespace dsmem;

template <int N, typename T, int M, class Op>
__device__ void reducescatter_ring(T* buf /*[N*M]*/, T* scratch /*[M]*/, uint32_t bar, Op op, Stamps st) {
    static_assert((M * sizeof(T)) % 16 == 0, "bulk copy needs a multiple of 16 bytes");
    constexpr uint32_t bytes = M * sizeof(T);
    const uint32_t rank = cluster_rank(), next = (rank + 1) % N;
    const uint32_t buf_a = smem_u32(buf), scr_a = smem_u32(scratch);
    uint32_t phase = 0;
    for (int s = 0; s < N - 1; ++s) {
        const int send_c = (rank - s - 1 + N) % N, recv_c = (rank - s - 2 + 2 * N) % N;
        if (threadIdx.x == 0) {
            mbar_arrive_expect_tx(bar, bytes);
            bulk_push(next, scr_a, buf_a + send_c * bytes, bytes, bar);   // partial of chunk send_c -> successor's scratch
        }
        mbar_wait_parity(bar, phase); phase ^= 1;
        for (int i = threadIdx.x; i < M; i += blockDim.x) buf[recv_c * M + i] = op(buf[recv_c * M + i], scratch[i]);
        fence_proxy_async_smem();
        cluster_sync();
        st.round(s);
    }
}

// Shared buffers of one CTA, taken from the dynamic shared-memory window (dsmem_primitives.cuh, dyn_smem);
// the launcher passes sizeof(CollectiveSmem) as the dynamic smem size (up to 227 KiB per CTA).
struct CollectiveSmem {
    static constexpr int N = CLUSTER_N, M = MSG_ELEMS;
    __align__(16) ELEM_T buf[N * M];
    __align__(16) ELEM_T scratch[M];
    __align__(8) uint64_t bar;
};

__global__ void __cluster_dims__(CLUSTER_N, 1, 1)
collective_kernel(const ELEM_T* in, ELEM_T* out, unsigned long long* ts, unsigned* smid_out) {
    constexpr int N = CLUSTER_N, M = MSG_ELEMS;
    CollectiveSmem& sm = dyn_smem<CollectiveSmem>();
    auto& buf = sm.buf; auto& scratch = sm.scratch; auto& bar = sm.bar;
    const uint32_t rank = cluster_rank();
    Stamps st{ts + rank * kStamps};

    load_block<ELEM_T, N * M>(buf, in + rank * N * M);
    fence_proxy_async_smem();
    if (threadIdx.x == 0) { mbar_init(smem_u32(&bar), 1); fence_mbar_init(); smid_out[rank] = smid(); }
    cluster_sync();
    st.mark(0);
    reducescatter_ring<N, ELEM_T, M>(buf, scratch, smem_u32(&bar), Sum{}, st);
    st.mark(1);
    store_block<ELEM_T, M>(out + rank * M, buf + rank * M);    // chunk `rank` is complete
    cluster_sync();
}
