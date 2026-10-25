/*
 * ReduceScatter - direct push (one round, every CTA scatters its chunks to their owners)
 *
 * 1. Op: CTA b ends with chunk b (M elements) of the element-wise sum of the N (N*M)-element inputs.
 *
 * 2. Algorithm: CTA b pushes chunk j into slot[b] of CTA j for every j != b (N-1 simultaneous
 *    messages), waits until the N-1 partials of chunk b have arrived in its own slots, then reduces
 *    them with its local copy. This is the first "shot" of the two-shot all-reduce and the row phase
 *    of the 2D-mesh all-reduce.
 *
 * 3. Pattern: fully connected push, 1 round. Messages: N-1 of M*sizeof(T) per CTA, total
 *    (N-1)*M*sizeof(T) per CTA (bandwidth-optimal, minimal rounds, N-1 concurrent flows per SM).
 *    Primitive: N-1 cp.async.bulk pushes completing on ONE receiver mbarrier armed with the
 *    aggregate byte count; parity wait; exit barrier.cluster.
 */
#include "dsmem_primitives.cuh"
using namespace dsmem;

template <int N, typename T, int M, class Op>
__device__ void reducescatter_direct_push(const T* buf /*[N*M]*/, T (*slots)[M], T* result, uint32_t bar, Op op, Stamps st) {
    static_assert((M * sizeof(T)) % 16 == 0, "bulk copy needs a multiple of 16 bytes");
    constexpr uint32_t bytes = M * sizeof(T);
    const uint32_t rank = cluster_rank();
    const uint32_t buf_a = smem_u32(buf), slots_a = smem_u32(&slots[0][0]);
    if (threadIdx.x == 0) {
        mbar_arrive_expect_tx(bar, (N - 1) * bytes);           // N-1 partials of my chunk will arrive
        for (uint32_t j = 1; j < N; ++j) {
            const uint32_t dst = (rank + j) % N;
            bulk_push(dst, slots_a + rank * bytes, buf_a + dst * bytes, bytes, bar);   // chunk dst -> slot[rank] of CTA dst
        }
    }
    mbar_wait_parity(bar, 0);
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        T acc = buf[rank * M + i];
        #pragma unroll
        for (int r = 0; r < N; ++r) if (r != (int)rank) acc = op(acc, slots[r][i]);
        result[i] = acc;
    }
    st.round(0);
}

__global__ void __cluster_dims__(CLUSTER_N, 1, 1)
collective_kernel(const ELEM_T* in, ELEM_T* out, unsigned long long* ts, unsigned* smid_out) {
    constexpr int N = CLUSTER_N, M = MSG_ELEMS;
    __shared__ __align__(16) ELEM_T buf[N * M];
    __shared__ __align__(16) ELEM_T slots[N][M];
    __shared__ ELEM_T result[M];
    __shared__ __align__(8) uint64_t bar;
    const uint32_t rank = cluster_rank();
    Stamps st{ts + rank * kStamps};

    load_block<ELEM_T, N * M>(buf, in + rank * N * M);
    fence_proxy_async_smem();
    if (threadIdx.x == 0) { mbar_init(smem_u32(&bar), 1); fence_mbar_init(); smid_out[rank] = smid(); }
    cluster_sync();
    st.mark(0);
    reducescatter_direct_push<N, ELEM_T, M>(buf, slots, result, smem_u32(&bar), Sum{}, st);
    st.mark(1);
    store_block<ELEM_T, M>(out + rank * M, result);
    cluster_sync();                                            // peers' pushes into my slots all landed; keep smem alive
}
