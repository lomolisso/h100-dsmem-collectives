/*
 * ReduceScatter - recursive halving (MPICH, first phase of Rabenseifner's all-reduce)
 *
 * 1. Op: CTA b ends with chunk b (M elements) of the element-wise sum of the N (N*M)-element inputs.
 *
 * 2. Algorithm (N power of two): log2(N) rounds with distance d = N/2, N/4, ..., 1, partner b XOR d.
 *    Each CTA keeps a live range (initially everything): the lower rank keeps the lower half and
 *    sends the upper half, the higher rank the opposite; the received half is added into the kept
 *    half. The live range halves each round and ends as chunk b.
 *
 * 3. Pattern: hypercube. Rounds: log2(N). Messages: N*M/2, N*M/4, ..., M elements
 *    (times sizeof(T)). Total per CTA: (N-1)*M*sizeof(T) (bandwidth-optimal, log rounds).
 *    Primitive: cp.async.bulk push of the discarded half into the partner's scratch + mbarrier
 *    complete_tx; parity wait; local add; barrier.cluster per round.
 */
#include "dsmem_primitives.cuh"
using namespace dsmem;

template <int N, typename T, int M, class Op>
__device__ void reducescatter_recursive_halving(T* buf /*[N*M]*/, T* scratch /*[N*M/2]*/, uint32_t bar, Op op, Stamps st) {
    static_assert((N & (N - 1)) == 0, "needs a power-of-two cluster size");
    static_assert((M * sizeof(T)) % 16 == 0, "bulk copy needs a multiple of 16 bytes");
    const uint32_t rank = cluster_rank();
    const uint32_t buf_a = smem_u32(buf), scr_a = smem_u32(scratch);
    uint32_t phase = 0;
    int lo = 0, hi = N * M, round = 0;
    for (int d = N / 2; d >= 1; d >>= 1, ++round) {
        const uint32_t partner = rank ^ d;
        const int mid = (lo + hi) / 2, len = hi - mid;
        const bool upper = (rank & d) != 0;
        const int send_lo = upper ? lo : mid, keep_lo = upper ? mid : lo;
        const uint32_t bytes = len * sizeof(T);
        if (threadIdx.x == 0) {
            mbar_arrive_expect_tx(bar, bytes);
            bulk_push(partner, scr_a, buf_a + send_lo * sizeof(T), bytes, bar);   // discarded half -> partner's scratch
        }
        mbar_wait_parity(bar, phase); phase ^= 1;
        for (int i = threadIdx.x; i < len; i += blockDim.x) buf[keep_lo + i] = op(buf[keep_lo + i], scratch[i]);
        lo = keep_lo; hi = keep_lo + len;
        fence_proxy_async_smem();
        cluster_sync();
        st.round(round);
    }
}

// Shared buffers of one CTA, taken from the dynamic shared-memory window (dsmem_primitives.cuh, dyn_smem);
// the launcher passes sizeof(CollectiveSmem) as the dynamic smem size (up to 227 KiB per CTA).
struct CollectiveSmem {
    static constexpr int N = CLUSTER_N, M = MSG_ELEMS;
    __align__(16) ELEM_T buf[N * M];
    __align__(16) ELEM_T scratch[N * M / 2];
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
    reducescatter_recursive_halving<N, ELEM_T, M>(buf, scratch, smem_u32(&bar), Sum{}, st);
    st.mark(1);
    store_block<ELEM_T, M>(out + rank * M, buf + rank * M);    // live range ended as chunk `rank`
    cluster_sync();
}
