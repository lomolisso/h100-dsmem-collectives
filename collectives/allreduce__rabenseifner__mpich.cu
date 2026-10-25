/*
 * AllReduce - Rabenseifner (recursive-halving reduce-scatter + recursive-doubling all-gather), MPICH large-message algorithm
 *
 * 1. Op: every CTA ends with the element-wise sum of N M-element vectors.
 *
 * 2. Algorithm (N power of two). Phase 1, recursive halving: log2(N) rounds with distance
 *    d = N/2, N/4, ..., 1 and partner p = b XOR d. Each CTA keeps a live range [lo, hi) (initially the
 *    whole vector); the CTA with the lower rank keeps the lower half and sends the upper half, the
 *    other does the opposite; the received half is added into the kept half, so the live range
 *    halves every round and ends as chunk `b` of size M/N. Phase 2, recursive doubling: distance
 *    d = 1, 2, ..., N/2, partner b XOR d; each CTA sends its live range to the partner's same
 *    position and receives the adjacent range, doubling the live range until it is the whole vector.
 *
 * 3. Pattern: hypercube (XOR partners), one send and one receive per CTA per round.
 *    Rounds: 2*log2(N). Message sizes: M/2, M/4, ..., M/N, then M/N, ..., M/2 (times sizeof(T)).
 *    Total per CTA: 2(N-1)/N * M*sizeof(T) (bandwidth-optimal like ring, but log rounds).
 *    Primitive: cp.async.bulk push into the partner's scratch (phase 1) or into its buffer at the same
 *    offset (phase 2); mbarrier complete_tx + parity wait; barrier.cluster per round.
 *
 * Source: Rabenseifner 2004; Thakur, Rabenseifner, Gropp 2005 (MPICH).
 */
#include "dsmem_primitives.cuh"
using namespace dsmem;

template <int N, typename T, int M, class Op>
__device__ void allreduce_rabenseifner(T* buf, T* scratch, uint32_t bar, Op op, Stamps st) {
    static_assert((N & (N - 1)) == 0, "needs a power-of-two cluster size");
    static_assert(M % N == 0 && ((M / N) * sizeof(T)) % 16 == 0, "final chunk must be a multiple of 16 bytes");
    const uint32_t rank = cluster_rank();
    const uint32_t buf_a = smem_u32(buf), scr_a = smem_u32(scratch);
    uint32_t phase = 0;
    int lo = 0, hi = M, round = 0;
    // phase 1: recursive halving reduce-scatter
    for (int d = N / 2; d >= 1; d >>= 1, ++round) {
        const uint32_t partner = rank ^ d;                     // hypercube partner
        const int mid = (lo + hi) / 2;
        const bool upper = (rank & d) != 0;                    // keep upper half if my bit d is set
        const int send_lo = upper ? lo : mid, send_hi = upper ? mid : hi;
        const int keep_lo = upper ? mid : lo;
        const uint32_t bytes = (send_hi - send_lo) * sizeof(T);
        if (threadIdx.x == 0) {
            mbar_arrive_expect_tx(bar, bytes);
            bulk_push(partner, scr_a, buf_a + send_lo * sizeof(T), bytes, bar);   // my discarded half -> partner's scratch
        }
        mbar_wait_parity(bar, phase); phase ^= 1;              // partner's copy of my kept half arrived
        const int len = hi - mid;
        for (int i = threadIdx.x; i < len; i += blockDim.x) buf[keep_lo + i] = op(buf[keep_lo + i], scratch[i]);
        lo = keep_lo; hi = keep_lo + len;
        fence_proxy_async_smem();
        cluster_sync();                                        // scratch consumed before the next push
        st.round(round);
    }
    // phase 2: recursive doubling all-gather of the live ranges
    for (int d = 1; d < N; d <<= 1, ++round) {
        const uint32_t partner = rank ^ d;
        const int len = hi - lo;
        const uint32_t bytes = len * sizeof(T);
        if (threadIdx.x == 0) {
            mbar_arrive_expect_tx(bar, bytes);
            bulk_push(partner, buf_a + lo * sizeof(T), buf_a + lo * sizeof(T), bytes, bar);   // my range -> partner, same offset
        }
        mbar_wait_parity(bar, phase); phase ^= 1;              // partner's range landed next to mine
        if (rank & d) lo -= len; else hi += len;               // live range doubles
        cluster_sync();
        st.round(round);
    }
}

// Shared buffers of one CTA, taken from the dynamic shared-memory window (dsmem_primitives.cuh, dyn_smem);
// the launcher passes sizeof(CollectiveSmem) as the dynamic smem size (up to 227 KiB per CTA).
struct CollectiveSmem {
    static constexpr int N = CLUSTER_N, M = MSG_ELEMS;
    __align__(16) ELEM_T buf[M];
    __align__(16) ELEM_T scratch[M / 2];
    __align__(8) uint64_t bar;
};

__global__ void __cluster_dims__(CLUSTER_N, 1, 1)
collective_kernel(const ELEM_T* in, ELEM_T* out, unsigned long long* ts, unsigned* smid_out) {
    constexpr int N = CLUSTER_N, M = MSG_ELEMS;
    CollectiveSmem& sm = dyn_smem<CollectiveSmem>();
    auto& buf = sm.buf; auto& scratch = sm.scratch; auto& bar = sm.bar;
    const uint32_t rank = cluster_rank();
    Stamps st{ts + rank * kStamps};

    load_block<ELEM_T, M>(buf, in + rank * M);
    fence_proxy_async_smem();
    if (threadIdx.x == 0) { mbar_init(smem_u32(&bar), 1); fence_mbar_init(); smid_out[rank] = smid(); }
    cluster_sync();
    st.mark(0);
    allreduce_rabenseifner<N, ELEM_T, M>(buf, scratch, smem_u32(&bar), Sum{}, st);
    st.mark(1);
    store_block<ELEM_T, M>(out + rank * M, buf);
    cluster_sync();
}
