/*
 * AllReduce - rooted: binomial-tree reduce to rank 0 followed by binomial-tree broadcast (MPICH short-message algorithm)
 *
 * 1. Op: every CTA ends with the element-wise sum of N M-element vectors.
 *
 * 2. Algorithm (N power of two). Reduce: rounds k = 0..log2(N)-1, distance d = 2^k. A CTA whose bit
 *    d is set sends its running vector to partner b - d and drops out; a CTA whose bit is clear
 *    receives from b + d and accumulates. After log2(N) rounds rank 0 holds the total. Broadcast:
 *    the mirror image, d = N/2 down to 1: a CTA that already holds the result (b mod 2d == 0) sends
 *    it to b + d. Simple, but the root's link carries log2(N) full messages and only half the CTAs
 *    are active in each round.
 *
 * 3. Pattern: binomial tree (rooted), log2(N) + log2(N) rounds, every message is the full
 *    M*sizeof(T). Total DSMEM traffic 2(N-1)*M*sizeof(T) across the cluster (per-CTA load is uneven).
 *    Primitive: cp.async.bulk push into the parent's scratch / child's buffer + mbarrier complete_tx;
 *    receivers wait on parity; barrier.cluster per round keeps the schedule and scratch simple.
 *
 * Source: MPICH MPI_Allreduce for short messages (reduce + bcast), Thakur et al. 2005.
 */
#include "dsmem_primitives.cuh"
using namespace dsmem;

template <int N, typename T, int M, class Op>
__device__ void allreduce_reduce_bcast_binomial(T* buf, T* scratch, uint32_t bar, Op op, Stamps st) {
    static_assert((N & (N - 1)) == 0, "binomial tree as written needs a power-of-two cluster size");
    static_assert((M * sizeof(T)) % 16 == 0, "bulk copy needs a multiple of 16 bytes");
    constexpr uint32_t bytes = M * sizeof(T);
    const uint32_t rank = cluster_rank();
    const uint32_t buf_a = smem_u32(buf), scr_a = smem_u32(scratch);
    uint32_t phase = 0;
    int round = 0;
    bool active = true;                                        // still holds a partial to contribute
    // reduce to rank 0
    for (int d = 1; d < N; d <<= 1, ++round) {
        if (active) {
            if (rank & d) {                                    // child: send and drop out
                if (threadIdx.x == 0) bulk_push(rank - d, scr_a, buf_a, bytes, bar);   // my partial -> parent's scratch
                active = false;
            } else {                                           // parent: receive from rank + d
                if (threadIdx.x == 0) mbar_arrive_expect_tx(bar, bytes);
                mbar_wait_parity(bar, phase); phase ^= 1;
                for (int i = threadIdx.x; i < M; i += blockDim.x) buf[i] = op(buf[i], scratch[i]);
                fence_proxy_async_smem();
            }
        }
        cluster_sync();                                        // round boundary (scratch reuse, schedule)
        st.round(round);
    }
    // broadcast from rank 0
    for (int d = N / 2; d >= 1; d >>= 1, ++round) {
        if ((rank % (2 * d)) == 0) {                           // holder: send to rank + d
            if (threadIdx.x == 0) bulk_push(rank + d, buf_a, buf_a, bytes, bar);   // result -> child's buffer
        } else if ((rank % (2 * d)) == (uint32_t)d) {          // receiver this round
            if (threadIdx.x == 0) mbar_arrive_expect_tx(bar, bytes);
            mbar_wait_parity(bar, phase); phase ^= 1;
            fence_proxy_async_smem();
        }
        cluster_sync();
        st.round(round);
    }
}

// Shared buffers of one CTA, taken from the dynamic shared-memory window (dsmem_primitives.cuh, dyn_smem);
// the launcher passes sizeof(CollectiveSmem) as the dynamic smem size (up to 227 KiB per CTA).
struct CollectiveSmem {
    static constexpr int N = CLUSTER_N, M = MSG_ELEMS;
    __align__(16) ELEM_T buf[M];
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

    load_block<ELEM_T, M>(buf, in + rank * M);
    fence_proxy_async_smem();
    if (threadIdx.x == 0) { mbar_init(smem_u32(&bar), 1); fence_mbar_init(); smid_out[rank] = smid(); }
    cluster_sync();
    st.mark(0);
    allreduce_reduce_bcast_binomial<N, ELEM_T, M>(buf, scratch, smem_u32(&bar), Sum{}, st);
    st.mark(1);
    store_block<ELEM_T, M>(out + rank * M, buf);
    cluster_sync();
}
