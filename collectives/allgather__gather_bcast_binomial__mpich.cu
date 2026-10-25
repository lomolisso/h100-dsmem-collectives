/*
 * AllGather - rooted: binomial-tree gather to rank 0, then binomial-tree broadcast of the whole buffer
 *
 * 1. Op: every CTA ends with the rank-ordered concatenation of the N M-element vectors.
 *
 * 2. Algorithm (N power of two). Gather: rounds d = 1, 2, ..., N/2; a CTA whose bit d is set sends
 *    the d slots it has accumulated (ranks b .. b+d-1, contiguous in its buffer) to parent b - d,
 *    which stores them at slots [d, 2d) of its own buffer; after log2(N) rounds rank 0 holds all N
 *    slots in natural order. Broadcast: d = N/2 .. 1, holders push the full N-slot buffer to b + d.
 *
 * 3. Pattern: binomial tree, 2*log2(N) rounds. Gather messages grow (d*M*sizeof(T)); broadcast
 *    messages are the full N*M*sizeof(T). Cluster-wide traffic (N-1)*M + (N-1)*N*M elements: the
 *    broadcast of the already-gathered buffer makes this the most traffic-hungry all-gather, which is
 *    why symmetric algorithms are preferred; included as the rooted baseline.
 *    Primitive: cp.async.bulk push into the parent's/child's buffer at the right slot offset +
 *    mbarrier complete_tx; parity wait; barrier.cluster per round.
 *
 * Source: MPICH gather/bcast composition (Thakur et al. 2005).
 */
#include "dsmem_primitives.cuh"
using namespace dsmem;

template <int N, typename T, int M>
__device__ void allgather_gather_bcast_binomial(T* D, uint32_t bar, Stamps st) {
    static_assert((N & (N - 1)) == 0, "binomial tree as written needs a power-of-two cluster size");
    static_assert((M * sizeof(T)) % 16 == 0, "bulk copy needs a multiple of 16 bytes");
    constexpr uint32_t blk = M * sizeof(T);
    const uint32_t rank = cluster_rank();
    const uint32_t D_a = smem_u32(D);
    uint32_t phase = 0;
    int round = 0;
    bool active = true;
    for (int d = 1; d < N; d <<= 1, ++round) {                 // gather
        if (active) {
            if (rank & d) {                                    // send my d slots to the parent's slots [d, 2d)
                if (threadIdx.x == 0) bulk_push(rank - d, D_a + d * blk, D_a, d * blk, bar);
                active = false;
            } else {
                if (threadIdx.x == 0) mbar_arrive_expect_tx(bar, d * blk);
                mbar_wait_parity(bar, phase); phase ^= 1;
            }
        }
        cluster_sync();
        st.round(round);
    }
    for (int d = N / 2; d >= 1; d >>= 1, ++round) {            // broadcast the full buffer
        if ((rank % (2 * d)) == 0) {
            if (threadIdx.x == 0) bulk_push(rank + d, D_a, D_a, N * blk, bar);
        } else if ((rank % (2 * d)) == (uint32_t)d) {
            if (threadIdx.x == 0) mbar_arrive_expect_tx(bar, N * blk);
            mbar_wait_parity(bar, phase); phase ^= 1;
        }
        cluster_sync();
        st.round(round);
    }
}

// Shared buffers of one CTA, taken from the dynamic shared-memory window (dsmem_primitives.cuh, dyn_smem);
// the launcher passes sizeof(CollectiveSmem) as the dynamic smem size (up to 227 KiB per CTA).
struct CollectiveSmem {
    static constexpr int N = CLUSTER_N, M = MSG_ELEMS;
    __align__(16) ELEM_T D[N * M];
    __align__(8) uint64_t bar;
};

__global__ void __cluster_dims__(CLUSTER_N, 1, 1)
collective_kernel(const ELEM_T* in, ELEM_T* out, unsigned long long* ts, unsigned* smid_out) {
    constexpr int N = CLUSTER_N, M = MSG_ELEMS;
    CollectiveSmem& sm = dyn_smem<CollectiveSmem>();
    auto& D = sm.D; auto& bar = sm.bar;
    const uint32_t rank = cluster_rank();
    Stamps st{ts + rank * kStamps};

    load_block<ELEM_T, M>(D, in + rank * M);                   // own data in slot 0 (slot i = rank b + i during gather)
    fence_proxy_async_smem();
    if (threadIdx.x == 0) { mbar_init(smem_u32(&bar), 1); fence_mbar_init(); smid_out[rank] = smid(); }
    cluster_sync();
    st.mark(0);
    allgather_gather_bcast_binomial<N, ELEM_T, M>(D, smem_u32(&bar), st);
    st.mark(1);
    store_block<ELEM_T, N * M>(out + rank * N * M, D);        // after bcast every buffer is rank 0's natural order
    cluster_sync();
}
