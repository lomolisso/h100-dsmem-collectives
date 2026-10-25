/*
 * AllGather - Bruck (MPICH short-message algorithm, any N)
 *
 * 1. Op: every CTA ends with the rank-ordered concatenation of the N M-element vectors.
 *
 * 2. Algorithm: buffer D of N slots, own data in slot 0. ceil(log2 N) rounds with d = 1, 2, 4, ...:
 *    CTA b sends its first min(d, N-d) slots to (b - d) mod N and receives the same count from
 *    (b + d) mod N into slots [d, d + count). After the loop slot i holds rank (b + i) mod N, so the
 *    final store un-rotates: out[r] = D[(r - b) mod N]. Unlike recursive doubling it needs no
 *    power-of-two N (the last round sends a partial block).
 *
 * 3. Pattern: ring with doubling shift (send to b-d, receive from b+d). Rounds: ceil(log2 N).
 *    Message in round k: min(2^k, N-2^k) * M*sizeof(T). Total per CTA: (N-1)*M*sizeof(T).
 *    Primitive: cp.async.bulk push of a contiguous prefix of D into the partner's D at offset d;
 *    mbarrier complete_tx + parity wait; barrier.cluster per round.
 *
 * Source: Bruck et al. 1997; MPICH MPI_Allgather (short messages, non power of two).
 */
#include "dsmem_primitives.cuh"
using namespace dsmem;

template <int N, typename T, int M>
__device__ void allgather_bruck(T* D, uint32_t bar, Stamps st) {
    static_assert((M * sizeof(T)) % 16 == 0, "bulk copy needs a multiple of 16 bytes");
    constexpr uint32_t blk = M * sizeof(T);
    const uint32_t rank = cluster_rank();
    const uint32_t D_a = smem_u32(D);
    uint32_t phase = 0;
    int round = 0;
    for (int d = 1; d < N; d <<= 1, ++round) {
        const int cnt = (d < N - d) ? d : N - d;               // blocks to move this round
        const uint32_t bytes = cnt * blk;
        if (threadIdx.x == 0) {
            mbar_arrive_expect_tx(bar, bytes);
            bulk_push((rank + N - d) % N, D_a + d * blk, D_a, bytes, bar);   // D[0:cnt] -> D[d:d+cnt] of CTA (b - d)
        }
        mbar_wait_parity(bar, phase); phase ^= 1;              // blocks from (b + d) landed at D[d:d+cnt]
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

    load_block<ELEM_T, M>(D, in + rank * M);                   // own data in slot 0
    fence_proxy_async_smem();
    if (threadIdx.x == 0) { mbar_init(smem_u32(&bar), 1); fence_mbar_init(); smid_out[rank] = smid(); }
    cluster_sync();
    st.mark(0);
    allgather_bruck<N, ELEM_T, M>(D, smem_u32(&bar), st);
    st.mark(1);
    ELEM_T* o = out + rank * N * M;                            // un-rotate: slot i = rank (b + i) mod N
    for (int i = threadIdx.x; i < N * M; i += blockDim.x) {
        const int k = i / M, e = i % M;
        o[((rank + k) % N) * M + e] = D[k * M + e];
    }
    cluster_sync();
}
