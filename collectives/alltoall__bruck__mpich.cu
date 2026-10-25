/*
 * AlltoAll - Bruck (MPICH short-message algorithm: log rounds, half the data per round)
 *
 * 1. Op: CTA b holds N chunks of M elements, chunk j destined to CTA j; afterwards CTA b holds
 *    chunk b of every CTA in source-rank order.
 *
 * 2. Algorithm (N power of two here): (i) local rotation: block j moves to position (j - b) mod N;
 *    (ii) log2(N) rounds, round k with d = 2^k: every block whose position has bit k set is sent to
 *    CTA (b + d) mod N and the blocks received from (b - d) mod N replace those positions;
 *    (iii) local inverse rotation: position i goes to source rank (b - i) mod N. Each round moves
 *    N/2 blocks, so the total is (N/2)*log2(N) blocks instead of N-1 - fewer rounds but more bytes.
 *
 * 3. Pattern: ring with doubling shift. Rounds: log2(N). Message per round: N/2 blocks of
 *    M*sizeof(T) (sent as N/2 separate bulk copies to one peer, received into a staging buffer so
 *    the blocks being sent are not overwritten mid-copy). Total per CTA: (N/2)*log2(N)*M*sizeof(T).
 *    Primitive: cp.async.bulk pushes completing on one aggregate-armed mbarrier; parity wait;
 *    barrier.cluster; local copy-back from the staging buffer; barrier.cluster (2 per round).
 *
 * Source: Bruck et al. 1997 (index algorithm); MPICH MPI_Alltoall for short messages.
 */
#include "dsmem_primitives.cuh"
using namespace dsmem;

template <int N, typename T, int M>
__device__ void alltoall_bruck(T* D /*[N*M], already rotated: position i holds the chunk for rank (b+i) mod N*/,
                               T* tmp /*[N*M/2]*/, uint32_t bar, Stamps st) {
    static_assert((N & (N - 1)) == 0, "this Bruck variant assumes a power-of-two cluster size");
    static_assert((M * sizeof(T)) % 16 == 0, "bulk copy needs a multiple of 16 bytes");
    constexpr uint32_t blk = M * sizeof(T);
    const uint32_t rank = cluster_rank();
    const uint32_t D_a = smem_u32(D), tmp_a = smem_u32(tmp);
    uint32_t phase = 0;
    int round = 0;
    for (int d = 1; d < N; d <<= 1, ++round) {
        const uint32_t to = (rank + d) % N;
        if (threadIdx.x == 0) {                                // blocks whose position has bit k set -> receiver's tmp[0..N/2)
            mbar_arrive_expect_tx(bar, (N / 2) * blk);
            int s = 0;
            for (int pos = 0; pos < N; ++pos) if (pos & d) {
                bulk_push(to, tmp_a + s * blk, D_a + pos * blk, blk, bar);   // block at `pos` -> tmp[s] of CTA (b + d)
                ++s;
            }
        }
        mbar_wait_parity(bar, phase); phase ^= 1;              // N/2 blocks from (b - d) landed in my tmp
        cluster_sync();                                        // every CTA's push completed: my sent blocks may be overwritten
        for (int i = threadIdx.x; i < (N / 2) * M; i += blockDim.x) {   // copy-back tmp[s] -> s-th position with bit k set
            const int s = i / M, e = i % M;
            int pos = -1, c = 0;
            for (int p = 0; p < N; ++p) if (p & d) { if (c == s) { pos = p; break; } ++c; }
            D[pos * M + e] = tmp[i];
        }
        fence_proxy_async_smem();
        cluster_sync();                                        // tmp free and D stable before the next round's pushes
        st.round(round);
    }
}

// Shared buffers of one CTA, taken from the dynamic shared-memory window (dsmem_primitives.cuh, dyn_smem);
// the launcher passes sizeof(CollectiveSmem) as the dynamic smem size (up to 227 KiB per CTA).
struct CollectiveSmem {
    static constexpr int N = CLUSTER_N, M = MSG_ELEMS;
    __align__(16) ELEM_T D[N * M];
    __align__(16) ELEM_T tmp[N * M / 2];
    __align__(8) uint64_t bar;
};

__global__ void __cluster_dims__(CLUSTER_N, 1, 1)
collective_kernel(const ELEM_T* in, ELEM_T* out, unsigned long long* ts, unsigned* smid_out) {
    constexpr int N = CLUSTER_N, M = MSG_ELEMS;
    CollectiveSmem& sm = dyn_smem<CollectiveSmem>();
    auto& D = sm.D; auto& tmp = sm.tmp; auto& bar = sm.bar;
    const uint32_t rank = cluster_rank();
    Stamps st{ts + rank * kStamps};

    // (i) rotation done while loading: chunk j of my input -> position (j - rank) mod N
    const ELEM_T* src = in + rank * N * M;
    for (int i = threadIdx.x; i < N * M; i += blockDim.x) {
        const int j = i / M, e = i % M;
        D[((j - (int)rank + N) % N) * M + e] = src[i];
    }
    __syncthreads();
    fence_proxy_async_smem();
    if (threadIdx.x == 0) { mbar_init(smem_u32(&bar), 1); fence_mbar_init(); smid_out[rank] = smid(); }
    cluster_sync();
    st.mark(0);
    alltoall_bruck<N, ELEM_T, M>(D, tmp, smem_u32(&bar), st);
    st.mark(1);
    // (iii) inverse rotation: position i holds the chunk from source rank (rank - i) mod N
    ELEM_T* o = out + rank * N * M;
    __syncthreads();
    for (int i = threadIdx.x; i < N * M; i += blockDim.x) {
        const int pos = i / M, e = i % M;
        o[((rank - pos + N) % N) * M + e] = D[i];
    }
    cluster_sync();
}
