/*
 * AlltoAll (N = 2) - pairwise half-buffer exchange (FlashAttention-4 2-CTA backward dS exchange)
 *
 * 1. Op: CTA b holds two chunks [c_0 | c_1] of M elements; afterwards CTA b holds chunk b of
 *    both CTAs: [own c_b | peer's c_b]. With N = 2 this is exactly AlltoAll (a 2x2 transpose).
 *    Origin: dS tile repack so that each CTA owns M/2 rows of dS with the full 2N reduction
 *    for the dQ = dS @ K MMA, halving the dQ global atomics.
 *
 * 2. Algorithm: single round. Peer = rank ^ 1. CTA b sends chunk (b ^ 1) (the half the peer
 *    owns) into the peer's receive buffer and keeps chunk b. The *sender* arms the
 *    *receiver's* mbarrier (remote arrive.expect_tx) and issues one bulk copy that completes
 *    the transaction bytes on it; the receiver waits on its own barrier.
 *    (In FA4 a relay warp then forwards the completion to the leader CTA's barrier with a
 *    remote mbarrier.arrive so the single tcgen05 MMA issuer can start; omitted here.)
 *
 * 3. Pattern: pair, 1 round, 1 message of M*sizeof(T) bytes per CTA (origin 16 KB bf16, or
 *    8 KB for the tile_n = 64 layout). Total DSMEM bytes per CTA: M*sizeof(T). No
 *    barrier.cluster inside the collective (only mbarrier); exit barrier added for DSMEM lifetime.
 *    Primitive: fence.proxy.async after the generic writes, mbarrier.arrive.expect_tx on the
 *    remote (peer) barrier via mapa, cp.async.bulk shared::cta -> shared::cluster complete_tx.
 *
 * Source: third_party/flash-attention/flash_attn/cute/flash_bwd_sm100.py (compute_loop,
 * "stage_copy_bytes", copy_utils.cpasync_bulk_s2cluster); Blackwell-only kernel, sm_90 primitives.
 */
#include "dsmem_primitives.cuh"
using namespace dsmem;

template <int N, typename T, int M>
__device__ void alltoall_pairwise_exchange(const T* chunks /*[2][M]*/, T* recv /*[M]*/, uint32_t bar, Stamps st) {
    static_assert(N == 2, "pairwise exchange is the N = 2 case; see alltoall__pairwise_xor__mpich.cu for general N");
    static_assert((M * sizeof(T)) % 16 == 0, "bulk copy needs a multiple of 16 bytes");
    constexpr uint32_t bytes = M * sizeof(T);
    const uint32_t rank = cluster_rank();
    const uint32_t peer = rank ^ 1;                            // peer_cta_rank_in_cluster = rank ^ 1
    const uint32_t stage = rank ^ 1;                           // exchange_stage: send the half the peer owns
    fence_proxy_async_smem();                                  // origin: fence_view_async_shared() after writing sdS_xchg
    __syncthreads();                                           // origin: compute_sync_barrier.arrive_and_wait()
    if (threadIdx.x == 0) {
        mbar_arrive_expect_tx_remote(bar, peer, bytes);        // arm the *peer's* dS_cluster_full barrier for `bytes`
        bulk_push(peer, smem_u32(recv), smem_u32(chunks + stage * M), bytes, bar);   // cpasync_bulk_s2cluster
    }
    mbar_wait_parity(bar, 0);                                  // my barrier completes when the peer's half has landed
    st.round(0);
}

// Shared buffers of one CTA, taken from the dynamic shared-memory window (dsmem_primitives.cuh, dyn_smem);
// the launcher passes sizeof(CollectiveSmem) as the dynamic smem size (up to 227 KiB per CTA).
struct CollectiveSmem {
    static constexpr int N = CLUSTER_N, M = MSG_ELEMS;
    __align__(16) ELEM_T chunks[N * M];  // [chunk for rank 0 | chunk for rank 1]
    __align__(16) ELEM_T recv[M];
    __align__(8) uint64_t bar;
};

__global__ void __cluster_dims__(CLUSTER_N, 1, 1)
collective_kernel(const ELEM_T* in, ELEM_T* out, unsigned long long* ts, unsigned* smid_out) {
    constexpr int N = CLUSTER_N, M = MSG_ELEMS;
    CollectiveSmem& sm = dyn_smem<CollectiveSmem>();
    auto& chunks = sm.chunks; auto& recv = sm.recv; auto& bar = sm.bar;
    const uint32_t rank = cluster_rank();
    Stamps st{ts + rank * kStamps};

    load_block<ELEM_T, N * M>(chunks, in + rank * N * M);
    if (threadIdx.x == 0) { mbar_init(smem_u32(&bar), 1); fence_mbar_init(); smid_out[rank] = smid(); }
    cluster_sync();
    st.mark(0);
    alltoall_pairwise_exchange<N, ELEM_T, M>(chunks, recv, smem_u32(&bar), st);
    st.mark(1);
    ELEM_T* o = out + rank * N * M;
    store_block<ELEM_T, M>(o + rank * M, chunks + rank * M);   // out[b][b] = own chunk b
    store_block<ELEM_T, M>(o + (rank ^ 1) * M, recv);          // out[b][peer] = peer's chunk b
    cluster_sync();
}
