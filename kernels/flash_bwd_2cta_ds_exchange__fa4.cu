/*
 * FlashAttention-4 2-CTA backward pass: the dS exchange step over DSMEM (CUDA C++ port of the exchange;
 * the surrounding MMA pipeline is Blackwell-only and is described, not ported).
 * Source: third_party/flash-attention/flash_attn/cute/flash_bwd_sm100.py (compute_loop, relay, mma) and
 * copy_utils.py (set_block_rank, cpasync_bulk_s2cluster). Verbatim excerpts: flash_bwd_2cta_ds_exchange__fa4.excerpt.py.
 * The original is CuTe-DSL Python for sm_100 (tcgen05 MMA, TMEM, cta_group::2); the DSMEM primitives it
 * uses are sm_90 PTX and are what this file exercises.
 *
 * 1. Computation (backward of O = softmax(Q K^T * scale) V, head dim d, tiles M = N = 128):
 *    per (KV block j fixed in the CTA pair, Q block i streamed):
 *      S^T = K Q^T,  P^T = exp(S^T - LSE),  dP^T = V dO^T,  dS^T = P^T o (dP^T - Delta)   [elementwise, bf16]
 *      dV += P^T dO,  dK += dS^T Q,  dQ_i += dS K  (dQ accumulated across KV blocks with global atomics)
 *    In 2-CTA mode the pair holds one 256-row KV tile (128 rows per CTA) and the five MMAs run as
 *    cta_group::2 UMMAs with M = 256 (K = 2N = 256 for dQ). The dQ MMA's reduction dimension is the KV axis,
 *    which is split across the pair, so each CTA needs the *other* CTA's dS rows for the Q columns it owns.
 *
 * 2. Split: CTA r in {0,1} owns KV rows [128r, 128r+128) of the tile and, for dQ, Q columns
 *    [64r, 64r+64) (M/2 rows of dQ, "exchange_stage = r ^ 1" is the half it must send). Each CTA writes the
 *    dS^T half for its own Q columns into its sdS (A operand of dQ = dS K) and the other half into a staging
 *    buffer sdS_xchg. After the exchange CTA r's sdS holds dS for its 64 Q columns over all 256 KV rows:
 *    rows [0,128) from CTA 0, rows [128,256) from CTA 1 -> operand (M/2) x (2N). Each CTA then reduces only
 *    its half of dQ into global memory, halving the number of global atomic adds.
 *
 * 3. Collective: 2-CTA AlltoAll ("pairwise exchange", see collectives/alltoall__pairwise_exchange__fa4.cu).
 *    One thread per CTA: (a) fence.proxy.async + CTA barrier so all warps' dS stores are visible to the
 *    async proxy; (b) mbarrier.arrive.expect_tx on the PEER's dS_cluster_full barrier (remote, via mapa)
 *    for stage_copy_bytes = 16 KB (bf16 128x64); (c) cp.async.bulk.shared::cluster.shared::cta from local
 *    sdS_xchg into the peer's sdS at offset rank*8192 elements, completing the tx bytes on that barrier.
 *    A relay warp in each CTA waits on its local full barrier and does a remote mbarrier.arrive on the
 *    LEADER's dS_cluster_leader barrier (count 2), which the leader's single MMA-issuing warp waits on before
 *    launching the pair-wide dQ UMMA. Buffer reuse is protected by the dS pipeline (producer_acquire) rather
 *    than by the unused dS_cluster_empty barrier.
 *
 * This file keeps the data layout and the three barriers, replaces the UMMAs by a plain smem reduction
 * (dQ_rows = dS_rows . K) so it can run on an H100 for inspection, and keeps the relay/leader handshake.
 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <stdint.h>

namespace cg = cooperative_groups;

// ---- primitives (1:1 with copy_utils.py) --------------------------------------------------------------
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
// copy_utils.set_block_rank: mapa.shared::cluster
__device__ __forceinline__ uint32_t set_block_rank(uint32_t smem_addr, uint32_t peer) {
    uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(smem_addr), "r"(peer)); return r;
}
// cute.arch.mbarrier_arrive_and_expect_tx(..., peer_cta_rank_in_cluster=peer): remote arrive + expect_tx
__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_remote(uint32_t bar, uint32_t bytes, uint32_t peer) {
    uint32_t rbar = set_block_rank(bar, peer);
    asm volatile("mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%0], %1;" :: "r"(rbar), "r"(bytes) : "memory");
}
// cute.arch.mbarrier_arrive(ptr, cta_rank): remote arrive (no tx)
__device__ __forceinline__ void mbarrier_arrive_remote(uint32_t bar, uint32_t peer) {
    uint32_t rbar = set_block_rank(bar, peer);
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" :: "r"(rbar) : "memory");
}
// copy_utils.cpasync_bulk_s2cluster
__device__ __forceinline__ void cpasync_bulk_s2cluster(uint32_t src, uint32_t dst, uint32_t bar, uint32_t bytes, uint32_t peer) {
    uint32_t rdst = set_block_rank(dst, peer), rbar = set_block_rank(bar, peer);
    asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(rdst), "r"(src), "r"(bytes), "r"(rbar) : "memory");
}
__device__ __forceinline__ void mbarrier_init(uint32_t bar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(count));
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void mbarrier_wait(uint32_t bar, uint32_t phase) {
    asm volatile("{\n.reg .pred P;\nW:\nmbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n@P bra D;\nbra W;\nD:\n}"
                 :: "r"(bar), "r"(phase) : "memory");
}
__device__ __forceinline__ void fence_view_async_shared() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }

// ---- tile shape (FA4 defaults: tile_m = tile_n = 128, bf16 dS; d = 128 for the K operand) --------------
constexpr int TILE_M = 128;           // Q rows of the tile pair
constexpr int TILE_N = 128;           // KV rows per CTA (pair holds 2*TILE_N)
constexpr int HEAD_D = 128;
constexpr int HALF_M = TILE_M / 2;    // Q columns owned by one CTA for dQ
constexpr int XCHG_ELEMS = TILE_N * HALF_M;              // 8192 bf16 = stage_copy_elems
constexpr uint32_t XCHG_BYTES = XCHG_ELEMS * sizeof(__half);  // 16384 B = stage_copy_bytes

struct __align__(16) SharedStorage {
    __half sdS[2 * XCHG_ELEMS];       // dS A-operand: [KV row 0..255][own 64 Q cols]; rows [128r,128r+128) come from CTA r
    __half sdS_xchg[XCHG_ELEMS];      // the half destined to the peer (its Q columns, my KV rows)
    __half sK[TILE_N * HEAD_D];       // this CTA's K rows (B operand of dQ = dS K is split across the pair)
    float  dQ_acc[HALF_M * HEAD_D];   // this CTA's (M/2, d) slice of dQ for one KV block
    uint64_t dS_cluster_full_mbar;    // count 1: peer's remote arrive.expect_tx + 16 KB complete_tx
    uint64_t dS_cluster_empty_mbar;   // initialised in FA4, never waited on (dS pipeline covers reuse)
    uint64_t dS_cluster_leader_mbar;  // count 2: one relay arrive per CTA (only meaningful on the leader)
};

// dS_in: [KV row (0..127 of this CTA)][Q col (0..127)] bf16 for one (i, j) tile; K_in: this CTA's 128 K rows.
// dQ_out: per CTA its (64 x d) rows of dQ_i (origin: TMEM accumulator then global atomics).
__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(256)
flash_bwd_2cta_ds_exchange(const __half* __restrict__ dS_in, const __half* __restrict__ K_in, float* __restrict__ dQ_out) {
    extern __shared__ __align__(128) uint8_t smem_raw[];
    SharedStorage& s = *reinterpret_cast<SharedStorage*>(smem_raw);
    cg::cluster_group cluster = cg::this_cluster();
    const uint32_t rank = cluster.block_rank();          // cta_rank_in_cluster
    const uint32_t peer = rank ^ 1;                      // peer_cta_rank_in_cluster
    const uint32_t exchange_stage = rank ^ 1;            // CTA 0 sends the bottom half, CTA 1 the top half
    const uint32_t full_bar = smem_u32(&s.dS_cluster_full_mbar);
    const uint32_t leader_bar = smem_u32(&s.dS_cluster_leader_mbar);
    const int tid = threadIdx.x, warp = tid / 32;

    if (tid == 0) {                                      // origin: warp 4 initialises the three barriers
        mbarrier_init(full_bar, 1);
        mbarrier_init(smem_u32(&s.dS_cluster_empty_mbar), 1);
        mbarrier_init(leader_bar, 2);
    }
    // stage K (B operand) and the dS tile, splitting dS by Q-column half: own half -> sdS rows [128*rank..),
    // peer's half -> sdS_xchg (origin: tdPrdS_cvt -> tRS_sdS[stage] or tdPrdS_xchg -> sdS_xchg)
    for (int i = tid; i < TILE_N * HEAD_D; i += blockDim.x) s.sK[i] = K_in[rank * TILE_N * HEAD_D + i];
    for (int i = tid; i < TILE_N * TILE_M; i += blockDim.x) {
        const int kv = i / TILE_M, q = i % TILE_M, stage = q / HALF_M, qq = q % HALF_M;
        const __half v = dS_in[rank * TILE_N * TILE_M + i];
        if (stage == (int)rank) s.sdS[(rank * TILE_N + kv) * HALF_M + qq] = v;   // my Q columns, my KV rows
        else                    s.sdS_xchg[kv * HALF_M + qq] = v;                 // peer's Q columns (exchange_stage)
    }
    __syncthreads();
    cluster.sync();                                      // barriers initialised in both CTAs before any remote arrive

    // ---- the DSMEM exchange (compute_loop, "2-CTA: DSMEM copy of the exchanged dS half") ----------------
    fence_view_async_shared();                           // generic smem stores -> visible to the async proxy
    __syncthreads();                                     // origin: compute_sync_barrier.arrive_and_wait()
    if (tid == 0) {
        // destination = peer's sdS at OUR rank's KV-row offset (the rows we own, its Q columns)
        const uint32_t dst = smem_u32(s.sdS) + rank * XCHG_BYTES;
        mbarrier_arrive_and_expect_tx_remote(full_bar, XCHG_BYTES, peer);          // arm the PEER's full barrier
        cpasync_bulk_s2cluster(smem_u32(s.sdS_xchg), dst, full_bar, XCHG_BYTES, peer);  // push 16 KB into the peer
    }

    // ---- relay warp: local full barrier -> remote arrive on the leader (relay()) --------------------------
    if (warp == 1) {
        mbarrier_wait(full_bar, 0);                      // "Wait for dS_xchg from peer CTA"
        if (tid % 32 == 0) mbarrier_arrive_remote(leader_bar, 0);   // "Arrive on MMA leader warp" (CTA 0)
    }

    // ---- MMA warp of the leader: wait for both relays, then the pair-wide dQ MMA (mma(), step 5) ---------
    // Here every CTA computes its own (M/2 x d) slice after the leader signal is observed cluster-wide; in
    // FA4 only CTA 0 issues one cta_group::2 UMMA that reads sdS and sK of BOTH CTAs.
    if (rank == 0 && warp == 0) mbarrier_wait(leader_bar, 0);
    cluster.sync();                                      // stand-in for the UMMA's cluster-wide operand visibility
    for (int i = tid; i < HALF_M * HEAD_D; i += blockDim.x) {   // dQ[q][d] = sum_kv dS[kv][q] * K[kv][d] over all 256 rows
        const int q = i / HEAD_D, d = i % HEAD_D;
        float acc = 0.f;
        for (int kv = 0; kv < 2 * TILE_N; ++kv) {
            const __half kval = (kv < TILE_N) ? s.sK[kv * HEAD_D + d] : K_in[kv * HEAD_D + d];  // peer's K rows read from global here
            acc += __half2float(s.sdS[kv * HALF_M + q]) * __half2float(kval);
        }
        s.dQ_acc[i] = acc;
    }
    __syncthreads();
    // origin: each CTA reduces only its (M/2, d) rows into global dQaccum -> half the global atomics
    for (int i = tid; i < HALF_M * HEAD_D; i += blockDim.x) atomicAdd(&dQ_out[rank * HALF_M * HEAD_D + i], s.dQ_acc[i]);
    cluster.sync();                                      // DSMEM lifetime: no CTA exits while the peer may still address its smem
}
