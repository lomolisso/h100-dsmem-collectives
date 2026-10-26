/*
 * dsmem_bw.cu — roofline R1: aggregate GPU-wide DSMEM (SM-to-SM network) bandwidth (DESIGN §2.1 "DSMEM bandwidth"):
 * every SM that can join a cluster of size CS hosts one CTA (max active clusters at 1 CTA/SM), every CTA streams 96 KiB
 * per pass from / to its ring neighbour (rank r → (r + 1) mod CS), for CS ∈ {2, 4, 8, 16}, in two directions of use:
 *   ld       pull: ld.shared::cluster.v4.u32 from the neighbour's source tile (1024 threads, ILP 4);
 *   tma_put  push: one thread per CTA issues cp.async.bulk.shared::cluster.shared::cta commands of 16 KiB into the
 *            neighbour's destination tile, completing on the neighbour's mbarrier (complete_tx); the neighbour's last
 *            warp waits for all bytes of the pass.
 *
 * Copied from MB/experiments/11__dsmem_bandwidth_vs_cluster_size/src at MB commit
 * 8f8fc7bec69f07db49f3a5a95bbdfb08de5d0ca8 (verbatim unless stated):
 *   common11.hpp   Op11/Pat11 names, Group11, Perm16, op_async, hash32, chunk_target, plan_groups, incoming_bytes.
 *                  Dropped: Discards11 (flagged launches are marked in the raw CSV instead).
 *   dsmem_bw.cu    kTile … kWaitTimeout constants, Rec11, P11, ld_stream, st_stream, tma_group, tma_issue, reduce_slot,
 *                  k_stream<OP, ILP, UNI>; only k_stream<OP_LD, 4, false> and k_stream<OP_TMA, 4, false> are instantiated.
 *                  Host: the launch of run_point() (explicit cluster attribute, 200 KB dynamic smem → 1 CTA per SM,
 *                  Default policy, nclusters = cudaOccupancyMaxActiveClusters = MB launch_mode "full", cudaEvent pair
 *                  around cudaLaunchKernelEx, mbarrier time-out check, %smid placement check, ld check word compared
 *                  with expected_chk()) and MB step S (ring +1, payload 96 KiB, passes {4, 8, 16, 32}, 1 sample per
 *                  launch; GPU-wide TB/s = 1 / slope of the event time over the launch's total bytes, which removes the
 *                  fixed cost of a launch: smem tile init, cluster barriers, launch overhead). Dropped: the GPC map,
 *                  steps A–F, P1, K4 (k_victim, k_p1), the per-record samples CSV, plan mode.
 *   MB experiments/harness/harness.cuh (newer than common/h100mb): st_dsmem_v4_u32, st_async_v4_b32,
 *                  bulk_red_s2s_add_u32 (referenced by the verbatim st_stream / tma_group templates); ld_dsmem_v4_u32
 *                  and record_place / Placement are in ceil_common.hpp.
 * This benchmark is independent of ../collectives/.
 *
 * Usage (from roofline/): harness/ceilings/bin/dsmem_bw {S|ncu|clockcheck} [--quick] [--label L]
 *   S    op ∈ {ld, tma_put} × CS ∈ {2, 4, 8, 16} × passes ∈ {4, 8, 16, 32} ({4, 16} with --quick): 2 warm-up + 10
 *        recorded launches (3 with --quick). A launch with two ranks of one cluster on the same SM (placement_bad) or a
 *        bad NVML clock sample (clock_bad) is written, not counted, and repeated.
 *   ncu  one launch per op × CS at passes 4, for ceilings_ncu.sh (-k regex:k_stream); manifest (expected payload bytes)
 *        → results/ceilings/ncu/manifest_dsmem_bw.csv. Expected (collectives C1, EXPERIMENTS §1): each payload byte is
 *        counted once in l1tex__m_l1tex2xbar_write_bytes_mem_dshared.sum and once in
 *        l1tex__m_xbar2l1tex_read_bytes_mem_dshared.sum (MB 11: "dshared byte counters = payload exactly").
 * → results/ceilings/dsmem_bw.csv (one row per launch): mandatory columns, op, pattern, cs, threads, ilp, chunk_bytes,
 *   payload_bytes, passes, nclusters, max_active_clusters, active_ctas, distinct_sms, total_bytes (issuer side, all
 *   CTAs), event_ns, makespan_ns (%globaltimer: earliest record start → latest record end), bpc_cta_median (median
 *   bytes/cycle of the bandwidth-side records: issuer for ld, target for tma_put), placement_bad, clock_bad.
 */
#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "ceil_common.hpp"

// ---------------------------------------------------------------- verbatim from MB experiments/harness/harness.cuh
// 16-B store into a peer's smem (.shared::cluster address, 16-B aligned).
__device__ __forceinline__ void st_dsmem_v4_u32(uint32_t a, uint4 v) {
    asm volatile("st.shared::cluster.v4.u32 [%0], {%1, %2, %3, %4};" :: "r"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory"); }
// 16-B st.async with complete_tx(16 B) on the peer's mbarrier (a, rbar: .shared::cluster addresses of one peer).
__device__ __forceinline__ void st_async_v4_b32(uint32_t a, uint4 v, uint32_t rbar) {
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                 :: "r"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "r"(rbar) : "memory"); }
// Bulk reduce-put (add.u32 element-wise) from own smem into a peer's smem; completion as bulk_s2s_dsmem.
__device__ __forceinline__ void bulk_red_s2s_add_u32(uint32_t dst, uint32_t src, uint32_t bytes, uint32_t rbar) {
    asm volatile("cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.add.u32 [%0], [%1], %2, [%3];"
                 :: "r"(dst), "r"(src), "r"(bytes), "r"(rbar) : "memory"); }

// ---------------------------------------------------------------- verbatim from MB 11 common11.hpp

enum Op11 { OP_LD = 0, OP_ST, OP_STA, OP_TMA, OP_TRED, OP_MIX_ST, OP_MIX_TMA, OP_N };
static const char* kOpName[] = {"ld", "st", "st_async", "tma_put", "tma_red", "mixed_ld_st", "mixed_ld_tma"};
enum Pat11 { PT_PAIR_UNI = 0, PT_PAIR_BI, PT_RING, PT_PACKED, PT_BCAST_IN, PT_BCAST_OUT, PT_A2A, PT_A2A_FULL, PT_RANDPERM,
             PT_UNIFORM, PT_MIX_SAME, PT_MIX_OPP, PT_N };
static const char* kPatName[] = {"pair_uni", "pair_bi", "ring", "packed_pairs", "broadcast_in", "broadcast_out", "all_to_all",
                                 "all_to_all_full", "randperm", "uniform_random", "mixed_same_dir", "mixed_opposite"};

constexpr int kMaxGroups = 15;
constexpr uint32_t kChunkTarget = 0xffu;          // per-16-KiB-chunk partner (uniform_random)
constexpr uint32_t kChunkElems = 16384u / 16u;    // elements per 16 KiB chunk

struct Group11 { uint32_t op, target, e0, e1, w0, w1; };
struct Perm16 { uint8_t p[16]; };

__host__ __device__ inline bool op_async(uint32_t op) { return op == OP_STA || op == OP_TMA || op == OP_TRED; }

__host__ __device__ inline uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}
// uniform_random partner of (rank, chunk) in pass `pass` of sample `sample`: uniformly one of the cs − 1 other ranks
__host__ __device__ inline uint32_t chunk_target(uint32_t seed, uint32_t sample, uint32_t pass, uint32_t rank, uint32_t chunk, uint32_t cs) {
    const uint32_t h = hash32(seed ^ hash32(sample * 0x9e3779b9u ^ hash32(pass * 0x85ebca6bu ^ hash32(rank * 977u + chunk))));
    return (rank + 1u + h % (cs - 1u)) % cs;
}

// Groups of the CTA with rank r (returns their number, 0 = idle as an issuer). op = kernel operation, nelem = payload / 16.
__host__ __device__ inline int plan_groups(uint32_t op, uint32_t pat, uint32_t cs, uint32_t k, uint32_t r, uint32_t nwarps, uint32_t nelem,
                                           const Perm16& perm, Group11* g) {
    auto one = [&](uint32_t o, uint32_t t, uint32_t w0, uint32_t w1) { g[0] = Group11{o, t, 0, nelem, w0, w1}; return 1; };
    switch (pat) {
    case PT_PAIR_UNI: return r == 0 ? one(op, 1, 0, nwarps) : 0;
    case PT_PAIR_BI: return r < 2 ? one(op, r ^ 1u, 0, nwarps) : 0;
    case PT_RING: return one(op, (r + k) % cs, 0, nwarps);
    case PT_PACKED: return ((r / k) % 2 == 0) ? one(op, r + k, 0, nwarps) : 0;
    case PT_BCAST_IN: return (r >= 1 && r <= k) ? one(op, 0, 0, nwarps) : 0;
    case PT_RANDPERM: return one(op, perm.p[r], 0, nwarps);
    case PT_UNIFORM: return one(op, kChunkTarget, 0, nwarps);
    case PT_BCAST_OUT:
    case PT_A2A:
    case PT_A2A_FULL: {
        if (pat == PT_BCAST_OUT && r != 0) return 0;
        const uint32_t G = pat == PT_BCAST_OUT ? k : cs - 1u, wpg = nwarps / G;
        for (uint32_t i = 0; i < G; ++i) {
            const uint32_t t = pat == PT_BCAST_OUT ? 1u + i : (r + 1u + i) % cs;
            const bool split = pat == PT_A2A;
            g[i] = Group11{op, t, split ? (uint32_t)((uint64_t)nelem * i / G) : 0u, split ? (uint32_t)((uint64_t)nelem * (i + 1) / G) : nelem,
                           i * wpg, (i + 1) * wpg};
        }
        return (int)G;
    }
    case PT_MIX_SAME:
    case PT_MIX_OPP: {
        const uint32_t second = op == OP_MIX_TMA ? (uint32_t)OP_TMA : (uint32_t)OP_ST, half = nwarps / 2;
        int n = 0;
        if (r == 0) g[n++] = Group11{OP_LD, 1, 0, nelem, 0, half};
        if (pat == PT_MIX_SAME && r == 1) g[n++] = Group11{second, 0, 0, nelem, half, nwarps};
        if (pat == PT_MIX_OPP && r == 0) g[n++] = Group11{second, 1, 0, nelem, half, nwarps};
        return n;
    }
    }
    return 0;
}

// Bytes per pass that CTA `me` receives from rank `src` through async operations (st_async / TMA): sum over src's groups.
__host__ __device__ inline uint32_t incoming_bytes(uint32_t op, uint32_t pat, uint32_t cs, uint32_t k, uint32_t src, uint32_t me,
                                                   uint32_t nwarps, uint32_t nelem, const Perm16& perm, uint32_t seed, uint32_t sample,
                                                   uint32_t pass) {
    Group11 g[kMaxGroups];
    const int n = plan_groups(op, pat, cs, k, src, nwarps, nelem, perm, g);
    uint32_t b = 0;
    for (int i = 0; i < n; ++i) {
        if (!op_async(g[i].op)) continue;
        if (g[i].target == kChunkTarget) {
            for (uint32_t c = g[i].e0 / kChunkElems; c * kChunkElems < g[i].e1; ++c)
                if (chunk_target(seed, sample, pass, src, c, cs) == me) {
                    const uint32_t a = c * kChunkElems > g[i].e0 ? c * kChunkElems : g[i].e0;
                    const uint32_t e = (c + 1) * kChunkElems < g[i].e1 ? (c + 1) * kChunkElems : g[i].e1;
                    b += (e - a) * 16u;
                }
        } else if (g[i].target == me) {
            b += (g[i].e1 - g[i].e0) * 16u;
        }
    }
    return b;
}

// ---------------------------------------------------------------- verbatim from MB 11 dsmem_bw.cu
constexpr uint32_t kTile = 96u * 1024u, kTileElems = kTile / 16u;
constexpr uint32_t kDstOff = kTile, kBarOff = 2u * kTile, kMaxBars = 512;  // MB: also kFlagOff (K4 only; dropped)
constexpr size_t kSmem = 200u * 1024u;
constexpr uint32_t kMaxCt = 256;  // uniform_random table entries (passes × chunks)
constexpr unsigned long long kWaitTimeout = 4000000000ull;

struct Rec11 { unsigned long long cycles, ns, t0_gt, bytes; uint32_t peer, valid, chk, stream; };
struct P11 {
    uint32_t pat, k, payload, chunk, samples, passes, seed, pad;
    uint64_t zero;
    Perm16 perm;
    Rec11* out;
    uint32_t* place;
    uint32_t* err;
};

// ---------------------------------------------------------------- device: streams of one warp group
template <int ILP, bool UNI>
__device__ __forceinline__ void ld_stream(const Group11& g, uint32_t warp, uint32_t lane, uint32_t passes, uint32_t src_u32,
                                          const uint8_t* ct, uint32_t zero, uint32_t& sy, uint32_t& sx) {
    const uint32_t Tg = (g.w1 - g.w0) * 32u, tg = (warp - g.w0) * 32u + lane;
    const uint32_t base = UNI ? 0u : mapa_u32(src_u32, g.target);
#pragma unroll 1
    for (uint32_t pass = 0; pass < passes; ++pass) {
        const uint32_t pb = base + (pass & zero);
#pragma unroll 1
        for (uint32_t e = g.e0 + tg; e < g.e1; e += ILP * Tg) {
            uint4 x[ILP];
#pragma unroll
            for (int j = 0; j < ILP; ++j) {
                const uint32_t ej = e + j * Tg;
                uint32_t a;
                if constexpr (UNI) a = mapa_u32(src_u32, ct[pass * 6u + ej / kChunkElems]) + ej * 16u + (pass & zero);
                else a = pb + ej * 16u;
                x[j] = ej < g.e1 ? ld_dsmem_v4_u32(a) : make_uint4(0, 0, 0, 0);
            }
#pragma unroll
            for (int j = 0; j < ILP; ++j) { sy += x[j].y; sx ^= x[j].x ^ x[j].z ^ x[j].w; }
        }
    }
}

template <int ILP, bool UNI, bool ASY>
__device__ __forceinline__ void st_stream(const Group11& g, uint32_t warp, uint32_t lane, uint32_t passes, uint32_t dst_u32,
                                          uint32_t bar_u32, uint32_t rank, uint32_t sample, const uint8_t* ct, uint32_t zero) {
    const uint32_t Tg = (g.w1 - g.w0) * 32u, tg = (warp - g.w0) * 32u + lane;
#pragma unroll 1
    for (uint32_t pass = 0; pass < passes; ++pass) {
        const uint32_t pb = UNI ? 0u : mapa_u32(dst_u32, g.target) + (pass & zero);
        const uint32_t rb = UNI ? 0u : mapa_u32(bar_u32 + (rank * passes + pass) * 8u, g.target);
#pragma unroll 1
        for (uint32_t e = g.e0 + tg; e < g.e1; e += ILP * Tg) {
#pragma unroll
            for (int j = 0; j < ILP; ++j) {
                const uint32_t ej = e + j * Tg;
                if (ej < g.e1) {
                    uint32_t a, r = rb;
                    if constexpr (UNI) {
                        const uint32_t t = ct[pass * 6u + ej / kChunkElems];
                        a = mapa_u32(dst_u32, t) + ej * 16u;
                        if constexpr (ASY) r = mapa_u32(bar_u32 + (rank * passes + pass) * 8u, t);
                    } else {
                        a = pb + ej * 16u;
                    }
                    const uint4 d = make_uint4(ej, pass, rank + 1u, sample);
                    if constexpr (ASY) st_async_v4_b32(a, d, r);
                    else st_dsmem_v4_u32(a, d);
                }
            }
        }
    }
}

// One thread issues the bulk commands of all TMA groups of its CTA: per pass, group after group, each group's
// commands back to back with the destination and barrier addresses in registers and 8 unguarded commands per loop
// iteration (uniform_random: target of every command from the per-chunk table).
template <bool RED>
__device__ __forceinline__ void tma_group(uint32_t d0, uint32_t src_u32, uint32_t off, uint32_t end, uint32_t chunk, uint32_t rb) {
#pragma unroll 1
    for (; off + 8u * chunk <= end; off += 8u * chunk) {  // 8 unguarded commands per iteration
#pragma unroll
        for (uint32_t u = 0; u < 8u; ++u) {
            if constexpr (RED) bulk_red_s2s_add_u32(d0 + off + u * chunk, src_u32 + off + u * chunk, chunk, rb);
            else bulk_s2s_dsmem(d0 + off + u * chunk, src_u32 + off + u * chunk, chunk, rb);
        }
    }
#pragma unroll 1
    for (; off < end; off += chunk) {
        const uint32_t bytes = min(chunk, end - off);
        if constexpr (RED) bulk_red_s2s_add_u32(d0 + off, src_u32 + off, bytes, rb);
        else bulk_s2s_dsmem(d0 + off, src_u32 + off, bytes, rb);
    }
}
template <bool RED, bool UNI>
__device__ __forceinline__ void tma_issue(const Group11* gs, int ng, uint32_t passes, uint32_t chunk, uint32_t src_u32,
                                          uint32_t dst_u32, uint32_t bar_u32, uint32_t rank, const uint8_t* ct) {
#pragma unroll 1
    for (uint32_t pass = 0; pass < passes; ++pass) {
#pragma unroll 1
        for (int i = 0; i < ng; ++i) {
            const Group11 g = gs[i];
            if (g.op != OP_TMA && g.op != OP_TRED) continue;
            const uint32_t end = g.e1 * 16u;
            if constexpr (UNI) {
#pragma unroll 1
                for (uint32_t off = g.e0 * 16u; off < end; off += chunk) {
                    const uint32_t t = ct[pass * 6u + off / 16384u], bytes = min(chunk, end - off);
                    const uint32_t d = mapa_u32(dst_u32, t) + off, rb = mapa_u32(bar_u32 + (rank * passes + pass) * 8u, t);
                    if constexpr (RED) bulk_red_s2s_add_u32(d, src_u32 + off, bytes, rb);
                    else bulk_s2s_dsmem(d, src_u32 + off, bytes, rb);
                }
            } else {
                tma_group<RED>(mapa_u32(dst_u32, g.target), src_u32, g.e0 * 16u, end, chunk,
                               mapa_u32(bar_u32 + (rank * passes + pass) * 8u, g.target));
            }
        }
    }
}

// warp reduction of one record slot (slot warp-uniform), then shared atomics by lane 0
__device__ __forceinline__ void reduce_slot(bool part, TimerPair a, TimerPair b, unsigned long long* c0, unsigned long long* c1,
                                            unsigned long long* n0, unsigned long long* n1, uint32_t sy, uint32_t* chk) {
    unsigned long long x0 = part ? a.cycles : ~0ull, x1 = part ? b.cycles : 0ull, y0 = part ? a.ns : ~0ull, y1 = part ? b.ns : 0ull;
    uint32_t s = part ? sy : 0u;
    for (int o = 16; o; o >>= 1) {
        x0 = min(x0, __shfl_xor_sync(0xffffffffu, x0, o));
        x1 = max(x1, __shfl_xor_sync(0xffffffffu, x1, o));
        y0 = min(y0, __shfl_xor_sync(0xffffffffu, y0, o));
        y1 = max(y1, __shfl_xor_sync(0xffffffffu, y1, o));
        s += __shfl_xor_sync(0xffffffffu, s, o);
    }
    if ((threadIdx.x & 31) == 0 && x1 != 0ull) {  // some lane of the warp participated
        atomicMin(c0, x0); atomicMax(c1, x1); atomicMin(n0, y0); atomicMax(n1, y1);
        if (chk) atomicAdd(chk, s);
    }
}

// ---------------------------------------------------------------- K1–K3 kernel
template <int OP, int ILP, bool UNI>
__global__ void __launch_bounds__(1024, 1) k_stream(P11 p) {
    extern __shared__ __align__(1024) uint8_t dyn[];
    __shared__ Group11 s_g[kMaxGroups];
    __shared__ int s_ng;
    __shared__ uint32_t s_exp[kMaxBars];      // bytes expected on barrier (source, pass) in this sample
    __shared__ uint32_t s_ph[kMaxBars / 32];  // phase parity per barrier
    __shared__ uint8_t s_ct[kMaxCt];          // uniform_random: target of (pass, chunk)
    __shared__ unsigned long long s_c0[3], s_c1[3], s_n0[3], s_n1[3];
    __shared__ uint32_t s_chk[2];
    constexpr bool ASYNC = OP == OP_STA || OP == OP_TMA || OP == OP_TRED || OP == OP_MIX_TMA;
    constexpr bool MIXED = OP == OP_MIX_ST || OP == OP_MIX_TMA;
    const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, nwarps = blockDim.x >> 5;
    const ClusterRegs c = read_cluster_regs();
    const uint32_t rank = c.ctarank, cs = c.nctarank;
    const uint32_t nelem = p.payload / 16u, passes = p.passes, nbars = cs * passes;
    const uint32_t src_u32 = smem_u32(dyn), dst_u32 = src_u32 + kDstOff, bar_u32 = src_u32 + kBarOff;
    const uint32_t zero = (uint32_t)p.zero;
    uint4* src = reinterpret_cast<uint4*>(dyn);
    for (uint32_t e = tid; e < kTileElems; e += blockDim.x) src[e] = make_uint4(e, rank + 1u, e ^ 0x5a5a5a5au, rank);
    record_place(p.place);
    if (tid == 0) s_ng = plan_groups(OP, p.pat, cs, p.k, rank, nwarps, nelem, p.perm, s_g);
    if (ASYNC) {
        for (uint32_t b = tid; b < nbars; b += blockDim.x) mbar_init(bar_u32 + b * 8u, 1);
        for (uint32_t w = tid; w < kMaxBars / 32; w += blockDim.x) s_ph[w] = 0;
        fence_mbarrier_init();
    }
    fence_proxy_async_shared_cta();  // source tile written by generic stores, read by the TMA unit
    __syncthreads();
    // my group (warp-uniform) and the TMA issuer
    int my = -1, first_tma = -1;
    const int ng = s_ng;
    for (int i = 0; i < ng; ++i) {
        if (warp >= s_g[i].w0 && warp < s_g[i].w1) my = i;
        if (first_tma < 0 && (s_g[i].op == OP_TMA || s_g[i].op == OP_TRED)) first_tma = i;
    }
    const bool tma_issuer = first_tma >= 0 && lane == 0 && warp == s_g[first_tma].w0;
    const bool waiter = ASYNC && warp == nwarps - 1 && lane == 0;
    cluster_sync();
#pragma unroll 1
    for (uint32_t sample = 0; sample < p.samples; ++sample) {
        // ---- arm (untimed)
        if (tid < 3) { s_c0[tid] = ~0ull; s_c1[tid] = 0; s_n0[tid] = ~0ull; s_n1[tid] = 0; }
        if (tid < 2) s_chk[tid] = 0;
        if (ASYNC && tid < cs) {
            const uint32_t s = tid;
            if (UNI) {
                for (uint32_t q = 0; q < passes; ++q)
                    s_exp[s * passes + q] = incoming_bytes(OP, p.pat, cs, p.k, s, rank, nwarps, nelem, p.perm, p.seed, sample, q);
            } else {
                const uint32_t b = incoming_bytes(OP, p.pat, cs, p.k, s, rank, nwarps, nelem, p.perm, p.seed, sample, 0);
                for (uint32_t q = 0; q < passes; ++q) s_exp[s * passes + q] = b;
            }
        }
        if (UNI)
            for (uint32_t i = tid; i < passes * 6u && i < kMaxCt; i += blockDim.x)
                s_ct[i] = (uint8_t)chunk_target(p.seed, sample, i / 6u, rank, i % 6u, cs);
        __syncthreads();
        if (ASYNC)
            for (uint32_t b = tid; b < nbars; b += blockDim.x)
                if (s_exp[b]) mbar_arrive_expect_tx(bar_u32 + b * 8u, s_exp[b]);
        __syncthreads();
        cluster_sync();
        // ---- timed region
        const TimerPair a = read_timers_pair();
        TimerPair b = a;
        bool part = false;
        uint32_t sy = 0, sx = 0;
        int slot = 0;
        if (my >= 0) {
            const Group11 g = s_g[my];
            slot = (MIXED && g.op != OP_LD) ? 1 : 0;
            if (g.op == OP_LD) {
                if constexpr (OP == OP_LD || MIXED) {
                    ld_stream<ILP, UNI>(g, warp, lane, passes, src_u32, s_ct, zero, sy, sx);
                    b = read_timers_after((uint64_t)(sy ^ sx), p.zero);
                    part = true;
                }
            } else if (g.op == OP_ST) {
                if constexpr (OP == OP_ST || OP == OP_MIX_ST) {
                    st_stream<ILP, UNI, false>(g, warp, lane, passes, dst_u32, bar_u32, rank, sample, s_ct, zero);
                    fence_acq_rel_cluster();
                    b = read_timers_pair();
                    part = true;
                }
            } else if (g.op == OP_STA) {
                if constexpr (OP == OP_STA) {
                    st_stream<ILP, UNI, true>(g, warp, lane, passes, dst_u32, bar_u32, rank, sample, s_ct, zero);
                    b = read_timers_pair();
                    part = true;
                }
            } else if (tma_issuer) {
                if constexpr (OP == OP_TMA || OP == OP_TRED || OP == OP_MIX_TMA) {
                    tma_issue<OP == OP_TRED, UNI>(s_g, ng, passes, p.chunk, src_u32, dst_u32, bar_u32, rank, s_ct);
                    b = read_timers_pair();
                    part = true;
                }
            }
        }
        if (waiter) {
            uint32_t done = 1;
            for (uint32_t bi = 0; bi < nbars; ++bi) {
                if (!s_exp[bi]) continue;
                const uint32_t ph = (s_ph[bi >> 5] >> (bi & 31)) & 1u;
                while (!(done = mbar_try_wait_parity_acq_cluster(bar_u32 + bi * 8u, ph)))
                    if (read_clock64() - a.cycles > kWaitTimeout) { atomicOr(p.err, 1u); break; }
                s_ph[bi >> 5] ^= 1u << (bi & 31);
            }
            const TimerPair t = read_timers_after((uint64_t)done, p.zero);
            s_c0[2] = a.cycles; s_c1[2] = t.cycles; s_n0[2] = a.ns; s_n1[2] = t.ns;
        }
        // ---- records (untimed)
        reduce_slot(part, a, b, &s_c0[slot], &s_c1[slot], &s_n0[slot], &s_n1[slot], sy, &s_chk[slot]);
        if (sx == 0x9e3779b9u && sy == 0x7f4a7c15u) atomicOr(p.err, 2u);  // keeps every loaded word live
        __syncthreads();
        if (tid == 0) {
            const size_t base = ((size_t)sample * gridDim.x + blockIdx.x) * 3;
            for (int sl = 0; sl < 2; ++sl) {
                unsigned long long bytes = 0;
                uint32_t peer = 0xffffffffu, n = 0, op = 0;
                for (int i = 0; i < ng; ++i) {
                    const int gsl = (MIXED && s_g[i].op != OP_LD) ? 1 : 0;
                    if (gsl != sl) continue;
                    bytes += (unsigned long long)(s_g[i].e1 - s_g[i].e0) * 16ull * passes;
                    peer = ++n == 1 && s_g[i].target != kChunkTarget ? s_g[i].target : 0xffffffffu;
                    op = s_g[i].op;
                }
                if (n && s_c1[sl])
                    p.out[base + sl] = Rec11{s_c1[sl] - s_c0[sl], s_n1[sl] - s_n0[sl], s_n0[sl], bytes, peer, 1u, s_chk[sl], op};
            }
            if (ASYNC) {
                unsigned long long bytes = 0;
                uint32_t peer = 0xffffffffu, n = 0;
                for (uint32_t s = 0; s < cs; ++s) {
                    unsigned long long bs = 0;
                    for (uint32_t q = 0; q < passes; ++q) bs += s_exp[s * passes + q];
                    if (bs) { bytes += bs; peer = ++n == 1 ? s : 0xffffffffu; }
                }
                if (bytes)
                    p.out[base + 2] = Rec11{s_c1[2] - s_c0[2], s_n1[2] - s_n0[2], s_n0[2], bytes, peer, 1u, 0u,
                                            (uint32_t)(OP == OP_MIX_TMA ? OP_TMA : OP)};
            }
        }
        __syncthreads();
    }
    cluster_sync();  // no CTA exits while a peer may still access its smem
}

// ---------------------------------------------------------------- end of the verbatim part

using KFn = void (*)(P11);

struct Cfg {
    int op = OP_LD, pat = PT_RING, k = 1, cs = 2;
    int threads = 1024, ilp = 4;
    uint32_t chunk = 16384, payload = kTile;
    int passes = 1;
};
static bool is_tma(int op) { return op == OP_TMA || op == OP_TRED; }

// Host value of the ld check word of CTA `rank`, slot 0, sample `sample` (verbatim from MB dsmem_bw.cu).
static uint32_t expected_chk(const Cfg& c, const Perm16& perm, uint32_t seed, int rank, int sample) {
    Group11 g[kMaxGroups];
    const int n = plan_groups(c.op, c.pat, c.cs, c.k, rank, c.threads / 32, c.payload / 16, perm, g);
    uint32_t s = 0;
    for (int i = 0; i < n; ++i) {
        if (g[i].op != OP_LD) continue;
        for (int q = 0; q < c.passes; ++q) {
            if (g[i].target == kChunkTarget) {
                for (uint32_t e = g[i].e0; e < g[i].e1; ++e) s += chunk_target(seed, sample, q, rank, e / kChunkElems, c.cs) + 1u;
            } else {
                s += (g[i].e1 - g[i].e0) * (g[i].target + 1u);
            }
        }
    }
    return s;
}

template <class T>
struct DevBuf {  // MB 09 common09.hpp DevBuf (verbatim)
    T* d = nullptr;
    size_t n = 0;
    explicit DevBuf(size_t n_) : n(n_) { CUDA_CHECK(cudaMalloc(&d, n * sizeof(T))); CUDA_CHECK(cudaMemset(d, 0, n * sizeof(T))); }
    ~DevBuf() { cudaFree(d); }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    std::vector<T> get(size_t m = 0) const {
        std::vector<T> h(m ? m : n);
        CUDA_CHECK(cudaMemcpy(h.data(), d, h.size() * sizeof(T), cudaMemcpyDeviceToHost));
        return h;
    }
    void zero(size_t m = 0) { CUDA_CHECK(cudaMemset(d, 0, (m ? m : n) * sizeof(T))); }
};

// One prepared configuration: full launch (max active clusters at 1 CTA/SM).
struct Launcher {
    Cfg c;
    KFn k = nullptr;
    hb::ClusterLaunch::Prepared prep;
    int maxact = 0, ncl = 0, nb = 0;
    size_t nrec = 0;
    P11 p{};
    float ms = 0;
    Launcher(const Cfg& c_, DevBuf<Rec11>& out, DevBuf<uint32_t>& place, DevBuf<uint32_t>& err) : c(c_) {
        k = c.op == OP_LD ? k_stream<OP_LD, 4, false> : k_stream<OP_TMA, 4, false>;
        hb::ClusterLaunch L;
        L.block = dim3(c.threads);
        L.cluster = dim3(c.cs, 1, 1);
        L.smem = kSmem;
        L.explicit_cluster = true;
        L.policy = cudaClusterSchedulingPolicyDefault;
        L.grid = dim3(c.cs);
        maxact = L.max_active_clusters(k);
        ncl = maxact;
        nb = ncl * c.cs;
        L.grid = dim3(nb);
        L.prepare(k, prep);
        nrec = (size_t)nb * 3;  // 1 sample
        if (op_async(c.op) && c.cs * c.passes > (int)kMaxBars) hb::die("too many barriers (MB check_cfg)");
        if (nrec > out.n || 4 * (size_t)nb > place.n) hb::die("record buffer too small");
        p.pat = c.pat; p.k = c.k; p.payload = c.payload; p.chunk = c.chunk; p.samples = 1; p.passes = c.passes;
        p.seed = 0; p.zero = 0; p.out = out.d; p.place = place.d; p.err = err.d;
        for (int i = 0; i < 16; ++i) p.perm.p[i] = (uint8_t)((i + 1) % c.cs);  // unused by the ring pattern
    }
};

static std::string label_of(const Cfg& c) {
    char b[160];
    std::snprintf(b, sizeof b, "dsmem_%s_%s%d_cs%d_t%d_i%d_c%u_p%u_x%d", kOpName[c.op], kPatName[c.pat], c.k, c.cs, c.threads, c.ilp, c.chunk,
                  c.payload, c.passes);
    return b;
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "S";
    const bool quick = hb::has_flag(argc, argv, "--quick");
    hb::Run run(kExp, argc, argv);
    const std::string label = hb::arg_str(argc, argv, "--label", mode.c_str());
    if (mode == "clockcheck") {
        run.lock_clocks_note(label, "check");
        run.complete();
        return 0;
    }
    const int sm = run.dev().sm_count;
    DevBuf<Rec11> out((size_t)sm * 3);
    DevBuf<uint32_t> place(4 * (size_t)sm), err(1);
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    Session s(run);

    auto launch_once = [&](Launcher& l) {
        CUDA_CHECK(cudaMemset(out.d, 0, l.nrec * sizeof(Rec11)));
        CUDA_CHECK(cudaMemset(place.d, 0, 4 * (size_t)l.nb * sizeof(uint32_t)));
        CUDA_CHECK(cudaEventRecord(e0));
        CUDA_CHECK(cudaLaunchKernelEx(&l.prep.cfg, l.k, l.p));
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventElapsedTime(&l.ms, e0, e1));
        const uint32_t e = err.get()[0];
        if (e & 1u) hb::die(label_of(l.c) + ": mbarrier wait timed out (expected bytes never arrived)");
        return Placement(place.get(4 * (size_t)l.nb), l.c.cs, l.ncl);
    };

    const int ops[] = {OP_LD, OP_TMA};
    const int css[] = {2, 4, 8, 16};
    if (mode == "S") {
        hb::CsvWriter csv(out_csv("dsmem_bw"),
                          hb::with_mandatory({"op", "pattern", "cs", "threads", "ilp", "chunk_bytes", "payload_bytes", "passes", "nclusters",
                                              "max_active_clusters", "active_ctas", "distinct_sms", "total_bytes", "event_ns", "makespan_ns",
                                              "bpc_cta_median", "placement_bad", "clock_bad"}));
        const std::vector<int> passes = quick ? std::vector<int>{4, 16} : std::vector<int>{4, 8, 16, 32};
        run.lock_clocks_note("dsmem_bw", "before");
        for (int op : ops)
            for (int cs : css) {
                std::vector<double> xs, ys;  // total bytes, event ns (good launches) → slope printout
                for (int q : passes) {
                    Cfg c;
                    c.op = op; c.cs = cs; c.passes = q;
                    Launcher l(c, out, place, err);
                    const std::string lab = label_of(c);
                    std::vector<uint32_t> last_sm;
                    const int bad = record(
                        s, lab, kWarmups, quick ? kQuickRecorded : kRecorded,
                        [&] {
                            const Placement pl = launch_once(l);
                            last_sm = pl.sm;
                            return pl.ok();
                        },
                        [&](const std::string& pre, const hb::ClockSample&, bool pb, bool cb) {
                            const std::vector<Rec11> o = out.get(l.nrec);
                            unsigned long long gmin = ~0ull, gmax = 0, total = 0;
                            std::vector<double> bpc;
                            std::vector<char> active(l.nb, 0);
                            for (int blk = 0; blk < l.nb; ++blk)
                                for (int sl = 0; sl < 3; ++sl) {
                                    const Rec11& r = o[(size_t)blk * 3 + sl];
                                    if (!r.valid) continue;
                                    gmin = std::min(gmin, r.t0_gt);
                                    gmax = std::max(gmax, r.t0_gt + r.ns);
                                    if (sl == 0 && c.op == OP_LD) {
                                        const uint32_t want = expected_chk(c, l.p.perm, l.p.seed, blk % c.cs, 0);
                                        if (r.chk != want)
                                            hb::die(lab + ": ld check of block " + std::to_string(blk) + " is " + std::to_string(r.chk) +
                                                    ", expected " + std::to_string(want));
                                    }
                                    if (sl < 2) { total += r.bytes; active[blk] = 1; }
                                    const bool bw_side = op_async(c.op) ? sl == 2 : sl == 0;
                                    if (bw_side && r.cycles) bpc.push_back((double)r.bytes / r.cycles);
                                }
                            int nact = 0;
                            for (char a : active) nact += a;
                            const int distinct = (int)std::set<uint32_t>(last_sm.begin(), last_sm.end()).size();
                            if (!pb && !cb) { xs.push_back((double)total); ys.push_back(l.ms * 1e6); }
                            char b[512];
                            std::snprintf(b, sizeof b, "%s,%s,%s,%d,%d,%d,%u,%u,%d,%d,%d,%d,%d,%llu,%.0f,%llu,%.4f,%d,%d", pre.c_str(), kOpName[c.op],
                                          kPatName[c.pat], c.cs, c.threads, is_tma(c.op) ? 0 : c.ilp, is_tma(c.op) ? c.chunk : 0u, c.payload,
                                          c.passes, l.ncl, l.maxact, nact, distinct, total, l.ms * 1e6, gmax - gmin, median_of(bpc), (int)pb,
                                          (int)cb);
                            csv.add(b);
                            csv.flush();
                        });
                    csv.sync();
                    if (bad) std::printf("  %s: %d flagged launches\n", lab.c_str(), bad);
                }
                // console only: least-squares slope of event time over bytes (the summary recomputes it per launch index)
                double mx = 0, my = 0, sxy = 0, sxx = 0;
                for (size_t i = 0; i < xs.size(); ++i) { mx += xs[i]; my += ys[i]; }
                mx /= xs.size(); my /= ys.size();
                for (size_t i = 0; i < xs.size(); ++i) { sxy += (xs[i] - mx) * (ys[i] - my); sxx += (xs[i] - mx) * (xs[i] - mx); }
                const double slope = sxy / sxx;  // ns per byte
                std::printf("%-8s cs %2d: %6.3f TB/s GPU-wide (event-time slope over %zu launches), fixed %.1f us\n", kOpName[op], cs,
                            1e-3 / slope, xs.size(), (my - slope * mx) * 1e-3);
                std::fflush(stdout);
            }
        run.lock_clocks_note("dsmem_bw", "after");
    } else if (mode == "ncu") {
        hb::CsvWriter man(out_csv("ncu/manifest_dsmem_bw"),
                          hb::with_mandatory({"op", "pattern", "cs", "nclusters", "threads", "ilp", "chunk_bytes", "payload_bytes", "passes",
                                              "issuer_bytes", "event_ns"}));
        run.lock_clocks_note("dsmem_bw_ncu", "before");
        for (int op : ops)
            for (int cs : css) {
                Cfg c;
                c.op = op; c.cs = cs; c.passes = 4;
                Launcher l(c, out, place, err);
                const Placement pl = launch_once(l);
                const std::vector<Rec11> o = out.get(l.nrec);
                unsigned long long bytes = 0;
                for (size_t i = 0; i < o.size(); ++i) if (o[i].valid && i % 3 < 2) bytes += o[i].bytes;
                const hb::ClockSample cs_ = run.clock_note("ncu_" + label_of(c), "launch", s.li);
                if (!pl.ok()) std::printf("  WARNING %s: two ranks of a cluster on one SM\n", label_of(c).c_str());
                char b[256];
                std::snprintf(b, sizeof b, ",%s,%s,%d,%d,%d,%d,%u,%u,%d,%llu,%.0f", kOpName[c.op], kPatName[c.pat], c.cs, l.ncl, c.threads,
                              is_tma(c.op) ? 0 : c.ilp, is_tma(c.op) ? c.chunk : 0u, c.payload, c.passes, bytes, l.ms * 1e6);
                man.add(run.prefix(s.li++, cs_) + b);
                man.flush();
            }
        man.sync();
        run.lock_clocks_note("dsmem_bw_ncu", "after");
    } else {
        hb::die("unknown mode " + mode);
    }
    std::printf("dsmem_bw %s: %d launches flagged (clock), %d (placement)\n", mode.c_str(), s.discarded_clock, s.discarded_placement);
    run.complete();
    return 0;
}
