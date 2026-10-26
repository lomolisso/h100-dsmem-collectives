/*
 * bw_l2.cu — roofline R1: L2 read bandwidth GPU-wide, steady state over a 16 MiB working set (fits the 60 MiB L2;
 * DESIGN §2.1 "L2 bandwidth").
 *
 * Kernels copied from MB/experiments/05__memory_hierarchy_bandwidth/src/bw_l2.cu at MB commit
 * 8f8fc7bec69f07db49f3a5a95bbdfb08de5d0ca8:
 *   k3_direct<ILP>   verbatim. MB configuration "mixed_direct": plain grid-stride stream, thread gt reads elements
 *                    gt + i·T + k·ILP·T (i < ILP) of the n4 = F/16 16-B elements with ld.global.cg.v4, `reps` passes,
 *                    (r & zero) added to the address so that ptxas keeps the rep-invariant loads in the rep loop;
 *                    2 × 132 blocks × 1024 threads, ILP 4, F = 16 MiB, 2048 reps (≈ 34 GB per launch, ≈ 4.7 ms).
 *   k3_stream<OP_LD, 4, false, C_CG>  (StreamArgs, warp_range, access_step, stream_body verbatim). MB configuration
 *                    "mixed" (the row l2_ld_mixed = 7.7 TB/s on GPU 2 / 7.75 on GPU 3 in MB AGENTS.md comes from it, not
 *                    from mixed_direct, which gave ≈ 7.3 TB/s raw in MB's runs): one group of all blocks over all lines
 *                    of F, each warp a fixed range of lines (8 lanes × 16 B per 128-B line, 4 lines per access, ILP 4
 *                    accesses per step), reps chosen for ≈ 8 GiB per launch. Measured here as a second L2 read method;
 *                    ceilings_summary.py reports both and l2_bw = the higher median.
 *   The home-labelling machinery (touch/probe kernels, line lists, local/remote variants, link, first-pass, store-copy,
 *   ncu modes) is dropped. c_part / c_gpc stay as __constant__ tables that are never filled (all 0) and reader_gpcs is
 *   0xFF, so every block is active and in group 0 — exactly MB's "mixed" variant.
 * No L2 flush: the 2 warm-up launches bring the working set into L2 (steady state, as in MB).
 * Placement: 2 blocks on every SM (host check from %smid), otherwise the launch is flagged and repeated (as in MB).
 *
 * Usage (from roofline/): harness/ceilings/bin/bw_l2 {k3|clockcheck} [--quick] [--label L]
 *   --quick: 256 reps (direct), ≈ 1 GiB (stream), 3 recorded launches.
 * → results/ceilings/bw_l2.csv (one row per launch): mandatory columns, variant (direct | stream), mb_variant
 *   (mixed_direct | mixed), footprint_bytes, blocks, threads, ilp, reps, bytes, ms_event, tbps_raw (floor not removed),
 *   sm_mhz_blocks, span_ns, placement_bad, clock_bad.
 */
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "ceil_common.hpp"

static constexpr uint32_t kLine = 128;
static constexpr uint64_t kMiB = 1ull << 20, kTargetBytes = 8ull << 30;

enum Op { OP_LD = 0, OP_ST = 1, OP_MIX = 2 };

__constant__ uint8_t c_part[256], c_gpc[256];  // MB: L2 partition / GPC of every SM; not filled here (all 0)

// ---------------------------------------------------------------- verbatim from MB bw_l2.cu
struct StreamArgs {
    char* buf;
    const uint32_t* list[2];  // LIST: line indices of group g
    uint32_t len[2];          // lines in group g's list
    uint32_t warps[2];        // warps of group g (expected)
    uint32_t reps;
    uint32_t by_partition;    // 0: one group (g = 0) of all blocks; 1: g = partition of the block's SM
    uint32_t reader_gpcs;     // bit g clear: blocks on SMs of GPC g exit at once
    uint32_t max_count;       // shared-memory list slots per warp (LIST)
    uint32_t* tickets;        // [2], zeroed before every launch
    BlockRec* rec;
    uint64_t zero;
};

__host__ __device__ inline void warp_range(uint32_t w, uint32_t W, uint32_t L, uint32_t step, uint32_t& start, uint32_t& count) {
    if (L == 0) {
        start = count = 0;
    } else if ((uint64_t)L >= (uint64_t)step * W) {
        start = (uint32_t)((uint64_t)w * L / W);
        count = (uint32_t)((uint64_t)(w + 1) * L / W) - start;
    } else {
        start = (uint32_t)((uint64_t)w * step % L);
        count = step;
    }
}

// One step of a warp: ILP accesses (lanes 8k…8k+7 cover line[i] in 16-B pieces), all issued before any loaded data is
// consumed; okm bit i = slot i is inside the warp's range.
template <int OP, int ILP, int C>
__device__ __forceinline__ void access_step(uint64_t base, const uint32_t (&line)[ILP], uint32_t okm, uint32_t r, uint32_t& acc) {
    V4 v[ILP];
#pragma unroll
    for (int i = 0; i < ILP; ++i) {
        if (!((okm >> i) & 1)) continue;
        const uint64_t ad = base + (uint64_t)line[i] * kLine;
        if (OP == OP_LD || (OP == OP_MIX && i % 2 == 0)) v[i] = ld_v4<C>(ad);  // folded after unrolling
        else st_v4<C_WB>(ad, r);
    }
#pragma unroll
    for (int i = 0; i < ILP; ++i)
        if (OP == OP_LD || (OP == OP_MIX && i % 2 == 0))
            if ((okm >> i) & 1) xor_into(acc, v[i]);
}

template <int OP, int ILP, bool LIST, int C>
__device__ __forceinline__ void stream_body(const StreamArgs& a) {
    extern __shared__ uint32_t s_idx[];
    __shared__ uint32_t s_ticket;
    __shared__ uint32_t s_lines;  // line accesses of the block (one 32-bit shared atomic per warp)
    constexpr uint32_t kStep = 4 * ILP;  // lines per warp step
    const uint32_t smid = read_smid(), part = c_part[smid];
    const uint32_t g = a.by_partition ? part : 0;
    const uint32_t warp = threadIdx.x / 32, lane = threadIdx.x % 32, nw = blockDim.x / 32;
    const bool active = (a.reader_gpcs >> c_gpc[smid]) & 1u;
    if (threadIdx.x == 0) {
        s_ticket = active ? atomicAdd(&a.tickets[g], 1u) : ~0u;
        s_lines = 0;
    }
    __syncthreads();
    if (!active) {
        if (threadIdx.x == 0) a.rec[blockIdx.x] = {0, 0, 0, 0, 0, smid, part, ~0u, 0};
        return;
    }
    const uint32_t W = a.warps[g], L = a.len[g];
    uint32_t wg = s_ticket * nw + warp;
    const uint32_t overflow = wg >= W;  // more blocks in this group than expected: placement check discards the launch
    wg %= W;
    uint32_t start, count;
    warp_range(wg, W, L, kStep, start, count);
    uint32_t* my = s_idx + warp * a.max_count;
    if constexpr (LIST) {
        const uint32_t padded = (count + kStep - 1) / kStep * kStep;
        for (uint32_t j = lane; j < padded; j += 32) {
            uint32_t pos = start + j;
            if (pos >= L) pos -= L;
            my[j] = j < count ? a.list[g][pos] : 0u;
        }
    }
    if (lane == 0) atomicAdd(&s_lines, count * a.reps);
    const uint32_t sub = (lane % 8) * 16, li = lane / 8;
    const uint64_t base = (uint64_t)a.buf + sub;
    uint32_t acc = 0;
    __syncthreads();
    TimerPair t0{};
    if (threadIdx.x == 0) t0 = read_timers_pair();
    if (count <= kStep) {
        // the warp's whole range is one step (every footprint up to 16 MiB mixed / 32 MiB local or remote with this
        // grid): its line indices stay in registers for all reps, and the timed loop contains only the accesses.
        // base + (r & zero) = base, but ptxas cannot prove it, so the rep-invariant loads stay inside the rep loop.
        uint32_t line[ILP], okm = 0;
#pragma unroll
        for (int i = 0; i < ILP; ++i) {
            const uint32_t j = 4 * i + li;
            okm |= (j < count) << i;
            if constexpr (LIST) line[i] = my[j];
            else { line[i] = start + j; if (line[i] >= L) line[i] -= L; }
        }
#pragma unroll 1
        for (uint32_t r = 0; r < a.reps; ++r) access_step<OP, ILP, C>(base + (r & a.zero), line, okm, r, acc);
    } else {
#pragma unroll 1
        for (uint32_t r = 0; r < a.reps; ++r) {
#pragma unroll 1
            for (uint32_t j0 = 0; j0 < count; j0 += kStep) {
                uint32_t line[ILP], okm = 0;  // 32-bit line indices first (the list slice is padded to whole steps)
#pragma unroll
                for (int i = 0; i < ILP; ++i) {
                    const uint32_t j = j0 + 4 * i + li;
                    okm |= (j < count) << i;
                    if constexpr (LIST) line[i] = my[j];
                    else { line[i] = start + j; if (line[i] >= L) line[i] -= L; }
                }
                access_step<OP, ILP, C>(base + (r & a.zero), line, okm, r, acc);
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const TimerPair t1 = read_timers_after(acc, a.zero);
        a.rec[blockIdx.x] = {t0.cycles, t1.cycles, t0.ns, t1.ns, (uint64_t)s_lines * kLine, smid, part, s_ticket, overflow};
    }
}

template <int OP, int ILP, bool LIST, int C>
__global__ void __launch_bounds__(1024, ILP >= 8 ? 1 : 2) k3_stream(StreamArgs a) { stream_body<OP, ILP, LIST, C>(a); }

// Plain grid-stride stream (no lists): thread gt reads elements gt + i·T + k·ILP·T (i < ILP) of n4 16-B elements.
template <int ILP>
__global__ void __launch_bounds__(1024, 2) k3_direct(const char* buf, uint64_t n4, uint32_t reps, BlockRec* rec, uint64_t zero) {
    __shared__ uint32_t s_units;  // 16-byte loads of the block over all reps
    const uint64_t T = (uint64_t)gridDim.x * blockDim.x, gt = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x == 0) s_units = 0;
    __syncthreads();
    uint32_t mine = 0;
    for (uint64_t b = gt; b < n4; b += ILP * T)
        for (int i = 0; i < ILP; ++i) mine += (b + i * T < n4);
    mine = __reduce_add_sync(0xffffffffu, mine * reps);
    if (threadIdx.x % 32 == 0) atomicAdd(&s_units, mine);
    uint32_t acc = 0;
    __syncthreads();
    TimerPair t0{};
    if (threadIdx.x == 0) t0 = read_timers_pair();
#pragma unroll 1
    for (uint32_t r = 0; r < reps; ++r) {
#pragma unroll 1
        for (uint64_t b = gt; b < n4; b += ILP * T) {
            V4 v[ILP];
#pragma unroll
            for (int i = 0; i < ILP; ++i)
                if (b + i * T < n4) v[i] = ld_v4<C_CG>((uint64_t)buf + (r & zero) + (b + i * T) * 16);
#pragma unroll
            for (int i = 0; i < ILP; ++i)
                if (b + i * T < n4) acc ^= xor4(v[i]);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const TimerPair t1 = read_timers_after(acc, zero);
        rec[blockIdx.x] = {t0.cycles, t1.cycles, t0.ns, t1.ns, (uint64_t)s_units * 16, read_smid(), c_part[read_smid()], blockIdx.x, 0};
    }
}
// ---------------------------------------------------------------- end of the verbatim part

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "k3";
    const bool quick = hb::has_flag(argc, argv, "--quick");
    hb::Run run(kExp, argc, argv);
    const std::string label = hb::arg_str(argc, argv, "--label", mode.c_str());
    if (mode == "clockcheck") {
        run.lock_clocks_note(label, "check");
        run.complete();
        return 0;
    }
    if (mode != "k3") hb::die("unknown mode " + mode);
    const int sm_count = run.dev().sm_count, blocks = 2 * sm_count, threads = 1024;
    const uint64_t F = 16 * kMiB;
    char* d_buf = nullptr;
    BlockRec* d_rec = nullptr;
    uint32_t* d_tickets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf, F));
    CUDA_CHECK(cudaMemset(d_buf, 0, F));
    CUDA_CHECK(cudaMalloc(&d_rec, blocks * sizeof(BlockRec)));
    CUDA_CHECK(cudaMalloc(&d_tickets, 2 * sizeof(uint32_t)));
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    // stream ("mixed") arguments, as MB's prepare() for V_MIXED with all GPCs working
    StreamArgs sa{};
    sa.buf = d_buf;
    sa.tickets = d_tickets;
    sa.rec = d_rec;
    sa.zero = 0;
    sa.reader_gpcs = 0xFF;
    sa.by_partition = 0;
    const uint32_t nF = (uint32_t)(F / kLine), wpb = threads / 32, step = 4 * 4;
    sa.len[0] = sa.len[1] = nF;
    sa.warps[0] = sa.warps[1] = blocks * wpb;
    sa.list[0] = sa.list[1] = nullptr;
    uint64_t bytes_per_rep = 0;
    uint32_t maxc = 0;
    for (uint32_t w = 0; w < sa.warps[0]; ++w) {
        uint32_t st, n;
        warp_range(w, sa.warps[0], nF, step, st, n);
        maxc = std::max(maxc, n);
        bytes_per_rep += (uint64_t)n * kLine;
    }
    sa.max_count = (maxc + step - 1) / step * step;
    const uint64_t target = quick ? (1ull << 30) : kTargetBytes;
    sa.reps = (uint32_t)std::max<uint64_t>(1, (uint64_t)std::llround(double(target) / bytes_per_rep));
    const uint32_t direct_reps = quick ? 256 : 2048;

    hb::CsvWriter csv(out_csv("bw_l2"), hb::with_mandatory({"variant", "mb_variant", "footprint_bytes", "blocks", "threads", "ilp", "reps", "bytes",
                                                             "ms_event", "tbps_raw", "sm_mhz_blocks", "span_ns", "placement_bad", "clock_bad"}));
    Session s(run);
    run.lock_clocks_note("bw_l2", "before");
    for (int v = 0; v < 2; ++v) {
        const bool direct = v == 0;
        const uint32_t reps = direct ? direct_reps : sa.reps;
        const std::string lab = direct ? "l2_direct_F16M" : "l2_stream_F16M";
        std::vector<BlockRec> r(blocks);
        std::vector<double> tb;
        auto launch = [&] {
            CUDA_CHECK(cudaMemsetAsync(d_tickets, 0, 2 * sizeof(uint32_t)));
            CUDA_CHECK(cudaEventRecord(ev0));
            if (direct) k3_direct<4><<<blocks, threads>>>(d_buf, F / 16, reps, d_rec, 0);
            else k3_stream<OP_LD, 4, false, C_CG><<<blocks, threads>>>(sa);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(ev1));
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(r.data(), d_rec, blocks * sizeof(BlockRec), cudaMemcpyDeviceToHost));
            bool overflow = false;
            for (const auto& b : r) overflow |= b.flags != 0;
            return !overflow && PlacementCheck(r, sm_count).exactly(2);
        };
        const int bad = record(s, lab, kWarmups, quick ? kQuickRecorded : kRecorded, launch,
                               [&](const std::string& pre, const hb::ClockSample&, bool pb, bool cb) {
                                   float ms = 0;
                                   CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
                                   uint64_t by = 0, t_lo = ~0ull, t_hi = 0;
                                   std::vector<double> mhz;
                                   for (const auto& b : r) {
                                       by += b.bytes;
                                       t_lo = std::min(t_lo, b.n0);
                                       t_hi = std::max(t_hi, b.n1);
                                       if (b.n1 > b.n0) mhz.push_back(1e3 * double(b.c1 - b.c0) / double(b.n1 - b.n0));
                                   }
                                   const double tbps = by / (ms * 1e-3) / 1e12;
                                   if (!pb && !cb) tb.push_back(tbps);
                                   char row[768];
                                   std::snprintf(row, sizeof row, "%s,%s,%s,%llu,%d,%d,4,%u,%llu,%.6f,%.6f,%.2f,%llu,%d,%d", pre.c_str(),
                                                 direct ? "direct" : "stream", direct ? "mixed_direct" : "mixed", (unsigned long long)F, blocks,
                                                 threads, reps, (unsigned long long)by, ms, tbps, median_of(mhz), (unsigned long long)(t_hi - t_lo),
                                                 (int)pb, (int)cb);
                                   csv.add(row);
                                   csv.flush();
                               });
        csv.sync();
        std::printf("%-16s reps %5u  %7.3f TB/s (median, event floor not removed)%s\n", lab.c_str(), reps, median_of(tb),
                    bad ? ("  flagged launches: " + std::to_string(bad)).c_str() : "");
    }
    run.lock_clocks_note("bw_l2", "after");
    std::printf("bw_l2: %d launches flagged (clock), %d (placement)\n", s.discarded_clock, s.discarded_placement);
    run.complete();
    return 0;
}
