/*
 * bw_hbm.cu — roofline R1: HBM read, write and copy bandwidth GPU-wide (DESIGN §2.1 "HBM bandwidth").
 *
 * Kernel k4<OP, C, ILP> and its launch configuration copied from MB/experiments/05__memory_hierarchy_bandwidth/src/
 * bw_dram.cu at MB commit 8f8fc7bec69f07db49f3a5a95bbdfb08de5d0ca8; read_step, write_step, k4_body, k4 are verbatim
 * (c_part stays as a __constant__ table that is never filled: every block records partition 0). Dropped: the GPC map /
 * L2-partition labelling, the per-block CSV, the active-SM sweep, the cs cache operator, ILP 1/2/8, OP_MIX (5R1W) and the
 * diag mode. Kept (MB "full grid"): 4 × 132 blocks × 1024 threads (2 resident per SM, 2 waves), ILP 4, .cg loads/stores,
 * buffers in/out of 2 GiB each (512 MiB with --quick), CleanL2 (flush + clean read, no dirty write-backs) before every
 * launch, warm-ups included, cudaEvent pair around the kernel only.
 *   op read   ld.global.cg.v4 of `in`, ILP loads issued before their data is XORed (MB "read")
 *   op write  st.global.cg.v4 of `out` (MB "write")
 *   op copy   MB "mix11": 1 read + 1 write per element, g < n4/2: reads in[g…], writes out[g…] (50/50 by bytes); the
 *             stored value is the element index, not the loaded word (two independent streams; MB's 1R1W kernel kept
 *             verbatim rather than writing a new data-dependent copy). bytes = read + written bytes.
 * Rates (ceilings_summary.py): bytes / (ms_event − event floor). Written lines still dirty in the L2 when the kernel ends
 * (MB 05: ≈ 25 MiB of 2 GiB) are taken out of the write/copy payload with the ncu dram__bytes_write of the same
 * configuration when results/ceilings/ncu/bw_hbm.csv exists (MB common05.hbm_write_factor).
 *
 * Usage (from roofline/): harness/ceilings/bin/bw_hbm {k4|ncu|clockcheck} [--quick] [--label L]
 *   k4   2 warm-up + 10 recorded launches (3 with --quick) per op.
 *   ncu  one launch per op (CleanL2 before each), for ceilings_ncu.sh (-k regex:k4, kernel replay, --cache-control all);
 *        manifest → results/ceilings/ncu/manifest_bw_hbm.csv (append, launch order).
 * → results/ceilings/bw_hbm.csv (one row per launch): mandatory columns, op, mb_op, cache_op, ilp, blocks, threads, bytes,
 *   read_bytes, write_bytes, ms_event, tbps_raw (bytes / event time, floor not removed), sm_mhz_blocks (median over blocks
 *   of %clock64 / %globaltimer), span_ns (earliest block start → latest block end), placement_bad, clock_bad.
 */
#include <algorithm>
#include <string>
#include <vector>

#include "ceil_common.hpp"

enum Op { OP_READ = 0, OP_WRITE = 1, OP_MIX = 2, OP_MIX11 = 3 };
static const char* kOpName[] = {"read", "write", "mix", "copy"};
static const char* kMbOpName[] = {"read", "write", "mix", "mix11"};

__constant__ uint8_t c_part[256];  // MB: L2 partition of every SM; not filled here (all 0)

// ---------------------------------------------------------------- verbatim from MB bw_dram.cu
// Element indices are 32-bit (n4 ≤ 2^27 here) and full steps run unguarded, with one guarded tail step: this keeps the
// ILP loads of a step back to back in SASS under the 32-register limit of 2 × 1024 threads per SM (checked in the SASS).
template <int C, int ILP>
__device__ __forceinline__ void read_step(uint64_t ib, uint32_t b, uint32_t T, uint32_t n, bool guard, uint32_t& acc) {
    V4 v[ILP];
#pragma unroll
    for (int i = 0; i < ILP; ++i)
        if (!guard || b + i * T < n) v[i] = ld_v4<C>(ib + (uint64_t)(b + i * T) * 16);
#pragma unroll
    for (int i = 0; i < ILP; ++i)
        if (!guard || b + i * T < n) xor_into(acc, v[i]);
}
template <int C, int ILP>
__device__ __forceinline__ void write_step(uint64_t ob, uint32_t b, uint32_t T, uint32_t n, bool guard) {
#pragma unroll
    for (int i = 0; i < ILP; ++i)
        if (!guard || b + i * T < n) st_v4<C>(ob + (uint64_t)(b + i * T) * 16, b);
}

template <int OP, int C, int ILP>
__device__ __forceinline__ void k4_body(const char* in, char* out, uint32_t n4, BlockRec* rec, uint64_t zero) {
    __shared__ uint32_t s_units;  // 16-byte elements read + written by the block
    const uint32_t T = gridDim.x * blockDim.x, gt = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t ib = (uint64_t)in, ob = (uint64_t)out;
    // elements of the strided stream: n for read/write, n4/2 for mix11 (read stream and write stream of equal size)
    const uint32_t n = OP == OP_MIX11 ? n4 / 2 : n4, G = n4 / 5;
    if (threadIdx.x == 0) s_units = 0;
    __syncthreads();
    // elements this thread moves, counted with the bounds of the timed loop; summed per warp, then one 32-bit shared
    // atomic per warp (a 64-bit shared atomicAdd from every thread is a CAS loop that costs tens of µs per block)
    uint32_t mine = 0;
    if constexpr (OP == OP_MIX) {
        for (uint32_t g = gt; g < G; g += T) mine += 6;
    } else {
        for (uint32_t b = gt; b < n; b += ILP * T)
            for (int i = 0; i < ILP; ++i) mine += (b + i * T < n) * (OP == OP_MIX11 ? 2 : 1);
    }
    mine = __reduce_add_sync(0xffffffffu, mine);
    if (threadIdx.x % 32 == 0) atomicAdd(&s_units, mine);
    uint32_t acc = 0;
    __syncthreads();
    TimerPair t0{};
    if (threadIdx.x == 0) t0 = read_timers_pair();
    if constexpr (OP == OP_MIX) {
#pragma unroll 1
        for (uint32_t g = gt; g < G; g += T) {
            V4 v[5];
#pragma unroll
            for (int k = 0; k < 5; ++k) v[k] = ld_v4<C>(ib + (uint64_t)(g + k * G) * 16);
            uint32_t s = 0;
#pragma unroll
            for (int k = 0; k < 5; ++k) xor_into(s, v[k]);
            st_v4<C>(ob + (uint64_t)g * 16, s);
            acc ^= s;
        }
    } else {
        uint32_t b = gt;
#pragma unroll 1
        for (; b + (ILP - 1) * T < n; b += ILP * T) {
            if constexpr (OP == OP_READ || OP == OP_MIX11) read_step<C, ILP>(ib, b, T, n, false, acc);
            if constexpr (OP == OP_WRITE || OP == OP_MIX11) write_step<C, ILP>(ob, b, T, n, false);  // mix11: data = index
        }
        if (b < n) {
            if constexpr (OP == OP_READ || OP == OP_MIX11) read_step<C, ILP>(ib, b, T, n, true, acc);
            if constexpr (OP == OP_WRITE || OP == OP_MIX11) write_step<C, ILP>(ob, b, T, n, true);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const TimerPair t1 = read_timers_after(acc, zero);
        const uint32_t sm = read_smid();
        rec[blockIdx.x] = {t0.cycles, t1.cycles, t0.ns, t1.ns, (uint64_t)s_units * 16, sm, c_part[sm], blockIdx.x, 0};
    }
}

template <int OP, int C, int ILP>
__global__ void __launch_bounds__(1024, ILP >= 8 ? 1 : 2) k4(const char* in, char* out, uint32_t n4, BlockRec* rec, uint64_t zero) {
    k4_body<OP, C, ILP>(in, out, n4, rec, zero);
}
// ---------------------------------------------------------------- end of the verbatim part

using K4Fn = void (*)(const char*, char*, uint32_t, BlockRec*, uint64_t);
static K4Fn pick(int op) {
    switch (op) {
        case OP_READ: return k4<OP_READ, C_CG, 4>;
        case OP_WRITE: return k4<OP_WRITE, C_CG, 4>;
        case OP_MIX11: return k4<OP_MIX11, C_CG, 4>;
    }
    hb::die("bad op");
}

struct Ctx {
    int sm_count = 0, blocks = 0;
    uint32_t n4 = 0;
    char *d_in = nullptr, *d_out = nullptr;
    BlockRec* d_rec = nullptr;
    CleanL2 flush;
    cudaEvent_t ev0{}, ev1{};
};

static void launch(Ctx& x, int op) {
    x.flush();
    CUDA_CHECK(cudaEventRecord(x.ev0));
    pick(op)<<<x.blocks, 1024>>>(x.d_in, x.d_out, x.n4, x.d_rec, 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(x.ev1));
    CUDA_CHECK(cudaDeviceSynchronize());
}

// payload split of one launch (16-B elements as counted by k4_body)
static void split_bytes(const Ctx& x, int op, uint64_t bytes, uint64_t& rd, uint64_t& wr) {
    if (op == OP_READ) { rd = bytes; wr = 0; }
    else if (op == OP_WRITE) { rd = 0; wr = bytes; }
    else { rd = bytes / 2; wr = bytes - rd; }
    (void)x;
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "k4";
    const bool quick = hb::has_flag(argc, argv, "--quick");
    hb::Run run(kExp, argc, argv);
    const std::string label = hb::arg_str(argc, argv, "--label", mode.c_str());
    if (mode == "clockcheck") {
        run.lock_clocks_note(label, "check");
        run.complete();
        return 0;
    }
    Ctx x;
    x.sm_count = run.dev().sm_count;
    x.blocks = 4 * x.sm_count;
    const size_t bytes = quick ? (512ull << 20) : (2ull << 30);
    x.n4 = (uint32_t)(bytes / 16);
    CUDA_CHECK(cudaMalloc(&x.d_in, bytes));
    CUDA_CHECK(cudaMalloc(&x.d_out, bytes));
    CUDA_CHECK(cudaMemset(x.d_in, 0, bytes));
    CUDA_CHECK(cudaMemset(x.d_out, 0, bytes));
    CUDA_CHECK(cudaMalloc(&x.d_rec, x.blocks * sizeof(BlockRec)));
    CUDA_CHECK(cudaEventCreate(&x.ev0));
    CUDA_CHECK(cudaEventCreate(&x.ev1));
    const int ops[] = {OP_READ, OP_WRITE, OP_MIX11};
    Session s(run);

    if (mode == "k4") {
        hb::CsvWriter csv(out_csv("bw_hbm"), hb::with_mandatory({"op", "mb_op", "cache_op", "ilp", "blocks", "threads", "bytes", "read_bytes",
                                                                  "write_bytes", "ms_event", "tbps_raw", "sm_mhz_blocks", "span_ns",
                                                                  "placement_bad", "clock_bad"}));
        run.lock_clocks_note("bw_hbm", "before");
        for (int op : ops) {
            const std::string lab = std::string("hbm_") + kOpName[op];
            std::vector<double> tb;
            std::vector<BlockRec> r(x.blocks);
            const int bad = record(
                s, lab, kWarmups, quick ? kQuickRecorded : kRecorded, [&] { launch(x, op); return true; },
                [&](const std::string& pre, const hb::ClockSample&, bool pb, bool cb) {
                    CUDA_CHECK(cudaMemcpy(r.data(), x.d_rec, x.blocks * sizeof(BlockRec), cudaMemcpyDeviceToHost));
                    float ms = 0;
                    CUDA_CHECK(cudaEventElapsedTime(&ms, x.ev0, x.ev1));
                    uint64_t by = 0, t_lo = ~0ull, t_hi = 0;
                    std::vector<double> mhz;
                    for (const auto& b : r) {
                        by += b.bytes;
                        t_lo = std::min(t_lo, b.n0);
                        t_hi = std::max(t_hi, b.n1);
                        if (b.n1 > b.n0) mhz.push_back(1e3 * double(b.c1 - b.c0) / double(b.n1 - b.n0));
                    }
                    uint64_t rd, wr;
                    split_bytes(x, op, by, rd, wr);
                    const double tbps = by / (ms * 1e-3) / 1e12;
                    if (!pb && !cb) tb.push_back(tbps);
                    char row[768];
                    std::snprintf(row, sizeof row, "%s,%s,%s,cg,4,%d,1024,%llu,%llu,%llu,%.6f,%.6f,%.2f,%llu,%d,%d", pre.c_str(), kOpName[op],
                                  kMbOpName[op], x.blocks, (unsigned long long)by, (unsigned long long)rd, (unsigned long long)wr, ms, tbps,
                                  median_of(mhz), (unsigned long long)(t_hi - t_lo), (int)pb, (int)cb);
                    csv.add(row);
                    csv.flush();
                });
            csv.sync();
            std::printf("%-12s %7.3f TB/s (median, event floor not removed)%s\n", lab.c_str(), median_of(tb),
                        bad ? ("  flagged launches: " + std::to_string(bad)).c_str() : "");
        }
        run.lock_clocks_note("bw_hbm", "after");
    } else if (mode == "ncu") {
        hb::CsvWriter man(out_csv("ncu/manifest_bw_hbm"),
                          hb::with_mandatory({"op", "mb_op", "cache_op", "ilp", "blocks", "threads", "bytes", "read_bytes", "write_bytes", "ms_event"}));
        run.lock_clocks_note("bw_hbm_ncu", "before");
        std::vector<BlockRec> r(x.blocks);
        for (int op : ops) {
            launch(x, op);
            const hb::ClockSample cs = run.clock_note(std::string("ncu_hbm_") + kOpName[op], "launch", s.li);
            CUDA_CHECK(cudaMemcpy(r.data(), x.d_rec, x.blocks * sizeof(BlockRec), cudaMemcpyDeviceToHost));
            uint64_t by = 0, rd, wr;
            for (const auto& b : r) by += b.bytes;
            split_bytes(x, op, by, rd, wr);
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, x.ev0, x.ev1));
            char row[512];
            std::snprintf(row, sizeof row, "%s,%s,%s,cg,4,%d,1024,%llu,%llu,%llu,%.6f", run.prefix(s.li, cs).c_str(), kOpName[op], kMbOpName[op], x.blocks,
                          (unsigned long long)by, (unsigned long long)rd, (unsigned long long)wr, ms);
            man.add(row);
            man.flush();
            ++s.li;
        }
        man.sync();
        run.lock_clocks_note("bw_hbm_ncu", "after");
    } else {
        hb::die("unknown mode " + mode);
    }
    std::printf("bw_hbm %s: %d launches flagged (clock), %d (placement)\n", mode.c_str(), s.discarded_clock, s.discarded_placement);
    run.complete();
    return 0;
}
