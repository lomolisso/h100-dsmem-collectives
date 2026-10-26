/*
 * compute_peak.cu — roofline R1: CUDA-core compute ceilings (DESIGN §2.1 "FP32 CUDA-core peak" and "INT32 peak").
 * New code (no MB counterpart); the timing / record pattern follows MB 05 (BlockRec, cudaEvent pair, NVML sample per
 * launch).
 *
 *   ffma_peak<ILP>  every thread runs ILP independent dependent chains acc[i] = fma(acc[i], a, b) (fma.rn.f32 in inline
 *                   PTX, a and b kernel arguments so nothing folds; a = 0.9999, b = 1e-4 keep the values near 1.0, no
 *                   denormals or overflow), 256 / ILP chain steps unrolled per loop iteration (256 FFMAs + 3 loop-control
 *                   instructions), ops_per_thread = 2^21 FFMAs per thread;
 *                   the folded result is stored per thread so the chains are live. FLOPs = 2 × FFMA count.
 *   imad_peak<ILP>  the same with mad.lo.u32 (IMAD) chains on 32-bit integers. Convention: one IMAD = 2 integer ops
 *                   (a multiply and an add), the integer analogue of FLOPs = 2 × FFMA; the top-k workload's int_ops are
 *                   counted against this ceiling.
 *   Grid: 4 blocks × 512 threads on every SM (528 blocks, one wave, 64 warps = 16 per SM sub-partition;
 *   __launch_bounds__(512, 4) → ≤ 32 registers), i.e. all 132 SMs. A launch whose blocks are not exactly 4 per SM (%smid)
 *   is flagged and repeated. ILP ∈ {1, 2, 4, 8}; ceilings_summary.py reports the ILP with the highest median.
 *   Timed by a cudaEvent pair around the kernel (≈ 10–40 ms, so the 4 µs event floor is < 0.05 %); per block, thread 0
 *   also reads %clock64 / %globaltimer after a bar.sync before and after the loop (SM clock of the launch; work per cycle
 *   per SM = rate / (132 × clock); expected peak 128 FFMA = 256 FLOP / cycle / SM).
 *
 * SASS check (cuobjdump -sass bin/compute_peak, `make sass` → results/sass/compute_peak.sass; nvcc 12.9.86, sm_90a):
 *   checked 2026-10-26: the timed loop of every instantiation (ffma_peak and imad_peak, ILP 1, 2, 4, 8) is exactly
 *   256 FFMA (resp. 256 IMAD, "IMAD Rk, Rk, Ra, Rb") + IADD3 + ISETP.GE.U32.AND + BRA (loop counter); no other
 *   instruction, no local memory; ILP independent accumulator registers interleaved round-robin; 20–27 registers,
 *   0 spills. Outside the loop: 1 FFMA (resp. 1 IMAD + 1 IMAD.WIDE address) per thread in set-up, not counted
 *   (relative 1/ops_per_thread ≤ 8e-6, visible only in the ncu instruction counters).
 *
 * Usage (from roofline/): harness/ceilings/bin/compute_peak {peak|ncu|clockcheck} [--quick] [--label L]
 *   peak  ffma and imad × ILP {1, 2, 4, 8}: 2 warm-up + 10 recorded launches (3 with --quick, which also uses 1/16 of
 *         the work per thread).
 *   ncu   one launch of ffma ILP 4 and imad ILP 4 (1/16 of the work) for ceilings_ncu.sh (-k regex:_peak); manifest
 *         (expected thread-level instruction counts) → results/ceilings/ncu/manifest_compute_peak.csv.
 * → results/ceilings/compute_peak.csv (one row per launch): mandatory columns, kind (ffma | imad), ilp, blocks, threads,
 *   ops_per_thread, inst_total (thread-level FFMA or IMAD count), work (2 × inst_total; FLOP or int ops), ms_event,
 *   rate_raw (work / event time, floor not removed), sm_mhz_blocks, span_ns, work_per_cycle_per_sm, placement_bad,
 *   clock_bad.
 */
#include <algorithm>
#include <string>
#include <vector>

#include "ceil_common.hpp"

constexpr int kThreads = 512, kBlocksPerSm = 4;
// chain instructions per loop iteration (all ILP chains): the loop control (IADD3 + ISETP + BRA) takes 3 of 259 issue
// slots (1.2 %); with 16 per iteration it took 3 of 19 at ILP 1 (an SMSP issues one warp instruction per cycle).
constexpr int kPerIter = 256;
constexpr uint32_t kOpsPerThread = 1u << 21;  // chain instructions per thread (all ILP chains together)

template <int ILP>
__global__ void __launch_bounds__(kThreads, kBlocksPerSm) ffma_peak(float a, float b, uint32_t steps, float* out, BlockRec* rec, uint64_t zero) {
    float acc[ILP];
#pragma unroll
    for (int i = 0; i < ILP; ++i) acc[i] = 1.0f + 1e-3f * (float)(threadIdx.x & 7) + 1e-4f * (float)i;
    __syncthreads();
    TimerPair t0{};
    if (threadIdx.x == 0) t0 = read_timers_pair();
#pragma unroll 1
    for (uint32_t s = 0; s < steps; ++s) {
#pragma unroll
        for (int u = 0; u < kPerIter / ILP; ++u)
#pragma unroll
            for (int i = 0; i < ILP; ++i) asm("fma.rn.f32 %0, %0, %1, %2;" : "+f"(acc[i]) : "f"(a), "f"(b));
    }
    float fold = 0.0f;
#pragma unroll
    for (int i = 0; i < ILP; ++i) fold += acc[i];
    out[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = fold;
    __syncthreads();
    if (threadIdx.x == 0) {
        const TimerPair t1 = read_timers_after(__float_as_uint(fold), zero);
        rec[blockIdx.x] = {t0.cycles, t1.cycles, t0.ns, t1.ns, (uint64_t)steps * kPerIter * blockDim.x, read_smid(), 0, blockIdx.x, 0};
    }
}

template <int ILP>
__global__ void __launch_bounds__(kThreads, kBlocksPerSm) imad_peak(uint32_t a, uint32_t b, uint32_t steps, uint32_t* out, BlockRec* rec, uint64_t zero) {
    uint32_t acc[ILP];
#pragma unroll
    for (int i = 0; i < ILP; ++i) acc[i] = threadIdx.x * 2654435761u + (uint32_t)i;
    __syncthreads();
    TimerPair t0{};
    if (threadIdx.x == 0) t0 = read_timers_pair();
#pragma unroll 1
    for (uint32_t s = 0; s < steps; ++s) {
#pragma unroll
        for (int u = 0; u < kPerIter / ILP; ++u)
#pragma unroll
            for (int i = 0; i < ILP; ++i) asm("mad.lo.u32 %0, %0, %1, %2;" : "+r"(acc[i]) : "r"(a), "r"(b));
    }
    uint32_t fold = 0;
#pragma unroll
    for (int i = 0; i < ILP; ++i) fold ^= acc[i];
    out[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = fold;
    __syncthreads();
    if (threadIdx.x == 0) {
        const TimerPair t1 = read_timers_after(fold, zero);
        rec[blockIdx.x] = {t0.cycles, t1.cycles, t0.ns, t1.ns, (uint64_t)steps * kPerIter * blockDim.x, read_smid(), 0, blockIdx.x, 0};
    }
}

struct Ctx {
    int sm_count = 0, blocks = 0;
    void* d_out = nullptr;
    BlockRec* d_rec = nullptr;
    cudaEvent_t ev0{}, ev1{};
};

static uint32_t steps_for(uint32_t ops) { return ops / kPerIter; }

static void launch(Ctx& x, bool fp, int ilp, uint32_t steps) {
    CUDA_CHECK(cudaEventRecord(x.ev0));
    if (fp) {
        float* o = static_cast<float*>(x.d_out);
        switch (ilp) {
            case 1: ffma_peak<1><<<x.blocks, kThreads>>>(0.9999f, 1e-4f, steps, o, x.d_rec, 0); break;
            case 2: ffma_peak<2><<<x.blocks, kThreads>>>(0.9999f, 1e-4f, steps, o, x.d_rec, 0); break;
            case 4: ffma_peak<4><<<x.blocks, kThreads>>>(0.9999f, 1e-4f, steps, o, x.d_rec, 0); break;
            case 8: ffma_peak<8><<<x.blocks, kThreads>>>(0.9999f, 1e-4f, steps, o, x.d_rec, 0); break;
            default: hb::die("bad ilp");
        }
    } else {
        uint32_t* o = static_cast<uint32_t*>(x.d_out);
        switch (ilp) {
            case 1: imad_peak<1><<<x.blocks, kThreads>>>(1664525u, 1013904223u, steps, o, x.d_rec, 0); break;
            case 2: imad_peak<2><<<x.blocks, kThreads>>>(1664525u, 1013904223u, steps, o, x.d_rec, 0); break;
            case 4: imad_peak<4><<<x.blocks, kThreads>>>(1664525u, 1013904223u, steps, o, x.d_rec, 0); break;
            case 8: imad_peak<8><<<x.blocks, kThreads>>>(1664525u, 1013904223u, steps, o, x.d_rec, 0); break;
            default: hb::die("bad ilp");
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(x.ev1));
    CUDA_CHECK(cudaDeviceSynchronize());
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "peak";
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
    x.blocks = kBlocksPerSm * x.sm_count;
    CUDA_CHECK(cudaMalloc(&x.d_out, (size_t)x.blocks * kThreads * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&x.d_rec, x.blocks * sizeof(BlockRec)));
    CUDA_CHECK(cudaEventCreate(&x.ev0));
    CUDA_CHECK(cudaEventCreate(&x.ev1));
    Session s(run);
    std::vector<BlockRec> r(x.blocks);

    if (mode == "peak") {
        hb::CsvWriter csv(out_csv("compute_peak"),
                          hb::with_mandatory({"kind", "ilp", "blocks", "threads", "ops_per_thread", "inst_total", "work", "ms_event", "rate_raw",
                                              "sm_mhz_blocks", "span_ns", "work_per_cycle_per_sm", "placement_bad", "clock_bad"}));
        const uint32_t ops = quick ? kOpsPerThread / 16 : kOpsPerThread;
        run.lock_clocks_note("compute_peak", "before");
        for (bool fp : {true, false})
            for (int ilp : {1, 2, 4, 8}) {
                const uint32_t steps = steps_for(ops);
                const std::string lab = std::string(fp ? "ffma" : "imad") + "_ilp" + std::to_string(ilp);
                std::vector<double> rates;
                auto go = [&] {
                    launch(x, fp, ilp, steps);
                    CUDA_CHECK(cudaMemcpy(r.data(), x.d_rec, x.blocks * sizeof(BlockRec), cudaMemcpyDeviceToHost));
                    return PlacementCheck(r, x.sm_count).exactly(kBlocksPerSm);
                };
                const int bad = record(s, lab, kWarmups, quick ? kQuickRecorded : kRecorded, go,
                                       [&](const std::string& pre, const hb::ClockSample&, bool pb, bool cb) {
                                           float ms = 0;
                                           CUDA_CHECK(cudaEventElapsedTime(&ms, x.ev0, x.ev1));
                                           uint64_t inst = 0, t_lo = ~0ull, t_hi = 0;
                                           std::vector<double> mhz;
                                           for (const auto& b : r) {
                                               inst += b.bytes;
                                               t_lo = std::min(t_lo, b.n0);
                                               t_hi = std::max(t_hi, b.n1);
                                               if (b.n1 > b.n0) mhz.push_back(1e3 * double(b.c1 - b.c0) / double(b.n1 - b.n0));
                                           }
                                           const double work = 2.0 * (double)inst, rate = work / (ms * 1e-3), m = median_of(mhz);
                                           if (!pb && !cb) rates.push_back(rate);
                                           char row[640];
                                           std::snprintf(row, sizeof row, "%s,%s,%d,%d,%d,%u,%llu,%.0f,%.6f,%.6e,%.2f,%llu,%.3f,%d,%d", pre.c_str(),
                                                         fp ? "ffma" : "imad", ilp, x.blocks, kThreads, steps * kPerIter, (unsigned long long)inst,
                                                         work, ms, rate, m, (unsigned long long)(t_hi - t_lo),
                                                         m > 0 ? rate / (m * 1e6 * x.sm_count) : 0.0, (int)pb, (int)cb);
                                           csv.add(row);
                                           csv.flush();
                                       });
                csv.sync();
                std::printf("%-10s %8.2f T%s/s (median, event floor not removed)%s\n", lab.c_str(), median_of(rates) / 1e12, fp ? "FLOP" : "IOP",
                            bad ? ("  flagged launches: " + std::to_string(bad)).c_str() : "");
                std::fflush(stdout);
            }
        run.lock_clocks_note("compute_peak", "after");
    } else if (mode == "ncu") {
        hb::CsvWriter man(out_csv("ncu/manifest_compute_peak"),
                          hb::with_mandatory({"kind", "ilp", "blocks", "threads", "ops_per_thread", "inst_total", "ms_event"}));
        run.lock_clocks_note("compute_peak_ncu", "before");
        for (bool fp : {true, false}) {
            const int ilp = 4;
            const uint32_t steps = steps_for(kOpsPerThread / 16);
            launch(x, fp, ilp, steps);
            const hb::ClockSample cs = run.clock_note(std::string("ncu_") + (fp ? "ffma" : "imad"), "launch", s.li);
            CUDA_CHECK(cudaMemcpy(r.data(), x.d_rec, x.blocks * sizeof(BlockRec), cudaMemcpyDeviceToHost));
            uint64_t inst = 0;
            for (const auto& b : r) inst += b.bytes;
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, x.ev0, x.ev1));
            char row[512];
            std::snprintf(row, sizeof row, "%s,%s,%d,%d,%d,%u,%llu,%.6f", run.prefix(s.li, cs).c_str(), fp ? "ffma" : "imad", ilp, x.blocks, kThreads,
                          steps * kPerIter, (unsigned long long)inst, ms);
            man.add(row);
            man.flush();
            ++s.li;
        }
        man.sync();
        run.lock_clocks_note("compute_peak_ncu", "after");
    } else {
        hb::die("unknown mode " + mode);
    }
    std::printf("compute_peak %s: %d launches flagged (clock), %d (placement)\n", mode.c_str(), s.discarded_clock, s.discarded_placement);
    run.complete();
    return 0;
}
