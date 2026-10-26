/*
 * tc_peak.cu — roofline R1: tensor-core instruction-throughput ceilings (DESIGN §2.1 "FP16/BF16 tensor-core peak",
 * "FP8 tensor-core peak"), measured with back-to-back wgmma.mma_async instead of a library GEMM. New code (no MB
 * counterpart); structure, timing and records as compute_peak.cu (BlockRec, cudaEvent pair, NVML sample per launch,
 * flagged launches repeated). Why: the cuBLAS-based ceilings (ceilings_tc.py / ceilings_tc_kernel.py) are below what
 * workload kernels reach (a DAK fp16 GEMM with fp16 accumulation 667 TFLOP/s against the cuBLAS-derived 618), so they are
 * not upper bounds.
 *
 *   tc_peak_kernel<KIND, N, NACC, WAIT>  one CTA per SM (160 KiB dynamic smem, so two CTAs never fit in the 228 KiB of an SM;
 *       grid = 132; a launch whose blocks are not exactly 1 per SM (%smid) is flagged and repeated), C ∈ {1, 2, 3}
 *       consumer warpgroups (blockDim = 128·C, __launch_bounds__(384, 1) → ≤ 168 registers per thread).
 *       Shared memory (1024-byte aligned, filled once before the timed loop with zeros, or in the "rand"
 *       pass of `peak`: hashed random bits with the top exponent bit cleared, i.e. finite values of magnitude < 2 in every
 *       format; contents do not change the instruction count, only the switching power):
 *         A tile per warpgroup  64 rows × 128 B (8 KiB), B tile per accumulator N rows × 128 B (shared by the warpgroups,
 *         as in a 128×N or 192×N CUTLASS tile), both K-major with the 128-byte swizzle (layout_type 1), SBO = 1024 B
 *         (8 rows × 128 B), LBO = 16 B (ignored for swizzled K-major), base_offset 0 (tiles 1024-byte aligned).
 *       Loop (steps iterations): wgmma.fence; 4 k-slices (32 B each: the 128-byte swizzle row = K 64 for f16/bf16, 128
 *       for e4m3; descriptor start address + 2 per slice, as DeepGEMM/CUTLASS advance along K) × NACC independent
 *       accumulators (accumulator a uses B tile a); wgmma.commit_group; wgmma.wait_group WAIT (0: drain each group
 *       before issuing the next; 1: one group in flight while the next is issued, CUTLASS's K_PIPE_MMAS = 1; chained
 *       wgmmas of one shape into the same accumulator need no wait, PTX ISA "wgmma.fence"). After the loop
 *       wait_group 0, every accumulator register folded per thread (sum / xor) and stored only if the runtime argument
 *       `store` is non-zero (always 0: the wgmmas are asm volatile, the fold keeps the accumulators live and feeds the
 *       end timer).
 *   KIND × N × NACC (accumulator registers per thread = NACC × N/2 for f32, NACC × N/4 for f16):
 *       fp16     m64nNk16.f32.f16.f16      N 256 (NACC 1), N 128 (NACC 1, 2)
 *       fp16acc  m64nNk16.f16.f16.f16      N 256 (NACC 1, 2), N 128 (NACC 1, 2)
 *       bf16     m64nNk16.f32.bf16.bf16    N 256 (NACC 1), N 128 (NACC 1, 2)
 *       fp8      m64nNk32.f32.e4m3.e4m3    N 256 (NACC 1), N 128 (NACC 1, 2)   (raw tensor-core accumulation, i.e. what
 *                cuBLAS calls fast accumulation; no CUDA-core promotion)
 *       (f32 N 256 NACC 2 would need 256 accumulator registers: not built.) Each × WAIT {0, 1} × C {1, 2, 3}.
 *   FLOPs = 2·M·N·K per wgmma (M = 64; K = 16, or 32 for e4m3) × wgmma count (state convention: one FMA = 2 FLOP).
 *   Work per launch: 2^37 FLOP per SM (2^33 with --quick), i.e. ≈ 20 ms for FP16 / 10 ms for FP8 at the architectural
 *   rate, so the 2.5 µs event floor is < 0.03 %. Timed by a cudaEvent pair around the kernel (the summary subtracts the
 *   floor); per block thread 0 reads %clock64 / %globaltimer after the bar.sync that ends the fill and after the
 *   bar.sync that follows the fold → SM clock of the launch (sm_mhz_blocks) and FLOP/clk/SM from cycles
 *   (flop_per_clk_sm, median over blocks).
 *   Architectural reference (GH100 SM: 4 tensor cores × 512 dense FP16 FMA/clk = 4096 FLOP/clk/SM; FP8 8192): H100 SXM
 *   989.4 TFLOP/s = 132 SMs × 1830 MHz × 4096. At the 1635 MHz lock: FP16/BF16 132 × 1.635e9 × 4096 = 884.0 TFLOP/s,
 *   FP8 1768.0 TFLOP/s. The H100 NVL datasheet's 835.5 (FP16 dense) = 132 × 4096 × 1545 MHz: it is not stated at the
 *   1785 MHz boost clock, so "datasheet × 1635/1785" (765) understates the per-clock peak.
 *
 * Power: dense back-to-back tensor work exceeds the 400 W limit and pulls the SM clock below the lock (R1, ceilings_tc.py).
 * Every launch is preceded by --idle-ms (default 1000) of host sleep; a launch is flagged clock_bad if its NVML sample
 * (after the launch) shows a throttle reason or a clock > 15 MHz below the lock (hb::ClockSample::bad) OR if the median
 * in-kernel clock of its blocks (cycles / %globaltimer ns over the timed loop) is > 15 MHz below the lock (kclock_bad;
 * catches throttling that ended before the NVML sample). Flagged launches are written, not counted, and repeated.
 *
 * SASS check (cuobjdump -sass bin/tc_peak, `make sass` → results/sass/tc_peak.sass; nvcc 12.9.86, sm_90a):
 *   checked 2026-10-26, all 26 instantiations: the timed loop is WARPGROUP.ARRIVE (wgmma.fence), then 4·NACC
 *   HGMMA.64xNx16.F32 / .F16 / .F32.BF16 (resp. QGMMA.64xNx32.F32.E4M3.E4M3) with both operands as gdesc[UR] and the
 *   accumulator in R (Rd = Rc, no scale-d predicate: the constant 1 folded), gsb0 on the last of the group, then
 *   WARPGROUP.DEPBAR.LE gsb0, WAIT (0x0 / 0x1 as instantiated) and the loop control VIADD/IADD3 + ISETP.GE.U32.AND + BRA;
 *   between the GMMAs only uniform-datapath descriptor arithmetic (ULOP3.LUT / UMOV / UIADD3 / USHF, 11–20 per
 *   iteration) and 0–6 NOPs ptxas inserts between back-to-back n256 GMMAs. Loop length 20–38 instructions per 4–8 GMMAs
 *   (each GMMA occupies the SM's tensor cores for ≥ 64 cycles at the architectural rate). No local memory, 0 spills;
 *   registers 58 (f16 acc N 128) / 90 (one f32 N 128 or f16 N 256 accumulator, or two f16 N 128) / 154 (128 accumulator
 *   registers); ptxas reports no "wgmma serialized" warning. The descriptors decode as intended: high word 0x40000040
 *   (layout_type 1 = 128B swizzle, SBO 64 × 16 B = 1024 B), low word (addr >> 4) | 0x10000 (LBO 1 × 16 B), + 2 per
 *   k-slice. The compute_90 PTX that nvcc also embeds traps (no wgmma below sm_90a); it is never selected on the H100.
 *
 * Usage (from roofline/): harness/ceilings/bin/tc_peak {peak|ncu|clockcheck} [--quick] [--label L] [--idle-ms MS]
 *                                                     [--kinds fp16,fp16acc,bf16,fp8]
 *   peak  every (KIND, N, NACC, WAIT, C) with zero fill: 1 warm-up + 5 recorded unflagged launches (3, and 1/16 of the
 *         work, with --quick); then, per KIND, the configuration with the highest median of this run again with
 *         random fill (column fill = rand).
 *   ncu   one launch per KIND at N 256, NACC 1, WAIT 1, C 2 (1/16 of the work) for ceilings_ncu.sh (-k regex:tc_peak);
 *         manifest → results/ceilings/ncu/manifest_tc_peak.csv (wgmma_total = warpgroup-level wgmma count).
 * → results/ceilings/tc_peak.csv (one row per launch): mandatory columns, kind, instr, n, k, nacc, wait_groups, consumers,
 *   blocks, threads, fill, steps, wgmma_total (warpgroup-level, all blocks), flop, ms_event, rate_raw (flop / event time,
 *   floor not removed), sm_mhz_blocks, span_ns, flop_per_clk_sm, placement_bad, kclock_bad, clock_bad (NVML or kernel).
 */
#include <algorithm>
#include <chrono>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "ceil_common.hpp"
#include "wgmma_ops.hpp"

enum Kind { K_FP16 = 0, K_FP16ACC = 1, K_BF16 = 2, K_FP8 = 3 };
static const char* kKindName[] = {"fp16", "fp16acc", "bf16", "fp8"};

constexpr int kMaxWG = 3;                       // consumer warpgroups per CTA (blockDim ≤ 384)
constexpr int kKSlices = 4;                     // wgmmas per accumulator per iteration (one 128-byte swizzle row)
constexpr int kATile = 64 * 128;                // bytes per A tile (64 rows × 128 B)
constexpr int kBTileMax = 256 * 128;            // bytes per B tile at N = 256
constexpr int kSmemUsed = kMaxWG * kATile + 2 * kBTileMax;  // 88 KiB
constexpr int kSmemAlloc = 160 * 1024;          // > 228 KiB / 2: one CTA per SM (+ 1 KiB alignment slack inside)
constexpr double kFlopPerSm = 137438953472.0;   // 2^37 FLOP per SM per launch
static_assert(kSmemUsed + 1024 <= kSmemAlloc && 2 * kSmemAlloc > 228 * 1024, "smem layout");

// ---------------------------------------------------------------- operation table
template <int KIND, int N> struct Op;
#define TC_OP(KIND_, N_, T_, R_, K_, FN_)                                                                   \
    template <> struct Op<KIND_, N_> {                                                                      \
        using T = T_;                                                                                       \
        static constexpr int R = R_, K = K_;                                                                \
        static __device__ __forceinline__ void mma(T (&d)[R_], uint64_t a, uint64_t b) { FN_(d, a, b, 1u); } \
    };
TC_OP(K_FP16, 256, float, 128, 16, wgmma_m64n256k16_f32_f16)
TC_OP(K_FP16, 128, float, 64, 16, wgmma_m64n128k16_f32_f16)
TC_OP(K_FP16ACC, 256, uint32_t, 64, 16, wgmma_m64n256k16_f16_f16)
TC_OP(K_FP16ACC, 128, uint32_t, 32, 16, wgmma_m64n128k16_f16_f16)
TC_OP(K_BF16, 256, float, 128, 16, wgmma_m64n256k16_f32_bf16)
TC_OP(K_BF16, 128, float, 64, 16, wgmma_m64n128k16_f32_bf16)
TC_OP(K_FP8, 256, float, 128, 32, wgmma_m64n256k32_f32_e4m3)
TC_OP(K_FP8, 128, float, 64, 32, wgmma_m64n128k32_f32_e4m3)
#undef TC_OP

static const char* instr_name(int kind, int n) {
    static char b[64];
    const char* t[] = {"f32.f16.f16", "f16.f16.f16", "f32.bf16.bf16", "f32.e4m3.e4m3"};
    std::snprintf(b, sizeof b, "m64n%dk%d.%s", n, kind == K_FP8 ? 32 : 16, t[kind]);
    return b;
}

// ---------------------------------------------------------------- device helpers
// Shared-memory matrix descriptor (PTX ISA "Matrix Descriptor Format"; CUTLASS GmmaDescriptor): bits 0–13 start address
// >> 4, 16–29 leading-dimension byte offset >> 4, 32–45 stride-dimension byte offset >> 4, 49–51 base offset, 62–63
// layout type (0 interleave/none, 1 128B swizzle, 2 64B, 3 32B).
__device__ __forceinline__ uint64_t make_desc_sw128(uint32_t saddr) {
    return (uint64_t)((saddr & 0x3FFFFu) >> 4) | ((uint64_t)(16u >> 4) << 16) | ((uint64_t)(1024u >> 4) << 32) | ((uint64_t)1 << 62);
}
__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int W>
__device__ __forceinline__ void wgmma_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(W) : "memory"); }
// CUTLASS warpgroup_fence_operand: pins an accumulator register in program order relative to the asm volatile wgmma
// statements (no reordering of ordinary reads/writes of it across them).
__device__ __forceinline__ void fence_operand(float& r) { asm volatile("" : "+f"(r) :: "memory"); }
__device__ __forceinline__ void fence_operand(uint32_t& r) { asm volatile("" : "+r"(r) :: "memory"); }

__device__ __forceinline__ uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}

template <int KIND, int N, int NACC, int WAIT>
__global__ void __launch_bounds__(128 * kMaxWG, 1)
tc_peak_kernel(uint32_t steps, uint32_t fill, uint32_t store, uint32_t* out, BlockRec* rec, uint64_t zero) {
#if defined(__CUDA_ARCH__) && !defined(__CUDA_ARCH_FEAT_SM90_ALL)
    // nvcc 12.9 -arch=sm_90a also embeds compute_90 PTX (no wgmma there); the sm_90a SASS is what runs on the H100.
    __trap();
#else
    using O = Op<KIND, N>;
    using T = typename O::T;
    extern __shared__ uint8_t smem_raw[];
    const uint32_t raw = smem_u32(smem_raw);
    const uint32_t base = (raw + 1023u) & ~1023u;
    uint32_t* w = reinterpret_cast<uint32_t*>(smem_raw + (base - raw));
    // f16/bf16: clear bit 14 of each half (top exponent bit); e4m3: clear bit 6 of each byte → finite, |x| < 2.
    const uint32_t mask = KIND == K_FP8 ? 0xBFBFBFBFu : 0xBFFFBFFFu;
    for (int i = threadIdx.x; i < kSmemUsed / 4; i += blockDim.x) w[i] = fill ? (hash32((uint32_t)i * 2654435761u + blockIdx.x) & mask) : 0u;
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");  // generic-proxy smem writes → visible to wgmma (async proxy)
    __syncthreads();

    const int wg = __shfl_sync(0xffffffffu, (int)(threadIdx.x >> 7), 0);  // warp-uniform (CUTLASS canonical_warp_group_idx)
    const uint64_t da = make_desc_sw128(base + wg * kATile);
    uint64_t db[NACC];
#pragma unroll
    for (int a = 0; a < NACC; ++a) db[a] = make_desc_sw128(base + kMaxWG * kATile + a * N * 128);
    T acc[NACC][O::R];
#pragma unroll
    for (int a = 0; a < NACC; ++a)
#pragma unroll
        for (int i = 0; i < O::R; ++i) {
            acc[a][i] = T(0);
            fence_operand(acc[a][i]);
        }
    TimerPair t0{};
    if (threadIdx.x == 0) t0 = read_timers_pair();
#pragma unroll 1
    for (uint32_t s = 0; s < steps; ++s) {
        wgmma_fence();
#pragma unroll
        for (int k = 0; k < kKSlices; ++k)
#pragma unroll
            for (int a = 0; a < NACC; ++a) O::mma(acc[a], da + 2 * k, db[a] + 2 * k);  // +32 B per k-slice
        wgmma_commit();
        wgmma_wait<WAIT>();
    }
    wgmma_wait<0>();
    uint32_t fold = 0;
#pragma unroll
    for (int a = 0; a < NACC; ++a)
#pragma unroll
        for (int i = 0; i < O::R; ++i) {
            fence_operand(acc[a][i]);
            if constexpr (std::is_same_v<T, float>) fold += __float_as_uint(acc[a][i]);
            else fold ^= (uint32_t)acc[a][i];
        }
    if (store) out[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = fold;
    __syncthreads();
    if (threadIdx.x == 0) {
        const TimerPair t1 = read_timers_after(fold, zero);
        const uint64_t wgmmas = (uint64_t)steps * kKSlices * NACC * (blockDim.x / 128);
        rec[blockIdx.x] = {t0.cycles, t1.cycles, t0.ns, t1.ns, wgmmas, read_smid(), 0, blockIdx.x, 0};
    }
#endif
}

// ---------------------------------------------------------------- host
using KernelFn = void (*)(uint32_t, uint32_t, uint32_t, uint32_t*, BlockRec*, uint64_t);
struct Variant {
    int kind, n, nacc, wait;
    KernelFn fn;
    int k() const { return kind == K_FP8 ? 32 : 16; }
    double flop_per_wgmma() const { return 2.0 * 64 * n * k(); }
};
#define TCV(KIND_, N_, NACC_, W_) Variant{KIND_, N_, NACC_, W_, tc_peak_kernel<KIND_, N_, NACC_, W_>}
#define TCV_W(KIND_, N_, NACC_) TCV(KIND_, N_, NACC_, 0), TCV(KIND_, N_, NACC_, 1)
static std::vector<Variant> all_variants() {
    return {TCV_W(K_FP16, 256, 1),    TCV_W(K_FP16, 128, 1),    TCV_W(K_FP16, 128, 2),
            TCV_W(K_FP16ACC, 256, 1), TCV_W(K_FP16ACC, 256, 2), TCV_W(K_FP16ACC, 128, 1), TCV_W(K_FP16ACC, 128, 2),
            TCV_W(K_BF16, 256, 1),    TCV_W(K_BF16, 128, 1),    TCV_W(K_BF16, 128, 2),
            TCV_W(K_FP8, 256, 1),     TCV_W(K_FP8, 128, 1),     TCV_W(K_FP8, 128, 2)};
}
#undef TCV_W
#undef TCV

struct Ctx {
    int sm_count = 0;
    unsigned lock_mhz = 0;
    int idle_ms = 1000;
    uint32_t* d_out = nullptr;
    BlockRec* d_rec = nullptr;
    cudaEvent_t ev0{}, ev1{};
};

static uint32_t steps_for(const Variant& v, int consumers, double flop_per_sm) {
    const double per_iter = v.flop_per_wgmma() * kKSlices * v.nacc * consumers;  // per SM (one CTA per SM)
    return (uint32_t)std::max(1.0, std::floor(flop_per_sm / per_iter));
}

static void launch(Ctx& x, const Variant& v, int consumers, uint32_t steps, uint32_t fill) {
    if (x.idle_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(x.idle_ms));  // start below the power limit
    CUDA_CHECK(cudaEventRecord(x.ev0));
    v.fn<<<x.sm_count, 128 * consumers, kSmemAlloc>>>(steps, fill, 0u, x.d_out, x.d_rec, 0ull);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(x.ev1));
    CUDA_CHECK(cudaDeviceSynchronize());
}

struct BlockStats {
    uint64_t wgmmas = 0, span_ns = 0;
    double mhz = 0, cycles_med = 0, wgmma_per_clk_sm = 0;
};
static BlockStats block_stats(const std::vector<BlockRec>& r) {
    BlockStats b;
    uint64_t t_lo = ~0ull, t_hi = 0;
    std::vector<double> mhz, cyc, wpc;
    for (const auto& e : r) {
        b.wgmmas += e.bytes;
        t_lo = std::min(t_lo, e.n0);
        t_hi = std::max(t_hi, e.n1);
        if (e.n1 > e.n0) mhz.push_back(1e3 * double(e.c1 - e.c0) / double(e.n1 - e.n0));
        if (e.c1 > e.c0) {
            cyc.push_back(double(e.c1 - e.c0));
            wpc.push_back(double(e.bytes) / double(e.c1 - e.c0));
        }
    }
    b.span_ns = t_hi - t_lo;
    b.mhz = median_of(mhz);
    b.cycles_med = median_of(cyc);
    b.wgmma_per_clk_sm = median_of(wpc);
    return b;
}

// record() of ceil_common.hpp with one more flag: the in-kernel clock (median over blocks) more than 15 MHz below the
// lock also makes the launch clock_bad. launch() returns {placement ok, kernel clock ok}.
struct LaunchOk { bool place, kclock; };
template <class L, class R>
int record_tc(Session& s, const std::string& label, int warmups, int recorded, L&& launch_fn, R&& row) {
    for (int w = 0; w < warmups; ++w) launch_fn();
    int good = 0, bad = 0;
    while (good < recorded) {
        const LaunchOk ok = launch_fn();
        const hb::ClockSample cs = s.run.clock_note(label, "launch", s.li);
        const bool clock_bad = cs.bad() || !ok.kclock;
        row(s.run.prefix(s.li, cs), !ok.place, !ok.kclock, clock_bad);
        ++s.li;
        if (!ok.place || clock_bad) {
            if (!ok.place) ++s.discarded_placement;
            else ++s.discarded_clock;
            if (++bad > kMaxDiscards) {  // e.g. the power limit with real data: keep the flagged rows, go on (R1)
                std::printf("%s: more than %d flagged launches, configuration abandoned with %d good\n", label.c_str(),
                            kMaxDiscards, good);
                break;
            }
            continue;
        }
        ++good;
    }
    return bad;
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
    const std::string kinds = std::string(",") + hb::arg_str(argc, argv, "--kinds", "fp16,fp16acc,bf16,fp8") + ",";
    auto kind_on = [&](int k) { return kinds.find(std::string(",") + kKindName[k] + ",") != std::string::npos; };
    Ctx x;
    x.sm_count = run.dev().sm_count;
    x.lock_mhz = run.dev().lock_mhz;
    x.idle_ms = (int)hb::arg_int(argc, argv, "--idle-ms", quick ? 200 : 1000);
    const std::vector<Variant> vars = all_variants();
    for (const auto& v : vars) CUDA_CHECK(cudaFuncSetAttribute(v.fn, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemAlloc));
    CUDA_CHECK(cudaMalloc(&x.d_out, (size_t)x.sm_count * 128 * kMaxWG * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&x.d_rec, x.sm_count * sizeof(BlockRec)));
    CUDA_CHECK(cudaEventCreate(&x.ev0));
    CUDA_CHECK(cudaEventCreate(&x.ev1));
    Session s(run);
    std::vector<BlockRec> r(x.sm_count);
    const double flop_per_sm = quick ? kFlopPerSm / 16 : kFlopPerSm;

    if (mode == "peak") {
        hb::CsvWriter csv(out_csv("tc_peak"),
                          hb::with_mandatory({"kind", "instr", "n", "k", "nacc", "wait_groups", "consumers", "blocks", "threads", "fill",
                                              "steps", "wgmma_total", "flop", "ms_event", "rate_raw", "sm_mhz_blocks", "span_ns",
                                              "flop_per_clk_sm", "placement_bad", "kclock_bad", "clock_bad"}));
        const int recorded = quick ? kQuickRecorded : 5, warmups = 1;
        run.lock_clocks_note("tc_peak", "before");
        auto run_cfg = [&](const Variant& v, int c, uint32_t fill) {
            const uint32_t steps = steps_for(v, c, flop_per_sm);
            char lab[96];
            std::snprintf(lab, sizeof lab, "%s_n%d_acc%d_w%d_c%d%s", kKindName[v.kind], v.n, v.nacc, v.wait, c, fill ? "_rand" : "");
            std::vector<double> rates;
            auto go = [&]() -> LaunchOk {
                launch(x, v, c, steps, fill);
                CUDA_CHECK(cudaMemcpy(r.data(), x.d_rec, x.sm_count * sizeof(BlockRec), cudaMemcpyDeviceToHost));
                const BlockStats b = block_stats(r);
                return {PlacementCheck(r, x.sm_count).exactly(1), !(x.lock_mhz > 0 && b.mhz + 15.0 < (double)x.lock_mhz)};
            };
            const int bad = record_tc(s, lab, warmups, recorded, go, [&](const std::string& pre, bool pb, bool kb, bool cb) {
                float ms = 0;
                CUDA_CHECK(cudaEventElapsedTime(&ms, x.ev0, x.ev1));
                const BlockStats b = block_stats(r);
                const double flop = (double)b.wgmmas * v.flop_per_wgmma(), rate = flop / (ms * 1e-3);
                if (!pb && !cb) rates.push_back(rate);
                char row[768];
                std::snprintf(row, sizeof row, "%s,%s,%s,%d,%d,%d,%d,%d,%d,%d,%s,%u,%llu,%.0f,%.6f,%.6e,%.2f,%llu,%.2f,%d,%d,%d", pre.c_str(),
                              kKindName[v.kind], instr_name(v.kind, v.n), v.n, v.k(), v.nacc, v.wait, c, x.sm_count, 128 * c,
                              fill ? "rand" : "zero", steps, (unsigned long long)b.wgmmas, flop, ms, rate, b.mhz,
                              (unsigned long long)b.span_ns, b.wgmma_per_clk_sm * v.flop_per_wgmma(), (int)pb, (int)kb, (int)cb);
                csv.add(row);
                csv.flush();
            });
            csv.sync();
            const double med = median_of(rates);
            std::printf("%-26s %8.1f TFLOP/s (median, event floor not removed) = %6.0f FLOP/clk/SM at %u MHz%s\n", lab, med / 1e12,
                        med / (x.sm_count * 1e6 * (x.lock_mhz ? x.lock_mhz : 1635)), x.lock_mhz,
                        bad ? ("  flagged launches: " + std::to_string(bad)).c_str() : "");
            std::fflush(stdout);
            return med;
        };
        for (int kind = 0; kind < 4; ++kind) {
            if (!kind_on(kind)) continue;
            double best = -1;
            const Variant* bv = nullptr;
            int bc = 0;
            for (const auto& v : vars) {
                if (v.kind != kind) continue;
                for (int c = 1; c <= kMaxWG; ++c) {
                    const double m = run_cfg(v, c, 0);
                    if (m > best) { best = m; bv = &v; bc = c; }
                }
            }
            if (bv) run_cfg(*bv, bc, 1);  // the best zero-fill configuration again with random operands (power)
        }
        run.lock_clocks_note("tc_peak", "after");
    } else if (mode == "ncu") {
        hb::CsvWriter man(out_csv("ncu/manifest_tc_peak"),
                          hb::with_mandatory({"kind", "instr", "n", "k", "nacc", "wait_groups", "consumers", "blocks", "threads", "steps",
                                              "wgmma_total", "flop", "ms_event"}));
        run.lock_clocks_note("tc_peak_ncu", "before");
        x.idle_ms = 0;
        for (int kind = 0; kind < 4; ++kind) {
            const Variant* v = nullptr;
            for (const auto& u : vars)
                if (u.kind == kind && u.n == 256 && u.nacc == 1 && u.wait == 1) v = &u;
            const int c = 2;
            const uint32_t steps = steps_for(*v, c, kFlopPerSm / 16);
            launch(x, *v, c, steps, 0);
            const hb::ClockSample cs = run.clock_note(std::string("ncu_tc_peak_") + kKindName[kind], "launch", s.li);
            CUDA_CHECK(cudaMemcpy(r.data(), x.d_rec, x.sm_count * sizeof(BlockRec), cudaMemcpyDeviceToHost));
            const BlockStats b = block_stats(r);
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, x.ev0, x.ev1));
            char row[512];
            std::snprintf(row, sizeof row, "%s,%s,%s,%d,%d,%d,%d,%d,%d,%d,%u,%llu,%.0f,%.6f", run.prefix(s.li, cs).c_str(), kKindName[kind],
                          instr_name(kind, v->n), v->n, v->k(), v->nacc, v->wait, c, x.sm_count, 128 * c, steps,
                          (unsigned long long)b.wgmmas, (double)b.wgmmas * v->flop_per_wgmma(), ms);
            man.add(row);
            man.flush();
            ++s.li;
        }
        man.sync();
        run.lock_clocks_note("tc_peak_ncu", "after");
    } else {
        hb::die("unknown mode " + mode);
    }
    std::printf("tc_peak %s: %d launches flagged (clock), %d (placement)\n", mode.c_str(), s.discarded_clock, s.discarded_placement);
    run.complete();
    return 0;
}
