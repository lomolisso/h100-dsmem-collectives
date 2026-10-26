/*
 * bw_smem.cu — roofline R1: shared-memory (L1/SMEM) load bandwidth per SM, all 132 SMs at once
 * (DESIGN §2.1 "L1/SMEM bandwidth": B/cycle/SM from the iteration slope × 132 SMs × measured SM clock).
 *
 * Kernel k12<LEVEL, OP, W, ILP, PAT, T> copied verbatim (with Words, consume, ld_one, st_one, slot_off) from
 * MB/experiments/05__memory_hierarchy_bandwidth/src/bw_smem_l1.cu at MB commit 8f8fc7bec69f07db49f3a5a95bbdfb08de5d0ca8.
 * Only the configuration of MB's SMEM-load ceiling is instantiated: LEVEL smem, OP ld, W = 16 B (ld.volatile.shared.v4),
 * pattern stride (bank-conflict-free: a warp reads 512 contiguous bytes per instruction), placement all_sms (132 blocks,
 * 64 KiB dynamic smem each; the launch is flagged and repeated unless all 132 %smid values are distinct), threads
 * T ∈ {256, 512, 1024} × ILP ∈ {1, 2, 4, 8} × iters ∈ {4096, 8192}. Dropped: L1 level, stores, 4-B width, single
 * placement, accelsim pattern, ncu mode.
 * Timed region (per block, thread 0): bar.sync → %clock64/%globaltimer → `iters` steps of ILP loads (XORed after the
 * step's last load is issued) → bar.sync → timers after its own last loaded value. bytes = T · iters · ILP · 16.
 * The analysis (ceilings_summary.py, MB common05.k12_slope) pairs the 4096- and 8192-step launches of every SM by launch
 * order: bpc_slope = Δbytes / Δcycles (the fixed cost of the timed region cancels); per launch pair the median over SMs;
 * the configuration with the highest median; GPU-wide B/s = bpc × 132 × the median SM clock of its blocks (cycles / ns).
 *
 * Usage (from roofline/): harness/ceilings/bin/bw_smem {k12|clockcheck} [--quick] [--label L]
 *   --quick: T ∈ {256, 1024}, ILP ∈ {1, 4}, 3 recorded launches.
 * → results/ceilings/bw_smem.csv (one row per block and launch): mandatory columns, level, op, width_bytes, threads, ilp,
 *   iters, pattern, placement, block, bytes, cycles, bpc_per_sm, ns, smid, placement_bad, clock_bad.
 */
#include <string>
#include <vector>

#include "ceil_common.hpp"

static constexpr uint32_t kRegion = 64 * 1024;

enum Level { L_SMEM = 0, L_L1 = 1 };
enum Op { OP_LD = 0, OP_ST = 1 };
enum Pattern { P_STRIDE = 0, P_ACCELSIM = 1 };

struct BlockOut { uint64_t cycles, ns; uint32_t smid, sink; };

// ---------------------------------------------------------------- verbatim from MB bw_smem_l1.cu
// One access of width W (4 or 16 B) at a shared (32-bit) or global (64-bit) address. Loads return the raw words
// (y = z = w = 0 for W = 4); the caller XORs them only after all ILP loads of a step are issued.
struct Words { uint32_t x, y, z, w; };
// acc ^= data as volatile PTX, so that it stays behind all loads of the step (bw_common.hpp, xor_into)
template <int W>
__device__ __forceinline__ void consume(uint32_t& acc, const Words& v) {
    if constexpr (W == 16) xor_into(acc, V4{v.x, v.y, v.z, v.w});
    else asm volatile("xor.b32 %0, %0, %1;" : "+r"(acc) : "r"(v.x));
}
template <int LEVEL, int W>
__device__ __forceinline__ Words ld_one(uint64_t a) {
    Words v{0, 0, 0, 0};
    if constexpr (W == 16) {
        if constexpr (LEVEL == L_SMEM)
            asm volatile("ld.volatile.shared.v4.f32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"((uint32_t)a) : "memory");
        else
            asm volatile("ld.global.ca.v4.f32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(a) : "memory");
    } else {
        if constexpr (LEVEL == L_SMEM) asm volatile("ld.volatile.shared.f32 %0, [%1];" : "=r"(v.x) : "r"((uint32_t)a) : "memory");
        else asm volatile("ld.global.ca.f32 %0, [%1];" : "=r"(v.x) : "l"(a) : "memory");
    }
    return v;
}
template <int LEVEL, int W>
__device__ __forceinline__ void st_one(uint64_t a, uint32_t v) {
    const float f = __uint_as_float(v);
    if constexpr (W == 16) {
        if constexpr (LEVEL == L_SMEM) asm volatile("st.volatile.shared.v4.f32 [%0], {%1, %1, %1, %1};" :: "r"((uint32_t)a), "f"(f) : "memory");
        else asm volatile("st.global.v4.f32 [%0], {%1, %1, %1, %1};" :: "l"(a), "f"(f) : "memory");
    } else {
        if constexpr (LEVEL == L_SMEM) asm volatile("st.volatile.shared.f32 [%0], %1;" :: "r"((uint32_t)a), "f"(f) : "memory");
        else asm volatile("st.global.f32 [%0], %1;" :: "l"(a), "f"(f) : "memory");
    }
}

// Byte offset of slot k of thread t: W·t + W·T·(k mod S) (stride) or 16·t + 128·k (accelsim) — T is a template
// parameter, so the slot part is a compile-time immediate in the SASS address ([R + imm]).
template <int W, int T, int PAT>
__device__ __forceinline__ constexpr uint32_t slot_off(int k) {
    return PAT == P_ACCELSIM ? 128u * k : (uint32_t)W * T * (k % (int)(kRegion / (W * T)));
}

template <int LEVEL, int OP, int W, int ILP, int PAT, int T>
__global__ void __launch_bounds__(T) k12(char* gbuf, int iters, uint64_t zero, BlockOut* out) {
    extern __shared__ __align__(16) unsigned char smem[];
    // Loads: one step of ILP loads per loop iteration, consumed before the next step → at most ILP loads outstanding per
    // thread. Stores never wait, so their steps are unrolled to ≥ 8 accesses per iteration to amortise the loop overhead;
    // copy u of a store step uses slots u·ILP … u·ILP + ILP − 1 (mod S), so consecutive stores never hit the same address.
    constexpr int kUnroll = (OP == OP_LD || ILP >= 8) ? 1 : 8 / ILP;
    const uint32_t t = threadIdx.x;
    const uint64_t base = ((LEVEL == L_SMEM) ? (uint64_t)smem_u32(smem) : (uint64_t)(gbuf + (size_t)blockIdx.x * kRegion)) +
                          (PAT == P_ACCELSIM ? 16u : (uint32_t)W) * t;
    uint32_t acc[ILP];
#pragma unroll
    for (int i = 0; i < ILP; ++i) acc[i] = 0;
    Words v[ILP];
    if constexpr (LEVEL == L_L1 && OP == OP_LD) {  // warm L1 with the exact lines of the timed loop
#pragma unroll
        for (int i = 0; i < ILP; ++i) v[i] = ld_one<LEVEL, W>(base + slot_off<W, T, PAT>(i));
#pragma unroll
        for (int i = 0; i < ILP; ++i) acc[i] ^= v[i].x ^ v[i].y ^ v[i].z ^ v[i].w;
    }
    __syncthreads();
    TimerPair t0{};
    if (t == 0) t0 = read_timers_pair();
#pragma unroll 1
    for (int r = 0; r < iters; r += kUnroll) {
        // + (r & zero) = + 0, but ptxas cannot prove the addresses loop-invariant: it would otherwise hoist the global
        // loads (LDG.STRONG.SM) out of the loop. Shared accesses are .volatile for the same reason (ld_one/st_one).
        const uint64_t p = base + ((uint64_t)r & zero);
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            if constexpr (OP == OP_LD) {
#pragma unroll
                for (int i = 0; i < ILP; ++i) v[i] = ld_one<LEVEL, W>(p + slot_off<W, T, PAT>(i));
#pragma unroll
                for (int i = 0; i < ILP; ++i) consume<W>(acc[i], v[i]);
            } else {
#pragma unroll
                for (int i = 0; i < ILP; ++i) st_one<LEVEL, W>(p + slot_off<W, T, PAT>(u * ILP + i), (uint32_t)(r + u));
            }
        }
    }
    uint32_t fold = 0;
#pragma unroll
    for (int i = 0; i < ILP; ++i) fold ^= acc[i];
    __syncthreads();
    if (t == 0) {
        const TimerPair t1 = read_timers_after(fold, zero);
        out[blockIdx.x] = {t1.cycles - t0.cycles, t1.ns - t0.ns, read_smid(), fold};
    }
}
// ---------------------------------------------------------------- end of the verbatim part

using KernelFn = void (*)(char*, int, uint64_t, BlockOut*);

template <int ILP>
static KernelFn pick_t(int threads) {
    switch (threads) {
        case 256: return k12<L_SMEM, OP_LD, 16, ILP, P_STRIDE, 256>;
        case 512: return k12<L_SMEM, OP_LD, 16, ILP, P_STRIDE, 512>;
        case 1024: return k12<L_SMEM, OP_LD, 16, ILP, P_STRIDE, 1024>;
    }
    hb::die("bad thread count " + std::to_string(threads));
}
static KernelFn pick(int ilp, int threads) {
    switch (ilp) {
        case 1: return pick_t<1>(threads);
        case 2: return pick_t<2>(threads);
        case 4: return pick_t<4>(threads);
        case 8: return pick_t<8>(threads);
    }
    hb::die("bad ilp " + std::to_string(ilp));
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "k12";
    const bool quick = hb::has_flag(argc, argv, "--quick");
    hb::Run run(kExp, argc, argv);
    const std::string label = hb::arg_str(argc, argv, "--label", mode.c_str());
    if (mode == "clockcheck") {
        run.lock_clocks_note(label, "check");
        run.complete();
        return 0;
    }
    if (mode != "k12") hb::die("unknown mode " + mode);
    const int nb = run.dev().sm_count;
    BlockOut* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, nb * sizeof(BlockOut)));
    hb::CsvWriter csv(out_csv("bw_smem"), hb::with_mandatory({"level", "op", "width_bytes", "threads", "ilp", "iters", "pattern", "placement",
                                                               "block", "bytes", "cycles", "bpc_per_sm", "ns", "smid", "placement_bad",
                                                               "clock_bad"}));
    const std::vector<int> T = quick ? std::vector<int>{256, 1024} : std::vector<int>{256, 512, 1024};
    const std::vector<int> I = quick ? std::vector<int>{1, 4} : std::vector<int>{1, 2, 4, 8};
    Session s(run);
    run.lock_clocks_note("bw_smem", "before");
    for (int it : {4096, 8192})
        for (int t : T)
            for (int ilp : I) {
                const KernelFn fn = pick(ilp, t);
                CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kRegion));
                const std::string lab = "smem_ld_w16_t" + std::to_string(t) + "_ilp" + std::to_string(ilp) + "_it" + std::to_string(it) + "_all";
                std::vector<BlockOut> h(nb);
                const double bytes = double(t) * it * ilp * 16;
                std::vector<double> bpc;
                auto launch = [&] {
                    fn<<<nb, t, kRegion>>>(nullptr, it, 0, d_out);
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaDeviceSynchronize());
                    CUDA_CHECK(cudaMemcpy(h.data(), d_out, nb * sizeof(BlockOut), cudaMemcpyDeviceToHost));
                    std::vector<uint32_t> sm(nb);
                    for (int b = 0; b < nb; ++b) sm[b] = h[b].smid;
                    return hb::all_distinct(sm);
                };
                const int bad = record(s, lab, kWarmups, quick ? kQuickRecorded : kRecorded, launch,
                                       [&](const std::string& pre, const hb::ClockSample&, bool pb, bool cb) {
                                           for (int b = 0; b < nb; ++b) {
                                               char r[512];
                                               std::snprintf(r, sizeof r, "%s,smem,ld,16,%d,%d,%d,stride,all_sms,%d,%.0f,%llu,%.4f,%llu,%u,%d,%d",
                                                             pre.c_str(), t, ilp, it, b, bytes, (unsigned long long)h[b].cycles,
                                                             bytes / h[b].cycles, (unsigned long long)h[b].ns, h[b].smid, (int)pb, (int)cb);
                                               csv.add(r);
                                               if (!pb && !cb) bpc.push_back(bytes / h[b].cycles);
                                           }
                                           csv.flush();
                                       });
                csv.sync();
                std::printf("%-36s %8.2f B/cycle/SM (median over blocks, fixed cost included)%s\n", lab.c_str(), median_of(bpc),
                            bad ? ("  flagged launches: " + std::to_string(bad)).c_str() : "");
            }
    run.lock_clocks_note("bw_smem", "after");
    std::printf("bw_smem: %d launches flagged (clock), %d (placement)\n", s.discarded_clock, s.discarded_placement);
    CUDA_CHECK(cudaFree(d_out));
    run.complete();
    return 0;
}
