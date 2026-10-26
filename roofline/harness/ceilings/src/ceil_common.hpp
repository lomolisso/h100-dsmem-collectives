/*
 * ceil_common.hpp — roofline R1 (ceilings): pieces shared by the ceiling binaries in this directory.
 *
 * Copied from h100-microbench (MB = ~/work/fa2026/iacoma/h100-microbench) at MB commit
 * 8f8fc7bec69f07db49f3a5a95bbdfb08de5d0ca8 (2026-10-26). Copied code is verbatim except where stated:
 *   MB/experiments/05__memory_hierarchy_bandwidth/src/bw_common.hpp
 *       BlockRec, V4, xor4, xor_into, CacheOp/cache_name, ld_v4<C>, st_v4<C>, PlacementCheck, clean_read_kernel, CleanL2.
 *       Dropped: the include of 04's gpc_map.hpp, kPartitionOfGpc and partition_of_sm (GPU-2 L2-partition labels; the
 *       ceilings need no SM -> GPC / partition map). BlockRec keeps its `part` field (always 0 here).
 *   MB/experiments/harness/harness.cuh (newer than the vendored copy in common/h100mb, which is MB 14875a5)
 *       ld_dsmem_v4_u32 (added to MB's harness by experiment 09; not in the vendored copy, so copied here instead of
 *       editing the vendored file).
 *   MB/experiments/09__dsmem_latency_vs_cluster_size/src/common09.hpp
 *       record_place, Placement (cluster rank -> %smid; ok() = ranks of every cluster on distinct SMs). Dropped: gpc_of,
 *       Grid09, Discards (discarded launches are flagged in each raw CSV instead).
 * Local (new) code, below the copied part:
 *   kExp / out_csv()   every binary is run from roofline/ (hb::Run writes results/roofline__runs.csv and
 *                      results/roofline__clock_log.csv); raw CSVs go to results/ceilings/<name>.csv.
 *   Session / record() the recorded-launch loop shared by all binaries: after every launch one NVML clock sample
 *                      (hb::Run::clock_note, one clock-log row per launch); a launch whose sample is bad (throttle reason
 *                      or SM clock > 15 MHz below the lock) is written with clock_bad = 1, one whose placement check
 *                      failed with placement_bad = 1; either is not counted and the launch is repeated until `recorded`
 *                      good launches exist (at most kMaxDiscards repeats per configuration, then the binary dies).
 */
#pragma once
#include <sys/stat.h>

#include <algorithm>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include "harness.cuh"
#include "host.hpp"

// ================================================================ copied from MB 05 bw_common.hpp (verbatim parts)
struct BlockRec {
    uint64_t c0, c1, n0, n1, bytes;
    uint32_t smid, part, ticket, flags;
};

struct V4 { uint32_t x, y, z, w; };
__device__ __forceinline__ uint32_t xor4(const V4& v) { return v.x ^ v.y ^ v.z ^ v.w; }
// acc ^= x ^ y ^ z ^ w as volatile PTX: it keeps its place after the (volatile) loads of the same step, so the compiler
// cannot interleave consumption with the loads and in-order issue never waits on a load before the step's last one.
__device__ __forceinline__ void xor_into(uint32_t& acc, const V4& v) {
    asm volatile("{\n\t.reg .b32 t;\n\txor.b32 t, %1, %2;\n\txor.b32 t, t, %3;\n\txor.b32 t, t, %4;\n\txor.b32 %0, %0, t;\n\t}"
                 : "+r"(acc) : "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
}

enum CacheOp { C_CA = 0, C_CG = 1, C_CS = 2, C_WB = 3, C_LU = 4, C_CV = 5 };
inline const char* cache_name(int c) { static const char* n[] = {"ca", "cg", "cs", "wb", "lu", "cv"}; return n[c]; }

template <int C>
__device__ __forceinline__ V4 ld_v4(uint64_t a) {
    V4 v;
    if constexpr (C == C_CA)
        asm volatile("ld.global.ca.v4.f32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(a) : "memory");
    else if constexpr (C == C_CG)
        asm volatile("ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(a) : "memory");
    else if constexpr (C == C_CS)
        asm volatile("ld.global.cs.v4.f32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(a) : "memory");
    else if constexpr (C == C_LU)
        asm volatile("ld.global.lu.v4.f32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(a) : "memory");
    else
        asm volatile("ld.global.cv.v4.f32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(a) : "memory");
    return v;
}
template <int C>
__device__ __forceinline__ void st_v4(uint64_t a, uint32_t d) {
    if constexpr (C == C_CG) asm volatile("st.global.cg.v4.f32 [%0], {%1, %1, %1, %1};" :: "l"(a), "r"(d) : "memory");
    else if constexpr (C == C_CS) asm volatile("st.global.cs.v4.f32 [%0], {%1, %1, %1, %1};" :: "l"(a), "r"(d) : "memory");
    else asm volatile("st.global.v4.f32 [%0], {%1, %1, %1, %1};" :: "l"(a), "r"(d) : "memory");
}

// Host: how many blocks ran on each SM.
struct PlacementCheck {
    std::vector<int> per_sm;
    explicit PlacementCheck(const std::vector<BlockRec>& r, int sm_count) : per_sm(sm_count, 0) {
        for (const auto& b : r) if (b.smid < (uint32_t)sm_count) ++per_sm[b.smid];
    }
    // every SM hosts exactly k blocks
    bool exactly(int k) const { for (int n : per_sm) if (n != k) return false; return true; }
    // no SM hosts more than one block
    bool at_most_one() const { for (int n : per_sm) if (n > 1) return false; return true; }
};

static __global__ void clean_read_kernel(const uint4* buf, size_t n, uint32_t* sink) {
    uint32_t acc = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) acc ^= buf[i].x;
    if (acc == 0x9e3779b9u) *sink = acc;  // practically never; keeps the loads
}
class CleanL2 {
  public:
    CleanL2() {
        CUDA_CHECK(cudaMalloc(&buf_, kBytes));
        CUDA_CHECK(cudaMemset(buf_, 0, kBytes));
        CUDA_CHECK(cudaMalloc(&sink_, sizeof(uint32_t)));
    }
    ~CleanL2() { cudaFree(buf_); cudaFree(sink_); }
    void operator()() {
        flush_();
        clean_read_kernel<<<132 * 4, 1024>>>(reinterpret_cast<const uint4*>(buf_), kBytes / sizeof(uint4), sink_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

  private:
    static constexpr size_t kBytes = 128ull << 20;
    hb::L2Flusher flush_;
    char* buf_ = nullptr;
    uint32_t* sink_ = nullptr;
};

// ================================================================ copied from MB experiments/harness/harness.cuh (09)
__device__ __forceinline__ uint4 ld_dsmem_v4_u32(uint32_t a) {
    uint4 v;
    asm volatile("ld.shared::cluster.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a) : "memory");
    return v;
}

// ================================================================ copied from MB 09 common09.hpp
__device__ __forceinline__ void record_place(uint32_t* place) {
    if ((threadIdx.x & 31) == 0 && threadIdx.x < 32) {
        const ClusterRegs c = read_cluster_regs();
        uint32_t* p = place + 4 * blockIdx.x;
        p[0] = read_smid();
        p[1] = c.clusterid_x;
        p[2] = c.ctarank;
        p[3] = c.nctarank;
    }
}

struct Placement {
    int cs = 1, nclusters = 0;
    std::vector<uint32_t> sm;  // sm[c * cs + r]
    Placement(const std::vector<uint32_t>& place, int cs_, int ncl) : cs(cs_), nclusters(ncl), sm((size_t)cs_ * ncl) {
        for (int b = 0; b < cs * ncl; ++b) {
            const uint32_t* p = &place[4 * (size_t)b];
            if ((int)p[1] != b / cs || (int)p[2] != b % cs || (int)p[3] != cs)
                hb::die("block " + std::to_string(b) + ": clusterid " + std::to_string(p[1]) + " rank " + std::to_string(p[2]) +
                        " nctarank " + std::to_string(p[3]) + " do not match the 1-D grid");
            sm[b] = p[0];
        }
    }
    uint32_t at(int c, int r) const { return sm[(size_t)c * cs + r]; }
    bool ok() const {
        for (int c = 0; c < nclusters; ++c) {
            const std::set<uint32_t> u(sm.begin() + (size_t)c * cs, sm.begin() + (size_t)(c + 1) * cs);
            if ((int)u.size() != cs) return false;
        }
        return true;
    }
};

// ================================================================ local (roofline R1)
static const char* kExp = "roofline";  // registry results/roofline__runs.csv, clock log results/roofline__clock_log.csv
constexpr int kWarmups = 2, kRecorded = 10, kQuickRecorded = 3, kMaxDiscards = 20;

// results/ceilings/<name>.csv (relative to roofline/, the working directory of every binary)
inline std::string out_csv(const std::string& name) {
    ::mkdir("results", 0775);
    ::mkdir("results/ceilings", 0775);
    ::mkdir("results/ceilings/ncu", 0775);
    return "results/ceilings/" + name + ".csv";
}

struct Session {
    hb::Run& run;
    long long li = 0;               // launch_idx of the next recorded launch (unique within the run)
    int discarded_clock = 0, discarded_placement = 0;
    explicit Session(hb::Run& r) : run(r) {}
};

// One configuration: `warmups` untimed launches, then launches until `recorded` good ones exist.
//   launch()                          runs one (synchronised) launch, returns true if its placement check passed;
//   row(prefix, sample, place_bad, clock_bad)  appends one CSV row (or several) for that launch (caller flushes).
// Every launch is recorded, flagged ones included. Returns the number of flagged launches.
template <class L, class R>
int record(Session& s, const std::string& label, int warmups, int recorded, L&& launch, R&& row) {
    for (int w = 0; w < warmups; ++w) launch();
    int good = 0, bad = 0;
    while (good < recorded) {
        const bool place_ok = launch();
        const hb::ClockSample cs = s.run.clock_note(label, "launch", s.li);
        const bool clock_bad = cs.bad();
        row(s.run.prefix(s.li, cs), cs, !place_ok, clock_bad);
        ++s.li;
        if (!place_ok || clock_bad) {
            if (!place_ok) ++s.discarded_placement;
            else ++s.discarded_clock;
            if (++bad > kMaxDiscards) hb::die(label + ": more than " + std::to_string(kMaxDiscards) + " flagged launches");
            continue;
        }
        ++good;
    }
    return bad;
}

inline double median_of(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
