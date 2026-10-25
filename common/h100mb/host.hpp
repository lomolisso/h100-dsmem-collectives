/*
 * host.hpp — host-side helpers shared by every experiment.
 *
 *   hb::DeviceInfo   device identity and the fixed configuration (cudaDeviceProp + NVML). open() refuses to
 *                    run unless the GPU is an H100 NVL, the enforced power limit is 400 W, ECC is enabled and
 *                    MIG is disabled. It also resets the persisting-L2 window. sample() reads the current
 *                    SM/memory clock, clocks_event_reasons, power and temperatures.
 *   hb::CsvWriter    crash-safe append-only CSV: header only on an empty file (checked against an existing
 *                    header), whole rows buffered and appended under an exclusive flock by flush()
 *                    (call it per launch), fsync by sync() (call it per configuration). Never truncates.
 *   hb::Run          one binary invocation: run_id (ISO timestamp), the run registry
 *                    results/<EXP>__runs.csv (started/complete rows), the clock log
 *                    results/<EXP>__clock_log.csv, and prefix(): the mandatory CSV columns
 *                    gpu_name, sm_count, sm_clock_mhz, mem_clock_mhz, cuda_version, driver_version, run_id, launch_idx.
 *   Run::clock_note  appends one clock-log row (label, phase, reasons, throttle/below-lock flags, power, temps)
 *                    and returns the sample. lock_clocks_note() also appends `nvidia-smi -q -d CLOCK,PERFORMANCE`
 *                    to results/<EXP>__nvidia_smi.log. A sample is "bad" (launch to be discarded) if a throttle
 *                    reason fired (SW/HW power cap, SW/HW thermal, power brake) or the SM clock is more than one
 *                    15 MHz step below the lock requested through tools/run (H100MB_LOCK_SM_MHZ).
 *   hb::ClusterLaunch  cudaLaunchKernelEx with optional cluster dims and an explicit scheduling policy
 *                    (explicit_cluster: also for a cluster of 1); prepare() fills the launch configuration
 *                    once so that a timed launch is a bare cudaLaunchKernelEx;
 *                    all_distinct()/distinct_within_clusters() check %smid placement.
 *   hb::SmClaim      host side of claim_target_sm: pin a one-warp block to a chosen SM without reserving its smem.
 *   hb::L2Flusher    streams a 128 MB buffer (read-modify-write) to evict L2.
 *   hb::warmup       runs a launch functor n times, synchronising after each.
 */
#pragma once
#include <cuda_runtime.h>
#include <nvml.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <set>
#include <string>
#include <utility>
#include <vector>

#define CUDA_CHECK(call)                                                                                   \
    do {                                                                                                   \
        cudaError_t e_ = (call);                                                                           \
        if (e_ != cudaSuccess) {                                                                           \
            std::fprintf(stderr, "CUDA error at %s:%d: %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e_)); \
            std::exit(2);                                                                                  \
        }                                                                                                  \
    } while (0)
#define NVML_CHECK(call)                                                                                   \
    do {                                                                                                   \
        nvmlReturn_t r_ = (call);                                                                          \
        if (r_ != NVML_SUCCESS) {                                                                          \
            std::fprintf(stderr, "NVML error at %s:%d: %s: %s\n", __FILE__, __LINE__, #call, nvmlErrorString(r_)); \
            std::exit(2);                                                                                  \
        }                                                                                                  \
    } while (0)

namespace hb {

[[noreturn]] inline void die(const std::string& msg) {
    std::fprintf(stderr, "ERROR: %s\n", msg.c_str());
    std::exit(3);
}

// Local time, millisecond precision, numeric UTC offset: 2026-10-23T14:03:05.123-05:00
inline std::string iso_now() {
    using namespace std::chrono;
    const auto now = system_clock::now();
    const long long ms = duration_cast<milliseconds>(now.time_since_epoch()).count() % 1000;
    const time_t t = system_clock::to_time_t(now);
    struct tm tm;
    localtime_r(&t, &tm);
    char base[32], tz[8], out[64];
    std::strftime(base, sizeof base, "%Y-%m-%dT%H:%M:%S", &tm);
    std::strftime(tz, sizeof tz, "%z", &tm);
    std::string z(tz);
    if (z.size() == 5) z.insert(3, ":");
    std::snprintf(out, sizeof out, "%s.%03lld%s", base, ms, z.c_str());
    return out;
}

inline std::string join(const std::vector<std::string>& v, char sep = ',') {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) { if (i) s += sep; s += v[i]; }
    return s;
}
inline std::string csv_quote(const std::string& s) {
    std::string q = "\"";
    for (char c : s) { if (c == '"') q += '"'; q += c; }
    return q + "\"";
}

// ---------------------------------------------------------------- argument helpers
inline const char* arg_str(int argc, char** argv, const char* key, const char* def) {
    for (int i = 1; i + 1 < argc; ++i) if (!std::strcmp(argv[i], key)) return argv[i + 1];
    return def;
}
inline long long arg_int(int argc, char** argv, const char* key, long long def) {
    const char* s = arg_str(argc, argv, key, nullptr);
    return s ? std::atoll(s) : def;
}
inline bool has_flag(int argc, char** argv, const char* key) {
    for (int i = 1; i < argc; ++i) if (!std::strcmp(argv[i], key)) return true;
    return false;
}

// ---------------------------------------------------------------- crash-safe CSV
class CsvWriter {
  public:
    CsvWriter(std::string path, const std::vector<std::string>& header) : path_(std::move(path)), header_(join(header)) {
        fd_ = ::open(path_.c_str(), O_WRONLY | O_APPEND | O_CREAT, 0664);
        if (fd_ < 0) die("cannot open " + path_ + ": " + std::strerror(errno));
        ::flock(fd_, LOCK_EX);
        struct stat st;
        ::fstat(fd_, &st);
        if (st.st_size > 0) {  // an existing file must have exactly our header
            FILE* f = std::fopen(path_.c_str(), "r");
            std::string first;
            for (int c; f && (c = std::fgetc(f)) != EOF && c != '\n';) first += (char)c;
            if (f) std::fclose(f);
            if (first != header_) { ::flock(fd_, LOCK_UN); die("header mismatch in " + path_ + "\n  file: " + first + "\n  ours: " + header_); }
        }
        ::flock(fd_, LOCK_UN);
    }
    CsvWriter(const CsvWriter&) = delete;
    CsvWriter& operator=(const CsvWriter&) = delete;
    ~CsvWriter() { sync(); ::close(fd_); }

    // One complete row (no trailing newline); buffered until flush().
    void add(const std::string& row) { buf_ += row; buf_ += '\n'; }
    // Append all buffered rows under an exclusive lock; the header goes first if the file is empty.
    void flush() {
        if (buf_.empty()) return;
        ::flock(fd_, LOCK_EX);
        struct stat st;
        ::fstat(fd_, &st);
        if (st.st_size == 0) write_all(header_ + "\n");
        write_all(buf_);
        ::flock(fd_, LOCK_UN);
        buf_.clear();
    }
    void sync() { flush(); ::fsync(fd_); }
    const std::string& path() const { return path_; }

  private:
    void write_all(const std::string& s) {
        size_t off = 0;
        while (off < s.size()) {
            ssize_t n = ::write(fd_, s.data() + off, s.size() - off);
            if (n < 0) { if (errno == EINTR) continue; die("write failed on " + path_ + ": " + std::strerror(errno)); }
            off += (size_t)n;
        }
    }
    std::string path_, header_, buf_;
    int fd_ = -1;
};

// ---------------------------------------------------------------- device identity, configuration, clocks
struct ClockSample {
    unsigned sm_mhz = 0, mem_mhz = 0;
    unsigned long long reasons = 0;
    double power_w = 0;
    unsigned gpu_temp_c = 0, mem_temp_c = 0;
    bool throttle = false, below_lock = false;
    bool bad() const { return throttle || below_lock; }
};
// NVML clocks_event_reasons bits that mean the clock was reduced by power/thermal limits:
// SwPowerCap 0x4, HwSlowdown 0x8, SwThermalSlowdown 0x20, HwThermalSlowdown 0x40, HwPowerBrakeSlowdown 0x80.
constexpr unsigned long long kThrottleMask = 0x4ull | 0x8ull | 0x20ull | 0x40ull | 0x80ull;

struct DeviceInfo {
    cudaDeviceProp prop{};
    std::string name, uuid, driver_version, cuda_version;
    int sm_count = 0, l2_bytes = 0;
    unsigned power_limit_mw = 0;
    bool ecc = false, mig = false;
    unsigned lock_mhz = 0;  // requested by tools/run --lock-sm-mhz (0 = none)
    nvmlDevice_t nvml{};

    static DeviceInfo open(bool enforce = true) {
        DeviceInfo d;
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaFree(nullptr));
        CUDA_CHECK(cudaGetDeviceProperties(&d.prop, 0));
        d.name = d.prop.name;
        d.sm_count = d.prop.multiProcessorCount;
        d.l2_bytes = d.prop.l2CacheSize;
        char u[64];
        const unsigned char* b = reinterpret_cast<const unsigned char*>(d.prop.uuid.bytes);
        std::snprintf(u, sizeof u, "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                      b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
        d.uuid = u;
        int rt = 0;
        CUDA_CHECK(cudaRuntimeGetVersion(&rt));
        d.cuda_version = std::to_string(rt / 1000) + "." + std::to_string((rt % 1000) / 10);

        NVML_CHECK(nvmlInit_v2());
        NVML_CHECK(nvmlDeviceGetHandleByUUID(d.uuid.c_str(), &d.nvml));
        char drv[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];
        NVML_CHECK(nvmlSystemGetDriverVersion(drv, sizeof drv));
        d.driver_version = drv;
        NVML_CHECK(nvmlDeviceGetEnforcedPowerLimit(d.nvml, &d.power_limit_mw));
        nvmlEnableState_t ecc_cur, ecc_pend;
        if (nvmlDeviceGetEccMode(d.nvml, &ecc_cur, &ecc_pend) == NVML_SUCCESS) d.ecc = ecc_cur == NVML_FEATURE_ENABLED;
        unsigned mig_cur = 0, mig_pend = 0;
        if (nvmlDeviceGetMigMode(d.nvml, &mig_cur, &mig_pend) == NVML_SUCCESS) d.mig = mig_cur == NVML_DEVICE_MIG_ENABLE;
        if (const char* l = std::getenv("H100MB_LOCK_SM_MHZ")) d.lock_mhz = (unsigned)std::atoi(l);

        if (enforce) {  // the fixed configuration of the project; refuse anything else
            if (d.name.find("H100 NVL") == std::string::npos) die("device is '" + d.name + "', expected an H100 NVL");
            if (d.power_limit_mw != 400000) die("enforced power limit is " + std::to_string(d.power_limit_mw / 1000) + " W, expected 400 W");
            if (!d.ecc) die("ECC is disabled; the fixed configuration requires ECC enabled");
            if (d.mig) die("MIG is enabled; the fixed configuration requires MIG disabled");
            if (const char* want = std::getenv("H100MB_GPU_UUID"))
                if (d.uuid != want) die("visible GPU " + d.uuid + " is not the selected " + want);
        }
        CUDA_CHECK(cudaCtxResetPersistingL2Cache());
        return d;
    }

    ClockSample sample() const {
        ClockSample s;
        nvmlDeviceGetClockInfo(nvml, NVML_CLOCK_SM, &s.sm_mhz);
        nvmlDeviceGetClockInfo(nvml, NVML_CLOCK_MEM, &s.mem_mhz);
        nvmlDeviceGetCurrentClocksEventReasons(nvml, &s.reasons);
        unsigned mw = 0;
        if (nvmlDeviceGetPowerUsage(nvml, &mw) == NVML_SUCCESS) s.power_w = mw / 1000.0;
        nvmlDeviceGetTemperature(nvml, NVML_TEMPERATURE_GPU, &s.gpu_temp_c);
        nvmlFieldValue_t fv{};
        fv.fieldId = NVML_FI_DEV_MEMORY_TEMP;
        if (nvmlDeviceGetFieldValues(nvml, 1, &fv) == NVML_SUCCESS && fv.nvmlReturn == NVML_SUCCESS) s.mem_temp_c = fv.value.uiVal;
        s.throttle = (s.reasons & kThrottleMask) != 0;
        s.below_lock = lock_mhz > 0 && s.sm_mhz + 15 < lock_mhz;
        return s;
    }
};

// ---------------------------------------------------------------- one benchmark invocation
inline std::vector<std::string> mandatory_cols() {
    return {"gpu_name", "sm_count", "sm_clock_mhz", "mem_clock_mhz", "cuda_version", "driver_version", "run_id", "launch_idx"};
}
inline std::vector<std::string> with_mandatory(const std::vector<std::string>& cols) {
    auto v = mandatory_cols();
    v.insert(v.end(), cols.begin(), cols.end());
    return v;
}

class Run {
  public:
    Run(const std::string& exp, int argc, char** argv, bool enforce = true)
        : exp_(exp), dev_(DeviceInfo::open(enforce)), run_id_(iso_now()),
          registry_(results(exp + "__runs.csv"), {"run_id", "started_iso", "finished_iso", "status", "gpu_uuid", "cmdline"}),
          clock_log_(results(exp + "__clock_log.csv"),
                     with_mandatory({"label", "phase", "clocks_event_reasons_hex", "throttle", "below_lock",
                                     "power_w", "gpu_temp_c", "mem_temp_c", "lock_sm_mhz"})) {
        std::vector<std::string> args(argv, argv + argc);
        cmdline_ = join(args, ' ');
        registry_.add(run_id_ + "," + iso_now() + ",,started," + dev_.uuid + "," + csv_quote(cmdline_));
        registry_.sync();
        std::printf("run_id %s  %s  %d SMs  lock %u MHz  driver %s  CUDA %s\n", run_id_.c_str(), dev_.name.c_str(),
                    dev_.sm_count, dev_.lock_mhz, dev_.driver_version.c_str(), dev_.cuda_version.c_str());
    }
    // Call after the last row of every results file has been synced.
    void complete() {
        registry_.add(run_id_ + ",," + iso_now() + ",complete," + dev_.uuid + "," + csv_quote(cmdline_));
        registry_.sync();
    }

    static std::string results(const std::string& file) { return "results/" + file; }
    const DeviceInfo& dev() const { return dev_; }
    const std::string& run_id() const { return run_id_; }

    // Mandatory column values for one launch; clocks are the measured values in `s`.
    std::string prefix(long long launch_idx, const ClockSample& s) const {
        char b[512];
        std::snprintf(b, sizeof b, "%s,%d,%u,%u,%s,%s,%s,%lld", dev_.name.c_str(), dev_.sm_count, s.sm_mhz, s.mem_mhz,
                      dev_.cuda_version.c_str(), dev_.driver_version.c_str(), run_id_.c_str(), launch_idx);
        return b;
    }

    // Sample clocks and append one row to the clock log (flushed immediately).
    ClockSample clock_note(const std::string& label, const std::string& phase, long long launch_idx = -1) {
        ClockSample s = dev_.sample();
        char b[256];
        std::snprintf(b, sizeof b, ",%s,%s,0x%016llx,%d,%d,%.2f,%u,%u,%u", label.c_str(), phase.c_str(), s.reasons,
                      (int)s.throttle, (int)s.below_lock, s.power_w, s.gpu_temp_c, s.mem_temp_c, dev_.lock_mhz);
        clock_log_.add(prefix(launch_idx, s) + b);
        clock_log_.flush();
        if (s.bad())
            std::fprintf(stderr, "WARNING [%s/%s] sm %u MHz (lock %u), reasons 0x%llx -> launch marked bad\n", label.c_str(),
                         phase.c_str(), s.sm_mhz, dev_.lock_mhz, s.reasons);
        return s;
    }
    // clock_note + the full `nvidia-smi -q -d CLOCK,PERFORMANCE` text appended to results/<EXP>__nvidia_smi.log.
    ClockSample lock_clocks_note(const std::string& label, const std::string& phase) {
        const std::string log = results(exp_ + "__nvidia_smi.log");
        int fd = ::open(log.c_str(), O_WRONLY | O_APPEND | O_CREAT, 0664);
        if (fd >= 0) {
            std::string text = "=== run_id " + run_id_ + " label " + label + " phase " + phase + " at " + iso_now() + "\n";
            if (FILE* p = popen("nvidia-smi -q -d CLOCK,PERFORMANCE 2>&1", "r")) {
                char buf[4096];
                size_t n;
                while ((n = std::fread(buf, 1, sizeof buf, p)) > 0) text.append(buf, n);
                pclose(p);
            }
            ::flock(fd, LOCK_EX);
            if (::write(fd, text.data(), text.size()) < 0) std::perror("nvidia_smi.log");
            ::flock(fd, LOCK_UN);
            ::fsync(fd);
            ::close(fd);
        }
        return clock_note(label, phase);
    }

  private:
    std::string exp_;
    DeviceInfo dev_;
    std::string run_id_, cmdline_;
    CsvWriter registry_, clock_log_;
};

// ---------------------------------------------------------------- launches and placement checks
struct ClusterLaunch {
    dim3 grid{1}, block{32}, cluster{1, 1, 1};
    size_t smem = 0;
    cudaClusterSchedulingPolicy policy = cudaClusterSchedulingPolicyDefault;
    cudaStream_t stream = nullptr;
    bool explicit_cluster = false;  // attach cluster attributes even for cluster_size() == 1

    int cluster_size() const { return (int)(cluster.x * cluster.y * cluster.z); }

    // Cluster attributes are attached for cluster_size() > 1, or for a size-1 cluster when explicit_cluster is
    // set (an explicit cluster launch of 1 CTA); otherwise none. Sets the dynamic-smem and non-portable-size
    // attributes when needed.
    template <typename... KArgs, typename... Args>
    cudaError_t launch(void (*k)(KArgs...), Args&&... args) const {
        cudaLaunchConfig_t cfg{};
        cudaLaunchAttribute attrs[2];
        fill(reinterpret_cast<const void*>(k), cfg, attrs);
        return cudaLaunchKernelEx(&cfg, k, std::forward<Args>(args)...);
    }
    // Configuration filled once (function attributes set here), so that a timed launch is a bare
    // cudaLaunchKernelEx(&p.cfg, k, args...). p.cfg.attrs points into p.attrs: keep p in place.
    struct Prepared {
        cudaLaunchConfig_t cfg{};
        cudaLaunchAttribute attrs[2];
    };
    template <typename... KArgs>
    void prepare(void (*k)(KArgs...), Prepared& p) const { fill(reinterpret_cast<const void*>(k), p.cfg, p.attrs); }

    template <typename... KArgs>
    int max_active_clusters(void (*k)(KArgs...)) const {
        cudaLaunchConfig_t cfg{};
        cudaLaunchAttribute attrs[2];
        fill(reinterpret_cast<const void*>(k), cfg, attrs);
        int n = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveClusters(&n, k, &cfg));
        return n;
    }

  private:
    void fill(const void* k, cudaLaunchConfig_t& cfg, cudaLaunchAttribute* attrs) const {
        if (smem > 48 * 1024) CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        if (cluster_size() > 8) CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
        cfg.gridDim = grid;
        cfg.blockDim = block;
        cfg.dynamicSmemBytes = smem;
        cfg.stream = stream;
        cfg.attrs = attrs;
        cfg.numAttrs = 0;
        if (cluster_size() > 1 || explicit_cluster) {
            attrs[0].id = cudaLaunchAttributeClusterDimension;
            attrs[0].val.clusterDim.x = cluster.x;
            attrs[0].val.clusterDim.y = cluster.y;
            attrs[0].val.clusterDim.z = cluster.z;
            attrs[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
            attrs[1].val.clusterSchedulingPolicyPreference = policy;
            cfg.numAttrs = 2;
        }
    }
};

inline bool all_distinct(const std::vector<uint32_t>& v) {
    std::set<uint32_t> s(v.begin(), v.end());
    return s.size() == v.size();
}
// smid[i] for block i; blocks [c*cs, (c+1)*cs) form cluster c (1-D grid and cluster).
inline bool distinct_within_clusters(const std::vector<uint32_t>& smid, int cs) {
    for (size_t c = 0; c + cs <= smid.size(); c += cs) {
        const std::set<uint32_t> s(smid.begin() + c, smid.begin() + c + cs);
        if ((int)s.size() != cs) return false;
    }
    return true;
}

// Host side of claim_target_sm (harness.cuh): a 2-word claim in device memory. Launch blocks() one-warp blocks;
// after the launch, ok(target) tells whether a block ran on `target` (if not, re-launch and count the retry).
class SmClaim {
  public:
    SmClaim() { CUDA_CHECK(cudaMalloc(&d_, 2 * sizeof(uint32_t))); }
    ~SmClaim() { cudaFree(d_); }
    SmClaim(const SmClaim&) = delete;
    SmClaim& operator=(const SmClaim&) = delete;
    uint32_t* ptr() const { return d_; }
    static int blocks(int sm_count) { return 2 * sm_count; }
    void reset(cudaStream_t s = nullptr) { CUDA_CHECK(cudaMemsetAsync(d_, 0, 2 * sizeof(uint32_t), s)); }
    bool ok(uint32_t target) const {
        uint32_t h[2];
        CUDA_CHECK(cudaMemcpy(h, d_, sizeof h, cudaMemcpyDeviceToHost));
        return h[0] != 0 && h[1] == target;
    }

  private:
    uint32_t* d_ = nullptr;
};

// ---------------------------------------------------------------- cache state and warm-up
static __global__ void hb_flush_kernel(uint4* buf, size_t n, unsigned salt) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) buf[i].x += salt;
}
class L2Flusher {
  public:
    explicit L2Flusher(size_t bytes = 128ull << 20) : n_(bytes / sizeof(uint4)) {
        CUDA_CHECK(cudaMalloc(&buf_, n_ * sizeof(uint4)));
        CUDA_CHECK(cudaMemset(buf_, 0, n_ * sizeof(uint4)));
    }
    ~L2Flusher() { cudaFree(buf_); }
    void operator()(cudaStream_t s = nullptr) {
        hb_flush_kernel<<<132 * 4, 1024, 0, s>>>(buf_, n_, ++salt_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(s));
    }

  private:
    uint4* buf_ = nullptr;
    size_t n_;
    unsigned salt_ = 0;
};

template <class F>
void warmup(F&& launch, int n = 2) {
    for (int i = 0; i < n; ++i) {
        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

}  // namespace hb
