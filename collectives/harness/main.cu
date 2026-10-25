/*
 * main.cu — host harness for one collective configuration (collectives/docs/TESTING.md §2–3, EXPERIMENTS.md §3, §5).
 *
 * One binary per (file, N, M, T): `#include COLLECTIVE_FILE` brings in
 *   __global__ void __cluster_dims__(CLUSTER_N,1,1) collective_kernel(const ELEM_T* in, ELEM_T* out,
 *                                                                    unsigned long long* ts, unsigned* smid);
 * Compile-time knobs (set by harness/driver.py): COLLECTIVE_FILE ("../<file>.cu"), COLL_NAME ("<file>"),
 * COLL_OP (0 allreduce, 1 allgather, 2 reducescatter, 3 alltoall), CLUSTER_N, MSG_ELEMS, ELEM_T (float|int).
 *
 * Launch: one cluster per launch, gridDim = CLUSTER_N, blockDim = --threads (256); the cluster size comes from the
 * kernel's compile-time __cluster_dims__ (plain <<<>>> launch; CLUSTER_N > 8 also sets
 * cudaFuncAttributeNonPortableClusterSizeAllowed — the launch path validated by harness/trap_cs16.cu).
 * Shared memory: the file's `struct CollectiveSmem` lives in dynamic shared memory; kSmemBytes =
 * sizeof(CollectiveSmem) is passed as the dynamic size of every launch (and of the occupancy query), with
 * cudaFuncAttributeMaxDynamicSharedMemorySize set to it. A configuration needing more than kSmemLimit =
 * 232448 B (227 KiB, the H100 per-block opt-in maximum) fails to compile with a static_assert whose message
 * starts with SMEM_LIMIT (driver.py: SKIP:smem). The value is also embedded in the binary as the string
 * "COLLECTIVE_SMEM_BYTES=<n>;" so driver.py can read it without running the binary.
 * Before every launch `out` and `ts` are zeroed and `smid` set to 0xffffffff (outside the kernel, so the
 * in-kernel %globaltimer stamps are not affected); a cudaEvent pair brackets the kernel alone (event_ms).
 *
 * Watchdog: the host polls cudaEventQuery on the end event; after --timeout-ms it prints `DEADLOCK …`, tries to
 * read `ts` through a separate non-blocking stream (bounded wait), appends a DEADLOCK row, and _exit(3)s.
 * harness/driver.py runs every configuration in a child process with its own timeout as a backstop.
 *
 * Inputs (§3): pattern A in[k] = T((k*7+3) % 17 - 8) over the whole `in` array (integer-valued, every sum exact);
 * pattern B: uniform floats in [-1, 1] from mt19937_64(seed 20261025), float only. Host reference from the
 * layouts at the bottom of dsmem_primitives.cuh, sums accumulated in double (float) or int64 (int):
 *   allreduce     out[b*M + i]           = sum_r in[r*M + i]
 *   allgather     out[b*N*M + r*M + i]   = in[r*M + i]
 *   reducescatter out[b*M + i]           = sum_r in[r*N*M + b*M + i]
 *   alltoall      out[b*N*M + r*M + i]   = in[r*N*M + b*M + i]
 * An element is bad if got != ref (pattern A, every int run, all data-movement ops: exact), or, for pattern B
 * reductions, if |got - ref| > 1e-5 * (1 + |ref|) * ceil(log2 N). max_abs_err = max |got - ref|;
 * max_rel_err = max |got - ref| / (1 + |ref|) (the normalisation of the tolerance).
 *
 * Modes and output (all CSVs start with the mandatory columns, hb::with_mandatory; launch_idx = rep):
 *   --mode correctness  --reps R (200) launches, every output checked →
 *       results/correctness.csv: file,op,alg,N,M,T,pattern,process,rep,status,max_abs_err,max_rel_err,n_bad,
 *       latency_ns,event_ms,smids,smid_distinct,clock_bad,max_active_clusters
 *       (status PASS | FAIL | DEADLOCK; latency_ns = max_r ts[r][exit] - min_r ts[r][entry]).
 *   --mode perf  20 warm-up + R timed launches, only the last output is checked →
 *       results/stamps.csv: file,op,alg,N,M,elem_bytes,payload_bytes,rep,rank,smid,stamp,t_ns,clock_bad
 *       results/host.csv:   file,op,alg,N,M,elem_bytes,rep,event_ms,clock_bad
 *       results/perf_check.csv: file,op,alg,N,M,T,pattern,process,reps,status,max_abs_err,max_rel_err,n_bad,n_clock_bad
 *       (stamp ∈ entry, exit, round<k> for every non-zero round slot; rows of clock-bad launches are kept and flagged).
 *   --mode ncu  R launches (default 1) for Nsight Compute; results printed, no result CSV (registry only).
 *       --baseline launches an empty kernel with the same grid, block and cluster instead (ncu baseline).
 * clock_bad = hb::ClockSample::bad() of the NVML sample taken right after the launch (throttle reason or SM clock
 * below the tools/run lock). Clock notes at the start and end of the configuration go to
 * results/collectives__clock_log.csv; the run is registered in results/collectives__runs.csv (started/complete).
 * Run from collectives/ (hb::Run writes ./results/).
 */
#include <host.hpp>

#include <cmath>
#include <memory>
#include <random>
#include <thread>
#include <type_traits>

#ifndef COLLECTIVE_FILE
#error "compile with -DCOLLECTIVE_FILE='\"../<file>.cu\"' (see harness/driver.py)"
#endif
#include COLLECTIVE_FILE

#ifndef COLL_NAME
#error "compile with -DCOLL_NAME='\"<file>\"'"
#endif
#ifndef COLL_OP
#error "compile with -DCOLL_OP=<0 allreduce|1 allgather|2 reducescatter|3 alltoall>"
#endif

// Same grid, block, compile-time cluster and dynamic smem as collective_kernel, does nothing: same-grid ncu baseline.
__global__ void __cluster_dims__(CLUSTER_N, 1, 1)
baseline_kernel(const ELEM_T*, ELEM_T*, unsigned long long*, unsigned*) {}

namespace {

constexpr int N = CLUSTER_N, M = MSG_ELEMS;
constexpr int OP = COLL_OP;
constexpr const char* kOpName[] = {"allreduce", "allgather", "reducescatter", "alltoall"};
constexpr bool kReduce = OP == 0 || OP == 2;
constexpr size_t kInElems = (OP == 0 || OP == 1) ? (size_t)N * M : (size_t)N * N * M;
constexpr size_t kOutElems = (OP == 0 || OP == 2) ? (size_t)N * M : (size_t)N * N * M;
constexpr size_t kPayloadBytes = ((OP == 0 || OP == 1) ? (size_t)M : (size_t)N * M) * sizeof(ELEM_T);
constexpr bool kIsInt = std::is_same<ELEM_T, int>::value;
static_assert(kIsInt || std::is_same<ELEM_T, float>::value, "ELEM_T must be float or int");
constexpr size_t kSmemBytes = sizeof(CollectiveSmem);
constexpr size_t kSmemLimit = 232448;  // cudaDevAttrMaxSharedMemoryPerBlockOptin on H100 (227 KiB)
static_assert(kSmemBytes <= kSmemLimit, "SMEM_LIMIT: CollectiveSmem exceeds the 227 KiB a block may opt in to");

// "COLLECTIVE_SMEM_BYTES=<kSmemBytes>;" as a host string kept in the binary (read by driver.py).
struct SmemTag {
    char s[48];
};
constexpr SmemTag make_smem_tag(size_t v) {
    SmemTag t{};
    const char head[] = "COLLECTIVE_SMEM_BYTES=";
    int n = 0;
    for (int i = 0; head[i]; ++i) t.s[n++] = head[i];
    char d[24]{};
    int k = 0;
    do { d[k++] = char('0' + v % 10); v /= 10; } while (v);
    while (k) t.s[n++] = d[--k];
    t.s[n++] = ';';
    return t;
}
__attribute__((used)) const SmemTag kSmemTag = make_smem_tag(kSmemBytes);

std::string alg_name() {  // <op>__<alg>__<src> → <alg>
    std::string f = COLL_NAME;
    size_t a = f.find("__"), b = f.rfind("__");
    return (a != std::string::npos && b > a) ? f.substr(a + 2, b - a - 2) : f;
}

void fill_input(std::vector<ELEM_T>& in, char pattern) {
    if (pattern == 'A') {
        for (size_t k = 0; k < in.size(); ++k) in[k] = ELEM_T((long long)((k * 7 + 3) % 17) - 8);
    } else {
        std::mt19937_64 g(20261025);
        std::uniform_real_distribution<double> u(-1.0, 1.0);
        for (auto& x : in) x = ELEM_T(u(g));
    }
}

// Reference in double (exact for pattern A and for int, whose sums are far below 2^53).
std::vector<double> reference(const std::vector<ELEM_T>& in) {
    std::vector<double> ref(kOutElems, 0.0);
    for (int b = 0; b < N; ++b)
        for (int i = 0; i < M; ++i) {
            if (OP == 0) {
                double s = 0;
                for (int r = 0; r < N; ++r) s += (double)in[(size_t)r * M + i];
                ref[(size_t)b * M + i] = s;
            } else if (OP == 1) {
                for (int r = 0; r < N; ++r) ref[(size_t)b * N * M + (size_t)r * M + i] = (double)in[(size_t)r * M + i];
            } else if (OP == 2) {
                double s = 0;
                for (int r = 0; r < N; ++r) s += (double)in[(size_t)r * N * M + (size_t)b * M + i];
                ref[(size_t)b * M + i] = s;
            } else {
                for (int r = 0; r < N; ++r)
                    ref[(size_t)b * N * M + (size_t)r * M + i] = (double)in[(size_t)r * N * M + (size_t)b * M + i];
            }
        }
    return ref;
}

struct Check {
    long long n_bad = 0;
    double max_abs = 0, max_rel = 0;
};

Check check(const std::vector<ELEM_T>& out, const std::vector<double>& ref, char pattern) {
    Check c;
    const bool tol = kReduce && !kIsInt && pattern == 'B';
    const double ln = std::ceil(std::log2((double)N));
    for (size_t k = 0; k < out.size(); ++k) {
        const double got = (double)out[k];
        const double want = kIsInt ? ref[k] : (double)(float)ref[k];  // exact cases compare in the element type
        const double err = std::fabs(got - ref[k]);
        bool bad;
        if (tol) bad = !(err <= 1e-5 * (1.0 + std::fabs(ref[k])) * ln);
        else bad = !(got == want);
        c.n_bad += bad;
        if (!(err <= c.max_abs)) c.max_abs = err;  // also propagates NaN
        const double rel = err / (1.0 + std::fabs(ref[k]));
        if (!(rel <= c.max_rel)) c.max_rel = rel;
    }
    return c;
}

std::string fmt(double v) {
    char b[32];
    std::snprintf(b, sizeof b, "%.6g", v);
    return b;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = hb::arg_str(argc, argv, "--mode", "correctness");
    const char pattern = hb::arg_str(argc, argv, "--pattern", "A")[0];
    const bool is_ncu = mode == "ncu", is_perf = mode == "perf";
    if (!is_ncu && !is_perf && mode != "correctness") hb::die("--mode must be correctness, perf or ncu");
    if (pattern != 'A' && pattern != 'B') hb::die("--pattern must be A or B");
    if (pattern == 'B' && kIsInt) hb::die("pattern B is float-only");
    const int reps = (int)hb::arg_int(argc, argv, "--reps", is_ncu ? 1 : 200);
    const int warmups = is_perf ? (int)hb::arg_int(argc, argv, "--warmups", 20) : 0;
    const int process = (int)hb::arg_int(argc, argv, "--process", 0);
    const int threads = (int)hb::arg_int(argc, argv, "--threads", 256);
    const long long timeout_ms = hb::arg_int(argc, argv, "--timeout-ms", is_ncu ? 600000 : 5000);
    const bool baseline = hb::has_flag(argc, argv, "--baseline");
    if (reps < 1 || threads < 32 || threads > 1024 || threads % 32) hb::die("bad --reps / --threads");
    if (baseline && !is_ncu) hb::die("--baseline is only for --mode ncu");

    hb::Run run("collectives", argc, argv);
    const std::string file = COLL_NAME, op = kOpName[OP], alg = alg_name();
    const std::string T = kIsInt ? "int" : "float";
    const std::string label = file + "__N" + std::to_string(N) + "_M" + std::to_string(M) + "_" + T;
    std::printf("%s  mode %s  pattern %c  reps %d  threads %d%s\n", label.c_str(), mode.c_str(), pattern, reps, threads,
                baseline ? "  BASELINE kernel" : "");

    auto kern = baseline ? baseline_kernel : collective_kernel;
    if (N > 8) CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    int optin = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    if (kSmemBytes > (size_t)optin)
        hb::die("CollectiveSmem needs " + std::to_string(kSmemBytes) + " B, the device allows " + std::to_string(optin));
    CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kSmemBytes));
    std::printf("dynamic shared memory per CTA: %zu B (device opt-in limit %d B)\n", kSmemBytes, optin);
    cudaLaunchConfig_t occ_cfg{};
    occ_cfg.gridDim = dim3(N);
    occ_cfg.blockDim = dim3(threads);
    occ_cfg.dynamicSmemBytes = kSmemBytes;
    int max_active = -1;
    CUDA_CHECK(cudaOccupancyMaxActiveClusters(&max_active, kern, &occ_cfg));
    std::printf("cudaOccupancyMaxActiveClusters = %d\n", max_active);
    if (max_active < 1) hb::die("cudaOccupancyMaxActiveClusters < 1: the cluster cannot be placed");

    std::vector<ELEM_T> h_in(kInElems), h_out(kOutElems);
    fill_input(h_in, pattern);
    const std::vector<double> ref = reference(h_in);
    ELEM_T *d_in, *d_out;
    unsigned long long* d_ts;
    unsigned* d_smid;
    CUDA_CHECK(cudaMalloc(&d_in, kInElems * sizeof(ELEM_T)));
    CUDA_CHECK(cudaMalloc(&d_out, kOutElems * sizeof(ELEM_T)));
    CUDA_CHECK(cudaMalloc(&d_ts, (size_t)N * kStamps * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_smid, N * sizeof(unsigned)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), kInElems * sizeof(ELEM_T), cudaMemcpyHostToDevice));
    std::vector<unsigned long long> h_ts((size_t)N * kStamps);
    std::vector<unsigned> h_smid(N);
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    // Result files of the selected mode (opened first so a header mismatch stops before any launch).
    std::unique_ptr<hb::CsvWriter> w_corr, w_stamps, w_host, w_pcheck;
    if (mode == "correctness")
        w_corr.reset(new hb::CsvWriter(hb::Run::results("correctness.csv"),
                                       hb::with_mandatory({"file", "op", "alg", "N", "M", "T", "pattern", "process", "rep",
                                                           "status", "max_abs_err", "max_rel_err", "n_bad", "latency_ns",
                                                           "event_ms", "smids", "smid_distinct", "clock_bad",
                                                           "max_active_clusters"})));
    if (is_perf) {
        w_stamps.reset(new hb::CsvWriter(hb::Run::results("stamps.csv"),
                                         hb::with_mandatory({"file", "op", "alg", "N", "M", "elem_bytes", "payload_bytes",
                                                             "rep", "rank", "smid", "stamp", "t_ns", "clock_bad"})));
        w_host.reset(new hb::CsvWriter(hb::Run::results("host.csv"),
                                       hb::with_mandatory({"file", "op", "alg", "N", "M", "elem_bytes", "rep", "event_ms",
                                                           "clock_bad"})));
        w_pcheck.reset(new hb::CsvWriter(hb::Run::results("perf_check.csv"),
                                         hb::with_mandatory({"file", "op", "alg", "N", "M", "T", "pattern", "process",
                                                             "reps", "status", "max_abs_err", "max_rel_err", "n_bad",
                                                             "n_clock_bad"})));
    }
    const std::string cfg_cols = file + "," + op + "," + alg + "," + std::to_string(N) + "," + std::to_string(M);

    hb::ClockSample last = run.clock_note(label, "start");
    long long n_fail = 0, n_clock_bad = 0;
    Check last_check;
    for (int it = -warmups; it < reps; ++it) {
        const int rep = it;  // negative = warm-up (not recorded)
        CUDA_CHECK(cudaMemset(d_out, 0, kOutElems * sizeof(ELEM_T)));
        CUDA_CHECK(cudaMemset(d_ts, 0, h_ts.size() * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d_smid, 0xff, N * sizeof(unsigned)));
        CUDA_CHECK(cudaEventRecord(e0));
        kern<<<N, threads, kSmemBytes>>>(d_in, d_out, d_ts, d_smid);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(e1));

        // Watchdog: poll the end event; a hang is reported and the process exits (the GPU is freed on exit).
        const auto t0 = std::chrono::steady_clock::now();
        cudaError_t q;
        while ((q = cudaEventQuery(e1)) == cudaErrorNotReady) {
            const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
            if (ms > timeout_ms) {
                std::printf("DEADLOCK %s N %d M %d T %s rep %d (no completion after %lld ms)\n", file.c_str(), N, M,
                            T.c_str(), rep, timeout_ms);
                cudaStream_t s;
                unsigned long long* pin = nullptr;
                if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess &&
                    cudaMallocHost(&pin, h_ts.size() * sizeof(unsigned long long)) == cudaSuccess &&
                    cudaMemcpyAsync(pin, d_ts, h_ts.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost, s) == cudaSuccess) {
                    const auto t1 = std::chrono::steady_clock::now();
                    while (cudaStreamQuery(s) == cudaErrorNotReady &&
                           std::chrono::steady_clock::now() - t1 < std::chrono::milliseconds(1000))
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    if (cudaStreamQuery(s) == cudaSuccess) {
                        for (int r = 0; r < N; ++r) {
                            std::printf("  rank %2d stamps:", r);
                            for (int k = 0; k < kStamps; ++k)
                                if (pin[r * kStamps + k]) std::printf(" [%d]=%llu", k, pin[r * kStamps + k]);
                            std::printf("\n");
                        }
                    } else {
                        std::printf("  stamps not readable while the kernel hangs\n");
                    }
                } else {
                    std::printf("  stamps not readable (side-stream copy could not be issued)\n");
                }
                if (w_corr) {
                    w_corr->add(run.prefix(rep, last) + "," + cfg_cols + "," + T + "," + pattern + "," +
                                std::to_string(process) + "," + std::to_string(rep) + ",DEADLOCK,,,,,,,," +
                                std::to_string(last.bad()) + "," + std::to_string(max_active));
                    w_corr->sync();
                }
                if (w_pcheck) {
                    w_pcheck->add(run.prefix(rep, last) + "," + cfg_cols + "," + T + "," + pattern + "," +
                                  std::to_string(process) + "," + std::to_string(reps) + ",DEADLOCK,,,," +
                                  std::to_string(n_clock_bad));
                    w_pcheck->sync();
                }
                std::fflush(stdout);
                _exit(3);
            }
            std::this_thread::sleep_for(std::chrono::microseconds(20));
        }
        if (q != cudaSuccess) {
            std::fprintf(stderr, "CUDA error in %s rep %d: %s\n", label.c_str(), rep, cudaGetErrorString(q));
            std::exit(2);
        }
        hb::ClockSample s = run.dev().sample();
        last = s;
        float ev_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ev_ms, e0, e1));
        if (it < 0) continue;
        n_clock_bad += s.bad();
        CUDA_CHECK(cudaMemcpy(h_ts.data(), d_ts, h_ts.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_smid.data(), d_smid, N * sizeof(unsigned), cudaMemcpyDeviceToHost));
        const bool need_out = !baseline && (!is_perf || it == reps - 1);
        Check c;
        if (need_out) {
            CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, kOutElems * sizeof(ELEM_T), cudaMemcpyDeviceToHost));
            c = check(h_out, ref, pattern);
            last_check = c;
            n_fail += c.n_bad > 0;
        }
        unsigned long long t_min = ~0ull, t_max = 0;
        for (int r = 0; r < N; ++r) {
            t_min = std::min(t_min, h_ts[r * kStamps + 0]);
            t_max = std::max(t_max, h_ts[r * kStamps + 1]);
        }
        const long long latency_ns = (long long)t_max - (long long)t_min;
        std::string smids;
        std::vector<uint32_t> smv(h_smid.begin(), h_smid.end());
        for (int r = 0; r < N; ++r) smids += (r ? "|" : "") + std::to_string(h_smid[r]);
        const std::string pre = run.prefix(rep, s);

        if (w_corr) {
            w_corr->add(pre + "," + cfg_cols + "," + T + "," + pattern + "," + std::to_string(process) + "," +
                        std::to_string(rep) + "," + (c.n_bad ? "FAIL" : "PASS") + "," + fmt(c.max_abs) + "," +
                        fmt(c.max_rel) + "," + std::to_string(c.n_bad) + "," + std::to_string(latency_ns) + "," +
                        fmt(ev_ms) + "," + smids + "," + std::to_string(hb::all_distinct(smv)) + "," +
                        std::to_string(s.bad()) + "," + std::to_string(max_active));
            w_corr->flush();
        }
        if (is_perf) {
            const std::string eb = std::to_string(sizeof(ELEM_T));
            for (int r = 0; r < N; ++r)
                for (int k = 0; k < kStamps; ++k) {
                    const unsigned long long t = h_ts[r * kStamps + k];
                    if (k >= 2 && t == 0) continue;
                    const std::string name = k == 0 ? "entry" : k == 1 ? "exit" : "round" + std::to_string(k - 2);
                    w_stamps->add(pre + "," + cfg_cols + "," + eb + "," + std::to_string(kPayloadBytes) + "," +
                                  std::to_string(rep) + "," + std::to_string(r) + "," + std::to_string(h_smid[r]) + "," +
                                  name + "," + std::to_string(t) + "," + std::to_string(s.bad()));
                }
            w_host->add(pre + "," + cfg_cols + "," + eb + "," + std::to_string(rep) + "," + fmt(ev_ms) + "," +
                        std::to_string(s.bad()));
            w_stamps->flush();
            w_host->flush();
        }
        if (is_ncu)
            std::printf("rep %d: %s n_bad %lld latency_ns %lld event_ms %s smids %s\n", rep,
                        baseline ? "baseline" : (c.n_bad ? "FAIL" : "PASS"), c.n_bad, latency_ns, fmt(ev_ms).c_str(),
                        smids.c_str());
    }
    if (w_pcheck) {
        w_pcheck->add(run.prefix(reps - 1, last) + "," + cfg_cols + "," + T + "," + pattern + "," + std::to_string(process) +
                      "," + std::to_string(reps) + "," + (last_check.n_bad ? "FAIL" : "PASS") + "," +
                      fmt(last_check.max_abs) + "," + fmt(last_check.max_rel) + "," + std::to_string(last_check.n_bad) +
                      "," + std::to_string(n_clock_bad));
    }
    for (auto* w : {w_corr.get(), w_stamps.get(), w_host.get(), w_pcheck.get()})
        if (w) w->sync();
    run.clock_note(label, "end");
    run.complete();
    std::printf("%s: %d reps, %lld failing, %lld clock-bad launches\n", label.c_str(), reps, n_fail, n_clock_bad);
    return n_fail ? 1 : 0;
}
