/*
 * event_floor.cu — roofline R1: the cudaEvent timing floor on this GPU, subtracted from every GPU-wide rate
 * (ceilings_summary.py: rate = bytes or FLOPs / (event time − floor)).
 *
 * Copy of K5 of MB/experiments/01__timers_and_harness/src/timers.cu (run_k5 and k5_empty) at MB commit
 * 8f8fc7bec69f07db49f3a5a95bbdfb08de5d0ca8. Changes: output file and columns (results/ceilings/event_floor.csv, one
 * clock_bad flag per row from the NVML sample after each mode), exp name "roofline"; the method is unchanged:
 *   single      1000 launches of the empty kernel k5_empty<<<1, 32>>>, each between its own event pair;
 *   batch       10 × (1000 back-to-back launches between one event pair) / 1000 (+ host enqueue time per launch);
 *   empty_pair  1000 event pairs with nothing in between — the floor used by MB 05's analysis (common05.event_floor_ms)
 *               and by ceilings_summary.py (median us_per_launch of the unflagged empty_pair rows of the latest run).
 * 12 warm-up launches first.
 *
 * Usage (from roofline/): harness/ceilings/bin/event_floor {floor|clockcheck} [--quick] [--label L]
 *   --quick: 100 instead of 1000 repetitions.
 * → results/ceilings/event_floor.csv: mandatory columns, mode, launches, us_per_launch, host_api_us_per_launch, clock_bad
 */
#include <chrono>
#include <string>
#include <vector>

#include "ceil_common.hpp"

__global__ void k5_empty() {}

static void run_k5(hb::Run& run, int n) {
    hb::CsvWriter csv(out_csv("event_floor"), hb::with_mandatory({"mode", "launches", "us_per_launch", "host_api_us_per_launch", "clock_bad"}));
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    auto launch = [] { k5_empty<<<1, 32>>>(); };
    hb::warmup(launch, 12);
    char b[512];

    // single: one event pair around each launch
    run.clock_note("k5_single", "before");
    std::vector<float> ms(n);
    for (int i = 0; i < n; ++i) {
        CUDA_CHECK(cudaEventRecord(e0));
        launch();
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        CUDA_CHECK(cudaEventElapsedTime(&ms[i], e0, e1));
    }
    hb::ClockSample s = run.clock_note("k5_single", "after");
    for (int i = 0; i < n; ++i) {
        std::snprintf(b, sizeof b, "%s,single,1,%.4f,,%d", run.prefix(i, s).c_str(), ms[i] * 1e3, (int)s.bad());
        csv.add(b);
    }
    csv.sync();

    // batch: n back-to-back launches between one event pair
    run.clock_note("k5_batch", "before");
    for (int r = 0; r < 10; ++r) {
        CUDA_CHECK(cudaEventRecord(e0));
        const auto h0 = std::chrono::steady_clock::now();
        for (int i = 0; i < n; ++i) launch();
        const auto h1 = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float t;
        CUDA_CHECK(cudaEventElapsedTime(&t, e0, e1));
        s = run.clock_note("k5_batch", "launch", n + r);
        const double host_us = std::chrono::duration<double, std::micro>(h1 - h0).count();
        std::snprintf(b, sizeof b, "%s,batch,%d,%.4f,%.4f,%d", run.prefix(n + r, s).c_str(), n, t * 1e3 / n, host_us / n, (int)s.bad());
        csv.add(b);
        csv.flush();
        std::printf("k5 batch %d: %.3f us/launch (event), %.3f us/launch (host enqueue)\n", r, t * 1e3 / n, host_us / n);
    }
    csv.sync();

    // empty_pair: event pair with nothing in between
    run.clock_note("k5_empty_pair", "before");
    for (int i = 0; i < n; ++i) {
        CUDA_CHECK(cudaEventRecord(e0));
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        CUDA_CHECK(cudaEventElapsedTime(&ms[i], e0, e1));
    }
    s = run.clock_note("k5_empty_pair", "after");
    for (int i = 0; i < n; ++i) {
        std::snprintf(b, sizeof b, "%s,empty_pair,0,%.4f,,%d", run.prefix(n + 10 + i, s).c_str(), ms[i] * 1e3, (int)s.bad());
        csv.add(b);
    }
    csv.sync();
    std::sort(ms.begin(), ms.end());
    std::printf("k5 empty_pair median %.3f us%s\n", ms[n / 2] * 1e3, s.bad() ? "  (clock flagged)" : "");
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "floor";
    const bool quick = hb::has_flag(argc, argv, "--quick");
    hb::Run run(kExp, argc, argv);
    const std::string label = hb::arg_str(argc, argv, "--label", mode.c_str());
    if (mode == "clockcheck") {
        run.lock_clocks_note(label, "check");
    } else if (mode == "floor") {
        run.lock_clocks_note("event_floor", "before");
        run_k5(run, quick ? 100 : 1000);
        run.lock_clocks_note("event_floor", "after");
    } else {
        hb::die("unknown mode " + mode);
    }
    run.complete();
    return 0;
}
