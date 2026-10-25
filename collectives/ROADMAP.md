# ROADMAP — collectives study (tasks C1–C7)

Imported by `collectives/CLAUDE.md`. One conversation = one task, run in order. How a conversation starts
and ends, the commit rule and the subagent rule: `../CLAUDE.md`. Environment: `../docs/ENVIRONMENT.md`.
Reusable code: `../docs/REUSE.md`. Design: `docs/TESTING.md` (correctness) and `docs/EXPERIMENTS.md`
(performance) are binding; the cards below say which sections each task executes.

## Task status
Status values: `todo` · `in progress` (current conversation) · `in review` (committed locally, waiting for the
user's review, not pushed) · `done` (approved and pushed). Update the row **before** the task's commit;
"Key results" records what later tasks consume (numbers, file paths, decisions).

| ID | Task | Status | Key results for later tasks |
|---|---|---|---|
| C1 | Environment, vendored harness, build matrix, GPU smoke test | todo | |
| C2 | Correctness: AllReduce (12 files) | todo | |
| C3 | Correctness: AllGather (7), ReduceScatter (4), AlltoAll (6) | todo | |
| C4 | Sanitizers, deadlock/race checks, placement and traffic sanity | todo | |
| C5 | Performance sweep (stamps, host times, ncu counters) | todo | |
| C6 | Notebook and final report | todo | |
| C7 | *Optional* — extensions taken over from h100-microbench experiment 17 | todo | |

## Dependencies
```
h100-microbench "GPU-3 bring-up" row (GPU lock value + GPU-3 SM→GPC map) ─┐  (needed for any GPU run)
C1 ─► C2 ─┬─► C4 ─► C5 ─► C6 ─► (C7, only if the user asks)
    └► C3 ─┘
```
C2 and C3 are independent of each other (either order). C1's build-only steps can run before the microbench
bring-up is done; its GPU smoke step cannot. This roadmap does not depend on `roofline/`.

## Shared facts for every task
- The 29 files, grouped: **AllReduce (12)** `allreduce__{recursive_doubling__clusterfusion, shift_accumulate_bulk__clusterfusion, shift_accumulate_atomic__clusterfusion, direct_replicated_push__credit, pull_reduce_scatter_push_allgather__hpcops, ring__nccl, rabenseifner__mpich, one_shot_pull__trtllm, two_shot_push__trtllm, reduce_bcast_binomial__mpich, atomic_all_push__generic, 2d_mesh__generic}`;
  **AllGather (7)** `allgather__{recursive_doubling__clusterfusion, direct_shift__clusterfusion, direct_replicated_push__credit, ring__nccl, bruck__mpich, direct_pull__generic, gather_bcast_binomial__mpich}`;
  **ReduceScatter (4)** `reducescatter__{ring__nccl, recursive_halving__mpich, direct_push__generic, direct_pull__generic}`;
  **AlltoAll (6)** `alltoall__{pairwise_exchange__fa4, direct_push__generic, pairwise_xor__mpich, shift__generic, bruck__mpich, direct_pull__generic}`.
  Per-file rounds, message sizes, traffic and N constraints: `README.md`.
- Commands: `TR=~/work/fa2026/iacoma/h100-microbench/tools/run`; builds `$TR --no-gpu -- …`; GPU runs
  `$TR --profile --lock-sm-mhz <LOCK> -- …` on GPU 3 (`<LOCK>` from h100-microbench `docs/context/10` §2).
- Results: `collectives/results/` (git-ignored), mandatory columns first, append-only, run registry
  `results/collectives__runs.csv`. Builds: `collectives/build/` (git-ignored).
- The algorithm files are frozen except for genuine bug fixes (logged in `CLAUDE.md` → Change log).

---

## C1 — Environment, vendored harness, build matrix, GPU smoke test
**Goal.** Everything later tasks need to build and launch any configuration safely, plus the list of legal
configurations.

**Prerequisites.** None for the build steps. For steps 7–8 (GPU): the h100-microbench GPU-3 bring-up row is
`done`; if not, do steps 1–6, commit, and tell the user.

**Read first.** `../docs/ENVIRONMENT.md`, `../docs/REUSE.md`, `CLAUDE.md` (file contract), `docs/TESTING.md`
§1–2, `dsmem_primitives.cuh`, two representative files (`allreduce__ring__nccl.cu`,
`allgather__direct_pull__generic.cu`), `h100-microbench/experiments/07__synchronization_primitives/src/sync_common.hpp`
(`ClusterOnce`).

**Steps.**
1. Record the tooling in `CLAUDE.md` → *Tooling*: `$TR --no-gpu -- nvcc --version`, `ncu --version`,
   image tag (`tools/env info`), driver; the commit of this repo.
2. Vendor `common/h100mb/` per `../docs/REUSE.md` §1/§4 (skip if roofline R1 already did; then only verify
   `PROVENANCE.md`).
3. Repository hygiene: `.claude/settings.json` (`../docs/ENVIRONMENT.md` §6); confirm `.gitignore` covers
   `collectives/build/` and `collectives/results/`.
4. `harness/main.cu` (TESTING §2): `#include COLLECTIVE_FILE`; allocation, deterministic input patterns A
   and B, host references for the four ops using the layouts of `dsmem_primitives.cuh`; launch of one
   cluster; watchdog on `cudaEventQuery` with `TIMEOUT_MS`; per-launch CSV line; `--mode correctness|perf`,
   `--reps`, `--pattern`. Use `hb::DeviceInfo`, `hb::Run`, `hb::CsvWriter`, `hb::with_mandatory` from the
   vendored `host.hpp`. Record `cudaOccupancyMaxActiveClusters`.
   *Trap to settle first:* the kernels carry a compile-time `__cluster_dims__(CLUSTER_N,1,1)`. Check with a
   minimal test whether `CLUSTER_N=16` compiles with a compile-time cluster size and launches with only
   `cudaFuncAttributeNonPortableClusterSizeAllowed` set (no launch-time cluster attribute), and whether
   `cudaLaunchKernelEx` accepts a cluster attribute equal to the compile-time one. Do not use
   `hb::ClusterLaunch` with `explicit_cluster` on these kernels unless that test passes. If CS 16 cannot work
   without editing the algorithm files, stop and ask the user.
5. `harness/driver.py`: enumerates (file, N ∈ {2,4,8,16}, M ∈ {16,64,256,1024,4096}, T ∈ {float, int for
   reduce files}); builds each into `build/<file>__N<N>_M<M>_<T>` with the TESTING §1 command plus
   `-Xptxas -v`; classifies `OK` / `SKIP:static_assert` / `SKIP:smem` / `ERROR` from the compiler output;
   runs configurations as child processes with a timeout; resumable (skips configurations already complete
   in the registry). Parallel builds capped at 16 jobs (the host is shared).
6. Build the full matrix (≈ 29 × 20 float + 16 × 20 int ≈ 900 builds; expect 1–2 h of CPU at 16 jobs) and
   write `results/build_matrix.csv` (`file, op, alg, N, M, T, status, static_smem_bytes, regs, spill_bytes,
   ptxas_msg`). Every `ERROR` that is not an expected `static_assert`/smem failure is a bug: fix it per the
   `CLAUDE.md` workflow or list it for C2/C3.
7. GPU smoke: one configuration per file (N = 4, or the file's only legal N; M = 256; pattern A; 5 reps).
8. ncu metric discovery (one run, `--profile`): `ncu --query-metrics` filtered for `dshared`, `cluster`,
   `cgas`; write the exact names into `docs/EXPERIMENTS.md` §1 and settle which DSMEM byte metric counts the
   payload of pushes (stores/bulk) and of pulls (loads) by profiling one push and one pull smoke config
   whose bytes are known.

**Deliverables.** `common/h100mb/` (if first), `harness/main.cu`, `harness/driver.py`, `harness/Makefile`
(or build rules in the driver), `.claude/settings.json`, `results/build_matrix.csv` (local), filled
*Tooling* and EXPERIMENTS §1.

**Done when.** Every (file, N, M, T) has a build status; the expected skips match TESTING §1 (differences
explained); the smoke run of all 29 files finishes without watchdog expiry (failures are listed, not fixed,
unless trivial); the DSMEM metric names are recorded. GPU time ≈ 15 min.

---

## C2 — Correctness: AllReduce (12 files)
**Goal.** Every legal configuration of the 12 AllReduce files is bit-exact under pattern A for 200 reps × 3
processes, with float and int; pattern B within tolerance.

**Prerequisites.** C1 `done`; GPU-3 bring-up `done`.

**Read first.** `docs/TESTING.md` §2–3 and §6; `README.md` rows of the 12 files; each file's docstring before
debugging it; C1's key results (skips, smoke failures).

**Steps.**
1. Run the matrix for the 12 files with `driver.py --mode correctness` (pattern A float + int, pattern B
   float; `REPS = 200`; 3 fresh processes per configuration). Write `results/correctness.csv`
   (mandatory columns + `file, op, alg, N, M, T, pattern, process, rep, status, max_abs_err, max_rel_err,
   latency_ns`) — or one summary row per (configuration, process) plus every failing rep in full; estimate
   the size first and keep every failing rep.
2. For each failure: reproduce with the smallest N/M, read the stamps, locate the bug (common causes: TESTING
   §4), apply the smallest fix that keeps rounds/partners/message sizes, log it in `CLAUDE.md` → Change log,
   rebuild and rerun the whole file.
3. `allreduce__atomic_all_push__generic` is order-nondeterministic in float: pattern A must still be exact.

**Deliverables.** `results/correctness.csv` (local), fixes in the algorithm files with change-log entries,
a per-file pass/skip table in the commit message and in "Key results".

**Done when.** All legal configurations of the 12 files pass TESTING §6's criterion (sanitizers come in C4).
GPU time ≈ 30–60 min (≈ 12 × 20 × 3 configurations × 3 processes, a few seconds each).

---

## C3 — Correctness: AllGather (7), ReduceScatter (4), AlltoAll (6)
**Goal / steps / deliverables.** As C2, for the 17 data-movement and reduce-scatter files. AllGather and
AlltoAll are pure data movement: bit-exact for every pattern. ReduceScatter uses `Sum`: int and pattern A
exact. Note the smem-heavy layouts (`N·M`, `2·N·M`): many large-N/large-M configurations are expected
`SKIP:smem` from C1.

**Prerequisites.** C1 `done`; GPU-3 bring-up `done`. Independent of C2.

**Done when.** All legal configurations of the 17 files pass TESTING §6's correctness criterion. GPU time
≈ 30–60 min.

---

## C4 — Sanitizers, deadlock/race checks, placement and traffic sanity
**Goal.** Close `docs/TESTING.md` §6: no sanitizer error, no watchdog expiry, placement and stamps sane,
measured DSMEM traffic matches the README model within 10 %.

**Prerequisites.** C2 and C3 `done`.

**Read first.** `docs/TESTING.md` §4–6; `docs/EXPERIMENTS.md` §1 and §4 (metric names from C1);
`../docs/ENVIRONMENT.md` §4 (ncu rules); `README.md` "Total DSMEM bytes per CTA" column.

**Steps.**
1. `compute-sanitizer --tool memcheck` on every passing configuration (use `--reps 3`; sanitizers are slow);
   `racecheck` and `synccheck` where they support cluster-scope barriers on this version — record
   "unsupported" with the tool's message otherwise. Run under `tools/run --profile` only if the tool needs
   it; otherwise plain GPU mode.
2. `CUDA_LAUNCH_BLOCKING=1` once per file.
3. Stamp/placement checks over all C2/C3 data: `ts[1] > ts[0]`, monotone round stamps, N distinct SMs all
   inside one GPC of the **GPU-3** SM→GPC map (path in `../docs/ENVIRONMENT.md` §1).
4. Traffic model: `harness/traffic_model.py` evaluates the README / docstring formulas per (file, N, M, T,
   rank) → `results/model.csv` (`EXPERIMENTS.md` §5 schema, mandatory columns first).
5. ncu traffic check, one mid-size configuration per file (N = 8 where legal, M = 1024): DSMEM bytes
   (kernel-wide `.sum`, plus `.max/.avg` as the hot-spot factor), compared with the model; subtract a
   same-grid baseline (an all-exit launch) as `../docs/ENVIRONMENT.md` §4 says. A mismatch > 10 % means the
   README or the implementation is wrong: investigate and fix the wrong one (log it), do not annotate.
   Check `lts__t_sectors_srcunit_tex.sum ≈ 0` (DSMEM never reaches L2).
6. Flakiness: latency p90/p50 across 200 reps < 20 % under the locked clock; outliers investigated (clock
   samples, throttle reasons, placement).

**Deliverables.** `results/sanitizer.csv`, `results/model.csv`, `results/ncu_traffic.csv` (local);
`harness/traffic_model.py`; the TESTING §6 skip list (every skipped configuration and why) in "Key results".

**Done when.** TESTING §6 holds; every exception is explained. GPU time ≈ 1–2 h (sanitizers dominate).

---

## C5 — Performance sweep
**Goal.** `docs/EXPERIMENTS.md` §3–5: latency and bandwidth of every legal configuration, plus ncu counters.

**Prerequisites.** C4 `done`.

**Read first.** `docs/EXPERIMENTS.md` (all), `../docs/ENVIRONMENT.md` §4–5, C4's key results (skips, metric
names, baseline traffic).

**Steps.**
1. `driver.py --mode perf`: 20 warm-up + 200 timed launches per configuration, one process per configuration,
   final correctness check after the timed loop; payloads 256 B … 64 KB per CTA (cap by the 48 KB static
   smem limit and record the cap per file); randomised configuration order; the full sweep twice (drift).
   Clock note before/after each configuration; launches below the lock are discarded and counted.
2. Output `results/stamps.csv`, `results/host.csv` (EXPERIMENTS §5, mandatory columns first). **Size
   estimate first**: ≈ 200 reps × N ranks × (2 + rounds) stamps per configuration; if a file would approach
   1 GiB, split it per op and sweep (`stamps_<op>_s<k>.csv`) rather than reducing — every stamp is kept
   (the notebook then concatenates `stamps*.csv`; adapt its loader in C6).
3. ncu pass (EXPERIMENTS §4, 5 replays per configuration, `--clock-control none`, metric list from §1) →
   `results/ncu.csv`.
4. Sanity against `../docs/REUSE.md` §3 (agent-only): e.g. a 1-round push collective at small M should sit
   near one cluster barrier + one `st.async`/bulk hand-off; latencies below one cluster barrier mean a stamp
   problem.

**Deliverables.** `results/{stamps,host,ncu}.csv` (local), the sweep command lines in `CLAUDE.md` → *Tooling*.

**Done when.** Both sweeps complete for every legal configuration (partial runs registered as partial and
re-run); drift between sweeps reported. GPU time ≈ 2–4 h (≈ 500 configurations × 2 sweeps + ncu replays).

---

## C6 — Notebook and final report
**Goal.** `analysis/dsmem_collectives.ipynb` executed with figures 1–6 of `docs/EXPERIMENTS.md` §6 stored in
the file, and a written report.

**Prerequisites.** C5 `done`.

**Steps.**
1. Fill the notebook (it exists as a skeleton: keep its structure, adapt to the real CSV columns):
   definitions (EXPERIMENTS §2), setup (device, clock, toolchain, run_ids used), figures 1–6, one or two
   sentences per figure, a table of the best algorithm per (op, N, payload) for latency and for bus bandwidth.
2. Write `results/summary_latency.csv` and `results/summary_bandwidth.csv` (medians, p10/p90, per-rank
   maxima) — these and the executed notebook are the only committed result files (force-add).
3. Execute: `$TR --no-gpu -- jupyter nbconvert --execute --to notebook --inplace collectives/analysis/dsmem_collectives.ipynb`.
4. Report (in the commit message and "Key results"): pass/skip/fix summary per file, best algorithms, where
   the figures are, what could not be measured.

**Done when.** The notebook runs end-to-end from the local CSVs and renders on GitHub without them.

---

## C7 — Optional: extensions from h100-microbench experiment 17 (start only when the user asks)
**Goal.** Cover what `h100-microbench/docs/experiments/17__dsmem_collectives.md` planned beyond C1–C6.
Read that spec first; it is reference input, not binding.

**Candidate items (agree the subset with the user in plan mode).**
- **Broadcast**: rooted algorithms the catalogue lacks — TMA multicast from the root
  (`cp.async.bulk…multicast::cluster`, as DAK/DeepGEMM use), binomial tree, direct push, pull — as new files
  `broadcast__<alg>__<src>.cu` following the same file contract and README columns.
- **Many clusters at once**: the max-active-clusters launch (≈ 66/30/15/7 clusters for CS 2/4/8/16 on GPU 2;
  re-measure on GPU 3) to expose fabric contention. This needs per-cluster input/output offsets, i.e. a
  small extension of the file contract (offset by `%clusterid`): propose it to the user before editing files.
- **Hot spots**: ncu `.max/.avg` of the DSMEM byte metrics per algorithm, against the per-rank model
  (`results/model.csv`).
- **Thread count**: 1024 vs 256 threads per CTA for the bandwidth-bound algorithms.

**Done when.** Agreed in the C7 plan.
