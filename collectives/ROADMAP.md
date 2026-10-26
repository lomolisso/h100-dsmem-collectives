# ROADMAP — collectives study (tasks C1–C7)

Imported by `collectives/CLAUDE.md`. One conversation = one task, run in order. How a conversation starts
and ends, the commit rule and the subagent rule: `../CLAUDE.md`. Environment: `../docs/ENVIRONMENT.md`.
Reusable code: `../docs/REUSE.md`. Design: `docs/TESTING.md` (correctness) and `docs/EXPERIMENTS.md`
(performance) are binding; the cards below say which sections each task executes.

## Task status
Status values: `todo` · `in progress` (current conversation) · `in review` (committed locally, waiting for the
user's review, not pushed) · `done` (approved and pushed). Update the row **before** the task's commit;
"Key results" records what later tasks consume (numbers, file paths, decisions).
Rows are listed in **run order**; the IDs are historical (C4 runs after C6, see *Dependencies*).

| ID | Task | Status | Key results for later tasks |
|---|---|---|---|
| C1 | Environment, vendored harness, build matrix, GPU smoke test | done | **Prereq:** microbench GPU-3 bring-up done in this conversation (commit `89ff996`, row `in review` there): lock **1635 MHz** on GPU 3; GPU-3 map = run `2026-10-25T19:56:47.043+00:00` of `03__sm_gpc_topology__map.csv`, identical to GPU 2 (SMs 124–131 only in CS ≤ 2; partition columns empty). **Harness:** `harness/main.cu` (modes correctness/perf/ncu, `--baseline`), `harness/driver.py` (build/run/summary), `harness/trap_cs16.cu`, `harness/ncu_discovery.sh`; vendored `common/h100mb/` (microbench `14875a5`, path fix only). **Launch path:** dynamic smem `<<<N, 256, sizeof(CollectiveSmem)>>>` after `cudaFuncAttributeMaxDynamicSharedMemorySize`; compile-time `__cluster_dims__(16)` compiles; plain `<<<N,256>>>` works once `NonPortableClusterSizeAllowed` is set (fails before), `cudaLaunchKernelEx` with an equal cluster attribute is also accepted; `cudaOccupancyMaxActiveClusters` works without the attribute (CS 4/8/16 at 256 thr, small smem: 248/124/58). Harness self-test (microbench 01 T1–T7) passes on the vendored copy. ~~First build (static shared memory, 48 KiB limit; `results/build_matrix.csv`): 900 configurations, 694 OK, 81 SKIP:static_assert, 125 SKIP:smem, 0 ERROR.~~ Superseded by the **C1 revision (2026-10-25, user request): all 29 files use dynamic shared memory up to 227 KiB per CTA** (`struct CollectiveSmem`, CLAUDE.md change log) and the docs declare the algorithm files modifiable. **Build matrix v2** (`results/build_matrix_v2.csv`; grid M ∈ {16,64,256,1024,4096,16384} = 1080 configurations plus `M_max` search points): **0 ERROR**; latest status 1181 OK, 118 SKIP:static_assert, 106 SKIP:smem. **Runnable set for C2/C3 = grid OK + the largest OK `M_max` per (file, N, T) = 1043 configurations**: allreduce 284 float + 284 int (C2); allgather 168, alltoall 119, reducescatter 94 float + 94 int (C3); `driver.py run` selects exactly these (138 lower `M_max` search builds are kept in the matrix but not run). `M_max` reaches 226–227 KiB for every (file, N, T) except algorithmic caps: shift_accumulate_atomic M ≤ 1024 (N = 2/4), shift_accumulate_bulk N = 4 only, fa4 N = 2 only; e.g. allreduce ring N = 16 → M = 54 656, hpcops → 58 112, alltoall direct push N = 16 → 1 812, allgather N = 16 → 3 416–3 628 (`driver.py summary`). Conversion check: for the 694 configurations that built statically, the dynamic size equals the static one (280) or +8 B struct padding (404); the 10 others are 2d_mesh N = 2, whose unused `col_slots` (0.5·M) the static build had dropped (TESTING §1). 97 configurations that were SKIP:smem now build. Skip formulas per file: TESTING §1. **Smoke** (N = 4, fa4 N = 2; M = 256 float; pattern A; 5 reps; lock 1635): **29/29 PASS**, 0 deadlocks, 0 clock-flagged launches, ranks always on distinct SMs (CS 4 → SMs 0/1/16/17 = G0, CS 2 → 124/125); median latency 224 ns (fa4 N = 2) … 2.75 µs (ring allreduce). **Smoke after the revision** (same small configuration + the largest-smem OK configuration of every file, up to 227 KiB/CTA at N = 16; 5 reps): **57/58 PASS**, 0 deadlocks, 0 clock-flagged; **FAIL: `allreduce__shift_accumulate_atomic__clusterfusion` N = 4, M = 1024, int** (5/5 reps): thread e owns element e (`if (threadIdx.x < M)`) and the harness runs 256 threads, so elements 256…1023 are never reduced — a genuine bug (the `M ≤ 1024` assert assumes 1024 threads), independent of the shared-memory change, left for C2. **ncu:** DSMEM payload = `l1tex__m_l1tex2xbar_write_bytes_mem_dshared.sum` = `l1tex__m_xbar2l1tex_read_bytes_mem_dshared.sum` = exactly the payload for bulk push (`_op_tma_st`), store push (`_op_st`) and pull (`_op_ld`); baseline 0; LSU `t_*` metrics miss bulk copies; `lts__t_sectors_srcunit_tex` counts the kernel's global I/O (not ≈ 0) — EXPERIMENTS §1. **Note:** 4 misaligned rows from a first driver test (extra empty mandatory field) are kept in `results/build_matrix__c1_driver_test_misaligned_2026-10-25.csv`, not used. GPU time ≈ 20 min. |
| C2 | Correctness: AllReduce (12 files) | done | **All 12 AllReduce files pass every runnable configuration** (TESTING §6 correctness; sanitizers in C4): **580 configurations** (290 float + 290 int = grid M ∈ {16…16384} + the largest M that fits, up to 227 KiB/CTA) × 3 fresh processes × 200 reps, pattern A bit-exact for float and int (580/580), pattern B float within tolerance (290/290; max error ≤ 1.64 % of `1e-5·(1+|ref|)·ceil(log2 N)`, worst atomic_all_push 6.6e-7 rel); 0 deadlocks, 0 clock-flagged launches (lock 1635 MHz), ranks always on distinct SMs. Per file (configurations per type): 2d_mesh, atomic_all_push, one_shot_pull, hpcops, recursive_doubling, reduce_bcast_binomial 28; rabenseifner, ring, two_shot_push 26 (M = 16 illegal at N ≥ 8); direct_replicated_push 23; shift_accumulate_atomic 14 (N = 2/4); shift_accumulate_bulk 7 (N = 4). `atomic_all_push` (order-nondeterministic float atomics) is exact under pattern A as required. **One fix:** `allreduce__shift_accumulate_atomic__clusterfusion` failed at M = 1024 (N = 2/4, 600/600 reps; elements ≥ blockDim = 256 never reduced): threads now stride over M with a `snap[M]` snapshot buffer; schedule, atomics and barriers unchanged; its `M ≤ 1024` cap is gone (M_max 29 056) — CLAUDE.md change log. Data (local): `results/correctness.csv` (119 MB, 532 036 rows incl. C1 smoke rows and the 12 superseded failing processes), driver runs `2026-10-25T20:40:43.672` (pattern A, 1704 processes: 1692 PASS + 12 FAIL before the fix), `21:19:57.737` (fixed file re-run, 84/84), `21:21:23.741` (pattern B, 870/870); triage `harness/correctness_report.py` → `results/correctness_summary__2026-10-25T163638.399-0500.csv`. `driver.py run --ops <op>` filters by collective (for C3). GPU time ≈ 55 min. |
| C3 | Correctness: AllGather (7), ReduceScatter (4), AlltoAll (6) | done | **All 17 files pass every runnable configuration, no fix needed** (TESTING §6 correctness; sanitizers in C4): **475 configurations** (381 float + 94 int; grid M ∈ {16…16384} + the largest M that fits 227 KiB/CTA) × 3 fresh processes × 200 reps; pattern A exact (475/475), pattern B float (381/381): AllGather/AlltoAll bit-exact (172 200 pattern-B rows, max_abs_err 0 in all), ReduceScatter within tolerance (max ≤ 1.29 % of `1e-5·(1+|ref|)·ceil(log2 N)`); 0 deadlocks, 0 clock-flagged launches (lock 1635 MHz), ranks always on distinct SMs. Per file (configurations per type): allgather ×7 24 each; alltoall bruck 24, direct_pull/direct_push/pairwise_xor/shift 22, pairwise_exchange (fa4, N = 2) 7; reducescatter direct_pull 24, direct_push 22, recursive_halving 24, ring 24 (float = int). Data (local): `results/correctness.csv` now 227 MB / 1 045 636 rows (C1 smoke + C2 + C3); driver runs `2026-10-25T21:39:19.912` (pattern A, 1425/1425 PASS), `22:04:19.984` (pattern B, 1143/1143 PASS); summary `results/correctness_summary__2026-10-25T172440.353-0500.csv`. GPU time ≈ 45 min. |
| C5 | Performance sweep (stamps, host times, ncu counters) and traffic model | done | **Sweep set** (`driver.py --set perf`): every file × legal N × payload per CTA {256 B, 1, 4, 16, 64 KiB} (allreduce/allgather M = P/4, reducescatter/alltoall M = P/(4N)) + the largest M that fits 227 KiB per (file, N), float = **615 configurations**; 33 payload points exceed 227 KiB (SKIP:smem); largest payload swept per (file, N) 12.6 KiB (replicated-push AR, N = 16) … 227 KiB (hpcops). 104 new reducescatter/alltoall payload configurations (M ∈ {4, 8, 32, 128, 512, 2048, 8192}) passed correctness first (pattern A, 312/312 processes × 200 reps; the C2/C3 set is now 1159). **Sweeps** s1/s2 (randomised order, seeds 1/2; 20 warm-up + 200 timed launches, 1 process per configuration, lock 1635 MHz): 615/615 complete each, every final check PASS, **246 000 timed launches kept, 0 discarded** (no clock flag); **drift s2/s1: median 0.00 %, p90 0.33 %, max 12.5 %** (sub-µs configurations, one 32 ns tick). Sanity: 8 configurations below 200 ns = N = 2 single-hand-off all-to-alls at small M (160–192 ns ≈ the 160 ns one-way hand-off), consistent, not a stamp problem. Headline (N = 8, 4 KiB, median L): allreduce 2d_mesh/two_shot 0.93 µs … ring 7.8 µs, atomic_all_push 13.2 µs; allgather replicated_push 2.2 µs … gather_bcast 6.5 µs; reducescatter direct_push 0.48 µs … ring 4.2 µs; alltoall direct_push 0.42 µs … bruck 4.2 µs — the files without a barrier inside the collective are fastest at small messages. Peak busbw: allreduce 25 GB/s (two_shot, N = 8, 64 KiB), allgather 15, reducescatter 22, alltoall 30 GB/s (direct_push, N = 8, 64 KiB). **ncu** (`harness/ncu_sweep.py`): 615 configurations + 8 same-grid baselines (per N, smallest- and largest-smem binary), 5 launches each, 0 failures → `results/ncu.csv` (means, 106 MB), `results/ncu_launches.csv` (per launch, 532 MB), raw `results/ncu_raw/`. **Model** (`harness/traffic_model.py`, written by a subagent from the code, verified; `--check`: invariants, closed forms, legality, C1's three ncu points exact) → `results/model.csv` (4398 rows, all 615 configurations; schema EXPERIMENTS §5: sent/recv/initiated/self/rounds/global I/O). It found two README errors, fixed (CLAUDE.md change log): gather_bcast_binomial total, hpcops rounds. Open for C4: whether ncu counts the `mapa` self-loads of one_shot_pull/reducescatter direct_pull/hpcops (model keeps them in `bytes_self`), and how remote atomics are counted. Data (local, 3.2 GB in `results/`): `stamps_<op>_s{1,2}.csv` (0.16–0.45 GB each), `host_s{1,2}.csv`, `perf_check_s{1,2}.csv`, `perf_summary__2026-10-25T203250.913-0500.csv` (`harness/perf_report.py`). Runs: build `2026-10-25T23:20:45.252`, correctness `23:25:22.881`, s1 `23:30:35.666`, s2 `23:44:35.715`, ncu `23:58:35.959` (UTC). GPU time ≈ 2.1 h (sweeps 28 min, ncu 1.6 h). |
| C6 | Notebook and report (preliminary result until C4 closes) | in review | **Notebook** `analysis/dsmem_collectives.ipynb` (executed, 1.6 MB, 7 figures + 4 tables embedded; all loading/plotting in `analysis/collectives_figures.py`, notebook cells are one-line calls; text = setup, definitions, algorithm table, findings, 1–3 sentences per figure). **Project status: preliminary result until C4 closes** (not stated in the notebook). **Payload definition changed to the nccl-tests size** (user decision): allgather P = N·M·B (was M·B), so allgather/reducescatter are comparable; allgather sweep topped up: 77 new configurations, correctness 231/231 processes, both sweeps (91 new configurations each), ncu 91/91 → perf set **643 configurations, 257 200 timed launches, 0 discarded** (C5's allgather headline numbers used the old definition; e.g. allgather N = 8, 4 KiB is now direct_replicated_push 0.70 µs). **Findings:** up to 4 KiB the fastest algorithm of every op is a single exchange without a per-round barrier (allreduce two_shot_push/2d_mesh 0.35–0.99 µs, reducescatter direct_push 0.22–0.80, alltoall direct_push/pairwise_exchange 0.16–0.70, allgather direct_replicated_push 0.42–0.86 µs, N = 2–16); a barrier per round costs ≈ 0.52–0.62 µs per round at 4 KiB (ring allreduce 1.25 → 17.5 µs from N = 2 to 16); peak busbw allreduce 25 GB/s (two_shot, N = 8, 64 KiB), allgather 29 (ring, N = 2, 227 KiB), reducescatter 22 (direct_push, N = 8, 64 KiB), alltoall 30 GB/s (direct_push, N = 4, 64 KiB); best per-SM injection at N = 16 16–20 GB/s; latency p90/p50 ≤ 1.094 for 95 % of configurations. **Traffic vs model (ncu): 507/643 within 10 %, none below**; above: transfers < 32 B counted as a 32-B sector (2×), remote atomics 3× (atomic_all_push: 2× op_atom + 1× op_ld), hpcops lane-interleaved scalar accesses 1–4× (grows with N), largest-payload points of bulk algorithms (sizes/offsets not multiples of 128 B) +6–38 % — for C4 to explain. **Committed results:** `results/summary_latency.csv`, `results/summary_bandwidth.csv` (643 rows each, mandatory columns first; force-added). Cache `results/derived/*.pkl` (local). No GPU time for the notebook; top-up ≈ 25 min GPU. |
| C4 | Sanitizers, deadlock/race checks, stamp/placement/traffic/flakiness checks on the sweep data; fix + redo loop | todo | |
| C7 | *Optional* — extensions taken over from h100-microbench experiment 17 | todo | |

## Dependencies
```
h100-microbench "GPU-3 bring-up" row (GPU lock value + GPU-3 SM→GPC map) ─┐  (needed for any GPU run)
C1 ─► C2 ─┬─► C5 ─► C6 ─► C4 ─► (C7, only if the user asks)
    └► C3 ─┘              ▲      │
                          └──────┘  a C4 finding: fix the file, re-run its C2/C3 correctness and its C5
                                    configurations, re-execute the C6 notebook
```
C2 and C3 are independent of each other (either order). The sweep (C5) and the notebook (C6) start as soon as
correctness (C2, C3) holds; the C6 notebook is the **preliminary** result, and C4 (sanitizers and the
TESTING §4–6 sanity checks, run largely on C5's data) is the gate that turns it into the final one. This is a
project-internal status only: the notebook itself never says "preliminary" or mentions C4. C1's build-only steps can run before the microbench
bring-up is done; its GPU smoke step cannot. This roadmap does not depend on `roofline/`.

## Deferred (out of the current scope; consider only after C7, when the user asks)
- **DSMEM-native variants** of the algorithms that synchronise with a `cluster_sync()` every round (ring AR/AG/RS,
  recursive doubling AR/AG, Rabenseifner, Bruck AG/A2A, pairwise XOR, shift, recursive halving, the binomial
  trees, ClusterFusion shift-accumulate bulk): same rounds, partners and messages, but per-round buffers and
  pairwise "credit" mbarriers instead of the per-round cluster barrier, as new files (source tag `dsmem`) next to
  the published/ported baselines, run through the same correctness → sweep → notebook pipeline. Motivation (C5):
  at small messages the files without a barrier inside the collective are the fastest (e.g. N = 8, 4 KiB:
  two-shot allreduce 0.93 µs vs ring 7.8 µs), so the current comparison partly measures synchronisation style.

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
- The algorithm files are modifiable: they are untested, and every task fixes what its checks show is wrong
  (`CLAUDE.md` → Workflow item 3; every change logged in `CLAUDE.md` → Change log).
- Shared memory is dynamic, up to 227 KiB per CTA (`CLAUDE.md` → File contract); the build matrix covers
  M ∈ {16, 64, 256, 1024, 4096, 16384} plus the largest M that fits per (file, N, T) (`M_max`).

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
5. `harness/driver.py`: enumerates (file, N ∈ {2,4,8,16}, M ∈ {16,64,256,1024,4096} — since the C1 revision
   {…, 16384} plus `M_max` — T ∈ {float, int for reduce files}); builds each into `build/<file>__N<N>_M<M>_<T>` with the TESTING §1 command plus
   `-Xptxas -v`; classifies `OK` / `SKIP:static_assert` / `SKIP:smem` / `ERROR` from the compiler output;
   runs configurations as child processes with a timeout; resumable (skips configurations already complete
   in the registry). Parallel builds capped at 16 jobs (the host is shared).
6. Build the full matrix (≈ 29 × 20 float + 16 × 20 int ≈ 900 builds; expect 1–2 h of CPU at 16 jobs) and
   write `results/build_matrix.csv` (since the C1 revision: `results/build_matrix_v2.csv`, + `grid`, `smem_bytes`) (`file, op, alg, N, M, T, status, static_smem_bytes, regs, spill_bytes,
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
exact. Note the smem-heavy layouts (`N·M`, `2·N·M`): their large-N/large-M configurations exceed 227 KiB and are
`SKIP:smem` in C1's matrix; their `M_max` points are the largest messages that fit.

**Prerequisites.** C1 `done`; GPU-3 bring-up `done`. Independent of C2.

**Done when.** All legal configurations of the 17 files pass TESTING §6's correctness criterion. GPU time
≈ 30–60 min.

---

## C5 — Performance sweep and traffic model
**Goal.** `docs/EXPERIMENTS.md` §3–5: latency and bandwidth of every legal configuration, ncu counters, and the
per-rank traffic model that C6 (figures 5–6) and C4 (traffic check) use.

**Prerequisites.** C2 and C3 `done`.

**Read first.** `docs/EXPERIMENTS.md` (all), `../docs/ENVIRONMENT.md` §4–5, C1's key results (metric names,
payload accounting, launch path), C2/C3 key results (runnable set, fixes).

**Steps.**
1. `driver.py --mode perf`: 20 warm-up + 200 timed launches per configuration, one process per configuration,
   final correctness check after the timed loop; payloads 256 B … 64 KB per CTA plus the largest
   payload that fits the 227 KiB shared-memory limit (the `M_max` points; record the cap per file and N); randomised configuration order; the full sweep twice (drift).
   Clock note before/after each configuration; launches below the lock are discarded and counted.
2. Output `results/stamps.csv`, `results/host.csv` (EXPERIMENTS §5, mandatory columns first). **Size
   estimate first**: ≈ 200 reps × N ranks × (2 + rounds) stamps per configuration; if a file would approach
   1 GiB, split it per op and sweep (`stamps_<op>_s<k>.csv`) rather than reducing — every stamp is kept
   (the notebook then concatenates `stamps*.csv`; adapt its loader in C6).
3. ncu pass (EXPERIMENTS §4, 5 replays per configuration, `--clock-control none`, metric list from §1) →
   `results/ncu.csv`, including one same-grid all-exit baseline launch (`main.cu --mode ncu --baseline`) per
   (N, dynamic smem size) so C4 can subtract the profiler's own traffic.
4. Traffic model: `harness/traffic_model.py` evaluates the README / docstring formulas per (file, N, M, T,
   rank) → `results/model.csv` (`EXPERIMENTS.md` §5 schema, mandatory columns first): `bytes_sent`,
   `bytes_recv` (pulls: bytes read), `rounds`; rooted trees are uneven per rank. Also the global-I/O model
   (`in` read, `out`/`ts`/`smid` written) that C4 compares `lts__t_sectors_srcunit_tex` against.
5. Sanity against `../docs/REUSE.md` §3 (agent-only): e.g. a 1-round push collective at small M should sit
   near one cluster barrier + one `st.async`/bulk hand-off; latencies below one cluster barrier mean a stamp
   problem.

**Deliverables.** `results/{stamps,host,ncu,model}.csv` (local), `harness/traffic_model.py`, the sweep command
lines in `CLAUDE.md` → *Tooling*.

**Done when.** Both sweeps complete for every legal configuration (partial runs registered as partial and
re-run); drift between sweeps reported; `model.csv` covers every swept (file, N, M, T, rank). GPU time
≈ 2–4 h.

---

## C6 — Notebook and report (preliminary result until C4 closes)
**Goal.** `analysis/dsmem_collectives.ipynb` executed with figures 1–6 of `docs/EXPERIMENTS.md` §6 stored in
the file, and a written report. Within the project this is the **preliminary** result: C4 runs the
remaining checks afterwards and re-executes the notebook if a fix changes any data. The notebook is written as
a final report and never mentions that status, C4, or the checks still to come.

**Prerequisites.** C5 `done`.

**Steps.**
1. Fill the notebook (it exists as a skeleton: keep its structure, adapt to the real CSV columns):
   definitions (EXPERIMENTS §2), setup (device, clock, toolchain, run_ids used), figures 1–6 (5 and 6 use
   C5's `model.csv`), one or two sentences per figure, a table of the best algorithm per (op, N, payload) for
   latency and for bus bandwidth.
2. Write `results/summary_latency.csv` and `results/summary_bandwidth.csv` (medians, p10/p90, per-rank
   maxima) — these and the executed notebook are the only committed result files (force-add).
3. Execute: `$TR --no-gpu -- jupyter nbconvert --execute --to notebook --inplace collectives/analysis/dsmem_collectives.ipynb`.
4. Report (in the commit message and "Key results"): pass/skip/fix summary per file, best algorithms, where
   the figures are, what could not be measured.

**Done when.** The notebook runs end-to-end from the local CSVs and renders on GitHub without them.

---

## C4 — Sanitizers, deadlock/race checks, stamp/placement/traffic/flakiness checks; fix + redo loop
**Goal.** Close `docs/TESTING.md` §6 on the final code and data: no sanitizer error, no watchdog expiry,
placement and stamps sane, measured DSMEM traffic within 10 % of the model, latency spread p90/p50 < 20 %.
C4 runs **after** C5 and C6 and checks their data; it turns the preliminary C6 result into the final one.

**Prerequisites.** C6 `done`.

**Read first.** `docs/TESTING.md` §4–6; `docs/EXPERIMENTS.md` §1 and §4 (metric names and payload accounting
from C1); `../docs/ENVIRONMENT.md` §4 (ncu rules); `README.md` "Total DSMEM bytes per CTA" column; C5 and C6
key results.

**Steps.**
1. `compute-sanitizer --tool memcheck` on every passing configuration (`--reps 3`; sanitizers are slow);
   `racecheck` and `synccheck` on a subset (every file × every legal N × {M = 256, the file's `M_max`}, float)
   where they support cluster-scope barriers on this version — probe one configuration first and record
   "unsupported" with the tool's message otherwise. Plain GPU mode unless a tool needs `--profile`.
   Sanitized runs write their correctness rows to `results/correctness__<tool>.csv`, not `correctness.csv`.
2. `CUDA_LAUNCH_BLOCKING=1` once per file.
3. Stamp/placement checks on C5's `stamps.csv` (and the `smids` of C2/C3): `exit > entry` per rank, round
   stamps non-decreasing (equal within the 32 ns tick is fine) and as many as the file's rounds, N distinct
   SMs all inside one GPC of the **GPU-3** SM→GPC map (path in `../docs/ENVIRONMENT.md` §1).
4. Traffic check from C5's `ncu.csv` against C5's `model.csv`: DSMEM bytes (kernel-wide `.sum`, plus
   `.max/.avg` as the hot-spot factor) minus the same-grid baseline (`../docs/ENVIRONMENT.md` §4). A mismatch
   > 10 % means the README or the implementation is wrong: investigate and fix the wrong one (log it), do not
   annotate. `lts__t_sectors_srcunit_tex.sum` must match the global-I/O model (DSMEM never reaches L2).
5. Flakiness from C5's stamps: latency p90/p50 across 200 reps < 20 % under the locked clock; outliers
   investigated (clock samples, throttle reasons, placement).
6. **Fix + redo loop.** Any finding that shows a file is wrong: fix it (CLAUDE.md → Workflow item 3, change
   log), re-run the file's C2/C3 correctness, re-run its C5 configurations (both sweeps, ncu), re-execute the
   C6 notebook, and re-check. Record which data were replaced (run_ids).

**Deliverables.** `results/sanitizer.csv`, the checks' output (local); the TESTING §6 skip list (every skipped
configuration and why) in "Key results"; fixes with change-log entries; the re-executed notebook if any data
changed.

**Done when.** TESTING §6 holds on the final data and every exception is explained; the notebook reflects the
final data. GPU time ≈ 2.5–3 h (sanitizers dominate) plus any redo.

---

## C7 — Optional: extensions from h100-microbench experiment 17 (start only when the user asks)
**Goal.** Cover what `h100-microbench/docs/experiments/17__dsmem_collectives.md` planned beyond C1–C6 and C4.
Read that spec first; it is reference input, not binding.

**Prerequisites.** C4 `done`.

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
