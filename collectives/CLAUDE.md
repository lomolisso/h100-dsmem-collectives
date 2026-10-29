# CLAUDE.md — H100 verification and benchmarking of `collectives/`

@ROADMAP.md

> **IMPORTANT — READ FIRST.** The server protocol is known and binding: `~/h100/H100.md` and
> `../docs/ENVIRONMENT.md`. Every build, run and notebook goes through `h100 run` (GPU runs queued, pinned to
> **GPU 3** with `--gpu 3` so the series stays on one card), with the locked SM clock for every timed or
> ncu run. Do not invent another protocol; if something on the server contradicts that file, stop, tell the
> user, and record the finding there.

## Scope

Only `collectives/` is in scope: every `<op>__<alg>__<src>.cu` must (1) compile for `sm_90a`,
(2) produce correct results for every legal configuration, (3) never deadlock or race, and (4) be
benchmarked for latency and bandwidth per `docs/EXPERIMENTS.md`, with figures in
`analysis/dsmem_collectives.ipynb`. **The 29 algorithm files are modifiable:** they are untested
implementations, and C2–C4 exist to find and fix their bugs (see *Workflow* item 3). `kernels/` and
`third_party/` are reference material and are not compiled here. This study also owns what was planned as experiment 17 of h100-microbench
(`h100-microbench/docs/experiments/17__dsmem_collectives.md`, reference input for task C7).

## Starting a new conversation
Follow `../CLAUDE.md` ("Starting a new conversation"): the task is the first row of the status table in
`ROADMAP.md` that is not `done`; plan mode first; read the task card and the documents it lists.

## H100 NVL server protocol (binding; details in `../docs/ENVIRONMENT.md`)

- Access: edit, run agents and use git directly in this NFS checkout on `nishil-4xh100-01`; the GPUs are
  shared with other users, ours are GPU 2 and GPU 3. The target GPU is set by the user (`h100-target`); this
  study's GPU runs are pinned to GPU 3 (`h100 run --gpu 3 …`), where C1 onwards ran.
- Checking and reserving the GPU: GPU jobs enter the user's FCFS queue. Its dispatcher runs the preflight (no
  foreign process, ≤ 1 GiB used, ≤ 5 % utilisation, no other container given the GPU) and holds the
  cooperative lock `/tmp/h100-gpu-locks/gpu<N>.lock` for the lifetime of the job's container; one GPU job at a
  time across this repository and h100-microbench. If anyone else uses the GPU, or a running job is ended from
  outside, the queue **freezes** (the interrupted job stays at the head and reruns after the user unfreezes):
  report `h100 queue`'s reason and wait. Status: `h100 status`.
- Limits: `--profile` is required for ncu counters (`RmProfilingAdminOnly=1`) and for the clock lock
  (`--lock-sm-mhz <LOCK>`, value from h100-microbench `docs/context/10` for GPU 3; GPU measurements wait
  until its GPU-3 bring-up row is `done`). Containers need ≥ 2 GB free scratch. No run-time limit is
  documented; keep sweeps restartable (skip configurations that already have a `complete` run) so an
  interrupted run never has to start over.
- Where to build and write: in-tree on NFS (`collectives/build/`, `collectives/results/`, both git-ignored);
  never the container's `/tmp`.
- Clean-up: none beyond the queue's own (containers removed after each job, lock released). Never `docker run` by
  hand, never prune, never touch other users' processes, containers or locks. Never delete results.

## Tooling (fill the run-time values when they are first used)

- Image: since 2026-10-29 the shared `h100:cuda12.9.1-<hash>` (`h100 image info`; same CUDA/ncu/nsys versions,
  plus PyTorch, clang-20, NVBit, CUTLASS). C1 ran in `h100-microbench:cuda12.9.1-26d4f436f8c5` (2026-10-25): CUDA 12.9.1, nvcc 12.9.86
  (`Build cuda_12.9.r12.9/compiler.36037853_0`), Nsight Compute 2025.2.1.0 (build 35987062), Nsight Systems
  2025.1.3.140, compute-sanitizer, cuda-gdb, Python 3.12.3 (numpy 2.5.4, pandas 3.1.0, matplotlib, scipy, jupyter).
  `h100 image verify --gpu 3 --profile` (2026-10-25): toolchain ok; GPU probe ok (1 visible device, driver API 13000,
  runtime 12090, H100 NVL cc 9.0, 132 SMs, 60 MiB L2, 227 KiB opt-in smem/block, PCI da:00); one-metric ncu probe ok.
- Driver 580.126.20 (CUDA 13.0 driver API); GPU 3 = `GPU-1241dcbc-33ea-80bf-5f87-37f7868c3126`, serial
  1793225008254. SM clock lock **1635 MHz** (confirmed for GPU 3 by the microbench GPU-3 bring-up, commit `89ff996`).
  Repo commit at C1 start: `bb9fa9c`. Vendored harness: `common/h100mb/` from microbench `14875a5`
  (`common/h100mb/PROVENANCE.md`).
- Exact build and run commands (C1; run from the repo root, `TR="h100 run"`; GPU commands pinned with `--gpu 3`):
  - build matrix: `$TR --no-gpu -- python3 collectives/harness/driver.py build --jobs 16` (per configuration the
    TESTING §1 command plus `-DCOLL_NAME='"<file>"' -DCOLL_OP=<0..3>`; writes `results/build_matrix_v2.csv`, grid +
    `M_max` points; see the driver docstring);
    status matrix: `python3 collectives/harness/driver.py summary`.
  - CS-16 launch check and vendored-harness self-test (no clock lock):
    `$TR -- bash -c './build/trap_cs16; ./build/harness_selftest'` from `collectives/` (binaries built with
    `nvcc -arch=sm_90a -std=c++17 -O3 -lineinfo -Xptxas -v -I ../common/h100mb -o build/trap_cs16 harness/trap_cs16.cu -L/usr/local/cuda/lib64/stubs -lcuda -lnvidia-ml`,
    and, from the repo root, `nvcc -arch=sm_90a -O3 -lineinfo -std=c++17 -I common/h100mb -o collectives/build/harness_selftest <MB>/experiments/01__timers_and_harness/src/harness_selftest.cu -L/usr/local/cuda/lib64/stubs -lcuda -lnvidia-ml`).
  - GPU smoke: `$TR --profile --lock-sm-mhz 1635 -- python3 collectives/harness/driver.py run --smoke --mode correctness --pattern A --reps 5 --processes 1 --child-timeout 120`.
  - ncu discovery: `$TR --profile --lock-sm-mhz 1635 -- bash collectives/harness/ncu_discovery.sh`.
  - C5 perf set (payload points 256 B…64 KiB per CTA + largest M per (file, N), float):
    `$TR --no-gpu -- python3 collectives/harness/driver.py build --set perf --jobs 16`; correctness of the new
    configurations `$TR --profile --lock-sm-mhz 1635 -- python3 collectives/harness/driver.py run --set default --ops reducescatter alltoall --T float --mode correctness --pattern A --reps 200 --processes 3`.
  - C5 sweeps: `$TR --profile --lock-sm-mhz 1635 -- python3 collectives/harness/driver.py run --set perf --mode perf --reps 200 --processes 1 --shuffle --seed <k> --extra "--tag s<k>"` (k = 1, 2).
  - C5 ncu pass: `$TR --profile --lock-sm-mhz 1635 -- python3 collectives/harness/ncu_sweep.py` (per configuration:
    `ncu --clock-control none --cache-control all --kernel-name collective_kernel --launch-skip 20 --launch-count 5 --section MemoryWorkloadAnalysis_Tables --metrics <list in ncu_sweep.py> --csv --page raw --print-units base build/<cfg> --mode ncu --reps 25`, plus `--baseline` / `baseline_kernel` runs).
  - C5 model and summary (no GPU): `python3 collectives/harness/traffic_model.py --set perf`;
    `python3 collectives/harness/perf_report.py`.

## File contract (what the harness may rely on)

- Every file includes `dsmem_primitives.cuh` and defines
  `__global__ void __cluster_dims__(CLUSTER_N,1,1) collective_kernel(const ELEM_T* in, ELEM_T* out, unsigned long long* ts, unsigned* smid)`.
- Compile-time knobs: `-DCLUSTER_N=<2|4|8|16> -DMSG_ELEMS=<M> -DELEM_T=<float|int>`.
  Launch one cluster: `gridDim = CLUSTER_N`, `blockDim = 256` (any multiple of 32 up to 1024).
  `CLUSTER_N=16` needs `cudaFuncSetAttribute(..., cudaFuncAttributeNonPortableClusterSizeAllowed, 1)`.
- Per-rank layouts (see the bottom of `dsmem_primitives.cuh`):
  allreduce `in/out[b*M .. b*M+M)`; allgather `in[b*M..)`, `out[b*N*M + r*M]`;
  reducescatter `in[b*N*M + j*M]`, `out[b*M..)`; alltoall `in[b*N*M + j*M]` → rank j, `out[b*N*M + r*M]`.
- Timestamps: `ts[b*32 + 0]` entry (after the entry `cluster_sync`), `ts[b*32 + 1]` exit, `ts[b*32 + 2 + r]`
  end of round r (where the algorithm has rounds); `smid[b]` = SM id of rank b. All in `%globaltimer` ns
  (32 ns tick on this device: small-message latencies are quantised; report medians over reps).
- `static_assert`s encode legal configurations (power-of-two N, N == 4, N == 2, N | M, 16-byte multiples).
  A `static_assert` failure is an *expected* compile failure for that configuration, not a bug.
- Shared memory is **dynamic**, up to the full **227 KiB (232 448 B) per CTA** a block may opt in to on H100:
  each file declares all its shared buffers in a namespace-scope `struct CollectiveSmem` and takes it with
  `CollectiveSmem& sm = dyn_smem<CollectiveSmem>();` (`dsmem_primitives.cuh`); no static `__shared__` arrays
  (static shared memory is capped at 48 KiB). The launcher passes `sizeof(CollectiveSmem)` as the dynamic size
  and sets `cudaFuncAttributeMaxDynamicSharedMemorySize`; the harness `static_assert`s the 227 KiB limit
  (message `SMEM_LIMIT`, classified `SKIP:smem`). New or rewritten files follow the same pattern.

## Workflow (the tasks of `ROADMAP.md`)

1. C1: record the tooling above, vendor the microbench harness (`../docs/REUSE.md`), build the harness
   described in `docs/TESTING.md` under `collectives/harness/`, and the compile matrix.
2. Run order after C1: C2–C3 correctness matrix; C5 performance sweep from `docs/EXPERIMENTS.md` plus the
   traffic model; C6 the notebook (the **preliminary** result — a project-internal status only, never written
   in the notebook); C4 sanitizers and the TESTING §4–6 sanity checks on the sweep's data, with the fix + redo
   loop (fix the file → re-run its correctness and sweep configurations → re-execute the notebook) that makes
   the result final; C7 (optional) the extensions taken over from microbench 17. IDs are historical; the
   `ROADMAP.md` status table lists the rows in run order.
3. The algorithm files are modifiable and are expected to need fixes: change a file whenever correctness,
   sanitizer, deadlock, traffic or performance-sanity checks show it is wrong (or when the user asks).
   Log every change in the *Change log* below (file, failing configuration or reason, root cause, what
   changed). Prefer the fix that keeps the algorithm the file is named after (its rounds, partners,
   message sizes); if a correct implementation needs a different pattern, extra barriers or different
   traffic, make the change and update the file's docstring and its `README.md` row in the same commit.
   Never make a failure disappear by skipping the configuration, loosening a tolerance beyond what
   `docs/TESTING.md` allows, or dropping data.
4. Report at the end of every task: which files pass which configurations, which were fixed and how, where
   the results and figures live, and anything that could not be measured and why.

## Rules of thumb

- A hung launch cannot be killed from inside the process: run every configuration in a child process
  with a timeout (see `docs/TESTING.md`) so one deadlock does not take down the sweep. A hung kernel keeps
  the GPU busy until its process dies: the driver must kill the child, and the next configuration starts
  only after the GPU is idle again.
- Use `%globaltimer`-based latency from the kernel's own stamps for the figures; host-side
  `cudaEvent` time includes launch overhead (≈ 4 µs event floor, 7–8 µs launch on this host) and is
  reported separately.
- Clocks: every timed or ncu run uses `h100 run --profile --lock-sm-mhz <LOCK>`; ncu always with
  `--clock-control none` (never `base`: it would override our lock). Record clocks and throttle reasons per
  configuration (`hb::Run::clock_note`) and discard launches that ran below the lock (count them).
- CSVs: mandatory columns first, append-only, flushed per launch, runs registered in
  `results/collectives__runs.csv` (`../docs/ENVIRONMENT.md` §5).
- Keep raw CSVs under `collectives/results/` (git-ignored). Commit only the executed notebook and the
  summary files C6 names (e.g. `results/summary_*.csv`, force-added), never raw stamps.
- Commit message per task: `collectives/C<n>: <what was done>; <main finding>`.

## Change log  *(append entries here)*

- 2026-10-25 (C1 revision, requested by the user): **all 29 files moved from static `__shared__` arrays to
  dynamic shared memory** (`struct CollectiveSmem` + `dyn_smem<>()`, `dsmem_primitives.cuh`) so every
  configuration can use up to 227 KiB per CTA instead of the 48 KiB static limit. Only the declarations
  changed (same buffers, names, types, sizes and alignment; struct padding adds ≤ 8 B); no algorithm logic
  changed. Harness: dynamic size + opt-in attribute + `SMEM_LIMIT` static_assert; build matrix M axis extended
  with 16384 and a per-(file, N, T) `M_max` point (largest M that fits). The docs now state that the
  algorithm files are modifiable.
- 2026-10-25 (C2) `allreduce__shift_accumulate_atomic__clusterfusion.cu`: **wrong result for M > blockDim**.
  Failing: N = 2 and 4, M = 1024, float and int, pattern A, 600/600 reps (n_bad 1446 at N = 2, 2982 at N = 4,
  = the elements ≥ 256 of every rank). Root cause: thread e owned element e (`if (threadIdx.x < M)`, guarded by
  `static_assert(M <= 1024)`, i.e. it assumed blockDim ≥ M), while the harness runs 256 threads, so elements
  256…M-1 were never reduced. Fix: threads stride over the M elements; the per-round snapshot of the running
  value moves from one register per thread to a `snap[M]` buffer in `CollectiveSmem` (all elements snapshotted
  before the first cluster barrier, as before). Unchanged: shift schedule, rounds (N-2, 1 for N = 2), one remote
  `atomicAdd` per element per round, 2 cluster barriers per round. The `M <= 1024` assert is gone (shared memory
  2M; M_max now 29 056 at N = 2/4). README row: also corrected the total for N = 2 (M·B, not 0).
- 2026-10-25 (C5, documentation only; found while writing `harness/traffic_model.py`, verified against the code):
  `allgather__gather_bcast_binomial__mpich` docstring and README total were wrong — the gather sends each
  non-root rank's whole subtree (lowbit(b) slots), so it moves (N/2)·log2 N blocks, not N−1 (e.g. N = 4: 4
  blocks): total (N/2)·log2 N·M·B + (N−1)·N·M·B. README `allreduce__pull_reduce_scatter_push_allgather__hpcops`
  rounds: 1 (pull, reduce and push fused in one loop, one `st.round`), not "2 phases, 1 round each";
  `allreduce__rabenseifner__mpich` N constraint also needs N | M and 16-byte chunks. README note added on
  self-addressed cluster accesses (`bytes_self`). No code changed.
- 2026-10-25 (C6, user decision): **payload definition → nccl-tests size for every op**. The allgather payload
  per CTA was M·B (input), now N·M·B (gathered output), like its dual reducescatter; allreduce M·B and
  reducescatter/alltoall N·M·B unchanged (EXPERIMENTS §2/§3). Harness: `main.cu` `kPayloadBytes`,
  `driver.py payload_m/payload_bytes`, `perf_report.py` and the notebook module recompute P from op, N, M (allgather
  stamp rows written before carry M·B in `payload_bytes`). The allgather sweep was topped up at the new payload
  points (77 new configurations, correctness 231/231, both sweeps, ncu).
