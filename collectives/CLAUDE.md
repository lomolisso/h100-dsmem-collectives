# CLAUDE.md — H100 verification and benchmarking of `collectives/`

@ROADMAP.md

> **IMPORTANT — READ FIRST.** The server protocol is known and binding: `../docs/ENVIRONMENT.md` (summary of
> `h100-microbench/docs/environment/README.md`). Every build, run and notebook goes through
> `~/work/fa2026/iacoma/h100-microbench/tools/run`, on **GPU 3**, with the locked SM clock for every timed or
> ncu run. Do not invent another protocol; if something on the server contradicts that file, stop, tell the
> user, and record the finding there.

## Scope

Only `collectives/` is in scope: every `<op>__<alg>__<src>.cu` must (1) compile for `sm_90a`,
(2) produce correct results for every legal configuration, (3) never deadlock or race, and (4) be
benchmarked for latency and bandwidth per `docs/EXPERIMENTS.md`, with figures in
`analysis/dsmem_collectives.ipynb`. **The 29 algorithm files are modifiable:** they are untested
implementations, and C2–C5 exist to find and fix their bugs (see *Workflow* item 3). `kernels/` and
`third_party/` are reference material and are not compiled here. This study also owns what was planned as experiment 17 of h100-microbench
(`h100-microbench/docs/experiments/17__dsmem_collectives.md`, reference input for task C7).

## Starting a new conversation
Follow `../CLAUDE.md` ("Starting a new conversation"): the task is the first row of the status table in
`ROADMAP.md` that is not `done`; plan mode first; read the task card and the documents it lists.

## H100 NVL server protocol (binding; details in `../docs/ENVIRONMENT.md`)

- Access: edit, run agents and use git directly in this NFS checkout on `nishil-4xh100-01`; the GPUs are
  shared with other users, ours are GPU 3 (default) and GPU 2 (only when the user names it).
- Checking and reserving the GPU: `tools/run` runs the preflight (no foreign process, ≤ 1 GiB used, ≤ 5 %
  utilisation, no other container given the GPU) and holds the cooperative lock
  `/tmp/h100-gpu-locks/gpu3.lock` for the lifetime of its container. The lock is shared with h100-microbench:
  one GPU job at a time; if it is held, wait and retry. Status: `make -C ~/work/fa2026/iacoma/h100-microbench gpu-status`.
- Limits: `--profile` is required for ncu counters (`RmProfilingAdminOnly=1`) and for the clock lock
  (`--lock-sm-mhz <LOCK>`, value from h100-microbench `docs/context/10` for GPU 3; GPU measurements wait
  until its GPU-3 bring-up row is `done`). Containers need ≥ 2 GB free scratch. No run-time limit is
  documented; keep sweeps restartable (skip configurations that already have a `complete` run) so an
  interrupted run never has to start over.
- Where to build and write: in-tree on NFS (`collectives/build/`, `collectives/results/`, both git-ignored);
  never the container's `/tmp`.
- Clean-up: none beyond `tools/run`'s own (`--rm` containers, lock released on exit). Never `docker run` by
  hand, never prune, never touch other users' processes, containers or locks. Never delete results.

## Tooling (fill the run-time values when they are first used)

- Image: `h100-microbench:cuda12.9.1-26d4f436f8c5` (`tools/env info`, 2026-10-25): CUDA 12.9.1, nvcc 12.9.86
  (`Build cuda_12.9.r12.9/compiler.36037853_0`), Nsight Compute 2025.2.1.0 (build 35987062), Nsight Systems
  2025.1.3.140, compute-sanitizer, cuda-gdb, Python 3.12.3 (numpy 2.5.4, pandas 3.1.0, matplotlib, scipy, jupyter).
  `tools/env verify --gpu 3 --profile` (2026-10-25): toolchain ok; GPU probe ok (1 visible device, driver API 13000,
  runtime 12090, H100 NVL cc 9.0, 132 SMs, 60 MiB L2, 227 KiB opt-in smem/block, PCI da:00); one-metric ncu probe ok.
- Driver 580.126.20 (CUDA 13.0 driver API); GPU 3 = `GPU-1241dcbc-33ea-80bf-5f87-37f7868c3126`, serial
  1793225008254. SM clock lock **1635 MHz** (confirmed for GPU 3 by the microbench GPU-3 bring-up, commit `89ff996`).
  Repo commit at C1 start: `bb9fa9c`. Vendored harness: `common/h100mb/` from microbench `14875a5`
  (`common/h100mb/PROVENANCE.md`).
- Exact build and run commands (C1; run from the repo root, `TR=~/work/fa2026/iacoma/h100-microbench/tools/run`):
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
2. C2–C3: correctness matrix; C4: deadlock/race checks and sanity checks; C5: performance sweep from
   `docs/EXPERIMENTS.md`; C6: the notebook; C7 (optional): the extensions taken over from microbench 17.
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
- Clocks: every timed or ncu run uses `tools/run --profile --lock-sm-mhz <LOCK>`; ncu always with
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
