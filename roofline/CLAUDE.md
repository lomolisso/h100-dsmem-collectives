# CLAUDE.md — H100 NVL roofline study of DSMEM vs non-DSMEM workloads

@ROADMAP.md

> **IMPORTANT — READ FIRST.** The server protocol is known and binding: `../docs/ENVIRONMENT.md` (summary of
> `h100-microbench/docs/environment/README.md`). Every build, run and notebook goes through `tools/run`
> (this study: the `roofline/tools/run` wrapper created in R0, which selects the roofline image and calls the
> microbench `tools/run`), on **GPU 3**, with the locked SM clock for every timed or ncu run. Do not invent
> another protocol; if something on the server contradicts that file, stop, tell the user, and record the
> finding there.

## Goal

One figure in the style of the TPUv1 paper's roofline (Jouppi et al. 2017, Fig. 3): log-log achieved
throughput vs operational intensity for an H100 NVL, with **one marker per workload variant**: the
DSMEM version of each workload and its non-DSMEM counterpart, joined by an arrow. Workloads are the
kernels catalogued in `../kernels/`; each is described by one manifest in `workloads/`. The study must be
extensible: adding a future DSMEM port means adding one manifest (see `docs/ADDING_A_WORKLOAD.md`).

Read, in this order: `docs/DESIGN.md` (scope decision, methodology, protocol, CSV schema),
`docs/WORKLOADS.md` (every pair: entry points, baselines, traps), `docs/ADDING_A_WORKLOAD.md`
(manifest reference), then the manifests. This study does not depend on `../collectives/`.

## Starting a new conversation
Follow `../CLAUDE.md` ("Starting a new conversation"): the task is the first row of the status table in
`ROADMAP.md` that is not `done`; plan mode first; read the task card and the documents it lists.

## H100 NVL server protocol (binding; details in `../docs/ENVIRONMENT.md`)

- Access: edit, run agents and use git directly in this NFS checkout on `nishil-4xh100-01`; the GPUs are
  shared with other users, ours are GPU 3 (default) and GPU 2 (only when the user names it).
- Checking and reserving the GPU: `tools/run` runs the preflight (no foreign process, ≤ 1 GiB used, ≤ 5 %
  utilisation, no other container given the GPU) and holds the cooperative lock
  `/tmp/h100-gpu-locks/gpu3.lock` for its container's lifetime. Shared with h100-microbench and
  `../collectives/`: one GPU job at a time; if the lock is held, wait and retry. Status:
  `make -C ~/work/fa2026/iacoma/h100-microbench gpu-status`.
- Limits: `--profile` is required for ncu counters (`RmProfilingAdminOnly=1`) and for the clock lock
  (`--lock-sm-mhz <LOCK>`, value from h100-microbench `docs/context/10` for GPU 3; GPU measurements wait until
  its GPU-3 bring-up row is `done`). Containers need ≥ 2 GB free scratch, image builds ≥ 40 GB. The 94 GB of
  HBM fit every workload; no model checkpoints are needed (random weights at the real shapes).
- Where to build and write: in-tree on NFS — `roofline/build/` (variant source worktrees, extensions,
  JIT caches) and `roofline/results/` (both git-ignored); never the container's `/tmp`. Python packages are
  baked into the image; artifact extensions are built in-tree and put on `PYTHONPATH` (containers are `--rm`,
  a `pip install` at run time is lost).
- Clean-up: none beyond `tools/run`'s own. Never `docker run` by hand, never prune, never touch other
  users' processes, containers or locks. Never delete results.

## Workflow (the tasks of `ROADMAP.md`)

1. R0: environment (submodules, roofline image, `tools/run` wrapper), fill `docs/DESIGN.md` §6. Record the
   commit hash of this repo.
2. R1: measure the ceilings (`docs/DESIGN.md` §2.1) and write `results/ceilings.csv`. Do this before any
   workload: a roofline with datasheet ceilings only is not acceptable.
3. R2–R7: the manifest runner, then the manifests in the priority order of `docs/WORKLOADS.md` §0. For
   each manifest: build every variant, validate outputs against the reference named in the manifest, collect
   the `points.csv` rows (operator scope, then layer scope where defined), run the ncu metric pass.
4. After each workload task and in R8: execute `analysis/roofline.ipynb`; check that every manifest appears
   (measured, or "reported" for `runnable: false`) and that the ncu roofline sections agree with the
   hand-computed points within 10 %.
5. Report (R8): ceilings table, per-workload table (time, FLOPs, bytes per level, intensity, throughput,
   speedup DSMEM/non-DSMEM, fraction of layer time), figures, and everything that could not be measured.

## Rules

- **Never modify `../collectives/`, `../kernels/` or the submodules in place.** Any edit to a submodule
  needed to get a variant (e.g. `CLUSTER_SIZE 1`, DeepGEMM cluster-1 override, hpc-ops `use_cluster=false`)
  is saved as `patches/<manifest>__<variant>.patch` (`git diff` inside the submodule) and referenced from the
  manifest's `build.patch` field, so a fresh checkout reproduces it. The runner applies a patch in a
  **per-variant git worktree of the submodule** (`roofline/build/src/<manifest>__<variant>/`, git-ignored),
  never in `third_party/<repo>` itself, which stays clean at its pinned commit.
- Ceilings are measured with microbenchmarks; datasheet numbers are drawn as dashed references only.
- Clocks: every timed or ncu run uses `tools/run --profile --lock-sm-mhz <LOCK>`; ncu always with
  `--clock-control none` (never `base`: it would override our lock), state the clock in the notebook.
  Flush L2 between timed iterations (`CleanL2` / 256 MB write) unless the workload's own harness already
  does; say which.
- Time kernels with ncu `gpu__time_duration.sum` / nsys kernel durations, not Python wall time —
  several artifacts call `cudaDeviceSynchronize` and rebuild TMA descriptors per call.
- A variant that does not validate (wrong results) is not plotted; log it in the change log below.
- CSVs: mandatory columns first, append-only, runs registered in `results/roofline__runs.csv`
  (`../docs/ENVIRONMENT.md` §5). Keep raw outputs under `results/` (git-ignored); commit only
  `results/ceilings.csv`, `results/points.csv` (force-added) and the executed notebook, when R8 is done or
  when a task's card says so.
- New Python or system dependency → add it to `roofline/docker/Dockerfile`, rebuild and re-tag the image
  (R0's build script), note it in `docs/DESIGN.md` §6. Never install at run time.
- Commit message per task: `roofline/R<n>: <what was done>; <main finding>`.

## Change log  *(append entries here)*

- _none yet_
