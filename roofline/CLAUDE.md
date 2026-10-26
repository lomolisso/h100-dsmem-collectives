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

## One conversation for the whole study
This study runs R0–R8 in **a single conversation**, task after task, without starting a new conversation
per task. This section replaces steps 2–3 of `../CLAUDE.md` "Starting a new conversation" and step 3 of
"Finishing a task"; everything else there (reading order, commit rules, subagent and GPU rules) applies.

1. **Start.** Read the files of `../CLAUDE.md` step 4, all of `ROADMAP.md` (every card) and the documents of
   the "Read first" lines of R0–R2. The starting task is the first row of the status table that is neither
   `in review` nor `done` (normally R0; a later row when resuming, see 6).
2. **One plan, in plan mode.** Present a plan for the whole remaining roadmap: detailed for the first two
   tasks (code to write, runs, GPU time, result files with sizes, commits), one paragraph per later task
   (what is built and run, GPU time, open risks), total GPU and image-build time, and the points where the
   conversation will stop for the user (3). After approval, run all tasks in auto mode.
3. **Stop and ask the user only** for: the h100-microbench `H100MB_IMAGE` commit (R0 step 4), a prerequisite
   that does not hold, a material deviation from the approved plan (a new dependency outside the image plan,
   a task needing more than 2× its GPU-time estimate, a manifest that cannot be measured beyond what its
   card already allows, a change to `docs/DESIGN.md` methodology), and anything the binding rules forbid
   doing alone. Otherwise continue to the next task without waiting.
4. **Per task.** Before starting a task, re-read its card and the "Key results" of the earlier rows; set the
   row to `in progress`. At its end, follow `../CLAUDE.md` "Finishing a task" steps 1–2: row `in review`,
   "Key results" filled, one commit `roofline/R<n>: …`, **no push**; then go on with the next task. If the
   user comments on a committed task mid-conversation, apply the change in a separate commit
   (`roofline/R<n>: review fixes; …`) and continue.
5. **Context.** The conversation will be summarised several times; durable state lives in files, not in the
   conversation: `ROADMAP.md` "Key results", `docs/DESIGN.md` §6, the change log below and
   `results/roofline__runs.csv`. Write them at the end of every task, before the commit. Keep long build
   and profiler output in files under `results/` or `build/` and read only the parts needed; delegate
   file-heavy reading and code writing to subagents (`../CLAUDE.md` "Subagents and tools") so that the main
   conversation keeps conclusions, not file dumps. GPU runs stay strictly sequential.
6. **Resume.** If the conversation ends before R8 (crash, user stop), a new conversation continues from the
   first row that is neither `in review` nor `done`; a row left `in progress` is checked against the files
   and `results/roofline__runs.csv` and either completed or restarted. It presents a plan for the remaining
   tasks as in 2.
7. **End.** After R8: summarise the whole study for the user (CLAUDE.md Workflow step 5, plus the list of
   commits and what each task did not achieve) and ask for **one review of the whole study**. Apply the
   requested changes, then, after explicit approval, set every row to `done`, commit
   (`roofline: approved; R0–R8 done`) and push. Never force-push.

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

- 2026-10-26 (R2) CREDIT, all six workloads: `patches/credit__roofline_hooks.patch` adds environment-controlled hooks
  (no effect when unset; kernels unchanged): `ROOFLINE_FLUSH_L2_MB` runs a clean-read L2 flush kernel before every
  launch of the timing loop (upstream launches back to back with a warm L2) and delimits iterations for nsys;
  `ROOFLINE_BLOCK_VARIANT=read|smem` benchmarks and verifies one one-CTA kernel (upstream keeps only the faster).
  Manifests: `flops` = FP32 operations per element as coded and `min_hbm_bytes` = compulsory traffic (both were
  wrong: the old bytes were CREDIT's `modeled_bytes()`, the read kernel's re-read traffic).
- 2026-10-26 (R3) DAK `gemv_multicast` (the DSMEM variant; `patches/dak_gemm_multicast__sync_fix.patch`): two races fixed.
  (1) No barrier between the consumer warpgroup's shared-memory writes of the output tile and thread 0's TMA store of
  it (and no barrier before the tile buffer is reused) — `gemv_multicast_prefill` has both (`__sync_barrier<13/15>`).
  (2) No cluster barrier before exit while the peer CTA still arrives remotely on the CTA's mbarriers: the unpatched
  kernel validated (bit-identical repetitions) but failed with `LaunchFailed` under ncu kernel replay at every shape.
  Fix: the two named barriers per warpgroup, and `__syncwarp(); cute::cluster_sync();` at kernel end. Cost 0.5–2.5 %
  (e.g. llama_qo_t1024 52.34 → 52.83 µs, opt30b_fc1 464.3 → 475.9 µs). The non-DSMEM twin is unpatched.
- 2026-10-26 (R4) ClusterFusion Llama-2-7B `LlamaDecoderLayerKernel` (sglang entry) — three races, all fixed in
  `patches/clusterfusion__residual_out_c4_fix.patch` (the DSMEM variant; combined diff against the pin, header with the
  evidence): (1) all 32 head clusters read `residual` and wrote `input + residual` back into it → a late reader
  normalised the wrong vector (whole heads wrong, nondeterministic; GPU-confirmed); fix: write to a new
  `residual_output` (as upstream's `kernel_batch_sglang.cuh`). (2) TMA double-buffer write-after-read in the Q/K/V
  GEMV and decode loops: thread 0 refills buffer `id % 2` while warps 1–3 may still read tile `id − 2` (GPU-localised:
  rows 32–127 of heads 6/7/30/31 only, 34/200 launches at cluster 4; 0/600 after one `block.sync()` per loop).
  (3) `include/dsm.cuh` cluster_reduce: `src += dst` before the CTA's own outgoing `cp.async.bulk.shared::cluster` has
  provably read `src` (no read-completion signal) → `cluster.sync()` per round + proxy/mbarrier-init fences (PTX
  memory model; never observed failing). Cost of all fixes ≈ 1–3 %. Control `clusterfusion__cluster_size_1.patch`:
  CLUSTER_SIZE 1 + RMSNorm prologue fix (only the last of the 4 chunks stayed in registers at cluster 1) + (1) + (2).
  Wrapper bug found and fixed in our harness: the fused variants had been validated against a second fused launch
  instead of the unfused sequence (determinism only); all fused validations were re-run against the unfused reference.
