# CLAUDE.md — H100 NVL roofline study of DSMEM vs non-DSMEM workloads

> **IMPORTANT — READ FIRST.** The Claude Code agent that runs this study on the shared H100 NVL
> server MUST, before building or launching anything, update this file (section *H100 NVL server
> protocol*) and `docs/DESIGN.md` §6 (*Tooling*) with the server's usage rules and the tooling it sets
> up. The server is a shared resource with restrictions and particularities that are not described
> here; discover them (operator, server docs, `nvidia-smi`, queue/lock conventions, module system,
> `/etc/motd`) and record them so every later session follows the same protocol. Every GPU run must
> comply with what you write there.

## Goal

One figure in the style of the TPUv1 paper's roofline (Jouppi et al. 2017, Fig. 3): log-log achieved
throughput vs operational intensity for an H100 NVL, with **one marker per workload variant**: the
DSMEM version of each workload and its non-DSMEM counterpart, joined by an arrow. Workloads are the
kernels catalogued in `../kernels/`; each is described by one manifest in `workloads/`. The study must be
extensible: adding a future DSMEM port means adding one manifest (see `docs/ADDING_A_WORKLOAD.md`).

Read, in this order: `docs/DESIGN.md` (scope decision, methodology, protocol, CSV schema),
`docs/WORKLOADS.md` (every pair: entry points, baselines, traps), `docs/ADDING_A_WORKLOAD.md`
(manifest reference), then the manifests.

## H100 NVL server protocol  *(to be filled by the remote agent)*

- Access / login, allowed hours, other users of the GPU: _TODO_
- How to check the GPU is free and how to reserve it (queue, lock file, `nvidia-smi`): _TODO_
- Limits (max run time, memory, concurrent processes, permission for `ncu` counters and clock locking): _TODO_
- Where to build and write results (scratch paths, quotas); where model weights may be stored: _TODO_
- Clean-up obligations: _TODO_

## Workflow

1. Fill in the protocol section above and `docs/DESIGN.md` §6. Record the commit hash of this repo.
2. Measure the ceilings (`docs/DESIGN.md` §2.1) and write `results/ceilings.csv`. Do this before any
   workload: a roofline with datasheet ceilings only is not acceptable.
3. Work through the manifests in the priority order of `docs/WORKLOADS.md` §0. For each manifest:
   build every variant, validate outputs against the reference named in the manifest, collect the
   `points.csv` rows (operator scope, then layer scope where defined), run the ncu metric pass.
4. Execute `analysis/roofline.ipynb`; check that every manifest appears (measured, or "reported" for
   `runnable: false`) and that the ncu roofline sections agree with the hand-computed points within 10 %.
5. Report: ceilings table, per-workload table (time, FLOPs, bytes per level, intensity, throughput,
   speedup DSMEM/non-DSMEM, fraction of layer time), figures, and everything that could not be measured.

## Rules

- **Never modify `../collectives/`, `../kernels/` or the submodules in place without a patch file.**
  Any edit to a submodule needed to get a variant (e.g. `CLUSTER_SIZE 1`, DeepGEMM cluster-1 override,
  hpc-ops `use_cluster=false`) is saved as `patches/<manifest>__<variant>.patch` (`git diff` inside the
  submodule) and referenced from the manifest's `build.patch` field, so a fresh checkout reproduces it.
- Ceilings are measured with microbenchmarks; datasheet numbers are drawn as dashed references only.
- Lock clocks for timed runs if the protocol allows; otherwise use `ncu --clock-control base` for
  counter runs and state the clock situation in the notebook. Flush L2 between timed iterations
  (256 MB write) unless the workload's own harness already does; say which.
- Time kernels with ncu `gpu__time_duration.sum` / nsys kernel durations, not Python wall time —
  several artifacts call `cudaDeviceSynchronize` and rebuild TMA descriptors per call.
- A variant that does not validate (wrong results) is not plotted; log it in the change log below.
- Keep raw outputs under `results/` (git-ignored); commit only `results/ceilings.csv`,
  `results/points.csv` and the executed notebook when the study is done.

## Change log  *(append entries here)*

- _none yet_
