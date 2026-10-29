# Environment and H100 server protocol (binding for `collectives/` and `roofline/`)

This is the only environment document in this repository. Since 2026-10-29 the server-wide tooling — the `h100`
command, the per-user FCFS GPU queue, the target-GPU switch and the single toolchain image — lives in `~/h100/`
and is described in **`~/h100/H100.md`**, which is authoritative for host facts. This file summarises what binds
here and adds what is specific to this repository. Facts dated 2026-10-25 unless stated.

## 1. The machine and the GPU
- Host `nishil-4xh100-01`: 4 × H100 NVL, shared with other users. Our group may use GPU 2 and GPU 3; `h100`
  refuses every other index. There is no default GPU: the user sets the target (`h100-target N`), and agents
  never change it.
- This repository's measurements ran on **GPU 3**: `GPU-1241dcbc-33ea-80bf-5f87-37f7868c3126`, PCI
  `00000000:DA:00.0`, serial 1793225008254; H100 NVL, 132 SMs, 60 MiB L2, 94 GB HBM3 (3.94 TB/s peak), power
  limit 400 W, ECC on, MIG off. Keep a measurement series on one card: GPU runs are pinned with `--gpu 3`
  (a pinned job waits in the queue rather than run on another card).
- Fixed run configuration: microbench `docs/context/10__device_config.md` (SM clock lock, memory clock,
  power limit, compiler flags). SM clock lock **1635 MHz** (selected on GPU 2, confirmed for GPU 3 by the
  microbench GPU-3 bring-up, commit `89ff996`); use it as `<LOCK>` below.
- Card-specific facts (SM→GPC map, SMs that never join clusters ≥ 4, CS-16-capable GPCs) come from the GPU-3
  run_ids of microbench experiments 02/03 (`h100-microbench/experiments/03__sm_gpc_topology/results/03__sm_gpc_topology__map.csv`,
  rows whose `notes` carry serial 1793225008254): map run `2026-10-25T19:56:47.043+00:00`, identical to the GPU-2
  map (8 GPCs 12/16/16/16/18/18/18/18, SMs 124–131 only in clusters ≤ 2, 7 GPCs host CS 16); its
  `partition`/`subnode` columns are empty. GPU 3's L2 partitions (microbench 12): GPCs {0,5,6,7} | {1,2,3,4}
  (G4 and G7 swapped compared with GPU 2).

## 2. Running anything: `h100 run`
Every build, GPU run and notebook execution goes through `h100` (`~/h100/bin`, on PATH), from any directory
under `$HOME`: each job is a fresh container of the shared image that mounts all of `$HOME` at the same path,
runs as your uid and `cd`s into `$PWD`.

```bash
h100 run --no-gpu -- make -C collectives/harness                          # build, no GPU, runs at once
h100 run --gpu 3 --profile --lock-sm-mhz <LOCK> -- <command>              # queued GPU run on GPU 3, locked clock, ncu allowed
h100 submit --gpu 3 --profile --lock-sm-mhz <LOCK> -- <command>           # same, returns the job id (h100 wait ID / h100 log -f ID)
h100 run --no-gpu -- jupyter nbconvert --execute --to notebook --inplace collectives/analysis/dsmem_collectives.ipynb
h100 shell                                                                 # interactive shell in the image (no GPU)
h100 queue                                                                 # queue state (active/frozen + reason), target, jobs
h100 status                                                                # who uses which GPU, locks, our containers
roofline/tools/run [same options] -- <command>                             # roofline: adds the study's cache/PYTHONPATH variables
```

- `--no-gpu`: builds and notebooks (no GPU visible, no queue, no lock).
- GPU jobs enter the **user's FCFS queue**, shared by both repositories: one GPU job at a time; waiting jobs
  hold no container. The dispatcher runs the preflight (no compute process on the GPU, memory ≤ 1 GiB,
  utilisation ≤ 5 %, no other container given the GPU, cooperative lock `/tmp/h100-gpu-locks/gpu<N>.lock` free)
  and holds the lock for the job.
- The queue **freezes** when the GPU is in use by someone else, when another user's process appears while our
  job runs (our job is stopped and stays at the head), or when a running job is ended from outside. Only the
  user unfreezes it (`h100 unfreeze`) or switches the target (`h100-target N`). Agents report the reason from
  `h100 queue` and wait. Keep sweeps restartable: an interrupted job reruns from the start.
- `--profile`: required for Nsight Compute (the driver has `RmProfilingAdminOnly=1`; a plain user gets
  `ERR_NVGPUCTRPERM`) and for clock locking. The entrypoint locks with `nvidia-smi -lgc`, resets on exit, and
  runs your command as your uid with `CAP_SYS_ADMIN`.
- `--lock-sm-mhz <LOCK>`: always pass it for timed and ncu runs; the harness records the measured clocks
  (`H100MB_LOCK_SM_MHZ` / `H100MB_GPU_UUID` are still set in the container for the vendored harness).
- Never `docker run` by hand, never any `docker … prune`, never kill/renice/touch other users' processes,
  containers, locks or allocations. A DCGM exporter runs on the node; "counters busy" from ncu is most likely it.
- Disk: containers need ≥ 2 GB free on the scratch volume (`/srv/local/scratch`), image builds ≥ 40 GB
  (`h100 image info`). `/scratch` has been full before: check first.
- The container's `/tmp` is RAM and is discarded: never write results there.

## 3. Image
Both studies use the one shared image `h100:cuda12.9.1-<hash>` (`~/h100/docker/`, `h100 image info`): CUDA 12.9.1
(nvcc 12.9.86, cuda-gdb, compute-sanitizer), Nsight Compute 2025.2.1, Nsight Systems 2025.1.3, clang-20, NVBit,
CUTLASS v4.4.2 source (`$CUTLASS_INCLUDE`), Python 3.12 venv with numpy/pandas/matplotlib/scipy/scikit-learn/
jupyter, PyTorch 2.11.0+cu129, Triton 3.6.0, flashinfer 0.7.2.post1, flash-attn 2.8.3.post2, pyyaml, ninja,
`libdw-dev`/`libelf-dev`, `TORCH_CUDA_ARCH_LIST=9.0a`, CUDA driver stubs on `LIBRARY_PATH`.
Earlier results ran in `h100-microbench:cuda12.9.1-26d4f436f8c5` (collectives C1) and in the derived
`h100-roofline:cuda12.9.1-83734cedf58e` (roofline R0–R5), both retired on 2026-10-29 with the same pinned
versions of everything they contained.

- New dependencies go into the shared image (`~/h100/docker/requirements.txt` → `h100 image lock` →
  `h100 image build`), with the user's approval; never `pip install` at run time.
- **Containers are deleted after each job**: anything installed into the image's venv at run time is lost.
  Artifact extensions (ClusterFusion, DAK, DeepGEMM, hpc-ops) are built in-tree on NFS (`python setup.py
  build_ext --inplace`, or `pip install --no-deps --no-build-isolation --prefix roofline/build/pyenv`) and put
  on `PYTHONPATH` by the runner. JIT caches that must persist across runs (DeepGEMM, Triton, torch extensions)
  are pointed at `roofline/build/cache/` on NFS by `roofline/tools/run`.
- NFS is slow for trees of thousands of small files: dependencies live in the image, not on NFS.

## 4. Nsight Compute on this setup (from microbench `docs/context/07`)
- Flags: `--clock-control none` (our own lock) `--csv --page raw --print-units base`; **do not** add
  `--print-metric-name name` (ncu 2025.2.1 rejects it together with `--page raw`). `--cache-control all|none`
  as each study specifies.
- No per-instance (per-SM, per-slice) values are exported for any hardware-unit counter: only `.sum`,
  `.avg`, `.max`, `.min` rollups. Per-rank numbers come from in-kernel accounting or models; hot spots from
  `.max/.avg`.
- One `lts__` counter is collected per replay pass (several `lts__` metrics = several passes; fine with
  kernel replay, but kernel replay saves/restores memory and wipes L2: warm-L2 counters need
  `--replay-mode application`).
- The profiler adds its own traffic that grows with the grid: subtract a same-grid "all-exit" baseline
  launch when absolute byte counts matter.
- DSMEM metrics use the qualifier `dshared` (e.g. `l1tex__m_l1tex2xbar_write_bytes_mem_dshared.sum`,
  `l1tex__m_xbar2l1tex_read_bytes_mem_dshared.sum`, `sm__sass_inst_executed_op_dshared_{ld,st,atom,redas,stas,tma_st,tma_red}.sum`);
  DSMEM traffic never appears in `lts__*`. Items still unverified on the device (which side carries the
  payload of a DSMEM load, `loopback` semantics, the rated link peak): microbench `docs/context/07` §8.
  Always filter metric lists through `ncu --query-metrics` first (template: `available()` in
  `h100-microbench/experiments/05__memory_hierarchy_bandwidth/src/run.sh`).

## 5. Raw data and CSVs (from microbench `docs/context/09` §6)
- Results go straight into `<study>/results/` on NFS (git-ignored), never `/tmp`. Never delete or overwrite a
  results file: append rows with a new `run_id`.
- Mandatory columns first in every CSV: `gpu_name, sm_count, sm_clock_mhz, mem_clock_mhz, cuda_version,
  driver_version, run_id, launch_idx` (`hb::with_mandatory` in the vendored `host.hpp`, `docs/REUSE.md`);
  the study's own schema follows.
- Crash safety: header only into an empty file, append-only, flush per launch, fsync per configuration,
  `flock` while appending; register every run in `results/<study>__runs.csv` (`started` row at start,
  `complete` row at the end; a run without `complete` is partial and is reported as such).
- Record clocks and throttle reasons before/after every configuration (`hb::Run::clock_note`); a launch
  whose clock fell below the lock or whose throttle reasons fired is discarded and counted.
- Commit only what each study's `CLAUDE.md` lists (code, executed notebooks, summary CSVs it names); never
  binaries, `.ncu-rep`/`.nsys-rep`, logs or raw CSVs.

## 6. Repository setup
- Submodules: `git submodule update --init` (needed by `roofline/`; `hpc-ops` and `flash-attention` are
  shallow; DeepGEMM additionally needs its own `third-party/{cutlass,deep_jit}` submodules initialised, see
  `roofline/docs/WORKLOADS.md` §8). Never commit changes inside a submodule; variants are built from patches (`roofline/CLAUDE.md`).
- Claude Code permissions: `.claude/settings.json` at the repository root allows `h100 run/submit/queue/log/
  wait/status/check` and `roofline/tools/run`, asks before `h100 unfreeze/freeze/cancel`, `h100-target N` and
  image builds, and keeps the deny list (`docker run`, every `docker … prune`, force-push).
- Edit, run agents and use git directly in this NFS checkout; no extra worktrees or branches unless the user asks.
