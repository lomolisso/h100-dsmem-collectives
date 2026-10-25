# Environment and H100 server protocol (binding for `collectives/` and `roofline/`)

This is the only environment document in this repository. The host, the GPUs, the Docker toolchain and the
safety rules are shared with the sibling repository `~/work/fa2026/iacoma/h100-microbench`. Its
`docs/environment/README.md` is authoritative for host facts; this file summarises what binds here and
adds what is specific to this repository. Facts dated 2026-10-25.

## 1. The machine and the GPU
- Host `nishil-4xh100-01`: 4 × H100 NVL, shared with other users. Our group may use **GPU 3 (default)** and
  GPU 2 (only when the user names it). Never any other index: `tools/run` refuses them.
- GPU 3: `GPU-1241dcbc-33ea-80bf-5f87-37f7868c3126`, PCI `00000000:DA:00.0`, serial 1793225008254; H100 NVL,
  132 SMs, 60 MiB L2, 94 GB HBM3 (3.94 TB/s peak), power limit 400 W, ECC on, MIG off.
- Fixed run configuration: microbench `docs/context/10__device_config.md` (SM clock lock, memory clock,
  power limit, compiler flags). The SM clock lock was selected on GPU 2 (1635 MHz); the microbench
  **GPU-3 bring-up** row of `h100-microbench/AGENTS.md` confirms or replaces it for GPU 3. **Until that row
  is `done`, do not start GPU measurements here** (builds are fine): tell the user. After it is done, use the
  lock value written in `docs/context/10` §2 as `<LOCK>` below.
- Card-specific facts (SM→GPC map, SMs that never join clusters ≥ 4, CS-16-capable GPCs) come from the GPU-3
  run_ids of microbench experiments 02/03 (`h100-microbench/experiments/03__sm_gpc_topology/results/03__sm_gpc_topology__map.csv`,
  rows whose `notes` carry serial 1793225008254). The GPU-2 values in `h100-microbench/AGENTS.md` are not
  valid for GPU 3 until re-checked.
- **GPU-3 bring-up done 2026-10-25** (microbench commit `89ff996`, row `in review`): lock **1635 MHz** confirmed for
  GPU 3; GPU-3 SM→GPC map = map run `2026-10-25T19:56:47.043+00:00` of the file above, identical to the GPU-2 map
  (8 GPCs 12/16/16/16/18/18/18/18, SMs 124–131 only in clusters ≤ 2, 7 GPCs host CS 16); its `partition`/`subnode`
  columns are empty (microbench 08 not re-run on GPU 3).

## 2. Running anything: `tools/run` from h100-microbench
Every build, GPU run and notebook execution goes through the microbench wrapper, called from this
repository (it works from any directory under `$HOME`: it mounts all of `$HOME` at the same path, runs as
your uid and `cd`s into `$PWD`):

```bash
TR=~/work/fa2026/iacoma/h100-microbench/tools/run
$TR --no-gpu -- make -C collectives/harness                         # build, no GPU
$TR --profile --lock-sm-mhz <LOCK> -- <command>                    # GPU run on GPU 3, locked clock, ncu allowed
$TR --no-gpu -- jupyter nbconvert --execute --to notebook --inplace collectives/analysis/dsmem_collectives.ipynb
$TR --no-gpu                                                        # interactive shell in the image
make -C ~/work/fa2026/iacoma/h100-microbench gpu-status             # who uses which GPU, locks, our containers
make -C ~/work/fa2026/iacoma/h100-microbench gpu-check              # preflight of GPU 3 only
```

- `--no-gpu`: builds and notebooks (no GPU visible, no lock).
- GPU mode (default GPU 3): preflight refuses if any process is on the GPU, memory > 1 GiB, utilisation > 5 %,
  another container was given the GPU, or the cooperative lock `/tmp/h100-gpu-locks/gpu3.lock` is held. The
  lock is shared with h100-microbench: **one GPU job at a time across both repositories**. If it is held,
  wait and retry; never switch GPU on your own.
- `--profile`: required for Nsight Compute (the driver has `RmProfilingAdminOnly=1`; a plain user gets
  `ERR_NVGPUCTRPERM`) and for clock locking. The entrypoint locks with `nvidia-smi -lgc`, resets on exit, and
  runs your command as your uid with `CAP_SYS_ADMIN`.
- `--lock-sm-mhz <LOCK>`: always pass it for timed and ncu runs; the harness records the measured clocks.
- Never `docker run` by hand, never any `docker … prune`, never kill/renice/touch other users' processes,
  containers, locks or allocations. A DCGM exporter runs on the node; "counters busy" from ncu is most likely it.
- Disk: containers need ≥ 2 GB free on the scratch volume (`/srv/local/scratch`), image builds ≥ 40 GB
  (`~/work/fa2026/iacoma/h100-microbench/tools/env info`). `/scratch` has been full before: check first.
- The container's `/tmp` is RAM and is discarded: never write results there.

## 3. Images
| Study | Image | Contents |
|---|---|---|
| `collectives/` | the microbench image as is (`h100-microbench:cuda12.9.1-<hash>`, selected by `tools/run`) | CUDA 12.9.1, nvcc 12.9.86, Nsight Compute 2025.2.1, Nsight Systems, cuda-gdb, compute-sanitizer, Python 3.12 venv (numpy, pandas, matplotlib, scipy, scikit-learn, jupyter, nbconvert) |
| `roofline/` | a derived image `FROM` the microbench image, built by roofline task R0 (`roofline/docker/Dockerfile`), selected through `roofline/tools/run`, which sets `H100MB_IMAGE` and calls the microbench `tools/run` | + PyTorch (CUDA 12.9 wheels), flashinfer, Triton, pyyaml, ninja, CUTLASS headers, the build dependencies of the six artifacts |

Rules for the derived image:
- Built only by R0's build script (content-hash tag, ≥ 40 GB free check, never prunes, removes only our own
  previous roofline tags). Its entrypoint is the microbench one (GPU UUID check, clock lock, capability drop),
  so do not override `ENTRYPOINT`.
- The `H100MB_IMAGE` override in `h100-microbench/tools/common.sh` (one line, added by R0) is honoured by
  `tools/run` only; `tools/env build` must keep building the microbench image.
- **Containers are `--rm`**: anything installed into the image's venv at run time is lost. Heavy Python
  dependencies are baked into the image. Artifact extensions (ClusterFusion, DAK, DeepGEMM, hpc-ops) are
  built in-tree on NFS (`python setup.py build_ext --inplace`, or `pip install --no-deps --no-build-isolation
  --prefix roofline/build/pyenv`) and put on `PYTHONPATH` by the runner. JIT caches that must persist across
  runs (DeepGEMM, Triton, torch extensions) are pointed at `roofline/build/cache/` on NFS.
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
- Claude Code permissions: the first task of each study adds `.claude/settings.json` at the repository root,
  copied from `h100-microbench/.claude/settings.json` with `tools/run` replaced by the absolute path above
  (and `roofline/tools/run`); the deny list (`docker run`, every `docker … prune`, force-push) stays.
- Edit, run agents and use git directly in this NFS checkout; no extra worktrees or branches unless the user asks.
