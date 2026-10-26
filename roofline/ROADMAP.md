# ROADMAP — roofline study (tasks R0–R8)

Imported by `roofline/CLAUDE.md`. **All tasks run in order in one conversation** (`CLAUDE.md` → "One
conversation for the whole study": single plan, when to stop for the user, resume, final review). Commit and
subagent rules: `../CLAUDE.md`. Environment: `../docs/ENVIRONMENT.md`.
Reusable code: `../docs/REUSE.md`. Design: `docs/DESIGN.md` (methodology, binding), `docs/WORKLOADS.md`
(per-workload facts and traps), `docs/ADDING_A_WORKLOAD.md` (manifest reference and done-checklist).

## Task status
Status values: `todo` · `in progress` (being worked on) · `in review` (committed locally, not pushed; the
next task may start, the user reviews the whole study after R8) · `done` (approved and pushed, set for all rows
at the end). Update the row **before** the task's commit; "Key results" records what later tasks consume
(numbers, file paths, decisions) and is the record that survives summarisation of the conversation.
A prerequisite "R<n> complete" holds when that row is `in review` or `done`.

| ID | Task | Manifests | Status | Key results for later tasks |
|---|---|---|---|---|
| R0 | Environment: submodules, roofline Docker image, `tools/run` wrapper | — | in review | **Image** `h100-roofline:cuda12.9.1-d08c91f3643b` (32.8 GB) `FROM h100-microbench:cuda12.9.1-26d4f436f8c5`; `roofline/tools/build_image.sh {tag,lock,build}` (content-hash tag over Dockerfile + lock + base + hpc-ops commit; ≥ 40 GB check; label `roofline.owner=$USER`, inherited `h100mb.owner` blanked so microbench `tools/env build` never removes it; removes only older `roofline.owner` images; never prunes). Pins: torch 2.11.0+cu129, triton 3.6.0, flashinfer-python 0.7.2.post1 (JIT), cmake 4.4.4, setuptools 81 (`roofline/docker/requirements.lock`, 68 lines, resolved on top of the microbench venv); CUTLASS 4.4.2 headers `/opt/cutlass/include` = `$CUTLASS_INCLUDE` (for DAK); `LIBRARY_PATH` = CUDA stubs; `TORCH_CUDA_ARCH_LIST=9.0a`. **Wrapper** `TR=roofline/tools/run` = microbench `tools/run` with `H100MB_IMAGE` + `env` prefix: `PYTHONPATH=roofline/harness[:$ROOFLINE_PYTHONPATH]`, `ROOFLINE_REPO`, JIT caches under `roofline/build/cache/`. **Microbench override** one line in MB `tools/run` (not `common.sh`): MB commit `8f8fc7b`, approved, local, not pushed. **Submodules** initialised at the pins (+ DeepGEMM `third-party/cutlass` v4.2.1 `f3fde58`, `deep_jit` `2efdab4`); all five `roofline/patches/*.patch` pass `git apply --check`. CREDIT built in worktree `roofline/build/src/CREDIT__pin` (`make all CUDA_ARCH=sm_90a`, 8 binaries, `--no-gpu`). **Smoke** (GPU 3, `--profile --lock-sm-mhz 1635`, clock read 1635/2619 MHz): torch fp16 4096³ matmul rel. err 3.3e-4 (~490 TFLOP/s rough), flashinfer rmsnorm JIT max abs err 3.8e-3, CREDIT `layernorm_backward` 2048×16384 P = 4: block_smem 0.205 ms, cluster_smem 0.150 ms (log `results/smoke/`). Versions in DESIGN §6. `.claude/settings.json` unchanged (already allows `roofline/tools/run`; an edit adding absolute paths was refused by the permission classifier). **Note for R0+:** never rewrite a running bash script in place (NFS keeps the inode): the MB edit was applied by write + rename. GPU time ≈ 2 min. |
| R1 | Ceilings → `results/ceilings.csv` | — | in review | **`results/ceilings.csv`** (latest row per name; summary run `roofline/harness/ceilings_summary.py`), all at **1635 MHz**, 0 clock-flagged launches in the used rows except where stated: HBM read **3.70** / write 3.62 / copy 3.39 TB/s (`hbm_read_bw` is the roofline slope); L2 **7.75** TB/s (`k3_stream`; `k3_direct` 7.53); SMEM **27.45** TB/s (128 B/cycle/SM); **DSMEM 4.04 TB/s** (bulk push, CS 2; pull 3.54; CS 4/8/16 push 3.24/2.93/2.49, pull 2.77/2.78/2.59 — `dsmem_bw_{ld,push}_cs<N>`), 88 % of the 21.33 B/cycle/SM send limit, only 9 % above HBM; FP32 **54.3** TFLOP/s (99 % of datasheet × 1635/1785), INT32 IMAD **27.5** T int-op/s (2 ops per IMAD, half the FFMA rate); tensor cores (cuBLAS 8192³ best launch): FP16 **479**, BF16 **526**, FP8 **955** TFLOP/s (`fp8_tc_peak` = fast-accum; default accum 848) = 62–69 % of datasheet at the lock. **Power limit:** back-to-back 8192³ GEMMs pull the SM clock below the lock (1605–1620, once 1485 MHz; first run 61 of 100 TC launches flagged, all kept in `results/ceilings/tc.csv`); `ceilings_tc.py` now idles 1 s per launch / 10 s per config (second run 0 flagged); TC ceilings = best unflagged launch over both complete runs (fp8 fast-accum 955 from run 1, 883 in run 2 — a 7 % run-to-run spread). Workload runs can hit the same limit: the runner re-runs clock-flagged timing commands and keeps the flags. **Datasheet correction:** DESIGN listed H100 SXM peaks (67/989/1979); H100 NVL dense = FP32 60, FP16/BF16 835.5, FP8 1670.5 TFLOP/s, HBM 3.9 TB/s (DESIGN §2.1, `ceilings_summary.py`). **ncu checks** (`results/ceilings/crosscheck.csv`): FFMA count ratio 1.0000; HBM read 1.000, write 0.986 (dirty-in-L2 at kernel end); DSMEM `l1tex2xbar_write` = `xbar2l1tex_read` = payload (1.000) for pull and push at CS 2–16 → payload rule in DESIGN §6. Event floor 2.53 µs on GPU 3. **Code:** `roofline/harness/ceilings/` (Makefile, `src/{ceil_common.hpp,event_floor,bw_hbm,bw_l2,bw_smem,dsmem_bw,compute_peak}.cu`, `run_all.sh`; kernels copied from MB 05/11 at MB `8f8fc7b`, provenance in `common/h100mb/PROVENANCE.md`), `harness/ceilings_{tc.py,ncu.sh,ncu_post.py,summary.py}`. **Notebook:** `analysis/roofline.ipynb` rewritten in the notebook style (logic in `analysis/roofline_figures.py`; §1 device roofline, ceilings and DSMEM tables). GPU time ≈ 8 min. |
| R2 | Manifest runner + the six CREDIT workloads | `credit_*` (6) | in review | **Runner** `harness/run_manifest.py` (+ `rooflib.py`: CSV/registry/ncu helpers copied from collectives; `flush.py`: Triton `roofline_l2_flush_kernel`; `wrapper.py`: wrapper CLI/loop) — conventions in `docs/ADDING_A_WORKLOAD.md` and the runner docstring: worktree per (submodule, patch) `build/src/<repo>__<patch|pin>`; validate (exit 0 + `expect` regexes; `skip:` per shape or `shape/P<n>`); time = nsys `cuda_gpu_trace`, iterations delimited by the flush kernel, sum of matched kernel durations, 20 warm-up + 100 reps; ncu `--clock-control none --cache-control all --kernel-name-base demangled`, launch-skip from the timing run's kernel counts; rows → `results/points.csv` (rep ≥ 0 timing, rep = −1 counters), `results/validation.csv`, `results/ncu_metrics.csv` (every metric of every launch), `results/runner_stages.csv`; clock samples before/after, flagged commands re-run (≤ 2) and kept with `clock_bad`. Per-variant `scope: layer` takes `scope_layer` formulas. **CREDIT** (patch `credit__roofline_hooks.patch`, worktree `CREDIT__credit__roofline_hooks`): 90 configurations = 6 × 3 shapes × (block_read, block_smem, cluster P 2/4/8): **80 PASS** (timed + ncu, 0 clock-flagged), **10 SKIP** (block_smem n64k ×6: row > 227 KiB; cluster P2 n64k ×4 for lars/wvar/pearson/softmax: slice > 227 KiB — the first validation recorded these 4 as FAIL (binary reports the variant invalid; rows kept in `validation.csv`) before the `n64k/P2` skip keys existed). **Findings:** n16k/n64k: DSMEM (best P: 2–8 at n16k, 8 at n64k) 1.17–1.55× faster than block_read; DSMEM HBM bytes = 0.96–1.05× compulsory vs block_read 1.5–2.5× (re-reads miss L2); but achieved HBM bandwidth drops from 3.3–3.6 TB/s (block_read) to 1.8–3.2 TB/s, also for the one-CTA staged control (1.4–3.0 TB/s at n16k) — mechanism not measured; DSMEM bytes ≤ 0.01 per HBM byte. n4k: block_read already ≈ compulsory (0.9–1.5×, L2 hits); DSMEM P2 slower for layernorm/softmax/quant (0.57–0.82×), ±5 % pearson/wvar, faster only lars (1.47×). **Formula checks:** FLOP formulas re-derived from the kernel code (flops: layernorm 16, wvar 21, pearson 18, softmax 40 (3 accurate expf), lars 14, quant 1 per element; min_hbm_bytes compulsory); ncu FP32 / formula = 1.00–1.06 for one-CTA kernels on long rows (quant 1.16 at n16k; short rows 1.03–1.63), DSMEM kernels 1.01–1.33 long / ≤ 2.25 short (per-row work replicated per CTA, ×P; quant n4k P8 6.0) — explained in the notebook. HBM bytes slightly below compulsory at n4k (outputs still dirty in L2 at kernel end). ncu kernel time / nsys median 0.96–1.08. Notebook §2 (3 figures/tables + FLOP check). GPU time ≈ 1.2 h. |
| R3 | DAK multicast GEMM | `dak_gemm_multicast` | todo | |
| R4 | ClusterFusion Llama-2-7B decoder (+ layer scope) and RMSNorm | `clusterfusion_llama2_7b_decoder`, `clusterfusion_rmsnorm` | todo | |
| R5 | DeepGEMM FP8 GEMM with TMA multicast | `deepgemm_fp8_gemm_multicast` | todo | |
| R6 | hpc-ops cluster-8 top-k and Lamport RMSNorm | `hpcops_topk_cluster8`, `hpcops_rmsnorm_lamport` | todo | |
| R7 | ClusterFusion DeepSeek-V2-Lite MLA (unfused baseline written here) | `clusterfusion_deepseek_v2_lite_mla` | todo | |
| R8 | FA4 reported markers, final notebook, ncu cross-check, report | `fa4_bwd_2cta_ds_exchange` + all | todo | |

## Dependencies
```
h100-microbench "GPU-3 bring-up" row (GPU lock value) ─┐  (needed for any GPU run)
R0 ─► R1 ─► R2 ─► R3 ─► R4 ─► R5 ─► R6 ─► R7 ─► R8
```
R3–R7 depend only on R2 (the runner) and could run in any order; the order above is `docs/WORKLOADS.md` §0
(value vs effort). R1 must be complete before any figure is shown to the user. This roadmap does not depend on
`../collectives/` (the DSMEM ceiling is measured in R1).

## Shared facts for every task
- Commands: after R0, `TR=roofline/tools/run` (from the repo root) wraps
  `~/work/fa2026/iacoma/h100-microbench/tools/run` with the roofline image. Builds `$TR --no-gpu -- …`; GPU
  runs `$TR --profile --lock-sm-mhz <LOCK> -- …` on GPU 3 (`<LOCK>`: h100-microbench `docs/context/10` §2).
- Layout: `roofline/harness/` (runner, ceiling kernels, wrappers), `roofline/build/` (git-ignored: per-variant
  source worktrees `build/src/<manifest>__<variant>/`, built extensions, `build/pyenv/`, JIT caches
  `build/cache/`), `roofline/results/` (git-ignored except the files `CLAUDE.md` names).
- The manifests' `build`/`run` commands were written without running anything (they still contain
  `pip install -e .`, in-place `git apply` and `# agent script` placeholders). Each workload task rewrites its
  manifests to the runner's conventions (worktree cwd, in-tree build, `PYTHONPATH`, real wrapper scripts
  under `roofline/harness/workloads/<manifest>.py`) and keeps the YAML schema of `docs/ADDING_A_WORKLOAD.md`.
- Every workload task ends with the `docs/ADDING_A_WORKLOAD.md` checklist satisfied for its manifests, the
  notebook re-executed with the new markers, and the per-workload numbers (time, FLOPs, bytes per level,
  intensity, speedup) in "Key results".
- A variant that fails validation is not plotted: log it in `CLAUDE.md` → Change log and in "Key results".

---

## R0 — Environment
**Goal.** A reproducible toolchain for all six artifacts, usable through `tools/run` with the same GPU
safety, lock and clock handling as h100-microbench.

**Prerequisites.** None for the image; the GPU smoke step needs the microbench GPU-3 bring-up `done`.

**Read first.** `../docs/ENVIRONMENT.md` (all), `h100-microbench/docs/environment/README.md`,
`h100-microbench/docker/{Dockerfile,entrypoint.sh}`, `h100-microbench/tools/{run,env,common.sh}`,
`docs/DESIGN.md` §6, `docs/WORKLOADS.md` (build lines of every artifact), the artifacts' own READMEs /
`setup.py` / `requirements*.txt` after step 1.

**Steps.**
1. `git submodule update --init` (check the pinned commits match `git submodule status`); DeepGEMM's own
   `third-party/` submodules. Never commit inside a submodule.
2. Dependency survey: for each artifact, the Python/CUDA/compiler requirements (ClusterFusion: CUDA ≥ 12.4,
   flashinfer; hpc-ops: CUDA ≥ 12.9, C++20; DAK: `setup.sh` installs CUTLASS through conda — replace with the
   CUTLASS headers in `third_party/hpc-ops/3rd/cutlass` or a pinned CUTLASS copy in the image; DeepGEMM:
   `develop.sh`, JIT; Triton/torch.compile baselines for CREDIT's `scripts/run_local.py`). Pick one PyTorch
   build for CUDA 12.9 that satisfies all of them and record the choice.
3. `roofline/docker/Dockerfile`: `FROM` the current microbench image tag (pinned), add the survey's packages
   into `/opt/venv` with a lock file (`roofline/docker/requirements.lock`), keep the microbench `ENTRYPOINT`.
   A build script `roofline/tools/build_image.sh`: content-hash tag (Dockerfile + lock + base tag), refuses
   below 40 GB free on the scratch volume, labels the image, removes only our previous roofline tags, never
   prunes.
4. One-line override in `~/work/fa2026/iacoma/h100-microbench/tools/common.sh`:
   `IMAGE=${H100MB_IMAGE:-$IMAGE_REPO:cuda12.9.1-$IMAGE_TAG}`, honoured by `tools/run` only — make sure
   `tools/env build` still builds the microbench image (unset `H100MB_IMAGE` there or keep a separate
   variable). Commit it in h100-microbench separately (`tools: H100MB_IMAGE override for derived images`)
   after the user approves (a stop point of the one-conversation protocol: ask, wait); do not push it
   without approval.
5. `roofline/tools/run`: `H100MB_IMAGE=<roofline tag> exec ~/work/fa2026/iacoma/h100-microbench/tools/run "$@"`,
   plus `PYTHONPATH`/cache variables pointing into `roofline/build/`. `tools/run` passes only `HOME`, `USER`
   and its own `H100MB_*` variables into the container (`h100-microbench/tools/run` l.54–79), so the wrapper
   prefixes the command with `env PYTHONPATH=… TRITON_CACHE_DIR=… DG_JIT_CACHE_DIR=… …` (or the image sets
   defaults with `ENV`). `image_present` in `tools/run` checks `$IMAGE`, so the override also covers it.
6. Repository hygiene: `.gitignore` entries `roofline/build/`, `roofline/results/*` (with the force-added
   exceptions), `common/h100mb` is tracked; `.claude/settings.json` (`../docs/ENVIRONMENT.md` §6, include
   `roofline/tools/run`).
7. Smoke (GPU 3): `torch.cuda.get_device_name()`, a 4096³ fp16 matmul, `import flashinfer`; build the CREDIT
   binaries with `--no-gpu` (`make all CUDA_ARCH=sm_90a` in a worktree) and run one CREDIT shape.
8. Fill `docs/DESIGN.md` §6 (versions of driver, CUDA, nvcc, ncu, PyTorch, flashinfer, Triton; image tag).

**Deliverables.** `roofline/docker/{Dockerfile,requirements.lock}`, `roofline/tools/{build_image.sh,run}`,
the microbench override commit, `.gitignore`, `.claude/settings.json`, DESIGN §6.

**Done when.** The image builds from scratch with the script; the smoke test passes on GPU 3 under
`--profile --lock-sm-mhz <LOCK>`; `tools/env build` in h100-microbench is unaffected. GPU time ≈ 10 min;
image build ≈ 30–60 min.

---

## R1 — Ceilings → `results/ceilings.csv`
**Goal.** Measured ceilings of DESIGN §2.1 on GPU 3 at the locked clock, plus datasheet reference rows.

**Prerequisites.** R0 complete; GPU-3 bring-up `done`.

**Read first.** `docs/DESIGN.md` §2.1 and §5, `../docs/REUSE.md` (§2 table rows for `05`), the docstrings of
`h100-microbench/experiments/05__memory_hierarchy_bandwidth/src/{bw_dram,bw_l2,bw_smem_l1}.cu`,
h100-microbench `docs/context/07` (metric names), `docs/context/09` §1 (timers).

**Steps.**
1. Vendor `common/h100mb/` if collectives C1 has not (`../docs/REUSE.md` §1); copy `bw_common.hpp`
   (`CleanL2`) and the needed kernels into `roofline/harness/ceilings/` with provenance comments.
2. Memory ceilings: HBM read/write/copy (`k4`, 2 GiB, 4 × 132 × 1024 threads, `CleanL2` before each launch),
   L2 read (`k3_direct`, 16–24 MiB working set), SMEM read (`k12`, slope method) — each × ≥ 5 launches after
   warm-ups; cudaEvent rates net of the event floor (measure the floor on GPU 3).
3. Compute ceilings (new kernels / scripts): FP32 FFMA chain (all SMs, ILP to saturate), INT32 IMAD chain,
   FP16 and BF16 dense tensor core (cuBLAS 8192³ via torch, best of 10), FP8 e4m3 (`torch._scaled_mm` /
   cuBLASLt, best of 10). Confirm the instruction mix with ncu (`sm__sass_thread_inst_executed_op_ffma_pred_on.sum`,
   `sm__inst_executed_pipe_tensor_op_hmma.sum`) and the SASS (FFMA / IMAD / HGMMA).
4. DSMEM ceiling (own microbenchmark, independent of `../collectives/`): every SM in a cluster of size
   CS ∈ {2, 4, 8, 16}, max active clusters at 1 CTA/SM, each CTA streaming from a peer with
   `ld.shared::cluster.v4` (pull) and with bulk `cp.async.bulk.shared::cluster` pushes; aggregate bytes/s
   GPU-wide from `%globaltimer` + cudaEvent, cross-checked with the `dshared` byte counters. Report the best
   CS and both directions.
5. Datasheet rows (`source = datasheet`): HBM 3.9 TB/s, FP32 67 TFLOP/s, FP16/BF16 989, FP8 1979 TFLOP/s
   (H100 NVL, boost clock). Note in the CSV `method` that measured rows are at the locked clock.
6. `results/ceilings.csv` (DESIGN §5 schema, mandatory columns first) and a raw file per ceiling.
7. Agent-only sanity: microbench 05 on GPU 2 measured HBM read 3.70 / write 3.62 TB/s, L2 read 7.7 TB/s,
   SMEM 27.5 TB/s at 1635 MHz; a GPU-3 value far from these needs investigation (clocks, placement, flush).

**Done when.** Every DESIGN §2.1 row has a measured value (or a recorded reason why not); the notebook's
ceiling cell renders. GPU time ≈ 30 min.

---

## R2 — Manifest runner + the six CREDIT workloads
**Goal.** One generic runner that takes a manifest from build to `points.csv`, proven on the simplest pairs.

**Prerequisites.** R1 complete.

**Read first.** `docs/DESIGN.md` §2.2–§3 and §5, `docs/ADDING_A_WORKLOAD.md`, `docs/WORKLOADS.md` §4 and §0,
the six `workloads/credit_*.yaml`, `third_party/CREDIT` README and `src/workloads/*.cu` CLI.

**Steps.**
1. `roofline/harness/run_manifest.py <manifest> [--variant] [--shape] [--stage build|validate|time|ncu|all]`:
   - parse the YAML, evaluate `flops`/`int_ops`/`min_hbm_bytes` per shape;
   - build: create `build/src/<manifest>__<variant>/` as a `git worktree` of the submodule at its pinned
     commit, apply `build.patch` there, run `build.cmd` there (no edits in `third_party/`);
   - validate: run the variant's validation, refuse to time a variant that fails;
   - time: locked clock, L2 flush between iterations unless the artifact does it (CREDIT does not: wrap it),
     20 warm-up + 100 reps, kernel durations from nsys or ncu `gpu__time_duration.sum`, p10/p50/p90;
   - ncu: `--clock-control none --cache-control all --csv --page raw --print-units base`, `-k regex:` from the
     manifest, one iteration per kernel of the sequence, byte metrics per level (DESIGN §2.2; one `lts__`
     counter per pass, so expect several passes);
   - append `points.csv` rows (mandatory columns first, DESIGN §5), register the run.
2. CREDIT: rewrite the six manifests to the runner's conventions; record both `block_read` and `block_smem`
   (`--profile-variant block` reports the faster one: read `block_name`), and every cluster size P ∈ {2,4,8}
   (the DSMEM marker is the best P per shape, the others are kept as rows).
3. Check: modelled bytes/element of the binary vs `dram__bytes`; FLOP formula vs FFMA/FADD/FMUL counters
   within 10 % (explain any gap in "Key results").

**Done when.** The six CREDIT manifests pass the `ADDING_A_WORKLOAD.md` checklist and appear in the notebook.
GPU time ≈ 1 h.

---

## R3 — DAK multicast GEMM
**Read first.** `docs/WORKLOADS.md` §7, `workloads/dak_gemm_multicast.yaml`, DAK `benchmark/benchmark_linear.py`,
`app/python/layers/linearLayerMulticast.py`, `include/task/gemv_multicast.cuh`.
**Key points.** `--offload 0`; tokens a multiple of 512; `NUM_SMS = 132` matches; the DSMEM variant needs
`LinearMulticast(prefill=False)` (default `True` runs the non-multicast twin): write a wrapper rather than
patching; replace the conda step of `setup.sh` with the image's CUTLASS; cuBLAS (`--type vanila`) as extra marker.
**Done when.** Checklist satisfied for both roles + the cuBLAS marker. GPU time ≈ 30 min.

## R4 — ClusterFusion Llama-2-7B decoder (+ layer scope) and RMSNorm
**Read first.** `docs/WORKLOADS.md` §1 and §3, both manifests, ClusterFusion `tests/test_llama.py`,
`tests/test_norm.py`, `include/H100/llama/{kernel_sglang.cuh,config.h}`, `chat/llama/model.py`,
`patches/clusterfusion__cluster_size_1.patch`.
**Key points.** Three variants (`unfused` → `fused_c1` control → `fused_c4`); `fused_c1` is untested upstream:
validate against `unfused` first; only cluster sizes 1 and 4 are correct; validation on the attention output
(`residual` is updated in place by every head cluster); layer scope adds FFN + residuals + second RMSNorm to
both variants and reports the fraction of layer time in the fused region; time with ncu/nsys (dispatchers
synchronise and re-encode TMA maps per call). RMSNorm: compare with `flashinfer.norm.rmsnorm`, **no**
cluster-1 rebuild. Build in-tree (no `pip install -e`).
**Done when.** Checklist satisfied for both manifests, operator and layer scope. GPU time ≈ 1 h.

## R5 — DeepGEMM FP8 GEMM with TMA multicast
**Read first.** `docs/WORKLOADS.md` §8, the manifest, `patches/deepgemm__force_cluster_1.patch`, DeepGEMM
`csrc/jit_kernels/heuristics/sm90.hpp`, `tests/generators.py`, `deep_gemm/testing` (`bench_kineto`).
**Key points.** The patch adds `DG_FORCE_CLUSTER_1`; confirm with `DG_PRINT_CONFIGS=1` that the DSMEM run
really uses `cluster_m/cluster_n = 2` (only when the grid has > 1 wave) and the control uses 1; JIT cache under
`build/cache/` on NFS; `DG_USE_NVIDIA_TOOLS=1` under ncu; dense shapes and the grouped MoE shape; cuBLASLt
as extra marker.
**Done when.** Checklist satisfied for every shape where the heuristic picks cluster 2 (list the others).
GPU time ≈ 1 h.

## R6 — hpc-ops cluster-8 top-k and Lamport RMSNorm
**Read first.** `docs/WORKLOADS.md` §5–6, both manifests, both hpc-ops patches, `src/topk/topk_filtered.cu`
(`choose_topk2048_plan`), `tests/test_topk.py`, `src/allreduce/fuse_allreduce_rmsnorm_low_latency.{h,cu}`.
**Key points.** Top-k is an integer workload against the INT32 ceiling (`int_ops`); two non-DSMEM routes
(padding `ke` to 17 rows, or the `use_cluster=false` patch); build the wheel in a worktree and install it
under `build/pyenv/`. Lamport RMSNorm is `runnable: conditional`: first check whether the fused op accepts
world size 1 on one GPU; if not, record that and skip the manifest (reported as not measurable).
**Done when.** Top-k checklist satisfied; Lamport either satisfied or documented as skipped. GPU time ≈ 45 min.

## R7 — ClusterFusion DeepSeek-V2-Lite MLA
**Read first.** `docs/WORKLOADS.md` §2, the manifest, `../kernels/fused_mla_decoder_layer__deepseek_v2_lite__clusterfusion.cu`
(docstring: shapes and dataflow), `patches/clusterfusion__deepseek_cluster_size_1.patch`.
**Key points.** No upstream test exists: write the unfused torch/flashinfer baseline
(`roofline/harness/workloads/deepseek_mla_unfused.py`: down-projections, RMSNorm, W_UK absorption, MQA
decode over the latent cache, W_UV, O-proj) and validate the fused kernel against it before timing;
compile-time `SEQ_LEN 4096`; nine TMA maps; `CLUSTER_SIZE 1` rebuild as control.
**Done when.** Checklist satisfied, or the manifest is marked not measurable with the reason. GPU time ≈ 45 min.

## R8 — FA4 reported markers, final notebook, cross-check, report
**Read first.** `docs/WORKLOADS.md` §9, `workloads/fa4_bwd_2cta_ds_exchange.yaml`, `docs/DESIGN.md` §3.5 and
§4, every earlier row's "Key results".
**Steps.** Fill the FA4 `reported` block from the paper (B200 Fig. 6 / Table 3, compulsory-traffic intensity)
and draw it as stars labelled "reported (B200)", separate from the H100 measurements; run the ncu
`SpeedOfLight_RooflineChart` / `SpeedOfLight_HierarchicalTensorRooflineChart` sections for every DSMEM kernel
and check them against the hand-computed points (10 %); finalise `analysis/roofline.ipynb` (setup cell with
device, clock, versions, run_ids; figures 1–3 of DESIGN §4; per-workload table), execute it with
`$TR --no-gpu -- jupyter nbconvert --execute --to notebook --inplace roofline/analysis/roofline.ipynb`;
force-add `results/ceilings.csv` and `results/points.csv`; final report (CLAUDE.md Workflow step 5).
**Done when.** Every manifest appears in the notebook (measured, reported, or listed as not measurable with
the reason) and the notebook renders on GitHub without the raw data.
