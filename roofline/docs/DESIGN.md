# DESIGN.md — roofline study of DSMEM vs non-DSMEM workloads on one H100 NVL

## 1. Scope of a roofline point: operator scope (primary) plus layer/step scope (secondary)

The question was whether to plot full workloads or only the kernels. The study does **both**, but the
headline markers are at **operator scope**; end-to-end serving (SGLang/vLLM) is not run.

- The roofline is a per-kernel model: achieved FLOP/s against FLOP per byte of *one* memory level. A
  full decode step through a serving framework mixes dozens of kernels, CUDA-graph gaps and CPU time,
  and is dominated by the weight-streaming GEMVs / FFN that are identical in both versions, so the two
  markers would nearly coincide (ClusterFusion itself: ~1.6× on the fused core modules vs 1.1–1.4×
  on time per output token).
- **Operator scope** = the region the DSMEM port changes: the fused DSMEM kernel vs the exact sequence
  of non-DSMEM kernels it replaces (or the same kernel at cluster size 1), on identical inputs and
  shapes. FLOPs, bytes and kernel times are summed over the sequence, so the non-DSMEM point is one
  marker too. This is where the DSMEM effect is large and the model is valid.
- **Layer/step scope** (hollow marker, same colour) answers "is it visible in the full workload": the
  remaining kernels of one decoder layer (or one decode step / one full row-wise pass) are timed with
  the same inputs and added to *both* variants. The notebook also reports the fraction of layer time
  spent in the optimized region, which is the Amdahl bound on the end-to-end gain.
- Feasibility: all runnable workloads fit one H100 NVL (94 GB; Llama-2-7B fp16 weights 13 GB,
  DeepSeek-V2-Lite ~31 GB); operator-scope kernels run in microseconds to milliseconds; the complete
  sweep is minutes of GPU time. Serving frameworks, model checkpoints beyond one layer's weights, and
  multi-GPU setups are not needed.

## 2. Roofline methodology

### 2.1 Ceilings (measured, datasheet as dashed reference)
Written to `results/ceilings.csv` (schema §5) by roadmap task R1, with microbenchmarks under
`harness/ceilings/` (memory kernels vendored from h100-microbench experiment 05, `../../docs/REUSE.md`),
on GPU 3 at the locked SM clock (`../../docs/ENVIRONMENT.md`). The datasheet references are at boost clock,
so a measured compute ceiling below them is expected; state both clocks in the notebook.

| Ceiling | Method | Datasheet reference (H100 NVL, per GPU) |
|---|---|---|
| HBM bandwidth | microbench 05 `bw_dram.cu` `k4` (read / write / copy over 2 GiB, 4 × 132 × 1024 threads, `CleanL2` before each launch), locked clock; report read, write and copy GB/s | 3.9 TB/s HBM3 |
| L2 bandwidth | microbench 05 `bw_l2.cu` `k3_direct`: repeated reads of a 16–24 MB working set (fits the 60 MiB L2), all SMs | — |
| L1/SMEM bandwidth | microbench 05 `bw_smem_l1.cu` `k12`: per-SM shared-memory read rate (iteration-slope method) × SM count (132) | — |
| DSMEM bandwidth | aggregate microbenchmark vendored from microbench 11 (`k_stream`; user decision in the R0–R8 plan, instead of a new kernel): max active clusters at 1 CTA/SM, CS = 2…16, every CTA streaming from a peer (`ld.shared::cluster.v4` pull and bulk push), best CS and direction, cross-checked with the `dshared` byte counters. Independent of `../collectives/` | — |
| FP32 CUDA-core peak | FMA-chain kernel, all SMs | 60 TFLOP/s |
| FP16/BF16 tensor-core peak (dense) | instruction throughput: back-to-back `wgmma` from shared memory (`harness/ceilings/src/tc_peak.cu`, R1 correction); cuBLAS GEMMs 2048–8192³ (kernel time) kept as lower references; plus the power-limited rate with random operands | 835.5 TFLOP/s dense (1671 with sparsity), quoted at 1545 MHz |
| FP8 tensor-core peak | as FP16 (`wgmma` e4m3, m64nNk32); cuBLASLt FP8 GEMM (`torch._scaled_mm`) as reference | 1670.5 TFLOP/s dense (3341 with sparsity) |
| INT32 peak | IMAD-chain kernel (for the top-k workload) | — |

Each ceiling row carries `method` and `source` (`measured` or `datasheet`). The datasheet column is the H100 NVL (PNY /
NVIDIA datasheet, per GPU, at the 1785 MHz maximum clock; tensor-core figures there are quoted with sparsity, dense =
half); the first version of this table listed the H100 SXM values (67 / 989 / 1979), corrected in R1. Ridge points
(intensity where compute = memory) are computed in the notebook from the measured values.

### 2.2 Point coordinates
For one variant at one scope and one shape: `y = work / time`, `x_L = work / bytes_L` for level
L ∈ {HBM, L2, L1/SMEM, DSMEM}.
- **work**: FLOPs from the manifest formula (floating-point workloads) or integer ops (`int_ops`,
  top-k). Cross-check with ncu: `sm__sass_thread_inst_executed_op_ffma_pred_on.sum` (×2) +
  `..._fadd` + `..._fmul` (+ `hfma2`/`hadd2` ×2 for fp16 CUDA-core math),
  `sm__inst_executed_pipe_tensor_op_hmma.sum` × FLOP per instruction for tensor cores. A disagreement
  > 10 % between the formula and the counters must be explained in the notebook.
- **bytes_HBM** = `dram__bytes_read.sum + dram__bytes_write.sum`; **bytes_L2** = `lts__t_bytes.sum`;
  **bytes_L1/SMEM** = `l1tex__t_bytes.sum` (+ shared-memory wavefront bytes,
  `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_{ld,st}.sum` × 128 B); **bytes_DSMEM** =
  `l1tex__m_l1tex2xbar_write_bytes_mem_dshared.sum` (+ `_op_tma_st` for bulk pushes) and
  `l1tex__m_xbar2l1tex_read_bytes_mem_dshared.sum`; which of the two carries the payload of DSMEM loads is
  unverified (h100-microbench `docs/context/07` §8) — R1 settles it with its DSMEM microbenchmark and
  records the rule in §6. TMA-multicast traffic (DAK, DeepGEMM) shows up as
  `l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld_dest_multicast.sum`, not as `dshared`. Only rollups
  exist on this setup (no per-SM instances); one `lts__` counter is collected per replay pass.
- **time** = ncu `gpu__time_duration.sum` per kernel with `--clock-control none` under the locked clock, summed over the
  kernels of the variant; a separate timed run (locked clocks, L2 flushed, 20 warm-up + 100 reps,
  cudaEvent or `%globaltimer` inside the kernel) gives the p10/p50/p90 used for error bars.
- **Aggregation over a kernel sequence** (unfused baselines, layer scope): sum work, bytes and kernel
  times; launch gaps are excluded from the point and reported separately as `wall_ns`.
- Compulsory traffic: the manifest's `min_hbm_bytes` (weights + activations that must cross HBM at least
  once) gives the *upper bound* of intensity; the notebook plots it as a faint vertical tick so the
  reader sees how far each variant is from the compulsory bound.

### 2.3 Variants and marker roles
- `role: non_dsmem` — the baseline (unfused kernel sequence, or same kernel at cluster size 1).
- `role: dsmem` — the version using DSMEM collectives / TMA multicast.
- `role: control` (optional) — fused but without DSMEM (e.g. ClusterFusion rebuilt with
  `CLUSTER_SIZE 1`): separates the kernel-fusion effect from the DSMEM effect. Arrows go
  non_dsmem → control → dsmem.
- `role: reference` (optional) — an extra library marker (cuBLAS, cuBLASLt) on the same inputs; drawn, but not part of
  the arrow chain (it is neither the replaced sequence nor a fusion control). Added in R3.
- `source: reported` — for `runnable: false` manifests, coordinates taken from the paper.

## 3. Measurement protocol
1. Ceilings first (§2.1), once per session, with the clock state recorded.
2. Per manifest, per variant, per shape: build (apply `patches/` if referenced), run the validation
   command (outputs must match the reference within the manifest's tolerance), timed run, ncu run.
   Use `ncu --cache-control all --clock-control none --metrics <list> --csv --page raw --print-units base`
   inside `tools/run --profile --lock-sm-mhz <LOCK>` (never `--clock-control base`: it overrides the lock;
   never `--print-metric-name name`: ncu 2025.2.1 rejects it with `--page raw`); for Python-launched
   kernels add `-k regex:<kernel names from the manifest>` and `--launch-skip/--launch-count` to
   capture exactly one iteration of each kernel of the sequence.
3. Append rows to `results/points.csv` (one row per rep for timing; counters replicated across reps or
   stored once with `rep = -1`).
4. Layer scope: run the extra kernels listed in `scope_layer.extra_kernels` the same way and add their
   work/bytes/time to both variants (rows with `scope = layer`).
5. Cross-check: ncu's `SpeedOfLight_RooflineChart` / `SpeedOfLight_HierarchicalTensorRooflineChart`
   sections for the DSMEM kernel must place it within 10 % of the hand-computed point.

## 4. Figures (`analysis/roofline.ipynb`)
1. **Classic roofline**: log-log, x = FLOP per HBM byte, y = TFLOP/s; measured ceilings (FP16-TC, FP8-TC,
   FP32, INT32 as separate horizontal lines; HBM slope), datasheet dashed; one marker per variant,
   arrows non_dsmem → (control →) dsmem; filled = operator scope, hollow = layer scope, star = reported.
2. **Hierarchical roofline**: per workload, the same y against intensity w.r.t. HBM, L2, L1/SMEM and DSMEM
   bytes, with the corresponding bandwidth ceilings — shows the bytes moving off HBM.
3. **Per-workload bars**: time, HBM bytes, DSMEM bytes per variant; fraction of layer time in the
   optimized region; speedup.

## 5. CSV schema
Every file starts with the mandatory columns `gpu_name, sm_count, sm_clock_mhz, mem_clock_mhz,
cuda_version, driver_version, run_id, launch_idx` (`../../docs/ENVIRONMENT.md` §5); the notebook reads the
columns below by name and ignores the others.

`results/ceilings.csv`
```
name,value,unit,method,source
hbm_read_bw,3.35e12,B/s,stream_copy_1GB_locked_clocks,measured
hbm_bw_datasheet,3.9e12,B/s,datasheet,datasheet
fp16_tc_peak,...,FLOP/s,cublas_gemm_8192,measured
```
`results/points.csv`
```
workload,variant_id,role,scope,shape_id,rep,time_ns,wall_ns,flops,int_ops,hbm_bytes,l2_bytes,l1_bytes,dsmem_bytes,kernels_n,source
```
`workload` = manifest `name`; `variant_id` / `role` from the manifest; `scope` ∈ {operator, layer};
`shape_id` = key of the manifest's `shapes` entry; `rep = -1` for counter-only rows; `source` ∈
{measured, reported}.

## 6. Tooling  *(known values pre-filled; run-time values filled by R0/R1 and each workload task)*
- Base image: h100-microbench `h100-microbench:cuda12.9.1-<hash>` (CUDA 12.9.1, nvcc 12.9.86, Nsight
  Compute 2025.2.1, Nsight Systems, Python 3.12); roofline image tag and Dockerfile: `roofline/docker/` (R0).
- Roofline image (R0, 2026-10-26): `h100-roofline:cuda12.9.1-d08c91f3643b` (32.8 GB) `FROM`
  `h100-microbench:cuda12.9.1-26d4f436f8c5`, built by `roofline/tools/build_image.sh build`; additions pinned in
  `roofline/docker/requirements.lock` (intent: `requirements.txt`), CUTLASS 4.4.2 headers at `/opt/cutlass/include`
  (`$CUTLASS_INCLUDE`, copied from `third_party/hpc-ops/3rd/cutlass` at hpc-ops `2a2e265`), `LIBRARY_PATH` = CUDA
  driver stubs, `TORCH_CUDA_ARCH_LIST=9.0a`. JIT caches (`TORCH_EXTENSIONS_DIR`, `TORCHINDUCTOR_CACHE_DIR`,
  `TRITON_CACHE_DIR`, `DG_JIT_CACHE_DIR`, `FLASHINFER_WORKSPACE_BASE`, `XDG_CACHE_HOME`) under
  `roofline/build/cache/`, set by `roofline/tools/run`. Not installed: conda, CUDA 13, fairscale, FA4 dependencies.
- Image revision (R5): `h100-roofline:cuda12.9.1-83734cedf58e` = R0 image + `libdw-dev`/`libelf-dev`
  0.190-1.1ubuntu0.1 (DeepGEMM's deep_jit includes elfutils headers; the runtime library was already present).
- Versions (R0 smoke on GPU 3): driver 580.126.20, CUDA 12.9.1 / nvcc 12.9.86, Nsight Compute 2025.2.1.0,
  Python 3.12.3, PyTorch 2.11.0+cu129 (cuDNN 9.17.1), Triton 3.6.0, flashinfer-python 0.7.2.post1 (JIT kernels).
- Clock state: SM clock locked by `tools/run --profile --lock-sm-mhz <LOCK>` on GPU 3 (value from
  h100-microbench `docs/context/10` §2), memory clock 2619 MHz, never changed. Value used: **1635 MHz** (R1: every ceiling launch at 1635 MHz; sustained
  8192³ tensor-core GEMMs back to back pull the clock below the lock through the 400 W limit, so `ceilings_tc.py` idles
  1 s before every timed GEMM and 10 s before every configuration).
- ncu metric names (R1/R2, all listed by `ncu --query-metrics` of ncu 2025.2.1 on GPU 3): HBM `dram__bytes_read.sum` +
  `dram__bytes_write.sum`; L2 `lts__t_bytes.sum`; L1/SMEM `l1tex__t_bytes.sum` + 128 ×
  `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_{ld,st}.sum`; DSMEM `l1tex__m_l1tex2xbar_write_bytes_mem_dshared.sum`
  (payload rule, verified in R1 on the DSMEM ceiling kernel: pull and bulk push both count each payload byte once in
  `l1tex2xbar_write` and once in `xbar2l1tex_read`, ratio 1.000 for CS 2–16, as collectives C1 found); TMA multicast
  `l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld_dest_multicast.sum`; FP32 work `2·…_op_ffma_pred_on + …_op_fadd_pred_on +
  …_op_fmul_pred_on` (`sm__sass_thread_inst_executed_…`, exact on the FFMA ceiling kernel); integer
  `sm__sass_thread_inst_executed_op_integer_pred_on.sum`; tensor cores `sm__inst_executed_pipe_tensor_op_gmma.sum` (= `…_hmma.sum` on
  SM90 cuBLAS kernels); time `gpu__time_duration.sum`. Not listed (dropped): `launch__grid_size`, `launch__cluster_dim_x`,
  `launch__block_size` as metrics. Write traffic still dirty in L2 at kernel end is missing from `dram__bytes_write`
  (−1.4 % on the 2 GiB write ceiling, more for outputs that fit L2).
- Build commands actually used per artifact (the manifests' `build.cmd`, run by `run_manifest.py --stage build` in the
  per-(submodule, patch) worktree): CREDIT `make all CUDA_ARCH=sm_90a -j8` (+ `patches/credit__roofline_hooks.patch`, R2);
  DAK `export NVCC_APPEND_FLAGS="-I$CUTLASS_INCLUDE" && make runtime.o .build_tile_n arch=90a TILE_N=8 && DAK_ARCH=90a
  DAK_TILE_N=8 MAX_JOBS=8 python3 setup.py build_ext --inplace` (no `setup.sh`/conda; CUTLASS 4.4.2 from the image; R3);
  DeepGEMM: symlink `third-party/{cutlass,deep_jit}` from the submodule into the worktree, link
  `deep_gemm/include/{cutlass,cute}`, `python3 setup.py build`, symlink `_C*.so` into `deep_gemm/` (develop.sh without the
  submodule update; R5); JIT cache `roofline/build/cache/deepgemm` (shared: the cache key includes the cluster size);
  hpc-ops `python3 {harness}/workloads/build_hpcops.py` (setup.py build_ext --inplace, SM_ARCH=90, -j16; R6);
  _others: R4, R7_
