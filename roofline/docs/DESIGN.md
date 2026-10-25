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
Written to `results/ceilings.csv` (schema §5) by microbenchmarks the agent writes under `harness/`:

| Ceiling | Method | Datasheet reference (H100 NVL, per GPU) |
|---|---|---|
| HBM bandwidth | STREAM-like copy/triad kernels over ≥ 1 GB, locked clocks; report read, write and copy GB/s | 3.9 TB/s HBM3 |
| L2 bandwidth | repeated reads of a 16–24 MB working set (fits the 50 MB L2), all SMs | — |
| L1/SMEM bandwidth | per-SM shared-memory read microbenchmark × SM count (132) | 33 TB/s class |
| DSMEM bandwidth | `../collectives/` results: aggregate injection bandwidth of `allgather__direct_pull`, `alltoall__direct_push` at N = 2…16, best case, × (132 / N) clusters | — |
| FP32 CUDA-core peak | FMA-chain kernel, all SMs | 67 TFLOP/s |
| FP16/BF16 tensor-core peak (dense) | large cuBLAS GEMM (8192³) best of 10 | 989 TFLOP/s dense |
| FP8 tensor-core peak | DeepGEMM / cuBLASLt FP8 GEMM best of 10 | 1979 TFLOP/s dense |
| INT32 peak | IMAD-chain kernel (for the top-k workload) | — |

Each ceiling row carries `method` and `source` (`measured` or `datasheet`). Ridge points
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
  **bytes_L1/SMEM** = `l1tex__t_bytes.sum` (+ shared-memory wavefront bytes); **bytes_DSMEM** = the
  distributed-shared-memory counters identified in `../collectives/docs/EXPERIMENTS.md` §4.
- **time** = ncu `gpu__time_duration.sum` per kernel with `--clock-control base`, summed over the
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
- `source: reported` — for `runnable: false` manifests, coordinates taken from the paper.

## 3. Measurement protocol
1. Ceilings first (§2.1), once per session, with the clock state recorded.
2. Per manifest, per variant, per shape: build (apply `patches/` if referenced), run the validation
   command (outputs must match the reference within the manifest's tolerance), timed run, ncu run.
   Use `ncu --cache-control all --clock-control base --metrics <list> --csv`; for Python-launched
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

## 6. Tooling  *(to be filled by the remote agent)*
- Driver / CUDA / nvcc / Nsight Compute / PyTorch / flashinfer / Triton versions: _TODO_
- Clock state and how it was locked: _TODO_
- Exact ncu metric names used for each byte level (after `ncu --query-metrics`): _TODO_
- Build commands actually used per artifact: _TODO_
