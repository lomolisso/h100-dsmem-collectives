# WORKLOADS.md — the DSMEM / non-DSMEM pairs

Facts below were established by reading the artifact code in `../../third_party/` (nothing was run).
A recurring finding: **no artifact ships an "off-chip" variant of its DSMEM collective**. The non-DSMEM
counterpart is always either the repo's own unfused / non-cluster kernels or the same kernel built with
cluster size 1. Manifests live in `../workloads/`.

## 0. Priority order (value vs effort) and roadmap task (`../ROADMAP.md`)
1. `credit_*` (6): one binary per workload, both variants in one run, no Python stack. → **R2**
2. `dak_gemm_multicast`: exact like-for-like pair in one extension. → **R3**
3. `clusterfusion_llama2_7b_decoder`: the headline LLM workload, three variants. → **R4**
4. `deepgemm_fp8_gemm_multicast`: needs a two-line heuristic patch and a rebuild. → **R5**
5. `hpcops_topk_cluster8`: needs the hpc-ops wheel; integer roofline. → **R6**
6. `clusterfusion_rmsnorm`: small memory-bound sanity point. → **R4** (same artifact build as 3)
7. `clusterfusion_deepseek_v2_lite_mla`: the unfused baseline must be written. → **R7**
8. `hpcops_rmsnorm_lamport`: multi-GPU op; only if world size 1 is accepted. → **R6** (same wheel as 5)
9. `fa4_bwd_2cta_ds_exchange`: Blackwell-only, reported from the paper. → **R8**

**Build conventions inside the roofline image** (`../../docs/ENVIRONMENT.md` §3, `../CLAUDE.md` Rules). The
build lines below and in the manifests are the artifacts' own instructions, written without running them;
the workload tasks translate them:
- Every variant is built in its own git worktree of the submodule (`roofline/build/src/<manifest>__<variant>/`),
  with the manifest's patch applied there; `third_party/<repo>` stays clean.
- `pip install -e .` / `pip install dist/*.whl` at run time is lost (`--rm` containers): build in-tree
  (`python setup.py build_ext --inplace`, `make pyext`) or `pip install --no-deps --no-build-isolation
  --prefix roofline/build/pyenv/<manifest>__<variant>`, and select the variant through `PYTHONPATH`.
  Third-party Python packages (flashinfer, …) belong in the image, not in a build step.
- DAK's `setup.sh` installs CUTLASS through conda: use the CUTLASS headers available in the image (or
  `third_party/hpc-ops/3rd/cutlass/include`) instead; no conda in the image unless R0 decides otherwise.
- hpc-ops needs CUDA ≥ 12.9 and C++20 (the image has 12.9.1); ClusterFusion needs CUDA ≥ 12.4.
- JIT caches (DeepGEMM, Triton, torch extensions) go to `roofline/build/cache/` on NFS when they must
  survive between runs.

## 1. ClusterFusion — Llama-2-7B fused decoder (`clusterfusion_llama2_7b_decoder`)
- **Computation** (one token, fp16): RMSNorm → QKV GEMV (4096 → 3·4096) → RoPE → flash-decoding over the
  KV cache (S cached tokens + the new one, × 32 heads × 128) → O-proj (4096 → 4096). FLOPs ≈ 2·4096·12288 +
  4·(S+1)·32·128 + 2·4096² (+ softmax, norm, RoPE: < 1 %); compulsory HBM bytes ≈ 2·(3·4096² + 4096²) +
  2·2·S·4096 (weights + K and V cache); exact formulas in the manifest.
- **DSMEM variant**: `clusterfusion.llama_decoder_layer_sglang` → `LlamaDecoderLayerKernel`
  (`include/H100/llama/kernel_sglang.cuh`), 32 head-clusters × 4 CTAs. Collectives: scalar AllReduce via
  remote atomics, vector AllReduce via bulk push (`../../kernels/fused_decoder_layer__llama2_7b__clusterfusion.cu`).
- **Non-DSMEM variants**:
  - `unfused` (role non_dsmem): the reference sequence in `tests/test_llama.py::llama_decode`:
    `flashinfer.fused_add_rmsnorm` → `nn.Linear` QKV → torch NEOX RoPE →
    `flashinfer.single_decode_with_kv_cache(use_tensor_cores=False)` → `nn.Linear` O-proj.
  - `fused_c1` (role control): the same fused kernel rebuilt with `#define CLUSTER_SIZE 1` in
    `include/H100/llama/config.h` (one CTA per head; ~105 KB smem; the reduce loops run zero rounds, which
    is correct for N = 1). Untested by the authors, and wrong as is (found in R4 by reading the code): the
    RMSNorm prologue of `kernel_sglang.cuh` keeps `x + residual` only in the registers of the last loop
    iteration, which is correct only when `DIM_PER_BLOCK = 4096 / CLUSTER_SIZE = 1024` (cluster 4);
    `patches/clusterfusion__cluster_size_1.patch` also stages it in shared memory (and carries the residual fix
    below). Validate against `unfused`.
  - TileLang port `tilelang_llama.py` accepts `cluster_size=1` but is marked untested; optional.
- **Layer scope**: add the FFN (gate/up/down GEMVs 4096↔11008 + SiLU), residual adds and the second RMSNorm
  from `chat/llama/model.py`; the FFN is identical in both variants.
- **Traps**: dispatchers call `cudaDeviceSynchronize` and re-encode TMA maps per call — time with ncu/nsys;
  only cluster sizes 1 and 4 are correct (2 runs zero reduce rounds, ≥ 8 double-counts);
  `kernel_sglang.cuh` updates `residual` in place from every head cluster, and this is a **race, confirmed on
  GPU 3 in R4**: a CTA that reads `residual` after another cluster's write normalises `2·input + residual`, so on
  each call a few random heads get wrong q/k/v (whole head, k/v rel. L2 ~8 %), at cluster 4 and 1, and repeated
  calls are not bit-identical. Fix used for both fused variants (residual_out, carried by
  `patches/clusterfusion__residual_out_c4_fix.patch` and `patches/clusterfusion__cluster_size_1.patch`): the sum
  goes to a separate `residual_output` (returned in place of `residual`, which is only read), as upstream's own
  `kernel_batch_sglang.cuh` does. A second race, **confirmed on GPU 3 in R4 at cluster 4 only**: the QKV GEMV and
  flash-decoding loops refill TMA buffer `id % 2` before warps 1–3 have finished reading tile `id − 2` from it
  (wrong k/v rows of warps 1–3 in heads 6, 7, 30, 31 in 1–5 of 20 launches; two CTAs share an SM there);
  fixed with one `block.sync()` per loop iteration in both fused builds (0 of 600 launches deviate). The cluster-4
  build also carries a `dsm.cuh` `cluster_reduce` hardening (a CTA could overwrite its bulk-push source before the
  outgoing copy is known to be read; missing proxy/mbarrier-init fences) — required by the PTX memory model, not
  observed. Validation checks 20 launches against the unfused sequence (until 2026-10-26 the wrapper compared
  fused variants with an extra launch of themselves), bit-identical k/v/residual across them, and the residual
  output exactly; upstream
  `tests/test_llama.py` builds the reference KV with `torch.cat` (a full cache copy per call) — the study's
  unfused baseline appends into a preallocated cache instead; build: `setup.py` reads the device architecture
  at import (`harness/workloads/build_clusterfusion.py` stubs it to sm_90 for `--no-gpu` builds).

## 2. ClusterFusion — DeepSeek-V2-Lite fused MLA decoder (`clusterfusion_deepseek_v2_lite_mla`)
- **Computation**: `../../kernels/fused_mla_decoder_layer__deepseek_v2_lite__clusterfusion.cu` docstring
  (hidden 2048, 16 heads, kv_lora_rank 512, rope 64, nope 128, compile-time SEQ_LEN 4096).
- **DSMEM variant**: `clusterfusion.deepseek_decoder_layer` (no Python test exists; nine TMA maps).
- **What it computes** (R7, read from `include/H100/deepseek/kernel.cuh`; details with line numbers in the
  wrapper docstring `../harness/workloads/clusterfusion_deepseek_v2_lite_mla.py`): absorbed MLA decode
  **without the rope score term and without a KV append**. q_pe and k_pe are projected and rotated but never
  read by the attention (the cache TMA boxes cover only the 512 latent columns of the 576-wide rows; the score is
  q_lat · c_kv / √192); the new latent replaces cache row S−1 in registers, the cache is never written. Output:
  `o` [1, 2048] only (fp16 atomicAdd over the 16 heads; no residual). Every head cluster recomputes c_kv | k_pe.
  The k_pe RoPE reads a wrong partner (l.305 misses `+ KV_LORA_RANK`; unobservable).
- **Non-DSMEM**: `unfused` written in R7 (same wrapper): flashinfer rmsnorm → cuBLAS q / kv_a GEMVs →
  flashinfer rmsnorm(c_kv) + RoPE writing the new token into cache row S−1 → cuBLAS bmm W_UK →
  `flashinfer.single_decode_with_kv_cache` at head_dim 512, group 16, K = V = latent cache → bmm W_UV → O-proj.
  flashinfer's MLA wrapper is not used (it always adds q_pe · k_pe). Control `fused_c1`: upstream's
  `CLUSTER_SIZE 1` build is wrong in three places (RMSNorm prologue registers, the Llama bug pattern; only 128 of
  the 512 W_UK outputs; only 128 of the 512 W_UV rows) — fixed in `patches/clusterfusion__deepseek_cluster_size_1.patch`.
  The Llama residual race does not exist here (no residual, inputs only read).
- **Traps**: compile-time `SEQ_LEN 4096` (one shape); only cluster sizes 1 (patched) and 4 are correct (N−2
  reduce rounds); shared-memory races of the shipped kernel, not observed here (pin 20/20) but fixed in both fused
  builds because the same pattern broke the Llama cluster-4 build: TMA double buffers refilled before every warp
  read the previous tile (one `block.sync()` per loop iteration) and CTA 0's q_lat / W_UV writes into
  local_qkv[0..128) racing other warps' reads of that range (`block.sync()` before each write); fused_c4 also
  carries the `dsm.cuh` hardening (spec-required, not observed) — validation checks 20 launches; P·V accumulated in fp16 registers (dominant error: emulated a_lat rel. L2 2.2e-3 at cluster 4,
  4.8e-3 at cluster 1).
- Lowest priority of the ClusterFusion entries.

## 3. ClusterFusion — RMSNorm (`clusterfusion_rmsnorm`)
- `clusterfusion.rmsnorm` (`norm/kernel.cuh`, cluster 2, batch 64 × hidden 8192, compile-time) vs
  `flashinfer.norm.rmsnorm` (`tests/test_norm.py`). Do **not** use a cluster-1 rebuild: the single
  shift-by-1 remote `atomicAdd` would add into the CTA's own sum. Memory-bound sanity point
  (FLOPs = 4 per element: x² accumulated by one FFMA + two FMUL for `x * rms_rcp * w`, `norm/kernel.cuh`;
  bytes ≈ 4 per element).

## 4. CREDIT — six reduction-reuse workloads (`credit_<workload>`)
- **Computation**: row-wise fp32 kernels (`../../kernels/<workload>__reduction_reuse__credit.cu`
  docstrings): layernorm backward, weighted-variance backward, Pearson backward, softmax-logits backward,
  LARS momentum, row-wise int8 quantisation. Rows 4096 (layernorm 2048), N ∈ {4K…64K}.
- **Binaries** (`make all CUDA_ARCH=sm_90a` → `build/<workload>`), CLI: `--csv --rows R --n-values a,b,c
  --cluster-size {2,4,8} --warmup W --iters I --profile-variant all|block|cluster [--no-verify]`. Output
  lines `RESULT,cols,rows,block_name,block_ms,block_gbps,cluster_name,cluster_ms,cluster_gbps,P,speedup`.
- **DSMEM variant**: `cluster_*_staged_kernel` (P CTAs per row, replicated-push AllReduce of the row
  statistics).
- **Non-DSMEM**: `block_*_read_kernel` (one CTA per row, re-reads the row from global memory for every
  reduction pass — the true no-DSMEM twin) and `block_*_staged_kernel` (one CTA stages the whole row in
  smem; only when it fits). The binary's `block_*` column reports the *faster* of the two; record both
  by running `--profile-variant block` and reading `block_name`. Triton and torch.compile baselines via
  `scripts/run_local.py` (optional extra markers).
- **Scope**: operator = whole workload; no layer scope.
- **Traps**: the timing loop never flushes L2 (use ncu `--cache-control all` for counters and add a
  256 MB flush between timed iterations in a wrapper, or accept and state warm-L2 timing); the modeled
  bytes/element (8, 10, 6, 7, 8 ×4 B; quant 9 B) are the compulsory traffic to compare with
  `dram__bytes`; best P must be chosen per N (the paper used the fastest of 2/4/8).

## 5. hpc-ops — cluster-8 exact top-k (`hpcops_topk_cluster8`)
- **Computation**: top-2048 indices of a row of n fp32 logits by radix select
  (`../../kernels/topk_filtered_cluster8__llm_sampling__hpcops.cu`); work measured as integer ops
  (`int_ops`, INT32 ceiling), not FLOPs.
- **DSMEM variant**: `hpc.topk(logits, ke, out, num_valid_rows, top_k=2048)` dispatches to
  `topk_filtered_cluster8_exact_kernel` when m ≤ 8 ∧ n ≥ 32768 or m ≤ 16 ∧ n ≥ 65536
  (`choose_topk2048_plan`, `src/topk/topk_filtered.cu:2860-2894`).
- **Non-DSMEM**: no flag exists. Either pad the row capacity to 17 rows (`ke.numel() = 17`,
  `num_valid_rows = m`) so the dispatcher picks the row-local single-CTA kernel (n ≤ 131072) or the
  4-way KV-split global-workspace kernel (n > 131072), or apply `patches/hpcops_topk__use_cluster_false.patch`
  (`use_cluster = false` at line 2868), the library's own dispatch on pre-SM90 GPUs. Measured in R6: with the patch,
  m = 8/16 and n ≤ 131072 run the same row-local kernel as the padding route (so `pad17` is skipped there), and only
  n = 244650 runs a split — an 8-way split with a 1/64 sample that predicts the threshold and merges through global
  memory (a different algorithm from the cluster's two exact scans, not an "exact 8-way analogue"); padding gives the
  4-way split there (`pad17`, role `reference`).
- Build (R6): `harness/workloads/build_hpcops.py` = the unmodified `setup.py build_ext --inplace` with `SM_ARCH=90` and
  `-j$MAX_JOBS` instead of the hard-coded `-j64`; module `hpc/_C_sm90.abi3.so` in the worktree, `HPC_SM_ARCH=90`.
  Tests only, no benchmark; shapes from `tests/test_topk.py` cluster section: (8, 32768), (16, 65536), (8, 244650).

## 6. hpc-ops — Lamport RMSNorm (`hpcops_rmsnorm_lamport`)
- `rmsNormLamport` is the second kernel of a **multi-GPU** fused AllReduce+RMSNorm; its DSMEM part is a
  replicated-push AllReduce of one float across a cluster of 8/4/2 CTAs per token (only for ≤ 33 tokens).
- Non-DSMEM: same kernel at cluster size 1 (automatic for ≥ 67 tokens, or patch `clusterSize = 8 → 1` at
  `fuse_allreduce_rmsnorm_low_latency.h:474`); or `rmsnorm_fusion=False` + `flashinfer.fused_add_rmsnorm`.
- **Not measurable on one GPU (R6, `runnable: false`):** `entry.cc:132` rejects world sizes outside 2–64; the launcher
  instantiates the two-shot kernel only for 2…64 ranks (`fuse_allreduce_rmsnorm_low_latency.cu:342-363`);
  `rmsNormLamport` has no input of its own — it spins on the Lamport broadcast buffer written by the multimem
  broadcast after the scatter from the other ranks (cu:40-41, 64-95, 114, 175-227). Running it on one GPU would need a
  different launcher (pre-filled buffer and flags), i.e. a different program.

## 7. DAK — multicast GEMM (`dak_gemm_multicast`)
- **Computation**: C[tokens, M] = X[tokens, K] · W (fp16, WGMMA 64×256×16), the OPT/Llama linear layers;
  FLOPs 2·tokens·M·K; compulsory bytes 2·(M·K + tokens·K + tokens·M).
- **DSMEM variant**: `gemv_multicast` (`include/task/gemv_multicast.cuh:186`, cluster 2, rank 0 TMA-multicasts
  8 KB weight tiles to both CTAs, remote mbarrier arrive back-pressure). Python: `offload.runtime.gemv_multicast`
  or `LinearMulticast` with `prefill=False`.
- **Non-DSMEM**: `gemv_multicast_prefill` (`gemv_multicast.cuh:870`) — identical tiles, smem layout, warp
  roles and MMA atom, plain per-CTA TMA instead of multicast. This is what `benchmark/benchmark_linear.py
  --type multicast` actually runs, because `LinearMulticast.prefill` defaults to `True`
  (`app/python/layers/linearLayerMulticast.py:59`). Extra markers: `gemv` (`gemv.cuh:372`, different tiling)
  and cuBLAS `nn.Linear` (`--type vanila`).
- **Run**: `--offload 0` (all weights in HBM; the host-offload path targets GH200 NVLink-C2C), tokens a
  multiple of 512 so both CTAs of a cluster share the weight tile, `num_copy_blocks = 132` (H100 NVL has
  132 SMs; `NUM_SMS` is hard-coded to 132 in `config.cuh`). Build: `bash setup.sh` (conda cutlass),
  `make pyext arch=90a`.

## 8. DeepGEMM — SM90 FP8 GEMM with TMA multicast (`deepgemm_fp8_gemm_multicast`)
- **Computation**: D = A·Bᵀ, fp8 e4m3 inputs with 1D×2D fp32 scales, bf16 output; FLOPs 2MNK; compulsory
  bytes M·K + N·K (fp8) + 2·M·N (bf16) + scales. Dense `fp8_gemm_nt` and grouped
  `m_grouped_fp8_gemm_nt_contiguous` (the MoE path ExpertPlex builds on).
- **DSMEM variant**: cluster 2 with multicast of A or B, chosen by the heuristic
  (`csrc/jit_kernels/heuristics/sm90.hpp:71-75`; only when the grid has > 1 wave; `DG_PRINT_CONFIGS=1`
  prints `cluster_m/cluster_n`).
- **Non-DSMEM**: the same JIT kernel with `kNumTMAMulticast = 1`. No env var exists: apply
  `patches/deepgemm__force_cluster_1.patch` (restrict the `cluster_m/cluster_n` candidate loop to 1 behind a
  `DG_FORCE_CLUSTER_1` env check) and rebuild with `./develop.sh`. cuBLASLt (`cublaslt_gemm_nt`) as extra marker.
- Shapes: `tests/generators.py` — M = 4096, (N,K) ∈ {(4096,7168), (7168,2048), (2112,7168), (24576,1536)};
  grouped (8 groups × 4096) × (4096,4096). `deep_gemm.testing.bench_kineto` flushes L2 (256 MB);
  `DG_USE_NVIDIA_TOOLS=1` for ncu. Initialise `third-party/{cutlass,deep_jit}` first.

## 9. FlashAttention-4 — 2-CTA backward (`fa4_bwd_2cta_ds_exchange`)
- Blackwell only (tcgen05/TMEM). `runnable: false`. Reported markers from the paper: backward TFLOPS on B200
  (Fig. 6, hd 128, causal and non-causal) for the 2-CTA kernel and the 1-CTA kernel (Table 3 gives the
  shared-memory traffic model: 3328 vs 2688 cycles). Intensity w.r.t. HBM from the FA backward
  compulsory traffic (Q, K, V, O, dO, dQ, dK, dV, LSE) at the paper's shapes. Plotted as stars labelled
  "reported (B200)", clearly separated from the H100 measurements.
