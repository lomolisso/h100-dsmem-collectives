# kernels/ — workload kernels that rely on DSMEM collectives

Each file is a standalone extraction of one kernel from the artifact repositories under `third_party/`,
prefixed with a docstring that explains (1) the computation and its shapes, (2) how it is split across the
N CTAs of a thread-block cluster, and (3) which collective resolves each inter-CTA dependency.
Lines that touch a cluster primitive are tagged `// [DSMEM] ...`. Code below the docstring is verbatim.

| File | Source | Cluster | Collectives used |
|---|---|---|---|
| `fused_decoder_layer__llama2_7b__clusterfusion.cu` | ClusterFusion `H100/llama/kernel.cuh` + `dsm.cuh` | 4 | AllReduce scalar (remote atomicAdd / fmaxf store, shift schedule), AllReduce vector (bulk push + mbarrier) |
| `fused_mla_decoder_layer__deepseek_v2_lite__clusterfusion.cu` | ClusterFusion `H100/deepseek/kernel.cuh` | 4 | same + AllGather (direct shift bulk push) |
| `rmsnorm__llama__clusterfusion.cu` | ClusterFusion `H100/norm/kernel.cuh` | 2 | AllReduce scalar (1 round remote atomicAdd) |
| `layernorm_backward__reduction_reuse__credit.cu` and 5 siblings (`weighted_var_backward`, `pearson_backward`, `softmax_logits_backward`, `lars_momentum`, `rowwise_quant`) | CREDIT `src/workloads/*.cu` | 2/4/8 | AllReduce of 1–5 scalars by direct replicated push (`map_shared_rank` + store, 2 `cluster.sync`) |
| `topk_filtered_cluster8__llm_sampling__hpcops.cu` | hpc-ops `src/topk/topk_filtered.cu` | 8 | AllReduce of a 2048-bin histogram (pull reduce-scatter + push all-gather); rooted gather to rank 0 (remote atomics, loads, stores) |
| `rmsnorm_lamport__fused_allreduce_rmsnorm__hpcops.cu` | hpc-ops `src/allreduce/fuse_allreduce_rmsnorm_low_latency.cu` | ≤8 | AllReduce scalar by replicated push; multi-GPU parts marked out of scope |
| `gemm_tma_multicast__opt_llama_linear__dak.cu` | DAK `include/task/gemv_multicast.cuh` | 2 | rooted: TMA multicast broadcast of weight tiles, remote mbarrier arrive back-pressure |
| `fp8_gemm_1d2d_tma_multicast__deepseek_v3_moe__deepgemm.cu` | DeepGEMM `impls/sm90_fp8_gemm_1d2d.cuh` | 2 | rooted: TMA multicast of A or B tiles; pairwise remote arrives on empty barriers |
| `flash_bwd_2cta_ds_exchange__fa4.cu` (+ `.excerpt.py`) | FlashAttention-4 `flash_bwd_sm100.py` (CuTe-DSL, sm100) | 2 | AlltoAll pairwise exchange of half of dS (remote expect_tx + bulk push), relay arrive on leader |

Build notes (the collectives roadmap compiles only `collectives/`; the roofline roadmap builds these workloads from `third_party/`, not from these files; the notes are for reference):
- ClusterFusion, CREDIT, FA4 port: `nvcc -arch=sm_90a -std=c++17 -lcuda`.
- hpc-ops files: add `-I third_party/hpc-ops -std=c++20` (they include `src/topk/*.h`, `src/utils/*`).
- DAK: `-I third_party/hpc-ops/3rd/cutlass/include -DDAK_SM_ARCH=90 -std=c++20` (CuTe/CUTLASS).
- DeepGEMM: `-I third_party/DeepGEMM/deep_gemm/include -I third_party/hpc-ops/3rd/cutlass/include -std=c++20`;
  the kernel is normally JIT-instantiated by DeepGEMM (launch attribute `cluster_dim` set by its runtime).
- The FA4 kernel is Blackwell-only in origin; the `.cu` here is a Hopper-runnable port of the exchange step
  with the UMMAs replaced by a plain smem reduction, the `.excerpt.py` holds the verbatim original lines.
