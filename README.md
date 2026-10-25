# h100-dsmem-collectives

Catalogue of collective-communication algorithms implemented over Hopper Distributed Shared Memory
(DSMEM, thread-block clusters) and of the LLM kernels that rely on them. Companion to a study of
3D-stacked GPCs where the SM-to-SM network is the stacking target.

| Directory | Content |
|---|---|
| `collectives/` | One CUDA file per algorithm (`<op>__<alg>__<src>.cu`) for AllReduce, AllGather, ReduceScatter, AlltoAll; shared `dsmem_primitives.cuh`; `README.md` index with rounds / message sizes / traffic / primitives; `CLAUDE.md` + `docs/` + `analysis/` for the H100 NVL verification and benchmarking study |
| `kernels/` | Docstring + verbatim extraction of the workload kernels that use these collectives (ClusterFusion, CREDIT, hpc-ops, DAK, DeepGEMM, FlashAttention-4) |
| `roofline/` | Design, workload registry (`workloads/*.yaml`) and notebook for the H100 NVL roofline study of each workload's DSMEM vs non-DSMEM version; `CLAUDE.md` briefs the agent that runs it |
| `third_party/` | The artifact repositories as pinned submodules (`git submodule update --init`; `hpc-ops` and `flash-attention` are shallow) |

Build target: `nvcc -arch=sm_90a -std=c++17`. Nothing in this repository has been compiled or run yet;
that is the job described in `collectives/CLAUDE.md`.
