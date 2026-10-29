# h100-dsmem-collectives

Catalogue of collective-communication algorithms implemented over Hopper Distributed Shared Memory
(DSMEM, thread-block clusters) and of the LLM kernels that rely on them. Companion to a study of
3D-stacked GPCs where the SM-to-SM network is the stacking target.

| Directory | Content |
|---|---|
| `collectives/` | One CUDA file per algorithm (`<op>__<alg>__<src>.cu`) for AllReduce, AllGather, ReduceScatter, AlltoAll; shared `dsmem_primitives.cuh`; `README.md` index with rounds / message sizes / traffic / primitives; `CLAUDE.md` + `ROADMAP.md` + `docs/` + `analysis/` for the H100 NVL verification and benchmarking study |
| `kernels/` | Docstring + verbatim extraction of the workload kernels that use these collectives (ClusterFusion, CREDIT, hpc-ops, DAK, DeepGEMM, FlashAttention-4) |
| `roofline/` | Design, workload registry (`workloads/*.yaml`) and notebook for the H100 NVL roofline study of each workload's DSMEM vs non-DSMEM version; `CLAUDE.md` + `ROADMAP.md` brief the agents that run it |
| `third_party/` | The artifact repositories as pinned submodules (`git submodule update --init`; `hpc-ops` and `flash-attention` are shallow) |
| `CLAUDE.md` | Entry point for Claude Code agents: which study, how a conversation starts and ends, shared rules |
| `docs/` | `ENVIRONMENT.md` (H100 server protocol, `h100 run`, Docker images, ncu and raw-data rules) and `REUSE.md` (code reused from `h100-microbench`, vendored into `common/h100mb/` by the first task) |

Build target: `nvcc -arch=sm_90a -std=c++17`. Nothing in this repository has been compiled or run yet.
The two studies are independent roadmaps for Claude Code agents: start with `CLAUDE.md`, then
`collectives/ROADMAP.md` or `roofline/ROADMAP.md`.
