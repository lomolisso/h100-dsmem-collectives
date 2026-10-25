# collectives/ — DSMEM collective algorithms for one H100 thread-block cluster

One algorithm per file, `<op>__<alg>__<src>.cu`. Every file includes `dsmem_primitives.cuh`
(documented wrappers for `mapa`, `ld/st/red.shared::cluster`, mbarrier, `cp.async.bulk` smem→smem,
`st.async`, `barrier.cluster`, `fence.proxy.async`) and exposes the same kernel:

```
__global__ void __cluster_dims__(CLUSTER_N,1,1)
collective_kernel(const ELEM_T* in, ELEM_T* out, unsigned long long* ts, unsigned* smid);
```
with compile-time knobs `-DCLUSTER_N= -DMSG_ELEMS= -DELEM_T=` (defaults 4, 1024, float), one cluster per
launch, per-rank input/output layouts and timestamp slots defined at the bottom of `dsmem_primitives.cuh`
(`ts[rank*32+0]` entry, `+1` exit, `+2+r` end of round r; `smid[rank]`). `M` below is `MSG_ELEMS`, the
per-CTA payload for allreduce/allgather and the chunk size for reducescatter/alltoall. `B = sizeof(ELEM_T)`.

| File | Op | Algorithm / logical topology | Rounds | Bytes per message | Total DSMEM bytes per CTA | Cluster barriers | Primitive | Source | N constraint |
|---|---|---|---|---|---|---|---|---|---|
| `allreduce__recursive_doubling__clusterfusion.cu` | AllReduce | ring, shift doubles (1,2,4,…), full vector each round | log2 N | M·B | M·B·log2 N | log2 N | bulk push + mbarrier | ClusterFusion paper Alg. 1 | pow2 |
| `allreduce__shift_accumulate_bulk__clusterfusion.cu` | AllReduce | ring, shift i = 1..N-2, running buffer forwarded | N-2 | M·B | (N-2)·M·B | N-2 | bulk push + mbarrier | ClusterFusion `dsm.cuh` (shipped) | N = 4 |
| `allreduce__shift_accumulate_atomic__clusterfusion.cu` | AllReduce | same shift schedule, scalar elements | N-2 (1 for N=2) | B per element | (N-2)·M·B | 2 per round | `map_shared_rank` + remote `atomicAdd` / `fmaxf` store | ClusterFusion `kernel.cuh`, `norm/kernel.cuh` | N ∈ {2,4} |
| `allreduce__direct_replicated_push__credit.cu` | AllReduce | fully connected, every CTA stores into every peer's slot, local reduce | 1 | M·B (N-1 msgs) | (N-1)·M·B | 2 | `map_shared_rank` + plain remote store | CREDIT `cluster_reduce_pair`; hpc-ops `rmsNormLamport` | any |
| `allreduce__pull_reduce_scatter_push_allgather__hpcops.cu` | AllReduce | fully connected: pull owned chunk from all peers, shuffle-reduce, push back | 2 phases, 1 round each | M/N·B | 2(N-1)/N·M·B (half reads, half writes) | 2 | `map_shared_rank` load/store + `__shfl_xor_sync` | hpc-ops `cluster_allreduce_histogram` | N \| 32, N \| M |
| `allreduce__ring__nccl.cu` | AllReduce | ring reduce-scatter + ring all-gather | 2(N-1) | M/N·B | 2(N-1)/N·M·B | 2(N-1) | bulk push + mbarrier | NCCL ring | N \| M |
| `allreduce__rabenseifner__mpich.cu` | AllReduce | hypercube: recursive halving RS + recursive doubling AG | 2 log2 N | M/2 … M/N … M/2 ·B | 2(N-1)/N·M·B | 2 log2 N | bulk push + mbarrier | MPICH / Rabenseifner | pow2 |
| `allreduce__one_shot_pull__trtllm.cu` | AllReduce | fully connected pull, local reduce | 1 | — (reads) | (N-1)·M·B reads | 2 | `mapa` + `ld.shared::cluster.v4` | TensorRT-LLM one-shot | any |
| `allreduce__two_shot_push__trtllm.cu` | AllReduce | direct RS push, then direct AG push | 2 | M/N·B (N-1 msgs each) | 2(N-1)/N·M·B | 0 (+exit) | bulk push, aggregate mbarrier | TensorRT-LLM two-shot | N \| M |
| `allreduce__reduce_bcast_binomial__mpich.cu` | AllReduce | rooted: binomial reduce to 0 + binomial bcast | 2 log2 N | M·B | uneven; 2(N-1)·M·B cluster-wide | 2 log2 N | bulk push + mbarrier | MPICH short-message | pow2 |
| `allreduce__atomic_all_push__generic.cu` | AllReduce | fully connected remote atomics into every peer | 1 | B per element | (N-1)·M·B | 2 | `mapa` + `red.shared::cluster.add.f32` / remote `atomicAdd` | generic | any |
| `allreduce__2d_mesh__generic.cu` | AllReduce | R×C grid: row RS push, column AR push, row AG push | 3 | M/C·B | (2(C-1)+(R-1))/C·M·B | 0 (+exit) | bulk push, per-phase mbarrier | 2D torus (TPU/Trainium style) | 16→4×4, 8→2×4, 4→2×2, 2→1×2 |
| `allgather__recursive_doubling__clusterfusion.cu` | AllGather | ring, shift doubles, block doubles | log2 N | 2^k·M·B | (N-1)·M·B | log2 N | bulk push + mbarrier | ClusterFusion paper Alg. 2 | pow2 |
| `allgather__direct_shift__clusterfusion.cu` | AllGather | ring, shift i, own segment only | N-1 | M·B | (N-1)·M·B | 2 per round | bulk push + mbarrier (re-init per round) | ClusterFusion `dsm.cuh` QUK stage (shipped) | any |
| `allgather__direct_replicated_push__credit.cu` | AllGather | fully connected push | 1 | M·B (N-1 msgs) | (N-1)·M·B | 1 (+exit) | `map_shared_rank` + remote store | CREDIT | any |
| `allgather__ring__nccl.cu` | AllGather | ring forward | N-1 | M·B | (N-1)·M·B | N-1 | bulk push + mbarrier | NCCL | any |
| `allgather__bruck__mpich.cu` | AllGather | ring, shift doubles, prefix sent, final un-rotation | ⌈log2 N⌉ | min(2^k, N-2^k)·M·B | (N-1)·M·B | ⌈log2 N⌉ | bulk push + mbarrier | Bruck / MPICH | any |
| `allgather__direct_pull__generic.cu` | AllGather | fully connected pull | 1 | — (reads) | (N-1)·M·B reads | 2 | `mapa` + `ld.shared::cluster.v4` | generic | any |
| `allgather__gather_bcast_binomial__mpich.cu` | AllGather | rooted: binomial gather + binomial bcast of full buffer | 2 log2 N | d·M·B then N·M·B | cluster-wide (N-1)·M·B + (N-1)·N·M·B | 2 log2 N | bulk push + mbarrier | MPICH | pow2 |
| `reducescatter__ring__nccl.cu` | ReduceScatter | ring | N-1 | M·B | (N-1)·M·B | N-1 | bulk push + mbarrier | NCCL | any |
| `reducescatter__recursive_halving__mpich.cu` | ReduceScatter | hypercube, halves each round | log2 N | N·M/2 … M ·B | (N-1)·M·B | log2 N | bulk push + mbarrier | MPICH | pow2 |
| `reducescatter__direct_push__generic.cu` | ReduceScatter | fully connected push | 1 | M·B (N-1 msgs) | (N-1)·M·B | 0 (+exit) | bulk push, aggregate mbarrier | generic (two-shot phase 1) | any |
| `reducescatter__direct_pull__generic.cu` | ReduceScatter | fully connected pull | 1 | — (reads) | (N-1)·M·B reads | 2 | `mapa` + `ld.shared::cluster.v4` | generic (hpc-ops phase 1) | any |
| `alltoall__pairwise_exchange__fa4.cu` | AlltoAll | pair swap of halves | 1 | M·B | M·B | 0 (+exit) | remote `mbarrier.arrive.expect_tx` + bulk push | FlashAttention-4 dS exchange | N = 2 |
| `alltoall__direct_push__generic.cu` | AlltoAll | fully connected push | 1 | M·B (N-1 msgs) | (N-1)·M·B | 0 (+exit) | bulk push, aggregate mbarrier | generic | any |
| `alltoall__pairwise_xor__mpich.cu` | AlltoAll | hypercube matchings, partner b^i | N-1 | M·B | (N-1)·M·B | N-1 | bulk push + mbarrier | MPICH long-message | pow2 |
| `alltoall__shift__generic.cu` | AlltoAll | ring, shift i | N-1 | M·B | (N-1)·M·B | N-1 | bulk push + mbarrier | generic | any |
| `alltoall__bruck__mpich.cu` | AlltoAll | ring, shift doubles, half the blocks per round, rotations | log2 N | N/2 blocks of M·B | N/2·log2 N·M·B | 2 per round | bulk pushes, aggregate mbarrier | Bruck / MPICH short-message | pow2 |
| `alltoall__direct_pull__generic.cu` | AlltoAll | fully connected pull | 1 | — (reads) | (N-1)·M·B reads | 2 | `mapa` + `ld.shared::cluster.v4` | generic | any |

Notes
- "Cluster barriers" counts `barrier.cluster` inside the collective; every kernel adds one entry and one
  exit barrier (the exit one keeps DSMEM alive until every peer is done).
- Push variants use one thread per CTA to issue `cp.async.bulk` and signal the receiver's mbarrier; pull
  variants use all threads with 16-byte `ld.shared::cluster` loads; scalar variants use generic pointers
  from `cluster.map_shared_rank`.
- Reduce-based files take a reduction functor (`Sum`, `Max`); the harness instantiates `Sum`.
- `N` constraints are enforced with `static_assert` and show up as expected compile failures in the sweep.
- Rooted building blocks seen in the workloads (TMA multicast broadcast in DAK/DeepGEMM, leader-CTA
  broadcast in ExpertPlex) are documented in `kernels/` and not duplicated here.
