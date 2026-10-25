# CLAUDE.md — H100 verification and benchmarking of `collectives/`

> **IMPORTANT — READ FIRST.** The Claude Code agent that runs this study on the shared H100 NVL
> server MUST, before compiling or launching anything, update this file (section *H100 NVL server
> protocol*) and `docs/EXPERIMENTS.md` (section *Tooling*) with the server's usage rules and the
> tooling it sets up. The server is a shared resource with restrictions and particularities that are
> not described here; discover them (ask the operator, read the server's own documentation, check
> `nvidia-smi`, scheduler/queue tools, `/etc/motd`, module system) and record them so later sessions
> follow the same protocol. Every GPU run must comply with what you write there.

## Scope

Only `collectives/` is in scope: every `<op>__<alg>__<src>.cu` must (1) compile for `sm_90a`,
(2) produce correct results for every legal configuration, (3) never deadlock or race, and (4) be
benchmarked for latency and bandwidth per `docs/EXPERIMENTS.md`, with figures in
`analysis/dsmem_collectives.ipynb`. `kernels/` and `third_party/` are reference material and are not
compiled here.

## H100 NVL server protocol  *(to be filled by the remote agent)*

- Access / login, allowed hours, who else uses the GPU: _TODO_
- How to check the GPU is free and how to reserve it (queue, lock file, `nvidia-smi`): _TODO_
- Limits (max run time, memory, concurrent processes, ncu permission / `--clock-control`): _TODO_
- Where to build and write results (scratch paths, quotas): _TODO_
- Clean-up obligations: _TODO_

## Tooling  *(to be filled by the remote agent; mirror in docs/EXPERIMENTS.md)*

- Driver / CUDA toolkit / `nvcc` version, Nsight Compute version, Python environment: _TODO_
- Exact build and run commands used: _TODO_

## File contract (what the harness may rely on)

- Every file includes `dsmem_primitives.cuh` and defines
  `__global__ void __cluster_dims__(CLUSTER_N,1,1) collective_kernel(const ELEM_T* in, ELEM_T* out, unsigned long long* ts, unsigned* smid)`.
- Compile-time knobs: `-DCLUSTER_N=<2|4|8|16> -DMSG_ELEMS=<M> -DELEM_T=<float|int>`.
  Launch one cluster: `gridDim = CLUSTER_N`, `blockDim = 256` (any multiple of 32 up to 1024).
  `CLUSTER_N=16` needs `cudaFuncSetAttribute(..., cudaFuncAttributeNonPortableClusterSizeAllowed, 1)`.
- Per-rank layouts (see the bottom of `dsmem_primitives.cuh`):
  allreduce `in/out[b*M .. b*M+M)`; allgather `in[b*M..)`, `out[b*N*M + r*M]`;
  reducescatter `in[b*N*M + j*M]`, `out[b*M..)`; alltoall `in[b*N*M + j*M]` → rank j, `out[b*N*M + r*M]`.
- Timestamps: `ts[b*32 + 0]` entry (after the entry `cluster_sync`), `ts[b*32 + 1]` exit, `ts[b*32 + 2 + r]`
  end of round r (where the algorithm has rounds); `smid[b]` = SM id of rank b. All in `%globaltimer` ns.
- `static_assert`s encode legal configurations (power-of-two N, N == 4, N == 2, N | M, 16-byte multiples).
  A `static_assert` failure is an *expected* compile failure for that configuration, not a bug.
- Static shared memory is used; configurations whose arrays exceed 48 KB fail in `ptxas` and are
  skipped (`docs/TESTING.md` gives the per-file formula). Do not convert files to dynamic smem unless the
  user asks.

## Workflow

1. Fill in the two TODO sections above. Record the commit hash of this repo you are testing.
2. Build the harness described in `docs/TESTING.md` under `collectives/harness/` (new directory; the
   algorithm files must stay untouched except for genuine bug fixes).
3. Run the correctness matrix; then the deadlock/race checks; then the performance sweep from
   `docs/EXPERIMENTS.md`; then execute the notebook.
4. If an algorithm file is wrong, fix the smallest thing that makes it correct, keep the algorithm's
   pattern (rounds, partners, message sizes) unchanged, and log the fix in the *Change log* below with
   the failing configuration and the root cause. Never "fix" a failure by skipping it, loosening a
   tolerance beyond what `docs/TESTING.md` allows, or adding barriers that change the algorithm's
   round structure without saying so in the docstring.
5. Report: which files pass which configurations, which were fixed and how, where the results and
   figures live, and anything that could not be measured and why.

## Rules of thumb

- A hung launch cannot be killed from inside the process: run every configuration in a child process
  with a timeout (see `docs/TESTING.md`) so one deadlock does not take down the sweep.
- Use `%globaltimer`-based latency from the kernel's own stamps for the figures; host-side
  `cudaEvent` time includes launch overhead and is reported separately.
- Lock clocks (or use `ncu --clock-control base`) for the performance sweep; note whether you could.
- Keep raw CSVs under `collectives/results/` (git-ignored except the final summary files you decide to commit).

## Change log  *(append entries here)*

- _none yet_
