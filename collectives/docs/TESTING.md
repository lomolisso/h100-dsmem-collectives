# TESTING.md — correctness, deadlock and sanity harness for `collectives/` (design, no code)

Roadmap tasks C1 (harness, build matrix), C2–C3 (correctness, §2–3) and C4 (§4–6) implement this design
under `collectives/harness/` (`ROADMAP.md`). Nothing here depends on the server; the server protocol is
`../../docs/ENVIRONMENT.md`, tool versions are recorded in `CLAUDE.md` → *Tooling*. Builds run through
`tools/run --no-gpu`, GPU runs through `tools/run --profile --lock-sm-mhz <LOCK>` on GPU 3; the vendored
microbench harness (`../../common/h100mb/`, `../../docs/REUSE.md`) provides the CSV writer, run registry,
device checks and clock notes.

## 1. Build matrix

One translation unit per (file, N, M, T): `harness/main.cu` does `#include COLLECTIVE_FILE` and is compiled as

```
nvcc -arch=sm_90a -std=c++17 -O3 -lineinfo -Xptxas -v -I ../common/h100mb \
     -DCOLLECTIVE_FILE='"../<file>.cu"' -DCLUSTER_N=<N> -DMSG_ELEMS=<M> -DELEM_T=<T> \
     -o build/<file>__N<N>_M<M>_<T> harness/main.cu -L$CUDA_HOME/lib64/stubs -lcuda -lnvidia-ml
```
(run from `collectives/` inside `tools/run --no-gpu`; `-lnvidia-ml` is needed by the vendored `host.hpp`).

| Axis | Values |
|---|---|
| file | every `collectives/*.cu` (29 files) |
| N | 2, 4, 8, 16 (16 is non-portable: set `cudaFuncAttributeNonPortableClusterSizeAllowed`) |
| M (elements) | 16, 64, 256, 1024, 4096 |
| T | `float` (all files), `int` (all reduce files with `Sum`: exact check) |

Expected compile failures (record them as `SKIP:static_assert` / `SKIP:smem`, never as errors):
- `static_assert` on N (pow2 files at N not pow2 — none in this matrix; `N == 4` and `N ∈ {2,4}` for
  the ClusterFusion shipped variants; `N == 2` for `alltoall__pairwise_exchange__fa4`; `N | M`, `N | 32`).
- Static smem over 48 KB (`ptxas error: ... uses too much shared data`). Static smem per file, in elements of T:
  allreduce files: 2M (one-shot/atomic/replicated: 2M–3M, slots N·M), ring: M + M/N, rabenseifner/recursive
  doubling/shift: 3M; allgather: N·M (+M); reducescatter: N·M + M (recursive halving: 1.5·N·M; direct push: 2·N·M);
  alltoall: 2·N·M (Bruck: 1.5·N·M). Compute the limit from `ptxas -v` output rather than trusting these figures.
  Measured by C1 (`ptxas -v`, `results/build_matrix.csv`): the figures hold except `allreduce__rabenseifner__mpich`
  = 1.5·M (never over the limit in this matrix) and `allreduce__2d_mesh__generic` = 2.5/3.5/2.75/3.25·M for
  N = 2/4/8/16 (over the limit at M = 4096 for N = 4 and 16). The limit is inclusive: exactly 49 152 B compiles
  (e.g. `allgather__direct_pull__generic` N = 2, M = 4096). Extra `static_assert`: `allreduce__shift_accumulate_atomic__clusterfusion`
  requires M ≤ 1024 (one element per thread), so its M = 4096 configurations are `SKIP:static_assert`.

## 2. Host harness (`harness/main.cu`)

1. Allocate `in`, `out`, `ts[N*32]`, `smid[N]` on the device; fill `in` on the host with a deterministic
   pattern (see §3), copy, zero `out`/`ts`.
2. Launch `collective_kernel<<<N, 256>>>` (the `__cluster_dims__` attribute sets the cluster); for N = 16
   use `cudaLaunchKernelEx` with the cluster-dimension attribute after setting the non-portable attribute.
   Check `cudaOccupancyMaxActiveClusters >= 1` first and record it.
3. Watchdog: record a `cudaEvent` after the launch and poll `cudaEventQuery` from the host; if it is not
   done after `TIMEOUT_MS` (default 5000) print `DEADLOCK <file> N M T` plus whatever `ts` stamps are
   readable via a *separate* non-blocking stream copy (they may not be), and `_exit(3)`. The driver
   script runs each configuration as a child process (`subprocess` with its own timeout) so a hang never
   blocks the sweep.
4. Copy `out`, `ts`, `smid` back; compare against the host reference (§3); write one CSV row
   `file,N,M,T,rep,status,max_abs_err,max_rel_err,latency_ns` where `latency_ns = max_r ts[r*32+1] - min_r ts[r*32+0]`,
   preceded by the mandatory columns (`hb::with_mandatory`, `../../docs/ENVIRONMENT.md` §5) and appended to
   `results/correctness.csv` with `hb::CsvWriter` (flush per launch).
5. Repeat the launch `REPS` times (default 200) in the same process without re-initialising `in`, checking
   `out` every time: intermittent failures indicate a race. Re-run the whole configuration 3 times in
   fresh processes to vary SM placement (`smid` differs between launches).

## 3. Reference and tolerance

- Reference computed on the host from the same `in` using the layouts in `dsmem_primitives.cuh`.
- Input pattern A (exact): `in[k] = T((k * 7 + 3) % 17 - 8)` — small integers, exactly representable;
  every reduction result is an integer < 2^24 → **bit-exact** comparison for `float` and `int`,
  regardless of summation order. All-gather / all-to-all are data movement only → always bit-exact.
- Input pattern B (random floats in [-1, 1], fixed seed): reductions compared with
  `|got - ref| <= 1e-5 * (1 + |ref|) * ceil(log2 N)`; this only guards against gross errors, pattern A is
  the real check.
- `allreduce__atomic_all_push__generic` is non-deterministic in float; pattern A still gives exact results.

## 4. Deadlock and race checks

- `compute-sanitizer --tool memcheck` on every passing configuration (catches out-of-range DSMEM
  addresses from wrong `mapa` offsets). `compute-sanitizer` ships in the microbench image; use few reps.
- `compute-sanitizer --tool racecheck` and `--tool synccheck` where the tool supports cluster-scope
  barriers on the installed version; record "unsupported" otherwise.
- Deadlock = watchdog expiry (§2.3). Typical causes to check in the file: an mbarrier armed with the
  wrong byte count (`expect_tx` ≠ bytes actually pushed), parity mismatch, a CTA exiting before its
  peers (missing exit `cluster_sync`), or `bulk_push` with a non-16-byte-aligned size.
- Run with `CUDA_LAUNCH_BLOCKING=1` once per file to rule out host-side ordering effects.

## 5. Sanity checks on the performance side (before the sweep)

- Stamps: every rank must have `ts[1] > ts[0]` and monotone round stamps; `smid` values must be N
  distinct SMs, all inside one GPC (ranks of one cluster never share an SM). Use the **GPU-3** SM→GPC map
  produced by the h100-microbench GPU-3 bring-up (`../../docs/ENVIRONMENT.md` §1), not the GPU-2 one.
  `%globaltimer` ticks every 32 ns: equal stamps within one tick are not a monotonicity violation.
- Traffic: for one mid-size configuration per file compare the DSMEM bytes measured by Nsight Compute
  (`docs/EXPERIMENTS.md` §4) with the per-CTA total in `README.md`; a mismatch > 10 % means the
  README or the implementation is wrong — investigate, do not just annotate. ncu exports only kernel-wide
  rollups (`.sum/.avg/.max/.min`, no per-SM instances): compare `.sum` with N × the per-CTA model and use
  `.max/.avg` for asymmetric (rooted) algorithms; subtract a same-grid baseline launch.
- Flakiness: latency spread (p90/p50) across 200 reps should be < 20 % with locked clocks.

## 6. What "done" means

All 29 files × all legal configurations pass §3 pattern A bit-exactly for 200 reps × 3 processes, no
sanitizer error, no watchdog expiry, and the §5 traffic check matches. List every configuration that was
skipped and why. Then proceed to `docs/EXPERIMENTS.md`.
