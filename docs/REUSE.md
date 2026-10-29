# Code reused from h100-microbench (vendored copy) — inventory and rules

`~/work/fa2026/iacoma/h100-microbench` (below `MB/`) already has a tested harness for this exact device:
crash-safe CSVs with the mandatory columns, a run registry, clock and throttle checks, cluster launches
(including non-portable CS 16), DSMEM/mbarrier/bulk-copy PTX wrappers, an L2 flush, and memory-bandwidth
kernels. Reuse it rather than rewriting it. The collectives' own `collectives/dsmem_primitives.cuh`
stays the primitive layer *inside* the algorithm files; the microbench code is for the harnesses around them.

## 1. Vendoring rule (decided by the user: pinned copy, not a path reference)
- The files are copied once into **`common/h100mb/`** at the repository root, by whichever study's first
  task runs first (collectives C1 or roofline R1); the other task only verifies the copy is there.
- `common/h100mb/PROVENANCE.md` records: the microbench commit hash (`git -C MB rev-parse HEAD`), the
  date, the source path of every copied file, and every local edit (path fixes only).
- Never edit a vendored file beyond the path fixes in §4. Study-specific code goes into
  `<study>/harness/`, which includes `common/h100mb/` headers. To pick up a later microbench fix, re-copy
  the file, update `PROVENANCE.md`, and say so in the commit message.
- Microbench's own rules for its notebooks (no citations, no pass/fail) do **not** carry over; each study's
  `CLAUDE.md` defines its own deliverables. Its CSV and crash-safety rules do (`docs/ENVIRONMENT.md` §5).

## 2. What to copy

| Source (`MB/experiments/…`) | Copy to | What it gives you |
|---|---|---|
| `harness/host.hpp` (457 lines, namespace `hb`) | `common/h100mb/host.hpp` | `CUDA_CHECK`, `NVML_CHECK`, `hb::die`; `iso_now`, `arg_str/arg_int/has_flag`; **`CsvWriter(path, header)`**: O_APPEND, header only into an empty file (dies on a different header), `add` + `flush` under `flock` (per launch), `sync` = fsync (per configuration); **`DeviceInfo::open(enforce)`**: refuses unless H100 NVL, 400 W, ECC on, MIG off and UUID = `H100MB_GPU_UUID` (set by `h100 run`), reads the lock from `H100MB_LOCK_SM_MHZ`, `sample()` → `ClockSample` with `bad()` (throttle mask or > 15 MHz below the lock); **`mandatory_cols()` / `with_mandatory(cols)`**; **`Run(exp, argc, argv)`**: `run_id`, registry `results/<exp>__runs.csv` (`started`/`complete` rows), clock log `results/<exp>__clock_log.csv`, `prefix(launch_idx, sample)`, `clock_note`, `lock_clocks_note` (+ `nvidia-smi -q` log); **`ClusterLaunch`** (`cudaLaunchKernelEx` with cluster dimension + scheduling policy, sets max dynamic smem > 48 KB and non-portable cluster size > 8 automatically; `prepare` for bare timed launches; `max_active_clusters`); `all_distinct`, `distinct_within_clusters(smid, cs)`; `SmClaim`; `L2Flusher(128 MB)`; `warmup(f, n)` |
| `harness/harness.cuh` (289 lines, device side) | `common/h100mb/harness.cuh` | Timers `read_clock64`, `read_globaltimer`, `read_timers_pair`, `read_clock64_after(dep, zero)`; `read_smid`, `read_nsmid`, `read_cluster_regs()` (cluster id, ctarank, nctarank); `mapa_u32`, `cluster_arrive{,_release,_relaxed}`, `cluster_wait{,_acquire}`, `cluster_sync`; mbarrier (`mbar_init`, `mbar_arrive_expect_tx`, `mbar_remote_arrive{,_relaxed}`, `mbar_wait_parity{,_acq_cluster}`, …); bulk copies `bulk_g2s`, `bulk_g2s_multicast`, `bulk_s2g`, `bulk_s2s_dsmem`; `tma_load_{1d,2d,3d}`; fences (`fence_proxy_async_*`, `fence_acq_rel_*`); `st_async_u32`, `red_async_add_u32`; cluster-scope flag loads/stores and atomics. Missing (write locally if needed): `cp.reduce.async.bulk`, v2/v4 `ld/st.shared::cluster` and `st.async`, FP `red.async`, multimem, wgmma/HMMA |
| `harness/Makefile.inc` (34 lines) | `common/h100mb/Makefile.inc` | `NVCCFLAGS = -arch=sm_90a -O3 -lineinfo -std=c++17 -Xptxas -v -Xptxas -warn-spills`; links `-lcuda -lnvidia-ml` through the toolkit stubs; `bin/<name>` from `src/<name>.cu` for each of `BINS`; `sass` target (`cuobjdump -sass` → `results/sass/`) |
| `07__synchronization_primitives/src/sync_common.hpp` | copy the needed pieces into `<study>/harness/` | **`ClusterOnce(cs, block, smem)`**: one cluster per launch with an explicit cluster attribute — the collectives' launch pattern; `bar_sync0_then_lds` (a clock read right after `bar.sync` can precede the barrier's completion: make t0 depend on a shared load after it); `ranks_distinct`; `DevBuf<T>` |
| `05__memory_hierarchy_bandwidth/src/bw_common.hpp` | roofline `harness/ceilings/` | **`CleanL2`** (`L2Flusher` + read of a clean 128 MB buffer: no dirty write-backs in the next timed kernel); `ld_v4<C>`/`st_v4<C>` with explicit cache operators; `BlockRec`; `PlacementCheck` |
| `05__memory_hierarchy_bandwidth/src/bw_dram.cu` (`k4<OP,C,ILP>`), `bw_l2.cu` (`k3_direct<ILP>`; skip the home-labelling machinery), `bw_smem_l1.cu` (`k12<…>`) | roofline `harness/ceilings/` | The HBM, L2 and SMEM/L1 ceiling kernels and their launch configurations (4 × 132 × 1024 threads for HBM; GPU-wide rate net of the 3.9 µs cudaEvent floor; SMEM from the 4096 → 8192-step slope) |
| `05__memory_hierarchy_bandwidth/src/run.sh`, `post_ncu.py`, `common05.py` | `<study>/harness/` | Runner template: `clockcheck` before/after, `available()` filter against `ncu --query-metrics`, ncu `-o` + `ncu -i … --csv --page raw --print-units base`, join of the ncu CSV to a binary-written launch manifest by launch order; Python `append_rows` (crash-safe), `latest_run`, `complete_runs`, `event_floor_ms` |
| `01__timers_and_harness/src/harness_selftest.cu` | reference only | Smoke tests T1–T7 (cluster + DSMEM, mbarrier + bulk, s2s DSMEM, remote arrive, multicast, flusher): run once on GPU 3 to validate the vendored copy |
| `MB/tools/export_results_summary.py` | optional, adapt | Notebook → plain-text summary for git readers (hardcodes microbench's layout and AGENTS.md table) |

## 3. Known numbers on this device type (agent-only sanity references; measured on GPU 2)
Use them to judge plausibility, not as expectations, and re-check card-specific ones on GPU 3
(`MB/AGENTS.md` rows 01–08 hold the details):
- `%globaltimer` ticks every **32 ns** (cross-SM offsets ≤ 1 tick); a cudaEvent pair alone costs 3.9 µs;
  an empty-kernel launch ≈ 7–8 µs host-bound.
- Cluster barrier: `cluster.sync()` 328–360 cycles (≤ 128 threads/CTA; 640–700 at 1024); signal propagation
  64 ns relaxed / 224 ns release; every release is a GPU-scope `MEMBAR` (264–284 cycles, ≈ 520 after a store).
- CS-2 hand-off round trips: `st.async`/`red.async` + mbarrier **565 cycles** (160 ns one way, cheapest
  correct hand-off), relaxed remote arrive 517, 2 × `cluster.sync` 712, store + volatile poll 754.
- mbarrier: init 32, arrive 29, `arrive.expect_tx` 38, completed-phase try_wait ≈ 30–39 cycles; an
  incomplete `try_wait` returns false after ≈ 4.75 µs.
- Bulk copy: ≈ 340 cycles + bytes / 107 B/cycle; issue 48 cycles per command per warp.
- Placement (GPU 2 card): 8 GPCs of 12/16/16/16/18/18/18/18 SMs; SMs 124–131 never join clusters ≥ 4; only
  7 GPCs host CS 16; max active clusters at 1 CTA/SM: CS 2 → 66, 4 → 30, 8 → 15, 16 → 7; a lone CS-2 cluster
  lands on SMs 124/125, CS 4/8 in GPC 0, CS 16 in GPC 1; ranks 2k, 2k+1 share a TPC.
- Memory (GPU 2, 1635 MHz): SMEM 128 B/cycle/SM (27.5 TB/s for 132 SMs), L2 read 7.7 TB/s, L2 store
  4.0 TB/s, per GPC ≤ 1.16 TB/s, HBM read 3.70 / write 3.62 TB/s.

## 4. Path fixes when vendoring (the only allowed edits; list them in PROVENANCE.md)
- `Makefile.inc` uses `-I../harness` and `../harness/*.cuh` prerequisites relative to an experiment
  directory: point them at `common/h100mb/` (e.g. a variable `H100MB_INC := $(abspath …/common/h100mb)`).
- `bw_common.hpp` includes `../../04__memory_hierarchy_latency/src/gpc_map.hpp`, which reads 03's map by a
  relative path; drop the include if the GPC map is not needed, or pass the absolute path of the GPU-3 map
  (`MB/experiments/03__sm_gpc_topology/results/03__sm_gpc_topology__map.csv`, which exists only in the local
  checkout).
- `hb::Run` writes into `./results/` of the current directory: run binaries from `<study>/` (or adapt in the
  study harness, not in the vendored file).
- `DeviceInfo::open(true)` enforces the H100 NVL / 400 W / ECC / MIG / UUID configuration — keep it enabled.
