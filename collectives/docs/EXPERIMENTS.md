# EXPERIMENTS.md — latency and bandwidth study of DSMEM collectives on an H100 NVL (design)

Runs only after `TESTING.md` is green (roadmap task C5; figures in C6, `../ROADMAP.md`). Produces
`collectives/results/*.csv`, consumed by `analysis/dsmem_collectives.ipynb`. Environment, clock lock and
ncu rules: `../../docs/ENVIRONMENT.md`.

## 1. Tooling  *(known values pre-filled; run-time values filled by C1/C5)*

- Toolchain: h100-microbench image, CUDA 12.9.1 (nvcc 12.9.86), Nsight Compute 2025.2.1; driver and image
  tag: see `CLAUDE.md` → *Tooling* (C1).
- Clock state: SM clock locked by `tools/run --profile --lock-sm-mhz <LOCK>` (GPU 3; value from
  h100-microbench `docs/context/10` §2), memory clock 2619 MHz never changed; the harness logs clocks and
  throttle reasons per configuration (`results/collectives__clock_log.csv`). Value used: _TODO (C5)_
- Exact commands used for the sweep and for `ncu`: _TODO (C5)_
- DSMEM metric names (candidates verified for gh100 by h100-microbench `docs/context/07` §2; confirm with
  `ncu --query-metrics` in C1 and record the final list here): outbound link bytes
  `l1tex__m_l1tex2xbar_write_bytes_mem_dshared.sum` (+ `_op_st`, `_op_atom`, `_op_redas`, and
  `l1tex__m_l1tex2xbar_write_bytes_mem_dshared_op_tma_st.sum` for bulk pushes), inbound
  `l1tex__m_xbar2l1tex_read_bytes_mem_dshared.sum` (+ `_op_…`), request side
  `l1tex__t_{requests,sectors,bytes}_pipe_lsu_mem_dshared(_op_{ld,st,atom,redas}).sum`, instruction counts
  `sm__sass_inst_executed_op_dshared_{ld,st,atom,redas,stas,tma_st,tma_red}.sum`, link utilisation
  `l1tex__m_{l1tex2xbar,xbar2l1tex}_throughput_pipe_lsu_mem_dshared.avg.pct_of_peak_sustained_elapsed`,
  cluster `launch__cluster_*`, `gpc__cgas_launched.sum`. ⚠ Unverified on the device (microbench `07` §8):
  which side carries the payload of a DSMEM *load* (request vs response bytes), the meaning of `loopback`,
  and the rated link peak — C1 settles the first two with one push and one pull configuration of known
  traffic. Final list: _TODO (C1)_

## 2. Definitions

All times come from `%globaltimer` stamps written by the kernel (ns, consistent across SMs); host
`cudaEvent` time is recorded only as a cross-check. For one launch of one configuration
(file = op+alg, N, M, T), with N ranks:

| Quantity | Definition |
|---|---|
| per-rank latency `L_r` | `ts[r][exit] - ts[r][entry]`; entry is stamped right after the entry `cluster_sync`, exit right after the last synchronisation of the collective |
| collective latency `L` | `max_r ts[r][exit] - min_r ts[r][entry]` (time until the slowest rank is done, measured from the moment the first rank entered) |
| skew | `max_r L_r - min_r L_r`; also report `argmax`/`argmin` rank and their SM ids |
| round latency `R_{r,k}` | `ts[r][2+k] - ts[r][2+k-1]` (round 0 from entry); shows which round dominates and whether the critical path moves between ranks |
| payload per CTA `P` | bytes each CTA contributes: allreduce/allgather `M·B`, reducescatter/alltoall `N·M·B` (`B = sizeof(T)`) |
| algorithm bandwidth `algbw` | `P / L` (what the user of the collective perceives) |
| bus bandwidth `busbw` | `algbw × f`, NCCL convention: allreduce `f = 2(N-1)/N`, allgather / reducescatter / alltoall `f = (N-1)/N`; comparable across ops and N |
| bytes sent by rank `S_r` | from the file's docstring formula (README "total DSMEM bytes per CTA"; for rooted trees use the per-rank value) |
| per-SM injection bandwidth | `S_r / L_r` — how hard rank r drives its SM-to-SM link; report per rank, not only the mean |
| per-SM ejection bandwidth | bytes received by rank r / `L_r` (pull algorithms: bytes read) |
| measured DSMEM traffic | Nsight Compute counters (§4), kernel-wide and, where available, per SM |

Report per configuration: median over reps of `L`, `algbw`, `busbw`; p10/p90 of `L`; per-rank
`L_r`, injection bw, `smid`; round breakdown. Always keep the per-rank table, not only averages.

## 3. Sweep

| Axis | Values |
|---|---|
| file | all 29 (skipping configurations illegal per `TESTING.md` §1) |
| N | 2, 4, 8, 16 |
| payload per CTA | 256 B, 1 KB, 4 KB, 16 KB, 64 KB (choose `MSG_ELEMS` accordingly; cap by the 48 KB static smem limit and record the cap) |
| T | `float` (add `__half` only if a file supports it) |
| reps | 20 warm-up + 200 timed launches per configuration; one process per configuration |

Procedure per configuration: build (TESTING.md command), run the harness in "perf" mode which writes one
row per (rep, rank, stamp) — no correctness check in the timed loop but a final check after it.
Randomise the order of configurations across the sweep and repeat the full sweep twice to detect drift.
Clocks: the SM clock is locked for every timed and ncu run (`tools/run --profile --lock-sm-mhz <LOCK>`);
the memory clock is left at its default. Launches whose clock sample is below the lock or whose throttle
reasons fired are discarded and counted (`hb::ClockSample::bad()`); state the clock situation in the notebook.

## 4. Nsight Compute recipe

1. Discover the metrics once: `ncu --query-metrics | grep -i -E "cluster|dsmem|distributed|shared"` and
   `ncu --list-sections | grep -i memory`; the *Memory Workload Analysis* tables contain a
   "Distributed Shared Memory" row on Hopper. Record the exact metric names in §1.
2. Per configuration (reps = 5, kernel replay), inside `tools/run --profile --lock-sm-mhz <LOCK>`:
   `ncu --clock-control none --cache-control all --launch-skip 20 --launch-count 5 --section MemoryWorkloadAnalysis_Tables --metrics <dsmem metrics>,sm__cycles_elapsed.avg,smsp__cycles_active.avg,sm__ctas_launched.sum,lts__t_sectors_srcunit_tex.sum --csv --page raw --print-units base ./build/<cfg> > results/ncu_<cfg>.csv`
   (never `--clock-control base`, which would override the lock; never `--print-metric-name name`, which
   ncu 2025.2.1 rejects with `--page raw`). Filter the metric list through `ncu --query-metrics` first.
3. Extract: DSMEM bytes read/written (kernel-wide `.sum`; divide by N for per-CTA), their `.max/.avg` (hot-spot
   factor of rooted or asymmetric algorithms), cycles elapsed. **No per-SM instance values exist on this
   setup** (microbench `07` §8): per-rank traffic comes from the model (`model.csv`), not from ncu. DSMEM
   traffic never appears in `lts__*`: `lts__t_sectors_srcunit_tex.sum` must stay at the baseline level.
   The profiler adds its own traffic: subtract a same-grid all-exit baseline launch. Compare with `S_r`.

## 5. CSV schema (consumed by the notebook)

Every file starts with the mandatory columns `gpu_name, sm_count, sm_clock_mhz, mem_clock_mhz,
cuda_version, driver_version, run_id, launch_idx` (`../../docs/ENVIRONMENT.md` §5), followed by the columns
below; the notebook ignores the mandatory columns except `run_id`.

`results/stamps.csv` (long format, one row per stamp):
```
file,op,alg,N,M,elem_bytes,payload_bytes,rep,rank,smid,stamp,t_ns
```
`stamp` ∈ {`entry`, `exit`, `round0`, `round1`, …}.

`results/host.csv`: `file,op,alg,N,M,elem_bytes,rep,event_ms` (cudaEvent kernel time).

`results/ncu.csv`: `file,op,alg,N,M,elem_bytes,metric,unit,value` (one row per metric, averaged over the 5 replays).

`results/model.csv`: `file,op,alg,N,M,elem_bytes,rank,bytes_sent,bytes_recv,rounds` — the docstring/README
traffic model evaluated by a small script, so the notebook can plot measured vs model.

## 6. Figures (produced by `analysis/dsmem_collectives.ipynb`)

1. Collective latency vs N, one panel per op, one line per algorithm, at payload 4 KB (log y).
2. Collective latency vs payload at N = 8 (and N = 16), per op.
3. Bus bandwidth vs payload, per op, N = 8; highlight the bandwidth-optimal algorithms.
4. Per-rank timeline for one configuration per op: entry/round/exit stamps per rank with SM ids —
   shows skew and which rank/SM is on the critical path.
5. Per-SM injection bandwidth distribution (box per algorithm) at N = 16, 16 KB.
6. Measured DSMEM bytes (ncu) vs the traffic model, per algorithm.

Keep the notebook text to the setup, the definitions above, and one or two sentences per figure.

## 7. Agent-only reference numbers (sanity checks, never quoted as expectations)
Measured on the same device type by h100-microbench (GPU 2 card, SM clock 1635 MHz; details in
`../../docs/REUSE.md` §3). Use them to judge whether a result is plausible; if something is far off,
investigate (stamps, placement, clocks, SASS) before accepting it.
- `%globaltimer` tick 32 ns: a single launch cannot resolve differences below that; use medians over reps.
- One `cluster.sync()` ≈ 330–360 cycles (≈ 0.2 µs) with 256 threads; every collective has an entry and an
  exit barrier plus its own, so per-round costs below one barrier indicate a stamp problem.
- Cheapest CS-2 data hand-off (`st.async`/bulk + remote mbarrier): 565-cycle round trip, ≈ 160 ns one way;
  a bulk copy costs ≈ 340 cycles + bytes / 107 B/cycle.
- Placement: a lone CS-2 cluster landed on SMs 124/125, CS 4/8 in GPC 0, CS 16 in GPC 1, ranks 2k/2k+1 on
  one TPC; only 7 GPCs host CS 16. Re-check these on GPU 3 (its map comes from the microbench bring-up).
