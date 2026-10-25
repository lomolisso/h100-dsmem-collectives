# EXPERIMENTS.md — latency and bandwidth study of DSMEM collectives on an H100 NVL (design)

Runs only after `TESTING.md` is green. Produces `collectives/results/*.csv`, consumed by
`analysis/dsmem_collectives.ipynb`.

## 1. Tooling  *(to be filled by the remote agent)*

- nvcc / driver / Nsight Compute versions, clock state (`nvidia-smi -q -d CLOCK`, locked or not): _TODO_
- Exact commands used for the sweep and for `ncu`: _TODO_
- DSMEM metric names found with `ncu --query-metrics` (see §4): _TODO_

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
Clocks: lock graphics and memory clocks if the server protocol allows; otherwise use
`ncu --clock-control base` for the counter runs and state the clock situation in the notebook.

## 4. Nsight Compute recipe

1. Discover the metrics once: `ncu --query-metrics | grep -i -E "cluster|dsmem|distributed|shared"` and
   `ncu --list-sections | grep -i memory`; the *Memory Workload Analysis* tables contain a
   "Distributed Shared Memory" row on Hopper. Record the exact metric names in §1.
2. Per configuration (reps = 5, kernel replay):
   `ncu --clock-control base --launch-skip 20 --launch-count 5 --section MemoryWorkloadAnalysis_Tables --metrics <dsmem metrics>,sm__cycles_elapsed.avg,smsp__cycles_active.avg,sm__ctas_launched.sum --csv ./build/<cfg> > results/ncu_<cfg>.csv`.
3. Extract: DSMEM bytes read/written (kernel-wide; divide by N for per-CTA), cycles elapsed, and any
   per-SM breakdown the tool offers (`--print-units base`, instance values). Compare with `S_r`.

## 5. CSV schema (consumed by the notebook)

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
