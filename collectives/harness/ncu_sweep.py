#!/usr/bin/env python3
"""ncu_sweep.py — C5 Nsight Compute pass over the perf set (EXPERIMENTS.md §4). Stdlib only.

Run from the repo root inside
  tools/run --profile --lock-sm-mhz <LOCK> -- python3 collectives/harness/ncu_sweep.py [--files ...] [--N ...]

Per configuration of `driver.runnable(..., "perf")` (float):
  ncu --clock-control none --cache-control all --kernel-name collective_kernel --launch-skip 20 --launch-count 5
      --section MemoryWorkloadAnalysis_Tables --metrics <METRICS> --csv --page raw --print-units base
      build/<cfg> --mode ncu --reps 25
and, per N, the same-grid all-exit baseline (`--baseline`, kernel baseline_kernel) of the configuration with the
smallest and of the one with the largest dynamic shared memory (the profiler's own traffic, ENVIRONMENT.md §4).
--clock-control none keeps the tools/run SM-clock lock; --print-metric-name is not used (ncu 2025.2.1 rejects it
with --page raw). The metric list is EXPERIMENTS.md §1's final list (C1), filtered against
results/ncu_query_metrics.txt when present.

Outputs (appended, never rewritten; a configuration whose raw file exists is skipped = resumable):
  results/ncu_raw/<cfg>[__baseline].csv  the ncu stdout of that run (raw page, all metrics, 5 launches; the
                                         harness's own stdout lines are mixed in and ignored by the parser)
  results/ncu_launches.csv  mandatory + file, op, alg, N, M, elem_bytes, kernel, launch, metric, unit, value
  results/ncu.csv           mandatory + file, op, alg, N, M, elem_bytes, kernel, metric, unit, value, n_launches
                            (value = mean over the launches for numeric metrics, else the first launch's value)
Mandatory columns: GPU name/SM count/CUDA/driver from nvidia-smi, sm/mem clock sampled by nvidia-smi right before
the configuration, run_id of this sweep, launch_idx = position of the configuration in this sweep.
"""
import argparse
import csv
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import driver  # noqa: E402

METRICS = [
    "l1tex__m_l1tex2xbar_write_bytes_mem_dshared.sum",
    "l1tex__m_l1tex2xbar_write_bytes_mem_dshared_op_st.sum",
    "l1tex__m_l1tex2xbar_write_bytes_mem_dshared_op_atom.sum",
    "l1tex__m_l1tex2xbar_write_bytes_mem_dshared_op_redas.sum",
    "l1tex__m_l1tex2xbar_write_bytes_mem_dshared_op_tma_st.sum",
    "l1tex__m_l1tex2xbar_write_bytes_mem_dshared_op_tma_red.sum",
    "l1tex__m_l1tex2xbar_write_bytes_mem_dshared_op_ld.sum",
    "l1tex__m_xbar2l1tex_read_bytes_mem_dshared.sum",
    "l1tex__m_xbar2l1tex_read_bytes_mem_dshared_op_ld.sum",
    "l1tex__m_xbar2l1tex_read_bytes_mem_dshared_op_st.sum",
    "l1tex__m_xbar2l1tex_read_bytes_mem_dshared_op_atom.sum",
    "l1tex__m_xbar2l1tex_read_bytes_mem_dshared_op_redas.sum",
    "l1tex__m_xbar2l1tex_read_bytes_mem_dshared_op_tma_st.sum",
    "l1tex__t_requests_pipe_lsu_mem_dshared.sum",
    "l1tex__t_sectors_pipe_lsu_mem_dshared.sum",
    "l1tex__t_bytes_pipe_lsu_mem_dshared.sum",
    "sm__sass_inst_executed_op_dshared_ld.sum",
    "sm__sass_inst_executed_op_dshared_st.sum",
    "sm__sass_inst_executed_op_dshared_atom.sum",
    "sm__sass_inst_executed_op_dshared_redas.sum",
    "sm__sass_inst_executed_op_dshared_stas.sum",
    "sm__sass_inst_executed_op_dshared_tma_st.sum",
    "sm__sass_inst_executed_op_dshared_tma_red.sum",
    "l1tex__m_l1tex2xbar_throughput_pipe_lsu_mem_dshared.avg.pct_of_peak_sustained_elapsed",
    "l1tex__m_xbar2l1tex_throughput_pipe_lsu_mem_dshared.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sectors_srcunit_tex.sum",
    "gpc__cgas_launched.sum",
    "sm__ctas_launched.sum",
    "sm__cycles_elapsed.avg",
    "smsp__cycles_active.avg",
]
COLS = ["file", "op", "alg", "N", "M", "elem_bytes", "kernel"]
LAUNCH_HDR = driver.MANDATORY + COLS + ["launch", "metric", "unit", "value"]
NCU_HDR = driver.MANDATORY + COLS + ["metric", "unit", "value", "n_launches"]
RAW = f"{driver.RES}/ncu_raw"


def available(metrics):
    q = f"{driver.RES}/ncu_query_metrics.txt"
    if not os.path.exists(q):
        return metrics
    names = {l.split()[0] for l in open(q) if l.strip()}
    keep = [m for m in metrics if m.split(".")[0] in names]
    for m in metrics:
        if m not in keep:
            print("not listed by ncu --query-metrics, dropped:", m)
    return keep


def smi(fields):
    p = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"], capture_output=True,
                       text=True)
    return [x.strip() for x in p.stdout.strip().splitlines()[0].split(",")] if p.returncode == 0 else []


def parse(path):
    """[(launch, {metric: (unit, value)})] from an ncu --csv --page raw stdout capture."""
    rows = [r for r in csv.reader(open(path, newline="")) if r]
    try:
        i = next(k for k, r in enumerate(rows) if r[0] == "ID")
    except StopIteration:
        return []
    hdr, unit = rows[i], rows[i + 1]
    out = []
    for n, r in enumerate(x for x in rows[i + 2:] if len(x) == len(hdr)):
        out.append((n, {h: (u, v) for h, u, v in zip(hdr, unit, r)}))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", nargs="*")
    ap.add_argument("--ops", nargs="*")
    ap.add_argument("--N", nargs="*", type=int)
    args = ap.parse_args()
    os.chdir(driver.COLL)
    os.makedirs(RAW, exist_ok=True)
    sel = argparse.Namespace(files=args.files, ops=args.ops, N=args.N, M=None, T=None)
    lb = driver.latest_build()
    cfgs = driver.runnable(lb, sel, "perf")
    jobs = [(c, False) for c in cfgs]
    for n in sorted({c[1] for c in cfgs}):  # baselines: smallest- and largest-smem configuration per N
        mine = sorted((int(lb[c]["smem_bytes"] or 0), c) for c in cfgs if c[1] == n)
        for c in {mine[0][1], mine[-1][1]}:
            jobs.append((c, True))
    metrics = ",".join(available(METRICS))
    name, drv = (smi("name,driver_version") + ["", ""])[:2]
    cuda = driver.nvcc_release()
    reg = driver.Registry(sys.argv)
    done = skipped = failed = 0
    for k, ((f, n, m, t), base) in enumerate(jobs):
        cfg = driver.cfg_name(f, n, m, t)
        raw = f"{RAW}/{cfg}{'__baseline' if base else ''}.csv"
        if os.path.exists(raw):
            skipped += 1
            continue
        clk = smi("clocks.sm,clocks.mem") or ["", ""]
        cmd = ["ncu", "--clock-control", "none", "--cache-control", "all", "--kernel-name",
               "baseline_kernel" if base else "collective_kernel", "--launch-skip", "20", "--launch-count", "5",
               "--section", "MemoryWorkloadAnalysis_Tables", "--metrics", metrics, "--csv", "--page", "raw",
               "--print-units", "base", f"build/{cfg}", "--mode", "ncu", "--reps", "25"] + (["--baseline"] if base else [])
        tmp = raw + ".part"
        with open(tmp, "w") as fh:
            p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True, timeout=1800)
        launches = parse(tmp)
        if p.returncode != 0 or len(launches) != 5:
            failed += 1
            print(f"[{k + 1}/{len(jobs)}] {cfg}{' baseline' if base else ''}: FAILED rc={p.returncode}, "
                  f"{len(launches)} launches parsed (kept {tmp})", flush=True)
            continue
        os.replace(tmp, raw)
        op, alg = driver.split(f)
        pre = [name, "132", clk[0], clk[1], cuda, drv, reg.run_id, k]
        key = [f, op, alg, n, m, 4, "baseline_kernel" if base else "collective_kernel"]
        lrows, agg = [], {}
        for li, vals in launches:
            for metric, (unit, v) in vals.items():
                if "__" not in metric:
                    continue
                lrows.append(pre + key + [li, metric, unit, v])
                agg.setdefault(metric, (unit, []))[1].append(v)
        nrows = []
        for metric, (unit, vs) in agg.items():
            try:
                val = f"{sum(float(x.replace(',', '')) for x in vs) / len(vs):.6g}"
            except ValueError:
                val = vs[0]
            nrows.append(pre + key + [metric, unit, val, len(vs)])
        driver.append_rows(f"{driver.RES}/ncu_launches.csv", LAUNCH_HDR, lrows)
        driver.append_rows(f"{driver.RES}/ncu.csv", NCU_HDR, nrows)
        done += 1
        print(f"[{k + 1}/{len(jobs)}] {cfg}{' baseline' if base else ''}: ok ({len(nrows)} metrics)", flush=True)
    if not failed:
        reg.complete()
    print(f"ncu sweep: {done} profiled, {skipped} already done, {failed} failed")


if __name__ == "__main__":
    main()
