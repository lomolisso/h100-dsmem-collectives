#!/usr/bin/env python3
"""perf_report.py — summary of the C5 sweeps (EXPERIMENTS.md §2–3). Stdlib only.

  python3 collectives/harness/perf_report.py [--sweeps s1 s2]

Reads results/stamps_<op>_<sweep>.csv (long format, one row per rep/rank/stamp). Per (file, N, M) and sweep it
uses the latest harness run_id only (a re-run supersedes earlier rows). Per launch (rep):
  L = max_r exit_r - min_r entry_r,   L_r = exit_r - entry_r,   skew = max_r L_r - min_r L_r.
Launches whose clock sample was bad (clock_bad = 1: throttle reason or SM clock below the lock) are discarded and
counted. Per configuration and sweep: median / p10 / p90 of L, algbw = P / median L, busbw = algbw * f
(allreduce 2(N-1)/N, others (N-1)/N; P = payload per CTA), median per-rank L_r (min and max over ranks), median skew.
Across sweeps: drift = median L(s2) / median L(s1) - 1. Sanity (agent-only): configurations whose median L is below
200 ns although the collective has a cluster barrier or a hand-off inside (every file) are listed — the cheapest
hand-off is ~160 ns one way and one cluster.sync() ~0.2 µs, so that would point to a stamp problem.

Output: results/perf_summary__<ts>.csv (mandatory + file, op, alg, N, M, payload_bytes, sweep, run_id, reps_kept,
reps_discarded, L_med_ns, L_p10_ns, L_p90_ns, algbw_GBps, busbw_GBps, Lr_med_min_ns, Lr_med_max_ns, skew_med_ns)
and a text summary on stdout.
"""
import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import driver  # noqa: E402

HDR = driver.MANDATORY + ["file", "op", "alg", "N", "M", "payload_bytes", "sweep", "run_id_sweep", "reps_kept",
                          "reps_discarded", "L_med_ns", "L_p10_ns", "L_p90_ns", "algbw_GBps", "busbw_GBps",
                          "Lr_med_min_ns", "Lr_med_max_ns", "skew_med_ns"]


def pct(v, q):
    v = sorted(v)
    if not v:
        return float("nan")
    k = (len(v) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def load(sweep):
    """{(file, N, M): (run_id, payload, {rep: {rank: [entry, exit, clock_bad]}})} for the latest run per config."""
    data, latest = {}, {}
    for path in sorted(glob.glob(f"{driver.RES}/stamps_*_{sweep}.csv")):
        with open(path, newline="") as fh:
            rd = csv.reader(fh)
            h = {k: i for i, k in enumerate(next(rd))}
            for r in rd:
                st = r[h["stamp"]]
                if st not in ("entry", "exit"):
                    continue
                key = (r[h["file"]], int(r[h["N"]]), int(r[h["M"]]))
                rid = r[h["run_id"]]
                if rid > latest.get(key, ""):
                    latest[key] = rid
                    data[key] = (rid, driver.payload_bytes(r[h["op"]], key[1], key[2], int(r[h["elem_bytes"]])),
                                 defaultdict(dict))  # recomputed: allgather rows before C6 carry M*B
                if rid != latest[key]:
                    continue
                rank = data[key][2][int(r[h["rep"]])].setdefault(int(r[h["rank"]]), [0, 0, 0])
                rank[0 if st == "entry" else 1] = int(r[h["t_ns"]])
                rank[2] |= r[h["clock_bad"]] == "1"
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweeps", nargs="*", default=["s1", "s2"])
    args = ap.parse_args()
    os.chdir(driver.COLL)
    rid = driver.iso_now()
    rows, med = [], defaultdict(dict)
    disc_total = kept_total = 0
    low = []
    for sw in args.sweeps:
        data = load(sw)
        print(f"sweep {sw}: {len(data)} configurations")
        for (f, n, m), (run_id, payload, reps) in sorted(data.items()):
            op, alg = driver.split(f)
            Ls, Lr, skew, disc = [], defaultdict(list), [], 0
            for rep, ranks in reps.items():
                if len(ranks) != n or any(v[2] for v in ranks.values()):
                    disc += 1
                    continue
                Ls.append(max(v[1] for v in ranks.values()) - min(v[0] for v in ranks.values()))
                lr = {r: v[1] - v[0] for r, v in ranks.items()}
                for r, x in lr.items():
                    Lr[r].append(x)
                skew.append(max(lr.values()) - min(lr.values()))
            disc_total += disc
            kept_total += len(Ls)
            if not Ls:
                continue
            Lm = pct(Ls, 0.5)
            f_bus = 2 * (n - 1) / n if op == "allreduce" else (n - 1) / n
            algbw = payload / Lm if Lm > 0 else float("nan")  # bytes/ns = GB/s
            lr_meds = [pct(v, 0.5) for v in Lr.values()]
            med[(f, n, m)][sw] = Lm
            if Lm < 200:
                low.append((sw, driver.cfg_name(f, n, m, "float"), Lm))
            rows.append(["", "", "", "", "", "", rid, -1, f, op, alg, n, m, payload, sw, run_id, len(Ls), disc,
                         f"{Lm:.1f}", f"{pct(Ls, 0.1):.1f}", f"{pct(Ls, 0.9):.1f}", f"{algbw:.4g}",
                         f"{algbw * f_bus:.4g}", f"{min(lr_meds):.1f}", f"{max(lr_meds):.1f}", f"{pct(skew, 0.5):.1f}"])
    out = f"{driver.RES}/perf_summary__{rid.replace(':', '')}.csv"
    driver.append_rows(out, HDR, rows)
    print(f"launches kept {kept_total}, discarded (clock-flagged or incomplete) {disc_total}")
    drift = sorted((v["s2"] / v["s1"] - 1, k) for k, v in med.items() if "s1" in v and "s2" in v and v["s1"] > 0)
    if drift:
        ad = sorted(abs(d) for d, _ in drift)
        print(f"drift s2/s1 over {len(drift)} configurations: median |d| {pct(ad, 0.5) * 100:.2f} %, "
              f"p90 {pct(ad, 0.9) * 100:.2f} %, max {ad[-1] * 100:.2f} %")
        for d, k in sorted(drift, key=lambda x: -abs(x[0]))[:10]:
            print(f"  {driver.cfg_name(*k, 'float')}: {d * 100:+.1f} %")
    print(f"median L below 200 ns: {len(low)}")
    for x in low[:20]:
        print("  ", *x)
    print("summary written to", out)


if __name__ == "__main__":
    main()
