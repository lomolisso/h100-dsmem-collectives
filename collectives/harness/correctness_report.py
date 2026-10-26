#!/usr/bin/env python3
"""correctness_report.py — triage of the correctness runs (C2/C3; TESTING.md §2–3, §6). Stdlib only.

  python3 collectives/harness/correctness_report.py --ops allreduce [--patterns A B] [--reps 200] [--processes 3]

Expected configurations = the set `driver.py run` selects (OK grid builds + the largest OK M_max point per
(file, N, T); pattern B: float only). For every (configuration, pattern, process) the *latest* driver outcome
(results/collectives__driver.csv) and the rows of the latest harness run_id in results/correctness.csv are used;
earlier runs (e.g. before a fix) stay in the files and are ignored here.

Configuration verdict per pattern:
  PASS      every process: driver outcome PASS and `reps` rows, all status PASS
  FAIL      some rep status FAIL (wrong output)            DEADLOCK  watchdog fired (row DEADLOCK / outcome)
  ERROR     child exited otherwise (CUDA error, ...)       KILLED    driver timeout
  INCOMPLETE fewer processes or reps than requested        MISSING   never run
Also counted over the latest runs: clock-flagged launches (clock_bad = 1) and launches whose ranks did not land
on distinct SMs (smid_distinct = 0).

Output: a per-file table on stdout, every non-PASS configuration with its first failing rep, and
results/correctness_summary__<timestamp>.csv (one row per configuration and pattern; new file per report).
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import driver  # noqa: E402  (configuration selection, CSV helpers)

SUMMARY_HDR = driver.MANDATORY + ["file", "op", "alg", "N", "M", "T", "pattern", "verdict", "processes_pass",
                                  "reps_total", "reps_fail", "max_n_bad", "max_abs_err", "clock_bad", "smid_not_distinct",
                                  "first_fail", "harness_run_ids"]


def expected(args):
    sel = driver.argparse.Namespace(files=None, ops=args.ops, N=None, M=None, T=None)
    return driver.runnable(driver.latest_build(), sel, "default")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ops", nargs="*", default=None)
    ap.add_argument("--patterns", nargs="*", default=["A", "B"])
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--processes", type=int, default=3)
    args = ap.parse_args()
    os.chdir(driver.COLL)

    # latest driver outcome per (cfg, pattern, process); file order is chronological
    outcome = {}
    for r in driver.read_csv(f"{driver.RES}/collectives__driver.csv"):
        if r["mode"] == "correctness":
            outcome[(r["cfg"], r["pattern"], int(r["process"]))] = r["outcome"]

    # rows of the latest harness run per (cfg, pattern, process)
    latest_run, rows = {}, defaultdict(list)
    with open(f"{driver.RES}/correctness.csv", newline="") as fh:
        rd = csv.reader(fh)
        h = {k: i for i, k in enumerate(next(rd))}
        for r in rd:
            key = (driver.cfg_name(r[h["file"]], r[h["N"]], r[h["M"]], r[h["T"]]), r[h["pattern"]], int(r[h["process"]]))
            rid = r[h["run_id"]]
            if rid > latest_run.get(key, ""):
                latest_run[key] = rid
                rows[key] = []
            if rid == latest_run[key]:
                rows[key].append(r)

    res = []
    for f, n, m, t in expected(args):
        cfg = driver.cfg_name(f, n, m, t)
        for pat in args.patterns:
            if pat == "B" and t != "float":
                continue
            v = dict(pp=0, reps=0, fail=0, nbad=0, abs=0.0, clk=0, place=0, first="", rids=[], verdicts=set())
            for p in range(args.processes):
                key = (cfg, pat, p)
                rr, oc = rows.get(key, []), outcome.get(key)
                if oc is None and not rr:
                    v["verdicts"].add("MISSING")
                    continue
                v["rids"].append(latest_run.get(key, ""))
                st = [x[h["status"]] for x in rr]
                v["reps"] += sum(s in ("PASS", "FAIL") for s in st)
                nfail = st.count("FAIL")
                v["fail"] += nfail
                for x in rr:
                    if x[h["status"]] == "FAIL":
                        v["nbad"] = max(v["nbad"], int(x[h["n_bad"]]))
                        v["abs"] = max(v["abs"], float(x[h["max_abs_err"]]))
                        if not v["first"]:
                            v["first"] = (f"p{p} rep {x[h['rep']]}: n_bad {x[h['n_bad']]}, max_abs_err "
                                          f"{x[h['max_abs_err']]}, smids {x[h['smids']]}")
                    if x[h["clock_bad"]] == "1":
                        v["clk"] += 1
                    if x[h["smid_distinct"]] == "0":
                        v["place"] += 1
                if "DEADLOCK" in st or oc == "DEADLOCK":
                    v["verdicts"].add("DEADLOCK")
                elif nfail:
                    v["verdicts"].add("FAIL")
                elif oc == "KILLED_TIMEOUT":
                    v["verdicts"].add("KILLED")
                elif oc not in ("PASS", None):
                    v["verdicts"].add("ERROR")
                elif len(st) < args.reps:
                    v["verdicts"].add("INCOMPLETE")
                else:
                    v["pp"] += 1
            order = ["DEADLOCK", "FAIL", "ERROR", "KILLED", "INCOMPLETE", "MISSING"]
            verdict = next((x for x in order if x in v["verdicts"]), "PASS")
            res.append((f, n, m, t, pat, verdict, v))

    # per-file table
    tab = defaultdict(lambda: defaultdict(int))
    for f, n, m, t, pat, verdict, v in res:
        tab[(f, t, pat)][verdict] += 1
        tab[(f, t, pat)]["configs"] += 1
        tab[(f, t, pat)]["clk"] += v["clk"]
        tab[(f, t, pat)]["place"] += v["place"]
    cols = ["configs", "PASS", "FAIL", "DEADLOCK", "ERROR", "KILLED", "INCOMPLETE", "MISSING", "clk", "place"]
    print(f"{'file':58s} {'T':5s} pat " + " ".join(f"{c[:8]:>8s}" for c in cols))
    for (f, t, pat), d in sorted(tab.items()):
        print(f"{f:58s} {t:5s} {pat:3s} " + " ".join(f"{d[c]:8d}" for c in cols))
    tot = defaultdict(int)
    for *_, verdict, v in res:
        tot[verdict] += 1
        tot["clock_bad_launches"] += v["clk"]
        tot["placement_not_distinct"] += v["place"]
    print("totals:", dict(tot))
    for f, n, m, t, pat, verdict, v in res:
        if verdict != "PASS":
            print(f"{verdict:10s} {driver.cfg_name(f, n, m, t)} pattern {pat}: processes PASS {v['pp']}/{args.processes}, "
                  f"failing reps {v['fail']}/{v['reps']}, max n_bad {v['nbad']}, max_abs_err {v['abs']:g}; {v['first']}")

    rid = driver.iso_now()
    out = f"{driver.RES}/correctness_summary__{rid.replace(':', '')}.csv"
    driver.append_rows(out, SUMMARY_HDR, [
        ["", "", "", "", "", "", rid, -1, f, *driver.split(f), n, m, t, pat, verdict, v["pp"], v["reps"], v["fail"],
         v["nbad"], f"{v['abs']:g}", v["clk"], v["place"], v["first"], "|".join(v["rids"])]
        for f, n, m, t, pat, verdict, v in res])
    print("summary written to", out)


if __name__ == "__main__":
    main()
