"""roofline R1: join an ncu raw export to the manifest written by the profiled program (called by ceilings_ncu.sh).

Adapted from MB/experiments/05__memory_hierarchy_bandwidth/src/post_ncu.py (MB commit 8f8fc7bec69f07db49f3a5a95bbdfb08de5d0ca8):
the profiled program appends one manifest row per profiled launch (mandatory columns + configuration) to
results/ceilings/ncu/manifest_<name>.csv; only the rows of the manifest's latest run_id are used. Changes from MB:
long output format (one row per manifest row × kernel × metric, so the header does not depend on which metrics
`ncu --query-metrics` lists), the ncu report path in every row, and --aggregate (all profiled kernels belong to the single
manifest row: a torch GEMM may launch more than one kernel).
    --ncu-csv  <report>.raw.csv   (ncu -i <report>.ncu-rep --csv --page raw --print-units base)
    --manifest results/ceilings/ncu/manifest_<name>.csv
    --metrics  a.sum,b.sum,...    the metrics passed to ncu (all their rollups are kept)
    --report   path of the .ncu-rep (recorded in every row)
    --out      results/ceilings/ncu/ncu_<name>.csv  (append only)
Output columns: manifest columns, ncu_report, kernel_idx, kernel_name, metric, value.
"""
import argparse
import csv
import fcntl
import io
import os
import sys


def append_rows(path, header, rows):
    """Crash-safe append (hb::CsvWriter semantics), copied from collectives/harness/driver.py."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a+", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            first = f.readline().rstrip("\n")
            buf = io.StringIO()
            w = csv.writer(buf, lineterminator="\n")
            if not first:
                w.writerow(header)
            elif first != ",".join(header):
                raise SystemExit(f"header mismatch in {path}\n  file: {first}\n  ours: {','.join(header)}")
            w.writerows(rows)
            f.seek(0, os.SEEK_END)
            f.write(buf.getvalue())
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncu-csv", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    with open(a.manifest, newline="") as f:
        rd = csv.reader(f)
        man_header = next(rd)
        man = list(rd)
    rid = man_header.index("run_id")
    latest = max(r[rid] for r in man)
    man = [r for r in man if r[rid] == latest]
    with open(a.ncu_csv, newline="") as f:
        rows = list(csv.reader([l for l in f if l.startswith('"')]))  # drop ==PROF== messages
    if not rows:
        raise SystemExit(f"no ncu rows in {a.ncu_csv}")
    header, data = rows[0], [r for r in rows[1:] if r and r[0].strip().isdigit()]  # numeric ID: drops the units row
    bases = {m.split(".")[0] for m in a.metrics.split(",") if m}
    cols = [i for i, c in enumerate(header) if c.split(".")[0] in bases]
    kname = header.index("Kernel Name")
    if a.aggregate:
        if len(man) != 1:
            raise SystemExit(f"--aggregate needs exactly one manifest row of run {latest}, found {len(man)}")
        pairs = [(man[0], k, r) for k, r in enumerate(data)]
    else:
        if len(data) != len(man):
            print(f"WARNING: {len(data)} profiled kernels, manifest run {latest} has {len(man)} rows; joining the first "
                  f"{min(len(data), len(man))} in launch order", file=sys.stderr)
        pairs = [(m, 0, r) for m, r in zip(man, data)]
    out = []
    for m, k, r in pairs:
        for i in cols:
            out.append(m + [a.report, k, r[kname], header[i], r[i]])
    append_rows(a.out, man_header + ["ncu_report", "kernel_idx", "kernel_name", "metric", "value"], out)
    print(f"ncu: {len(pairs)} kernel(s) x {len(cols)} metric columns -> {a.out} (manifest run {latest})")


if __name__ == "__main__":
    main()
