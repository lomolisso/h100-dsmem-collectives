#!/usr/bin/env python3
"""driver.py — build matrix and child-process runner for the collectives harness (TESTING.md §1–2).

Run inside the microbench container from anywhere (it chdirs to collectives/):
  tools/run --no-gpu -- python3 collectives/harness/driver.py build [--jobs 16] [--files F ...] [--N ...] [--M ...] [--T ...] [--force] [--no-mmax]
  tools/run --profile --lock-sm-mhz <LOCK> -- python3 collectives/harness/driver.py run --mode correctness \
        --pattern A --reps 200 --processes 3 [--smoke] [--files ...] [--N ...] [--M ...] [--T ...] [--shuffle --seed S]
  tools/run --no-gpu -- python3 collectives/harness/driver.py summary

build: one binary per (file, N, M, T): N in {2,4,8,16}; M on the grid {16,64,256,1024,4096,16384} plus, per
  (file, N, T), the largest M whose shared memory fits the 227 KiB per-block limit ("M_max", see below); T = float
  for every file and int for the Sum files (allreduce, reducescatter). Command (TESTING §1, plus the -DCOLL_* knobs
  main.cu needs):
    nvcc -arch=sm_90a -std=c++17 -O3 -lineinfo -Xptxas -v -I ../common/h100mb -DCOLLECTIVE_FILE='"../<file>.cu"'
         -DCOLL_NAME='"<file>"' -DCOLL_OP=<k> -DCLUSTER_N=N -DMSG_ELEMS=M -DELEM_T=T
         -o build/<file>__N<N>_M<M>_<T> harness/main.cu -L$CUDA_HOME/lib64/stubs -lcuda -lnvidia-ml
  Compiler output → build/logs/<cfg>.log. Status: OK (exit 0); SKIP:static_assert (a static assertion of the
  algorithm file, i.e. an N/M constraint; takes precedence); SKIP:smem (only main.cu's "SMEM_LIMIT" static_assert:
  sizeof(CollectiveSmem) > 232448 B, or a ptxas "uses too much shared data"); ERROR (anything else). smem_bytes = sizeof(CollectiveSmem),
  read from the "COLLECTIVE_SMEM_BYTES=<n>;" string main.cu embeds in the binary (OK builds only); from the
  collective_kernel entry of the ptxas -v output: static smem bytes (0 for the dynamic-smem files), registers,
  spill-store bytes. One row per built configuration is appended to results/build_matrix_v2.csv (grid = 1 for grid
  points, 0 for M_max points); a configuration whose log is newer than every source it depends on and that already
  has a row is not rebuilt unless --force. At most --jobs (16) parallel nvcc.
  M_max: for fixed (file, N, T) the shared memory is affine in M, bytes = a*M + b. a, b come from the two largest
  OK grid builds; the fit floor((232448 - b) / a) is rounded down to a multiple of 4 elements (16 B) first; if that
  build fails a divisibility static_assert it is retried rounded to 4*N, then 16*N (16*N satisfies every N | M,
  16-byte-chunk and C | M constraint of the files); if it comes out SKIP:smem (padding) it is lowered by the
  granularity, at most 3 times. Only points above the largest OK grid M that are not grid values are built. An
  algorithmic static_assert (e.g. M <= 1024) ends the search. `run` uses the grid points and, per (file, N, T),
  only the largest OK non-grid point.
run: every OK configuration (grid and M_max) selected by the filters (--smoke: per file the M = 256 float build at
  N = 4, or its smallest OK N there, plus the OK build with the largest smem_bytes), --processes fresh child
  processes each, in file order or shuffled (--shuffle --seed). Child: build/<cfg> --mode ... --pattern ... --reps
  ... --process k [--timeout-ms ...], cwd collectives/, stdout+stderr appended to results/logs/<cfg>.log. A child
  whose exact command line already has a `complete` row in results/collectives__runs.csv is skipped (resumable). A
  child that outlives --child-timeout (s) is killed; the driver then waits until no compute process is left on the
  GPU before the next child. Outcomes go to results/collectives__driver.csv: PASS (exit 0), FAIL (exit 1), DEADLOCK
  (exit 3 and the watchdog line), ERROR (other exit codes), KILLED_TIMEOUT.
summary: per file and T, the N x M grid of the latest build status (. OK, a static_assert, s smem, E error), the
  M_max point per N (M and smem bytes), totals per status, and every ERROR with its first error line.
Every CSV starts with the mandatory columns (docs/ENVIRONMENT.md §5); for build-only rows the GPU fields are
empty, cuda_version is the nvcc release, launch_idx = -1. Both commands register themselves in
results/collectives__runs.csv (started / complete). results/build_matrix.csv is the C1 record of the earlier
static-shared-memory build (48 KiB limit) and is no longer written.
"""
import argparse
import concurrent.futures as cf
import csv
import fcntl
import glob
import io
import os
import random
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime

COLL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # collectives/
OPS = {"allreduce": 0, "allgather": 1, "reducescatter": 2, "alltoall": 3}
SUM_OPS = {"allreduce", "reducescatter"}
NS, MS = [2, 4, 8, 16], [16, 64, 256, 1024, 4096, 16384]
SMEM_LIMIT = 232448  # bytes per block, H100 opt-in maximum (227 KiB); main.cu static_asserts the same value
MANDATORY = ["gpu_name", "sm_count", "sm_clock_mhz", "mem_clock_mhz", "cuda_version", "driver_version", "run_id",
             "launch_idx"]
REG_HDR = ["run_id", "started_iso", "finished_iso", "status", "gpu_uuid", "cmdline"]
BUILD_HDR = MANDATORY + ["file", "op", "alg", "N", "M", "T", "grid", "status", "smem_bytes", "static_smem_bytes",
                         "regs", "spill_bytes", "ptxas_msg"]
DRIVER_HDR = MANDATORY + ["cfg", "mode", "pattern", "process", "returncode", "outcome", "wall_s", "cmdline"]
RES = "results"
MATRIX = f"{RES}/build_matrix_v2.csv"


def iso_now():
    """Same format as hb::iso_now: local time, ms, numeric UTC offset."""
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def append_rows(path, header, rows):
    """Crash-safe append (hb::CsvWriter semantics): header only into an empty file, checked otherwise; flock; fsync."""
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


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


class Registry:
    """started/complete rows in results/collectives__runs.csv, same columns as hb::Run."""

    def __init__(self, argv):
        self.run_id = iso_now()
        self.cmd = " ".join(argv)
        append_rows(f"{RES}/collectives__runs.csv", REG_HDR,
                    [[self.run_id, iso_now(), "", "started", os.environ.get("H100MB_GPU_UUID", ""), self.cmd]])

    def complete(self):
        append_rows(f"{RES}/collectives__runs.csv", REG_HDR,
                    [[self.run_id, "", iso_now(), "complete", os.environ.get("H100MB_GPU_UUID", ""), self.cmd]])


def files():
    return sorted(os.path.basename(p)[:-3] for p in glob.glob(os.path.join(COLL, "*__*__*.cu")))


def split(f):
    """<op>__<alg>__<src> -> (op, alg)"""
    parts = f.split("__")
    return parts[0], parts[1]


def types_of(f):
    return ["float", "int"] if split(f)[0] in SUM_OPS else ["float"]


def cfg_name(f, n, m, t):
    return f"{f}__N{n}_M{m}_{t}"


def selected(args, f, n, t):
    return (not args.files or f in args.files) and (not args.N or n in args.N) and (not args.T or t in args.T)


def grid(args):
    return [(f, n, m, t) for f in files() for t in types_of(f) for n in NS for m in MS
            if selected(args, f, n, t) and (not args.M or m in args.M)]


def nvcc_release():
    try:
        v = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout
        m = re.search(r"release (\d+\.\d+)", v)
        return m.group(1) if m else ""
    except FileNotFoundError:
        return ""


def parse_log(text):
    """(static_smem_bytes, regs, spill_bytes) of collective_kernel from ptxas -v, and (status, msg) of a failed build."""
    smem = regs = spill = ""
    m = re.search(r"Compiling entry function '_Z17collective_kernel[^']*'.*?\n(.*?)Compile time", text, re.S)
    if m:
        blk = m.group(1)
        r = re.search(r"Used (\d+) registers", blk)
        regs = r.group(1) if r else ""
        s = re.search(r"(\d+) bytes smem", blk)
        smem = s.group(1) if s else ("0" if r else "")
        sp = re.search(r"(\d+) bytes spill stores", blk)
        spill = sp.group(1) if sp else ""
    errs = [l.strip() for l in text.splitlines() if re.search(r"error", l, re.I)]
    algo = [l for l in errs if "static assertion failed" in l and "SMEM_LIMIT" not in l]
    if algo:  # an N/M constraint of the file makes the configuration illegal whatever its size
        status, hit = "SKIP:static_assert", algo
    elif "SMEM_LIMIT" in text or "uses too much shared data" in text:
        status, hit = "SKIP:smem", [l for l in errs if "SMEM_LIMIT" in l or "shared data" in l]
    else:
        status, hit = "ERROR", errs or ([text.strip().splitlines()[-1]] if text.strip() else [""])
    return smem, regs, spill, status, (hit[0] if hit else "")[:300]


def build_one(f, n, m, t, cuda_home):
    """Compile one configuration → (status, smem_bytes, static_smem, regs, spill, msg)."""
    cfg = cfg_name(f, n, m, t)
    cmd = ["nvcc", "-arch=sm_90a", "-std=c++17", "-O3", "-lineinfo", "-Xptxas", "-v", "-I", "../common/h100mb",
           f'-DCOLLECTIVE_FILE="../{f}.cu"', f'-DCOLL_NAME="{f}"', f"-DCOLL_OP={OPS[split(f)[0]]}",
           f"-DCLUSTER_N={n}", f"-DMSG_ELEMS={m}", f"-DELEM_T={t}", "-o", f"build/{cfg}", "harness/main.cu",
           f"-L{cuda_home}/lib64/stubs", "-lcuda", "-lnvidia-ml"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    text = p.stdout + p.stderr
    with open(f"build/logs/{cfg}.log", "w") as lf:
        lf.write(" ".join(cmd) + "\n" + text + f"\nexit code {p.returncode}\n")
    static, regs, spill, status, msg = parse_log(text)
    if p.returncode == 0:
        with open(f"build/{cfg}", "rb") as bf:
            tag = re.search(rb"COLLECTIVE_SMEM_BYTES=(\d+);", bf.read())
        return "OK", (tag.group(1).decode() if tag else ""), static, regs, spill, ""
    if os.path.exists(f"build/{cfg}"):
        os.remove(f"build/{cfg}")  # never leave a stale binary behind a failed build
    return status, "", static, regs, spill, msg


def latest_build():
    """Latest row of every (file, N, M, T) in build_matrix_v2.csv from complete build runs."""
    done = {r["run_id"] for r in read_csv(f"{RES}/collectives__runs.csv") if r["status"] == "complete"}
    latest = {}
    for r in read_csv(MATRIX):
        if r["run_id"] in done:
            latest[(r["file"], int(r["N"]), int(r["M"]), r["T"])] = r  # file order = chronological
    return latest


class Builder:
    def __init__(self, args, reg):
        if not shutil.which("nvcc"):
            raise SystemExit("nvcc not found: run inside tools/run --no-gpu")
        self.cuda_home = os.path.dirname(os.path.dirname(shutil.which("nvcc")))
        self.args, self.reg, self.rel = args, reg, nvcc_release()
        self.common = max(os.path.getmtime(p) for p in ("harness/main.cu", "dsmem_primitives.cuh",
                                                         "../common/h100mb/host.hpp"))
        self.have = {(r["file"], int(r["N"]), int(r["M"]), r["T"]) for r in read_csv(MATRIX)}
        self.counts = {}

    def fresh(self, f, n, m, t):
        log = f"build/logs/{cfg_name(f, n, m, t)}.log"
        return (not self.args.force and (f, n, m, t) in self.have and os.path.exists(log)
                and os.path.getmtime(log) > max(self.common, os.path.getmtime(f"{f}.cu")))

    def run(self, cfgs, is_grid):
        """Build cfgs in parallel, append their rows; returns {cfg tuple: status}."""
        todo = [c for c in cfgs if not self.fresh(*c)]
        print(f"build ({'grid' if is_grid else 'M_max'}): {len(todo)} to compile, {len(cfgs) - len(todo)} up to date, "
              f"{self.args.jobs} jobs", flush=True)
        out, t0 = {}, time.time()
        with cf.ThreadPoolExecutor(max_workers=self.args.jobs) as ex:
            futs = {ex.submit(build_one, *c, self.cuda_home): c for c in todo}
            for i, fu in enumerate(cf.as_completed(futs), 1):
                f, n, m, t = futs[fu]
                status, smem, static, regs, spill, msg = fu.result()
                op, alg = split(f)
                append_rows(MATRIX, BUILD_HDR, [["", "", "", "", self.rel, "", self.reg.run_id, -1, f, op, alg, n, m,
                                                 t, int(is_grid), status, smem, static, regs, spill, msg]])
                self.have.add((f, n, m, t))
                out[(f, n, m, t)] = status
                self.counts[status] = self.counts.get(status, 0) + 1
                if status == "ERROR" or i % 100 == 0 or i == len(todo):
                    print(f"[{i}/{len(todo)} {time.time() - t0:.0f}s] {cfg_name(f, n, m, t)}: {status} {msg}",
                          flush=True)
        return out


def mmax_fit(lb, args):
    """(file, N, T) -> (unrounded largest M that fits, largest OK grid M), from the two largest OK grid builds."""
    fits = {}
    for f in files():
        for t in types_of(f):
            for n in NS:
                if not selected(args, f, n, t):
                    continue
                ok = sorted((m, int(r["smem_bytes"])) for (ff, nn, m, tt), r in lb.items()
                            if (ff, nn, tt) == (f, n, t) and m in MS and r["status"] == "OK" and r["smem_bytes"])
                if len(ok) < 2:
                    continue
                (m1, s1), (m2, s2) = ok[-2], ok[-1]
                a = (s2 - s1) / (m2 - m1)
                if a > 0:
                    fits[(f, n, t)] = (int((SMEM_LIMIT - (s2 - a * m2)) // a), m2)
    return fits


def cmd_build(args):
    os.makedirs("build/logs", exist_ok=True)
    reg = Registry(sys.argv)
    b = Builder(args, reg)
    b.run(grid(args), True)
    if not args.no_mmax:
        # Per (file, N, T): granularity 4 elements (16 B) first; on a divisibility static_assert retry with 4*N, then
        # 16*N; on SKIP:smem (padding) step down by the granularity, at most 3 times.
        state = {key: [raw, top, [4, 4 * key[1], 16 * key[1]], 0, 0]
                 for key, (raw, top) in mmax_fit(latest_build_including(reg.run_id), args).items()}
        while state:
            cands = {}
            for key, (raw, top, gs, gi, down) in state.items():
                m = raw // gs[gi] * gs[gi] - down * gs[gi]
                if m > top and m not in MS:
                    cands[key] = m
            if not cands:
                break
            res = b.run([(f, n, m, t) for (f, n, t), m in cands.items()], False)
            lb = latest_build_including(reg.run_id)
            nxt = {}
            for (f, n, t), m in cands.items():
                st = res.get((f, n, m, t)) or lb[(f, n, m, t)]["status"]
                raw, top, gs, gi, down = state[(f, n, t)]
                if st == "SKIP:static_assert" and gi + 1 < len(gs):
                    nxt[(f, n, t)] = [raw, top, gs, gi + 1, 0]
                elif st == "SKIP:smem" and down < 3:
                    nxt[(f, n, t)] = [raw, top, gs, gi, down + 1]
            state = nxt
    reg.complete()
    print("build done:", b.counts)


def latest_build_including(run_id):
    """latest_build() plus the rows of the (still running, not yet complete) build run_id."""
    done = {r["run_id"] for r in read_csv(f"{RES}/collectives__runs.csv") if r["status"] == "complete"} | {run_id}
    latest = {}
    for r in read_csv(MATRIX):
        if r["run_id"] in done:
            latest[(r["file"], int(r["N"]), int(r["M"]), r["T"])] = r
    return latest


def gpu_idle(timeout_s=120):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        p = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True,
                           text=True)
        if p.returncode == 0 and not p.stdout.strip():
            return True
        time.sleep(1)
    return False


def cmd_run(args):
    lb = latest_build()
    ok = {c: r for c, r in lb.items() if r["status"] == "OK"}
    best = {}
    for (f, n, m, t), r in ok.items():  # of the M_max points keep only the largest OK one per (file, N, T)
        if m not in MS and m > best.get((f, n, t), 0):
            best[(f, n, t)] = m
    ok = {c: r for c, r in ok.items() if c[2] in MS or best[(c[0], c[1], c[3])] == c[2]}
    sel = []
    if args.smoke:
        for f in files():
            if args.files and f not in args.files:
                continue
            ns = sorted(n for (ff, n, m, t) in ok if ff == f and m == 256 and t == "float")
            if ns:
                sel.append((f, 4 if 4 in ns else ns[0], 256, "float"))
            else:
                print(f"smoke: {f} has no OK build at M = 256 float")
            mine = [(int(r["smem_bytes"] or 0), c[1], c) for c, r in ok.items() if c[0] == f]
            if mine:
                big = max(mine)[2]
                if big not in sel:
                    sel.append(big)
    else:
        sel = sorted(c for c in ok if selected(args, c[0], c[1], c[3]) and (not args.M or c[2] in args.M))
    if args.pattern == "B":
        sel = [c for c in sel if c[3] == "float"]  # pattern B is float-only
    if args.shuffle:
        random.Random(args.seed).shuffle(sel)
    done = {r["cmdline"] for r in read_csv(f"{RES}/collectives__runs.csv") if r["status"] == "complete"}
    os.makedirs(f"{RES}/logs", exist_ok=True)
    reg = Registry(sys.argv)
    stats, k = {}, 0
    print(f"run: {len(sel)} configurations x {args.processes} processes, mode {args.mode}, pattern {args.pattern}")
    for f, n, m, t in sel:
        cfg = cfg_name(f, n, m, t)
        if not os.path.exists(f"build/{cfg}"):
            print(f"{cfg}: binary missing (rebuild)")
            stats["MISSING"] = stats.get("MISSING", 0) + 1
            continue
        for proc in range(args.processes):
            cmd = [f"build/{cfg}", "--mode", args.mode, "--pattern", args.pattern, "--reps", str(args.reps),
                   "--process", str(proc)] + (["--timeout-ms", str(args.timeout_ms)] if args.timeout_ms else [])
            line = " ".join(cmd)
            if line in done and not args.rerun:
                stats["SKIPPED_COMPLETE"] = stats.get("SKIPPED_COMPLETE", 0) + 1
                continue
            t0 = time.time()
            with open(f"{RES}/logs/{cfg}.log", "a") as lf:
                lf.write(f"=== {iso_now()} {line}\n")
                lf.flush()
                try:
                    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                       timeout=args.child_timeout)
                    out, rc = p.stdout, p.returncode
                    outcome = {0: "PASS", 1: "FAIL"}.get(rc, "DEADLOCK" if rc == 3 and "DEADLOCK" in out else "ERROR")
                except subprocess.TimeoutExpired as e:
                    out = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
                    rc, outcome = -9, "KILLED_TIMEOUT"
                lf.write(out)
            wall = time.time() - t0
            append_rows(f"{RES}/collectives__driver.csv", DRIVER_HDR,
                        [["", "", "", "", "", "", reg.run_id, k, cfg, args.mode, args.pattern, proc, rc, outcome,
                          f"{wall:.2f}", line]])
            k += 1
            stats[outcome] = stats.get(outcome, 0) + 1
            tail = out.strip().splitlines()[-1] if out.strip() else ""
            print(f"[{k}] {cfg} p{proc}: {outcome} ({wall:.1f}s) {tail if outcome != 'PASS' else ''}", flush=True)
            if outcome in ("KILLED_TIMEOUT", "DEADLOCK") and not gpu_idle():
                print("GPU still busy 120 s after a killed child: stopping (run is partial)")
                print("run stats:", stats)
                sys.exit(4)
    reg.complete()
    print("run stats:", stats)


def cmd_summary(args):
    lb = latest_build()
    code = {"OK": ".", "SKIP:static_assert": "a", "SKIP:smem": "s", "ERROR": "E"}
    tot = {}
    for r in lb.values():
        tot[r["status"]] = tot.get(r["status"], 0) + 1
    print("grid status per (N, M): . OK  a static_assert  s smem (> 227 KiB)  E error  - not built;"
          f" each N block lists M = {MS}; then M_max (largest M that fits, smem bytes) per N")
    print(f"{'file':58s} T     " + " | ".join(f"N={n:<2d}   " for n in NS) + " | M_max N=2 / 4 / 8 / 16")
    for f in files():
        for t in types_of(f):
            cells, mx = [], []
            for n in NS:
                cells.append("".join(code.get(lb[(f, n, m, t)]["status"], "?") if (f, n, m, t) in lb else "-"
                                     for m in MS))
                oks = [(m, r["smem_bytes"]) for (ff, nn, m, tt), r in lb.items()
                       if (ff, nn, tt) == (f, n, t) and r["status"] == "OK"]
                mx.append(f"{max(oks)[0]}({int(max(oks)[1]) // 1024}K)" if oks else "-")
            print(f"{f:58s} {t:5s} " + " | ".join(f"{c:7s}" for c in cells) + " | " + " / ".join(mx))
    print("totals (grid + M_max points):", tot)
    for (f, n, m, t), r in sorted(lb.items()):
        if r["status"] == "ERROR":
            print(f"ERROR {cfg_name(f, n, m, t)}: {r['ptxas_msg']}")


def main():
    os.chdir(COLL)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "run", "summary"):
        p = sub.add_parser(name)
        p.add_argument("--files", nargs="*")
        p.add_argument("--N", nargs="*", type=int)
        p.add_argument("--M", nargs="*", type=int)
        p.add_argument("--T", nargs="*")
        if name == "build":
            p.add_argument("--jobs", type=int, default=16)
            p.add_argument("--force", action="store_true")
            p.add_argument("--no-mmax", action="store_true", help="grid only, no M_max points")
        if name == "run":
            p.add_argument("--mode", default="correctness", choices=["correctness", "perf"])
            p.add_argument("--pattern", default="A", choices=["A", "B"])
            p.add_argument("--reps", type=int, default=200)
            p.add_argument("--processes", type=int, default=3)
            p.add_argument("--timeout-ms", type=int, default=0, help="child watchdog (0 = harness default 5000)")
            p.add_argument("--child-timeout", type=float, default=300.0)
            p.add_argument("--smoke", action="store_true")
            p.add_argument("--shuffle", action="store_true")
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--rerun", action="store_true", help="also run command lines that are already complete")
    args = ap.parse_args()
    if args.cmd == "build":
        args.jobs = min(args.jobs, 16)  # the host is shared
    {"build": cmd_build, "run": cmd_run, "summary": cmd_summary}[args.cmd](args)


if __name__ == "__main__":
    main()
