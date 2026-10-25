#!/usr/bin/env python3
"""driver.py — build matrix and child-process runner for the collectives harness (TESTING.md §1–2).

Run inside the microbench container from anywhere (it chdirs to collectives/):
  tools/run --no-gpu -- python3 collectives/harness/driver.py build [--jobs 16] [--files F ...] [--N ...] [--M ...] [--T ...] [--force]
  tools/run --profile --lock-sm-mhz <LOCK> -- python3 collectives/harness/driver.py run --mode correctness \
        --pattern A --reps 200 --processes 3 [--smoke] [--files ...] [--N ...] [--M ...] [--T ...] [--shuffle --seed S]
  tools/run --no-gpu -- python3 collectives/harness/driver.py summary

build: one binary per (file, N, M, T), N in {2,4,8,16}, M in {16,64,256,1024,4096}, T = float for every file and
  int for the Sum files (allreduce, reducescatter). Command (TESTING §1, plus the -DCOLL_* knobs main.cu needs):
    nvcc -arch=sm_90a -std=c++17 -O3 -lineinfo -Xptxas -v -I ../common/h100mb -DCOLLECTIVE_FILE='"../<file>.cu"'
         -DCOLL_NAME='"<file>"' -DCOLL_OP=<k> -DCLUSTER_N=N -DMSG_ELEMS=M -DELEM_T=T
         -o build/<file>__N<N>_M<M>_<T> harness/main.cu -L$CUDA_HOME/lib64/stubs -lcuda -lnvidia-ml
  Compiler output → build/logs/<cfg>.log. Status: OK (exit 0); SKIP:static_assert ("static assertion failed");
  SKIP:smem ("uses too much shared data"); ERROR (anything else). From the collective_kernel entry of the
  ptxas -v output: registers, static smem bytes (for SKIP:smem: the bytes in the ptxas error) and spill-store bytes.
  One row per built configuration is appended to results/build_matrix.csv; a configuration whose log is newer than
  every source it depends on is not rebuilt (and gets no new row) unless --force. At most --jobs (16) parallel nvcc.
run: every OK configuration selected by the filters (--smoke: M = 256, float, N = 4 or, if the file has no OK
  N = 4 build there, its smallest OK N), --processes fresh child processes each, in file order or shuffled
  (--shuffle --seed). Child: build/<cfg> --mode ... --pattern ... --reps ... --process k [--timeout-ms ...],
  cwd collectives/, stdout+stderr appended to results/logs/<cfg>.log. A child whose exact command line already has
  a `complete` row in results/collectives__runs.csv is skipped (resumable). A child that outlives --child-timeout
  (s) is killed; the driver then waits until no compute process is left on the GPU before the next child.
  Outcomes go to results/collectives__driver.csv: PASS (exit 0), FAIL (exit 1), DEADLOCK (exit 3 and the watchdog
  line), ERROR (other exit codes), KILLED_TIMEOUT.
summary: per file and T, the N x M matrix of the latest build status (. OK, a static_assert, s smem, E error),
  totals per status, and every ERROR with its first error line.
Every CSV starts with the mandatory columns (docs/ENVIRONMENT.md §5); for build-only rows the GPU fields are
empty, cuda_version is the nvcc release, launch_idx = -1. Both commands register themselves in
results/collectives__runs.csv (started / complete).
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
REPO = os.path.dirname(COLL)
OPS = {"allreduce": 0, "allgather": 1, "reducescatter": 2, "alltoall": 3}
SUM_OPS = {"allreduce", "reducescatter"}
NS, MS = [2, 4, 8, 16], [16, 64, 256, 1024, 4096]
MANDATORY = ["gpu_name", "sm_count", "sm_clock_mhz", "mem_clock_mhz", "cuda_version", "driver_version", "run_id",
             "launch_idx"]
REG_HDR = ["run_id", "started_iso", "finished_iso", "status", "gpu_uuid", "cmdline"]
BUILD_HDR = MANDATORY + ["file", "op", "alg", "N", "M", "T", "status", "static_smem_bytes", "regs", "spill_bytes",
                         "ptxas_msg"]
DRIVER_HDR = MANDATORY + ["cfg", "mode", "pattern", "process", "returncode", "outcome", "wall_s", "cmdline"]
RES = "results"


def iso_now():
    """Same format as hb::iso_now: local time, ms, numeric UTC offset."""
    s = datetime.now().astimezone().isoformat(timespec="milliseconds")
    return s


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


def cfg_name(f, n, m, t):
    return f"{f}__N{n}_M{m}_{t}"


def matrix(args):
    out = []
    for f in files():
        if args.files and f not in args.files:
            continue
        op, _ = split(f)
        for t in (["float", "int"] if op in SUM_OPS else ["float"]):
            if args.T and t not in args.T:
                continue
            for n in NS:
                if args.N and n not in args.N:
                    continue
                for m in MS:
                    if args.M and m not in args.M:
                        continue
                    out.append((f, n, m, t))
    return out


def nvcc_release():
    try:
        v = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout
        m = re.search(r"release (\d+\.\d+)", v)
        return m.group(1) if m else ""
    except FileNotFoundError:
        return ""


def parse_log(text):
    """(status, static_smem_bytes, regs, spill_bytes, msg) from nvcc/ptxas output of one build."""
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
    if "static assertion failed" in text:
        sa = [l for l in errs if "static assertion failed" in l]
        return "SKIP:static_assert", smem, regs, spill, (sa[0] if sa else errs[0])[:300]
    big = re.search(r"uses too much shared data \((0x[0-9a-f]+) bytes", text)
    if big:
        return "SKIP:smem", str(int(big.group(1), 16)), regs, spill, [l for l in errs if "shared data" in l][0][:300]
    return "ERROR", smem, regs, spill, (errs[0] if errs else text.strip().splitlines()[-1] if text.strip() else "")[:300]


def build_one(f, n, m, t, cuda_home):
    cfg = cfg_name(f, n, m, t)
    op, _ = split(f)
    cmd = ["nvcc", "-arch=sm_90a", "-std=c++17", "-O3", "-lineinfo", "-Xptxas", "-v", "-I", "../common/h100mb",
           f'-DCOLLECTIVE_FILE="../{f}.cu"', f'-DCOLL_NAME="{f}"', f"-DCOLL_OP={OPS[op]}", f"-DCLUSTER_N={n}",
           f"-DMSG_ELEMS={m}", f"-DELEM_T={t}", "-o", f"build/{cfg}", "harness/main.cu",
           f"-L{cuda_home}/lib64/stubs", "-lcuda", "-lnvidia-ml"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    text = p.stdout + p.stderr
    with open(f"build/logs/{cfg}.log", "w") as lf:
        lf.write(" ".join(cmd) + "\n" + text + f"\nexit code {p.returncode}\n")
    if p.returncode == 0:
        _, smem, regs, spill, _ = parse_log(text)
        return cfg, ("OK", smem, regs, spill, "")
    if os.path.exists(f"build/{cfg}"):
        os.remove(f"build/{cfg}")  # never leave a stale binary behind a failed build
    return cfg, parse_log(text)


def cmd_build(args):
    if not shutil.which("nvcc"):
        raise SystemExit("nvcc not found: run inside tools/run --no-gpu")
    cuda_home = os.path.dirname(os.path.dirname(shutil.which("nvcc")))
    os.makedirs("build/logs", exist_ok=True)
    reg = Registry(sys.argv)
    rel = nvcc_release()
    common = [os.path.getmtime(p) for p in ("harness/main.cu", "dsmem_primitives.cuh", "../common/h100mb/host.hpp")]
    have = {(r["file"], r["N"], r["M"], r["T"]) for r in read_csv(f"{RES}/build_matrix.csv")}
    todo = []
    for f, n, m, t in matrix(args):
        log = f"build/logs/{cfg_name(f, n, m, t)}.log"
        fresh = os.path.exists(log) and os.path.getmtime(log) > max(common + [os.path.getmtime(f"{f}.cu")])
        if args.force or not fresh or (f, str(n), str(m), t) not in have:
            todo.append((f, n, m, t))
    print(f"build: {len(todo)} configurations to compile ({len(matrix(args)) - len(todo)} up to date), {args.jobs} jobs")
    counts, t0 = {}, time.time()
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(build_one, *c, cuda_home): c for c in todo}
        for i, fu in enumerate(cf.as_completed(futs), 1):
            f, n, m, t = futs[fu]
            cfg, (status, smem, regs, spill, msg) = fu.result()
            op, alg = split(f)
            append_rows(f"{RES}/build_matrix.csv", BUILD_HDR,
                        [["", "", "", "", rel, "", reg.run_id, -1, f, op, alg, n, m, t, status, smem, regs, spill,
                          msg]])
            counts[status] = counts.get(status, 0) + 1
            if status == "ERROR" or i % 50 == 0 or i == len(todo):
                print(f"[{i}/{len(todo)} {time.time() - t0:.0f}s] {cfg}: {status} {msg}", flush=True)
    reg.complete()
    print("build done:", counts)


def latest_build():
    rows = read_csv(f"{RES}/build_matrix.csv")
    done = {r["run_id"] for r in read_csv(f"{RES}/collectives__runs.csv") if r["status"] == "complete"}
    latest = {}
    for r in rows:
        if r["run_id"] in done:
            latest[(r["file"], int(r["N"]), int(r["M"]), r["T"])] = r  # file order = chronological
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
    sel = []
    if args.smoke:
        for f in files():
            if args.files and f not in args.files:
                continue
            ok = sorted(n for (ff, n, m, t), r in lb.items() if ff == f and m == 256 and t == "float"
                        and r["status"] == "OK")
            if ok:
                sel.append((f, 4 if 4 in ok else ok[0], 256, "float"))
            else:
                print(f"smoke: {f} has no OK build at M = 256 float")
    else:
        for c in matrix(args):
            r = lb.get(c)
            if r and r["status"] == "OK":
                sel.append(c)
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
                reg_note = "GPU still busy 120 s after a killed child: stopping (run is partial)"
                print(reg_note)
                print("run stats:", stats)
                sys.exit(4)
    reg.complete()
    print("run stats:", stats)


def cmd_summary(args):
    lb = latest_build()
    code = {"OK": ".", "SKIP:static_assert": "a", "SKIP:smem": "s", "ERROR": "E"}
    tot = {}
    print("status per (N, M): . OK  a static_assert  s smem  E error  - not built")
    print("each N block lists M =", MS, "left to right")
    print(f"{'file':58s} T     " + " | ".join(f"N={n:<2d}      " for n in NS))
    for f in files():
        op, _ = split(f)
        for t in (["float", "int"] if op in SUM_OPS else ["float"]):
            cells = []
            for n in NS:
                s = ""
                for m in MS:
                    r = lb.get((f, n, m, t))
                    s += code.get(r["status"], "?") if r else "-"
                    if r:
                        tot[r["status"]] = tot.get(r["status"], 0) + 1
                cells.append(f"     {s}")
            print(f"{f:58s} {t:5s} " + " | ".join(cells))
    print("M columns:", MS, "; totals:", tot)
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
