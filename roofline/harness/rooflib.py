"""Shared helpers of the roofline harness: crash-safe CSVs, run registry, GPU/clock queries, ncu CSV parsing.

`append_rows`, `iso_now`, `read_csv`, `Registry` (renamed results path), `available` and `parse` are copied from
`collectives/harness/driver.py` and `collectives/harness/ncu_sweep.py` (repo commit c2520a1); semantics unchanged
(hb::CsvWriter / hb::Run conventions, ../../docs/ENVIRONMENT.md §5)."""
import csv
import fcntl
import io
import os
import re
import subprocess
from datetime import datetime

REPO = os.environ.get("ROOFLINE_REPO") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOF = os.path.join(REPO, "roofline")
RES = os.path.join(ROOF, "results")
BUILD = os.path.join(ROOF, "build")
HARNESS = os.path.join(ROOF, "harness")

MANDATORY = ["gpu_name", "sm_count", "sm_clock_mhz", "mem_clock_mhz", "cuda_version", "driver_version", "run_id",
             "launch_idx"]
REG_HDR = ["run_id", "started_iso", "finished_iso", "status", "gpu_uuid", "cmdline"]
# clocks_event_reasons bits that invalidate a timed launch (same mask as hb::ClockSample::bad): SW power cap,
# HW slowdown, SW thermal, HW thermal, HW power brake.
BAD_REASONS = 0x4 | 0x8 | 0x20 | 0x40 | 0x80
CLOCK_TOL_MHZ = 15


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
    """started/complete rows in results/roofline__runs.csv, same columns as hb::Run."""

    def __init__(self, argv):
        self.run_id = iso_now()
        self.cmd = " ".join(argv)
        self.path = os.path.join(RES, "roofline__runs.csv")
        append_rows(self.path, REG_HDR, [[self.run_id, iso_now(), "", "started", os.environ.get("H100MB_GPU_UUID", ""),
                                          self.cmd]])

    def complete(self, status="complete"):
        append_rows(self.path, REG_HDR, [[self.run_id, "", iso_now(), status, os.environ.get("H100MB_GPU_UUID", ""),
                                          self.cmd]])


def smi(fields):
    p = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"], capture_output=True,
                       text=True)
    return [x.strip() for x in p.stdout.strip().splitlines()[0].split(",")] if p.returncode == 0 and p.stdout else []


def clock_sample():
    """{'sm': MHz, 'mem': MHz, 'reasons': int, 'bad': bool} against the lock in H100MB_LOCK_SM_MHZ."""
    v = smi("clocks.sm,clocks.mem,clocks_event_reasons.active")
    if len(v) < 3:
        return {"sm": "", "mem": "", "reasons": -1, "bad": True}
    sm, mem = int(float(v[0])), int(float(v[1]))
    reasons = int(v[2], 16) if v[2].startswith("0x") else int(v[2])
    lock = int(os.environ.get("H100MB_LOCK_SM_MHZ") or 0)
    bad = bool(reasons & BAD_REASONS) or (lock > 0 and sm < lock - CLOCK_TOL_MHZ)
    return {"sm": sm, "mem": mem, "reasons": reasons, "bad": bad}


_DEV = {}


def device():
    """gpu_name, sm_count, cuda_version, driver_version (cached); sm_count from the CUDA runtime via torch."""
    if not _DEV:
        name, drv = (smi("name,driver_version") + ["", ""])[:2]
        p = subprocess.run(["python3", "-c", "import torch; print(torch.cuda.get_device_properties(0).multi_processor_count)"],
                           capture_output=True, text=True)
        sms = p.stdout.strip() if p.returncode == 0 else ""
        nv = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout
        m = re.search(r"release (\d+\.\d+)", nv)
        _DEV.update(gpu_name=name, sm_count=sms, cuda_version=m.group(1) if m else "", driver_version=drv)
    return _DEV


def mandatory(run_id, launch_idx, clk):
    d = device()
    return [d["gpu_name"], d["sm_count"], clk["sm"], clk["mem"], d["cuda_version"], d["driver_version"], run_id,
            launch_idx]


def available(metrics, query_file):
    """Keep only metrics whose base name `ncu --query-metrics` lists (MB 05 run.sh `available()`)."""
    if not os.path.exists(query_file):
        return metrics
    names = {l.split()[0] for l in open(query_file) if l.strip()}
    keep = [m for m in metrics if m.split(".")[0] in names]
    for m in metrics:
        if m not in keep:
            print("not listed by ncu --query-metrics, dropped:", m)
    return keep


def parse(path):
    """[(launch, {metric: (unit, value)})] from an ncu --csv --page raw capture."""
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


def num(x):
    try:
        return float(str(x).replace(",", ""))
    except ValueError:
        return float("nan")
