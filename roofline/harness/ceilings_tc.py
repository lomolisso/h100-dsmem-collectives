"""roofline R1: dense tensor-core peaks with torch (DESIGN §2.1 "FP16/BF16 tensor-core peak", "FP8 tensor-core peak").

Run inside the roofline image (torch 2.11), from roofline/, on GPU 3 at the locked clock:
    python3 harness/ceilings_tc.py [--quick] [--size 8192] [--configs fp16,bf16,fp8,fp8_fastacc]
    python3 harness/ceilings_tc.py --ncu <config>      (one profiled GEMM; used by harness/ceilings_ncu.sh)

Configurations (M = N = K = --size, default 8192; inputs torch.randn on the GPU, no checkpoint):
    fp16         torch.matmul(a, b, out=c), a, b, c float16 (cuBLAS, fp32 accumulation)
    bf16         the same in bfloat16
    fp8          torch._scaled_mm(a, b, scale_a, scale_b, out_dtype=bfloat16, use_fast_accum=False):
                 a float8_e4m3fn row-major M x K, b float8_e4m3fn column-major K x N (b = b_t.t(), b_t N x K contiguous),
                 per-tensor fp32 scales 1.0 (cuBLASLt FP8 GEMM)
    fp8_fastacc  the same with use_fast_accum=True (Hopper fast-accumulation mode; the summary takes the faster of the
                 two fp8 rows as fp8_tc_peak and names the variant in `method`)
FLOPs per GEMM = 2·M·N·K. Every config: one correctness check of the top-left 256 x 256 block against an fp32 matmul of
the same (rounded) inputs (max |c − ref| / max |ref|, column max_rel_err; valid = 1 if below 2e-2), 5 warm-up GEMMs,
then (after --cool-s idle seconds per config and --gap-s before every launch, so each GEMM starts below the 400 W
limit: back-to-back 8192³ GEMMs pull the SM clock below the lock, R1 run 2026-10-26T15:15) GEMMs each between its own cudaEvent pair until 10 unflagged launches exist (3 with --quick); after every launch one
NVML clock sample: clock_bad = 1 if a throttle reason fired (SW/HW power cap, SW/HW thermal, power brake) or the SM clock
is more than 15 MHz below H100MB_LOCK_SM_MHZ — the launch is written, not counted, and repeated (at most 20 flagged per
config, then the config is abandoned with fewer rows). A config that raises (e.g. an fp8 path missing in this torch
build) gets one row with impl = "NOT MEASURED: <error>" and empty timing; the run continues. No L2 flush (operands of 64–128 MiB per matrix
exceed the 60 MiB L2; the GEMM is compute-bound). ceilings_summary.py takes the best of the unflagged launches of the
latest complete run ("best of 10", DESIGN §2.1) net of the cudaEvent floor and reports the median too.

Rows (append-only, crash-safe like hb::CsvWriter; mandatory columns first, docs/ENVIRONMENT.md §5):
    results/ceilings/tc.csv   MANDATORY + config, dtype, impl, m, n, k, flops, ms_event, tflops_raw, max_rel_err, valid,
                              clock_bad
    results/roofline__runs.csv (started / complete rows, same columns as hb::Run), results/roofline__clock_log.csv (one row
    per launch and before/after, same columns as hb::Run::clock_note).
    --ncu: results/ceilings/ncu/manifest_tc.csv  MANDATORY + config, dtype, impl, m, n, k, flops, ms_event; the profiled
    GEMM runs between torch.cuda.profiler.start()/stop() (ncu --profile-from-start off).
Device checks as hb::DeviceInfo::open(true): H100 NVL, enforced power limit 400 W, ECC on, MIG off, UUID = H100MB_GPU_UUID.
"""
import argparse
import time
import csv
import fcntl
import io
import os
import sys
from datetime import datetime

import pynvml
import torch

RES = "results"
MANDATORY = ["gpu_name", "sm_count", "sm_clock_mhz", "mem_clock_mhz", "cuda_version", "driver_version", "run_id", "launch_idx"]
REG_HDR = ["run_id", "started_iso", "finished_iso", "status", "gpu_uuid", "cmdline"]
CLOCK_HDR = MANDATORY + ["label", "phase", "clocks_event_reasons_hex", "throttle", "below_lock", "power_w", "gpu_temp_c",
                         "mem_temp_c", "lock_sm_mhz"]
TC_HDR = MANDATORY + ["config", "dtype", "impl", "m", "n", "k", "flops", "ms_event", "tflops_raw", "max_rel_err", "valid",
                      "clock_bad"]
MAN_HDR = MANDATORY + ["config", "dtype", "impl", "m", "n", "k", "flops", "ms_event"]
THROTTLE_MASK = 0x4 | 0x8 | 0x20 | 0x40 | 0x80  # SwPowerCap, HwSlowdown, SwThermal, HwThermal, HwPowerBrake (host.hpp)
WARMUPS, RECORDED, QUICK_RECORDED, MAX_FLAGGED = 5, 10, 3, 20
TOL = 2e-2
CONFIGS = ["fp16", "bf16", "fp8", "fp8_fastacc"]


def iso_now():
    """Same format as hb::iso_now: local time, ms, numeric UTC offset."""
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def append_rows(path, header, rows):
    """Crash-safe append (hb::CsvWriter semantics): header only into an empty file, checked otherwise; flock; fsync.
    (copied from collectives/harness/driver.py)"""
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


def _s(x):
    return x.decode() if isinstance(x, bytes) else x


class Device:
    """hb::DeviceInfo in Python: identity, fixed-configuration checks, NVML clock samples."""

    def __init__(self, enforce=True):
        torch.cuda.init()
        props = torch.cuda.get_device_properties(0)
        self.name = torch.cuda.get_device_name(0)
        self.sm_count = props.multi_processor_count
        self.cuda_version = torch.version.cuda  # runtime the GEMMs use, e.g. "12.9" (hb: cudaRuntimeGetVersion)
        pynvml.nvmlInit()
        self.uuid = os.environ.get("H100MB_GPU_UUID", "")
        if self.uuid:
            self.h = pynvml.nvmlDeviceGetHandleByUUID(self.uuid)
        else:  # h100 run always sets it; outside h100 run use the only visible GPU
            self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.uuid = _s(pynvml.nvmlDeviceGetUUID(self.h))
        self.driver_version = _s(pynvml.nvmlSystemGetDriverVersion())
        self.lock_mhz = int(os.environ.get("H100MB_LOCK_SM_MHZ", "0") or 0)
        if enforce:
            if "H100 NVL" not in self.name:
                raise SystemExit(f"device is '{self.name}', expected an H100 NVL")
            pl = pynvml.nvmlDeviceGetEnforcedPowerLimit(self.h)
            if pl != 400000:
                raise SystemExit(f"enforced power limit is {pl // 1000} W, expected 400 W")
            ecc_cur, _ = pynvml.nvmlDeviceGetEccMode(self.h)
            if ecc_cur != pynvml.NVML_FEATURE_ENABLED:
                raise SystemExit("ECC is disabled; the fixed configuration requires ECC enabled")
            try:
                mig_cur, _ = pynvml.nvmlDeviceGetMigMode(self.h)
            except pynvml.NVMLError:
                mig_cur = 0
            if mig_cur == pynvml.NVML_DEVICE_MIG_ENABLE:
                raise SystemExit("MIG is enabled; the fixed configuration requires MIG disabled")

    def sample(self):
        h = self.h
        s = {"sm": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
             "mem": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM)}
        get_reasons = getattr(pynvml, "nvmlDeviceGetCurrentClocksEventReasons", None) or \
            pynvml.nvmlDeviceGetCurrentClocksThrottleReasons
        s["reasons"] = int(get_reasons(h))
        try:
            s["power_w"] = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
        except pynvml.NVMLError:
            s["power_w"] = 0.0
        s["gpu_temp"] = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        s["mem_temp"] = 0
        try:
            fv = pynvml.nvmlDeviceGetFieldValues(h, [pynvml.NVML_FI_DEV_MEMORY_TEMP])[0]
            if fv.nvmlReturn == pynvml.NVML_SUCCESS:
                s["mem_temp"] = fv.value.uiVal
        except (pynvml.NVMLError, AttributeError):
            pass
        s["throttle"] = (s["reasons"] & THROTTLE_MASK) != 0
        s["below_lock"] = self.lock_mhz > 0 and s["sm"] + 15 < self.lock_mhz
        s["bad"] = s["throttle"] or s["below_lock"]
        return s


class Run:
    """hb::Run in Python: run_id, registry rows, clock log, mandatory prefix."""

    def __init__(self, dev, argv):
        self.dev, self.run_id, self.cmd = dev, iso_now(), " ".join(argv)
        append_rows(f"{RES}/roofline__runs.csv", REG_HDR, [[self.run_id, iso_now(), "", "started", dev.uuid, self.cmd]])
        print(f"run_id {self.run_id}  {dev.name}  {dev.sm_count} SMs  lock {dev.lock_mhz} MHz  driver {dev.driver_version}  "
              f"CUDA {dev.cuda_version}  torch {torch.__version__}", flush=True)

    def complete(self):
        append_rows(f"{RES}/roofline__runs.csv", REG_HDR, [[self.run_id, "", iso_now(), "complete", self.dev.uuid, self.cmd]])

    def prefix(self, launch_idx, s):
        d = self.dev
        return [d.name, d.sm_count, s["sm"], s["mem"], d.cuda_version, d.driver_version, self.run_id, launch_idx]

    def clock_note(self, label, phase, launch_idx=-1):
        s = self.dev.sample()
        append_rows(f"{RES}/roofline__clock_log.csv", CLOCK_HDR,
                    [self.prefix(launch_idx, s) + [label, phase, f"0x{s['reasons']:016x}", int(s["throttle"]), int(s["below_lock"]),
                                                    f"{s['power_w']:.2f}", s["gpu_temp"], s["mem_temp"], self.dev.lock_mhz]])
        if s["bad"]:
            print(f"WARNING [{label}/{phase}] sm {s['sm']} MHz (lock {self.dev.lock_mhz}), reasons 0x{s['reasons']:x} -> launch marked bad",
                  file=sys.stderr, flush=True)
        return s


def make_gemm(config, n):
    """Returns (dtype name, impl string, fn() running one GEMM, check() -> max_rel_err, tensors to keep alive)."""
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(20261026)
    if config in ("fp16", "bf16"):
        dt = torch.float16 if config == "fp16" else torch.bfloat16
        a = torch.randn(n, n, device=dev, dtype=dt, generator=g)
        b = torch.randn(n, n, device=dev, dtype=dt, generator=g)
        c = torch.empty(n, n, device=dev, dtype=dt)

        def fn():
            torch.matmul(a, b, out=c)

        def check():
            fn()
            torch.cuda.synchronize()
            ref = a[:256].float() @ b[:, :256].float()
            return ((c[:256, :256].float() - ref).abs().max() / ref.abs().max()).item()

        impl = "torch.matmul(out=), cuBLAS, fp32 accumulate"
        return str(dt).replace("torch.", ""), impl, fn, check, (a, b, c)
    if config in ("fp8", "fp8_fastacc"):
        fast = config == "fp8_fastacc"
        f8 = torch.float8_e4m3fn
        a = torch.randn(n, n, device=dev, dtype=torch.float32, generator=g).to(f8)          # M x K row-major
        b = torch.randn(n, n, device=dev, dtype=torch.float32, generator=g).to(f8).t()      # K x N column-major
        sa = torch.tensor(1.0, device=dev, dtype=torch.float32)
        sb = torch.tensor(1.0, device=dev, dtype=torch.float32)
        out = {}

        def fn():
            r = torch._scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16, use_fast_accum=fast)
            out["c"] = r[0] if isinstance(r, tuple) else r  # older torch returned (out, amax)

        def check():
            fn()
            torch.cuda.synchronize()
            ref = a[:256].float() @ b[:, :256].float()
            return ((out["c"][:256, :256].float() - ref).abs().max() / ref.abs().max()).item()

        impl = f"torch._scaled_mm, cuBLASLt, e4m3 x e4m3 -> bf16, per-tensor scales 1.0, use_fast_accum={fast}"
        return "float8_e4m3fn", impl, fn, check, (a, b, sa, sb, out)
    raise SystemExit(f"unknown config {config}")


def timed(fn):
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    fn()
    e1.record()
    e1.synchronize()
    return e0.elapsed_time(e1)  # ms


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--size", type=int, default=8192)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--gap-s", type=float, default=1.0, help="idle seconds before every timed GEMM (power-limit recovery)")
    ap.add_argument("--cool-s", type=float, default=10.0, help="idle seconds before every config")
    ap.add_argument("--ncu", metavar="CONFIG", help="one GEMM of CONFIG between cudaProfilerStart/Stop, manifest row")
    args = ap.parse_args()
    os.makedirs(f"{RES}/ceilings/ncu", exist_ok=True)
    dev = Device(enforce=True)
    run = Run(dev, sys.argv)
    n = args.size
    flops = 2 * n * n * n
    li = 0
    if args.ncu:
        dt, impl, fn, check, keep = make_gemm(args.ncu, n)
        err = check()  # also the warm-up (cuBLAS heuristics, workspace) outside the profiled range
        run.clock_note(f"tc_ncu_{args.ncu}", "before")
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        ms = timed(fn)
        torch.cuda.profiler.stop()
        s = run.clock_note(f"tc_ncu_{args.ncu}", "launch", li)
        append_rows(f"{RES}/ceilings/ncu/manifest_tc.csv", MAN_HDR,
                    [run.prefix(li, s) + [args.ncu, dt, impl, n, n, n, flops, f"{ms:.6f}"]])
        print(f"ncu {args.ncu}: {ms:.3f} ms under the profiler, max_rel_err {err:.2e}")
        run.complete()
        return
    rec_n = QUICK_RECORDED if args.quick else RECORDED
    run.clock_note("tc", "before")
    for cfg in args.configs.split(","):
        try:
            dt, impl, fn, check, keep = make_gemm(cfg, n)
            err = check()
        except (RuntimeError, TypeError, AttributeError) as e:  # e.g. an fp8 path this torch build does not offer
            msg = f"NOT MEASURED: {type(e).__name__}: {e}".replace("\n", " ")[:300]
            print(f"{cfg}: {msg}", flush=True)
            s = run.clock_note(f"tc_{cfg}", "error", li)
            append_rows(f"{RES}/ceilings/tc.csv", TC_HDR,
                        [run.prefix(li, s) + [cfg, "", msg, n, n, n, flops, "", "", "", 0, int(s["bad"])]])
            li += 1
            torch.cuda.empty_cache()
            continue
        valid = int(err < TOL)
        if not valid:
            print(f"{cfg}: max_rel_err {err:.3e} >= {TOL} — rows written with valid = 0, not used by the summary", flush=True)
        time.sleep(args.cool_s)
        for _ in range(WARMUPS):
            fn()
        torch.cuda.synchronize()
        good, flagged, best, rows = 0, 0, 0.0, []
        while good < rec_n:
            time.sleep(args.gap_s)
            ms = timed(fn)
            s = run.clock_note(f"tc_{cfg}", "launch", li)
            tf = flops / (ms * 1e-3) / 1e12
            append_rows(f"{RES}/ceilings/tc.csv", TC_HDR,
                        [run.prefix(li, s) + [cfg, dt, f"{impl}; gap {args.gap_s:g} s", n, n, n, flops, f"{ms:.6f}", f"{tf:.3f}", f"{err:.3e}", valid,
                                              int(s["bad"])]])
            li += 1
            if s["bad"]:
                flagged += 1
                if flagged > MAX_FLAGGED:
                    print(f"{cfg}: more than {MAX_FLAGGED} clock-flagged launches — abandoned", flush=True)
                    break
                continue
            good += 1
            best = max(best, tf)
        print(f"{cfg:12s} best {best:7.1f} TFLOP/s of {good} unflagged launches ({flagged} flagged), max_rel_err {err:.2e}  "
              f"[{impl}]", flush=True)
        del keep, fn, check
        torch.cuda.empty_cache()
    run.clock_note("tc", "after")
    run.complete()


if __name__ == "__main__":
    main()
