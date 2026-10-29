"""roofline R1: ceilings summary → results/ceilings.csv (no GPU needed).

    roofline/tools/run --no-gpu -- python3 roofline/harness/ceilings_summary.py [--quick] [--dry-run]

Reads, for every raw file under results/ceilings/, the rows of the latest *complete* run of the producing command
(results/roofline__runs.csv: `started` + `complete` rows; --quick runs are ignored unless --quick is given), drops the
launches flagged clock_bad / placement_bad (counted in n_flagged) and computes:
  event floor  median us_per_launch of the empty_pair rows (event_floor floor; MB 01 K5 / common05.event_floor_ms)
  hbm_{read,write,copy}_bw  k4 bytes / (ms_event − floor) per launch; write and copy payloads minus the written bytes still
               dirty in L2 at kernel end (= write payload − ncu dram__bytes_write of the same op, latest ncu_hbm.csv run;
               MB common05.hbm_write_factor); without ncu data no correction (stated in `method`)
  l2_bw_direct, l2_bw_stream, l2_bw   k3_direct / k3_stream bytes / (ms_event − floor); l2_bw = the higher median
  l1_smem_bw   k12: per configuration (threads, ILP) the 4096- and 8192-step launches are paired by launch order and SM:
               slope = Δbytes / Δcycles (MB common05.k12_slope); per pair the median over SMs; the configuration with the
               highest median; B/s = B/cycle/SM × sm_count × median SM clock of its blocks (cycles / %globaltimer ns);
               also l1_smem_bw_per_sm_cycle (B/cycle/SM)
  dsmem_bw_{ld,push}_cs<N>, dsmem_bw   per (op, CS) and launch index k: least-squares slope of event_ns over total_bytes
               across the passes {4, 8, 16, 32} (MB 11 step S: GPU-wide TB/s = 1 / slope; the floor and every fixed cost
               cancel); dsmem_bw = the highest median over op and CS (both named in `method`)
  fp32_peak, int32_peak   work / (ms_event − floor), median per ILP, the best ILP (work = 2 × FFMA or 2 × IMAD)
  {fp16,bf16}_tc_peak, fp8_tc_peak (+ fp8_tc_peak_{default,fast}_accum)   torch GEMM 8192³: best of the unflagged launches
               (DESIGN: best of 10), 2·M·N·K / (ms_event − floor); median / p10 / p90 in the extra columns; rows with
               valid = 0 (correctness check failed) are not used; a config that could not run → value empty, source
               not_measured, the error in `method`
  {fp16,fp16acc,bf16,fp8}_tc_peak_wgmma   tc_peak peak (back-to-back wgmma.mma_async, instruction-throughput ceiling):
               per configuration (N, accumulators, wait_group, warpgroups, fill) the median of flop / (ms_event − floor) over
               the unflagged launches; the configuration with the highest median (named in `method`, with the FLOP/clk/SM
               from %clock64 against the architectural 4096 / 8192). fp16_tc_peak, bf16_tc_peak, fp8_tc_peak are then the
               max of the torch/cuBLAS value above and the wgmma value(s) (fp16_tc_peak: fp16 and fp16acc wgmma, i.e. the
               ceiling of every fp16-input GEMM whatever its accumulator); `method` names the winner
  datasheet rows (source = datasheet; H100 NVL, dense, boost clock)
value = median over launches unless stated (tensor cores: best), p10 / p90 / n / n_flagged / source_run_ids as extra columns
(DESIGN §5: the notebook reads name, value, unit, method, source by name). Mandatory columns: gpu_name, sm_count, cuda and
driver version of the source rows, sm_clock_mhz / mem_clock_mhz = median NVML sample of the launches used, run_id = this
summary's run_id (registered in results/roofline__runs.csv), launch_idx = -1.
Also: ncu cross-checks (results/ceilings/ncu/ncu_*.csv, latest run each) → results/ceilings/crosscheck.csv and a printed
table; a comparison with the GPU-2 references of MB 05 / 11.
Append-only; nothing is overwritten.
"""
import argparse
import csv
import fcntl
import io
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # roofline/
RES = "results"
CEIL = f"{RES}/ceilings"
MANDATORY = ["gpu_name", "sm_count", "sm_clock_mhz", "mem_clock_mhz", "cuda_version", "driver_version", "run_id", "launch_idx"]
REG_HDR = ["run_id", "started_iso", "finished_iso", "status", "gpu_uuid", "cmdline"]
OUT_COLS = ["name", "value", "unit", "method", "source", "p10", "p90", "n", "n_flagged", "source_run_ids"]
XC_COLS = ["check", "key", "metric", "measured", "expected", "ratio", "note", "source_run_ids"]
DATASHEET = [("hbm_bw_datasheet", 3.9e12, "B/s"), ("fp32_peak_datasheet", 60e12, "FLOP/s"),
             ("fp16_tc_peak_datasheet", 835.5e12, "FLOP/s"), ("bf16_tc_peak_datasheet", 835.5e12, "FLOP/s"),
             ("fp8_tc_peak_datasheet", 1670.5e12, "FLOP/s")]
# H100 NVL (PNY/NVIDIA datasheet): FP32 60 TFLOP/s; tensor cores 1671 (FP16/BF16) and 3341 (FP8) TFLOP/s "with sparsity"
# -> dense = half. (DESIGN first listed the H100 SXM values 67 / 989 / 1979; corrected in R1.)
GPU2_REF = {"hbm_read_bw": 3.70e12, "hbm_write_bw": 3.62e12, "l2_bw": 7.7e12, "l1_smem_bw": 27.5e12}  # MB 05 (GPU 2)
GPU2_DSMEM = (2.0e12, 4.0e12)  # MB 11: 2.0–4.0 TB/s GPU-wide (ring, all co-resident clusters)


def iso_now():
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


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


# ---------------------------------------------------------------- run selection
def runs():
    r = pd.read_csv(f"{RES}/roofline__runs.csv", dtype=str, keep_default_na=False)
    started = r[r.status == "started"].set_index("run_id")
    done = set(r.loc[r.status == "complete", "run_id"])
    out = started[["started_iso", "cmdline"]].copy()
    out["complete"] = [i in done for i in out.index]
    return out.reset_index()


def latest_run(tool, mode=None, quick=False, exclude=()):
    """run_id of the latest complete invocation whose first token ends with /<tool> (and second token == mode)."""
    r = runs()
    toks = r.cmdline.str.split()

    def ok(t):
        if not t or not (t[0] == tool or t[0].endswith("/" + tool)):
            return False
        if mode is not None and (len(t) < 2 or t[1] != mode):
            return False
        if any(x in t for x in exclude):
            return False
        return ("--quick" in t) == quick

    sel = r[toks.map(ok) & r.complete]
    if sel.empty:
        return None
    return sel.sort_values("started_iso").run_id.iloc[-1]


def read_run(file, run_id):
    path = f"{CEIL}/{file}"
    if run_id is None or not os.path.exists(path):
        return None
    d = pd.read_csv(path, keep_default_na=False, na_values=[""])
    return d[d.run_id == run_id].reset_index(drop=True)


def good(d):
    bad = np.zeros(len(d), bool)
    for c in ("clock_bad", "placement_bad"):
        if c in d:
            bad |= d[c].fillna(0).astype(int).to_numpy() != 0
    return d[~bad], int(bad.sum())


def q(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return np.nan, np.nan, np.nan, 0
    return float(np.median(v)), float(np.percentile(v, 10)), float(np.percentile(v, 90)), len(v)


# ---------------------------------------------------------------- summary
class Summary:
    def __init__(self, quick):
        self.quick = quick
        self.rows, self.xc = [], []
        self.meta = None  # first source frame (device identity)
        self.lock = None
        self.clock_samples = []

    def note_src(self, d):
        if d is not None and len(d):
            if self.meta is None:
                self.meta = d.iloc[0]
            self.clock_samples.append(d[["sm_clock_mhz", "mem_clock_mhz"]])

    def add(self, name, value, unit, method, source="measured", p10=np.nan, p90=np.nan, n=0, flagged=0, rids=""):
        self.rows.append([name, value, unit, method, source, p10, p90, n, flagged, rids])

    def lock_note(self, rids):
        """Lock requested by h100 run, from the clock-log rows of the source runs."""
        try:
            cl = pd.read_csv(f"{RES}/roofline__clock_log.csv", usecols=["run_id", "lock_sm_mhz"])
            v = cl[cl.run_id.isin(rids)].lock_sm_mhz
            self.lock = int(v.mode().iloc[0]) if len(v) else None
        except (FileNotFoundError, ValueError):
            self.lock = None
        return f"SM clock locked at {self.lock} MHz" if self.lock else "SM clock lock unknown (no clock-log rows)"


def main():
    ap = argparse.ArgumentParser(description="roofline R1 ceilings summary")
    ap.add_argument("--quick", action="store_true", help="use the latest --quick runs (dry check only)")
    ap.add_argument("--dry-run", action="store_true", help="print, do not append to results/ceilings.csv")
    ap.add_argument("--root", default=ROOT, help="directory holding results/ (default: roofline/; for tests)")
    a = ap.parse_args()
    os.chdir(a.root)
    S = Summary(a.quick)
    src = {}
    # ---- event floor
    rf = latest_run("event_floor", "floor", a.quick)
    fl = read_run("event_floor.csv", rf)
    if fl is None or fl.empty:
        raise SystemExit("no complete event_floor run: run harness/ceilings/run_all.sh (step floor) first")
    flg, nfl = good(fl)
    floor_ms = flg[flg["mode"] == "empty_pair"].us_per_launch.median() / 1e3
    src["floor"] = rf
    print(f"event floor: {floor_ms * 1e3:.3f} us (median empty event pair, run {rf}; {nfl} flagged rows)")

    # ---- HBM
    rh = latest_run("bw_hbm", "k4", a.quick)
    hb_ = read_run("bw_hbm.csv", rh)
    ncu = load_ncu("hbm")
    if hb_ is not None and len(hb_):
        S.note_src(hb_)
        g, nf = good(hb_)
        for op in ["read", "write", "copy"]:
            d = g[g.op == op]
            payload = d.bytes.astype(float)
            dirty = 0.0
            corr = "reads only" if op == "read" else "no dirty-line correction (no ncu data)"
            if op != "read" and ncu is not None:
                m = ncu[(ncu.op == op) & (ncu.metric == "dram__bytes_write.sum")]
                if len(m):
                    wexp = float(m.write_bytes.iloc[0])
                    dirty = max(0.0, wexp - float(m.value.iloc[0]))
                    corr = f"payload minus {dirty / 2**20:.1f} MiB still dirty in L2 at kernel end (ncu run {m.run_id.iloc[0]})"
            rate = (payload - dirty) / ((d.ms_event - floor_ms) * 1e-3)
            med, p10, p90, n = q(rate)
            S.add(f"hbm_{op}_bw", med, "B/s",
                  f"k4 {op} (MB {d.mb_op.iloc[0] if len(d) else op}) .cg ILP 4, 4x132x1024 thr, 2 GiB, CleanL2 before each launch; "
                  f"median of {n} launches, bytes / (event time - {floor_ms * 1e3:.2f} us floor); {corr}; LOCK",
                  p10=p10, p90=p90, n=n, flagged=nf, rids=rh)
        src["hbm"] = rh
    else:
        S.add("hbm_read_bw", np.nan, "B/s", "no complete bw_hbm k4 run", "not_measured")

    # ---- L2
    rl = latest_run("bw_l2", "k3", a.quick)
    l2 = read_run("bw_l2.csv", rl)
    if l2 is not None and len(l2):
        S.note_src(l2)
        g, nf = good(l2)
        best = None
        for v, mb in [("direct", "mixed_direct"), ("stream", "mixed")]:
            d = g[g.variant == v]
            med, p10, p90, n = q(d.bytes / ((d.ms_event - floor_ms) * 1e-3))
            reps = int(d.reps.iloc[0]) if len(d) else 0
            meth = (f"k3_{v} (MB {mb}) ld.global.cg.v4, 16 MiB working set, 2x132x1024 thr, ILP 4, {reps} reps, warm L2; "
                    f"median of {n} launches net of the {floor_ms * 1e3:.2f} us event floor; LOCK")
            S.add(f"l2_bw_{v}", med, "B/s", meth, p10=p10, p90=p90, n=n, flagged=nf, rids=rl)
            if np.isfinite(med) and (best is None or med > best[0]):
                best = (med, p10, p90, n, meth)
        if best:
            S.add("l2_bw", best[0], "B/s", "higher of l2_bw_direct / l2_bw_stream: " + best[4], p10=best[1], p90=best[2],
                  n=best[3], flagged=nf, rids=rl)
        src["l2"] = rl
    else:
        S.add("l2_bw", np.nan, "B/s", "no complete bw_l2 k3 run", "not_measured")

    # ---- SMEM (slope)
    rs = latest_run("bw_smem", "k12", a.quick)
    sm = read_run("bw_smem.csv", rs)
    if sm is not None and len(sm):
        S.note_src(sm)
        g, nf_rows = good(sm)
        nf = int(sm[(sm.clock_bad != 0) | (sm.placement_bad != 0)].launch_idx.nunique())
        g = g.copy()
        g["mhz"] = g.cycles / g.ns * 1e3
        keys = ["threads", "ilp"]
        g["k"] = g.groupby(keys + ["iters", "smid"]).launch_idx.rank(method="first")
        a4 = g[g.iters == 4096].set_index(keys + ["smid", "k"])[["bytes", "cycles"]]
        a8 = g[g.iters == 8192].set_index(keys + ["smid", "k"])[["bytes", "cycles"]]
        j = a8.merge(a4, left_index=True, right_index=True, suffixes=("_8192", "_4096")).reset_index()
        j["bpc"] = (j.bytes_8192 - j.bytes_4096) / (j.cycles_8192 - j.cycles_4096)
        per_pair = j.groupby(keys + ["k"]).bpc.median().reset_index()
        med_cfg = per_pair.groupby(keys).bpc.median()
        t, i = med_cfg.idxmax()
        sel = per_pair[(per_pair.threads == t) & (per_pair.ilp == i)].bpc
        bpc, p10, p90, n = q(sel)
        mhz = g[(g.threads == t) & (g.ilp == i)].mhz.median()
        smc = int(S.meta.sm_count)
        conv = smc * mhz * 1e6
        meth = (f"k12 ld.volatile.shared.v4, 132 blocks on distinct SMs, {t} thr, ILP {i} (best of "
                f"{len(med_cfg)} configs); slope 4096->8192 steps, median over SMs per launch pair, median of {n} pairs; "
                f"x {smc} SMs x {mhz:.1f} MHz (median block clock); LOCK")
        S.add("l1_smem_bw", bpc * conv, "B/s", meth, p10=p10 * conv, p90=p90 * conv, n=n, flagged=nf, rids=rs)
        S.add("l1_smem_bw_per_sm_cycle", bpc, "B/cycle/SM", meth, p10=p10, p90=p90, n=n, flagged=nf, rids=rs)
        src["smem"] = rs
    else:
        S.add("l1_smem_bw", np.nan, "B/s", "no complete bw_smem k12 run", "not_measured")

    # ---- DSMEM (slope over passes)
    rd = latest_run("dsmem_bw", "S", a.quick)
    ds = read_run("dsmem_bw.csv", rd)
    if ds is not None and len(ds):
        S.note_src(ds)
        g, nf = good(ds)
        g = g.copy()
        g["k"] = g.groupby(["op", "cs", "passes"]).launch_idx.rank(method="first")
        best = None
        for op, tag in [("ld", "ld"), ("tma_put", "push")]:
            for cs in sorted(g.cs.unique()):
                d = g[(g.op == op) & (g.cs == cs)]
                rates = []
                for k, dk in d.groupby("k"):
                    if dk.passes.nunique() < 2:
                        continue
                    x, y = dk.total_bytes.to_numpy(float), dk.event_ns.to_numpy(float)
                    slope = np.polyfit(x, y, 1)[0]  # ns per byte
                    rates.append(1e9 / slope)
                med, p10, p90, n = q(rates)
                act = int(d.active_ctas.median()) if len(d) else 0
                ncl = int(d.nclusters.median()) if len(d) else 0
                desc = "ld.shared::cluster.v4 pull" if op == "ld" else "cp.async.bulk.shared::cluster push (16 KiB commands)"
                meth = (f"k_stream {desc}, ring +1, CS {cs}, {ncl} clusters = {act} CTAs (1 per SM, max active clusters), "
                        f"96 KiB per pass, passes {sorted(d.passes.unique())}; 1 / slope of event time over bytes per launch "
                        f"index, median of {n}; LOCK")
                S.add(f"dsmem_bw_{tag}_cs{cs}", med, "B/s", meth, p10=p10, p90=p90, n=n,
                      flagged=int(((ds.op == op) & (ds.cs == cs) & ((ds.clock_bad != 0) | (ds.placement_bad != 0))).sum()), rids=rd)
                if np.isfinite(med) and (best is None or med > best[0]):
                    best = (med, p10, p90, n, meth)
        if best:
            S.add("dsmem_bw", best[0], "B/s", "best CS and direction: " + best[4], p10=best[1], p90=best[2], n=best[3],
                  flagged=nf, rids=rd)
        src["dsmem"] = rd
    else:
        S.add("dsmem_bw", np.nan, "B/s", "no complete dsmem_bw S run", "not_measured")

    # ---- FFMA / IMAD
    rc = latest_run("compute_peak", "peak", a.quick)
    cp = read_run("compute_peak.csv", rc)
    if cp is not None and len(cp):
        S.note_src(cp)
        g, nf = good(cp)
        for kind, name, unit, what in [("ffma", "fp32_peak", "FLOP/s", "FFMA (2 FLOP each)"),
                                       ("imad", "int32_peak", "intop/s", "IMAD (2 integer ops each: multiply + add)")]:
            d = g[g.kind == kind].copy()
            d["rate"] = d.work / ((d.ms_event - floor_ms) * 1e-3)
            med_ilp = d.groupby("ilp").rate.median()
            if med_ilp.empty:
                S.add(name, np.nan, unit, "no unflagged launches", "not_measured")
                continue
            ilp = int(med_ilp.idxmax())
            dd = d[d.ilp == ilp]
            med, p10, p90, n = q(dd.rate)
            wpc = med / (int(S.meta.sm_count) * dd.sm_mhz_blocks.median() * 1e6)
            S.add(name, med, unit,
                  f"{kind}_peak: {what} dependent chains, ILP {ilp} (best of {sorted(med_ilp.index.tolist())}: "
                  + ", ".join(f"{k}: {v / 1e12:.2f}T" for k, v in med_ilp.items())
                  + f"), 4x512 thr per SM on all SMs; median of {n} launches net of the event floor; {wpc:.1f} ops/cycle/SM at "
                  f"{dd.sm_mhz_blocks.median():.0f} MHz; LOCK", p10=p10, p90=p90, n=n,
                  flagged=int(((cp.kind == kind) & ((cp.clock_bad != 0) | (cp.placement_bad != 0))).sum()), rids=rc)
        src["compute"] = rc
    else:
        S.add("fp32_peak", np.nan, "FLOP/s", "no complete compute_peak run", "not_measured")
        S.add("int32_peak", np.nan, "intop/s", "no complete compute_peak run", "not_measured")

    # ---- tensor cores
    # Tensor cores: every complete run (a ceiling is an upper bound; the first R1 run's back-to-back GEMMs were partly
    # clock-flagged by the 400 W limit, the second idles between launches): best unflagged launch over all of them.
    rr = runs()
    rr = rr[rr.complete & rr.cmdline.str.contains("ceilings_tc.py") & ~rr.cmdline.str.contains("--ncu")
            & (rr.cmdline.str.contains("--quick") == a.quick)]
    rt = ";".join(rr.sort_values("started_iso").run_id) or None
    tcs = [read_run("tc.csv", r) for r in rr.run_id]
    tc = pd.concat([t for t in tcs if t is not None], ignore_index=True) if tcs else None
    kbest = {}
    kpath = f"{CEIL}/tc_kernel.csv"
    if os.path.exists(kpath):
        kd = pd.read_csv(kpath)
        done = set(runs().query("complete").run_id)
        kd = kd[kd.run_id.isin(done)]
        for cfg, g in kd.groupby("config"):
            i = g.flops_per_s.idxmax()
            kbest[cfg] = (float(g.flops_per_s[i]), int(g["size"][i]), ",".join(map(str, sorted(g["size"].unique()))),
                          str(g.kernel[i])[:40])
    if tc is not None and len(tc):
        S.note_src(tc)
        fp8 = []
        for cfg, name in [("fp16", "fp16_tc_peak"), ("bf16", "bf16_tc_peak"), ("fp8", "fp8_tc_peak_default_accum"),
                          ("fp8_fastacc", "fp8_tc_peak_fast_accum")]:
            d0 = tc[tc.config == cfg]
            if d0.empty:
                S.add(name, np.nan, "FLOP/s", f"{cfg} not in run {rt}", "not_measured")
                continue
            if d0.ms_event.isna().all():
                S.add(name, np.nan, "FLOP/s", str(d0.impl.iloc[0]), "not_measured", rids=rt)
                continue
            g, nf = good(d0)
            g = g[(g.valid == 1) & g.ms_event.notna()]
            if g.empty:
                S.add(name, np.nan, "FLOP/s", f"{cfg}: no valid unflagged launch (max_rel_err {d0.max_rel_err.iloc[0]})",
                      "not_measured", flagged=nf, rids=rt)
                continue
            rate = g.flops / ((g.ms_event - floor_ms) * 1e-3)
            med, p10, p90, n = q(rate)
            best = float(rate.max())
            best_rid = str(g.run_id.loc[rate.idxmax()])
            kb = kbest.get(cfg)
            ev_note = ""
            if kb is not None and kb[0] > best:  # kernel duration (nsys), the timer of the workload points
                ev_note = (f"; kernel duration (nsys, ceilings_tc_kernel.py) of the best launch over sizes {kb[2]}: "
                           f"{kb[0] / 1e12:.1f} T at {kb[1]}^3 (kernel {kb[3]}), above the event-timed best "
                           f"{best / 1e12:.1f} T")
                best = kb[0]
            m = int(g.m.iloc[0])
            meth = (f"{g.impl.iloc[0]}, {m}^3, best of {n} launches after 5 warm-ups (median {med / 1e12:.1f} T), "
                    f"2*M*N*K / (event time - floor), best launch from run {best_rid}, max_rel_err {float(g.max_rel_err.iloc[0]):.1e}"
                    f"{ev_note}; LOCK")
            S.add(name, best, "FLOP/s", meth, p10=p10, p90=p90, n=n, flagged=nf, rids=rt)
            if cfg.startswith("fp8"):
                fp8.append((best, cfg, meth, p10, p90, n, nf))
        if fp8:
            b = max(fp8)
            S.add("fp8_tc_peak", b[0], "FLOP/s", f"faster fp8 variant ({b[1]}): {b[2]}", p10=b[3], p90=b[4], n=b[5],
                  flagged=b[6], rids=rt)
        src["tc"] = rt
    else:
        for name in ["fp16_tc_peak", "bf16_tc_peak", "fp8_tc_peak"]:
            S.add(name, np.nan, "FLOP/s", "no complete ceilings_tc.py run", "not_measured")

    # ---- tensor cores, instruction throughput (tc_peak: back-to-back wgmma, operands in smem, accumulators in registers)
    rw = latest_run("tc_peak", "peak", a.quick)
    tw = read_run("tc_peak.csv", rw)
    WG = [("fp16", "fp16_tc_peak_wgmma", 4096), ("fp16acc", "fp16acc_tc_peak_wgmma", 4096), ("bf16", "bf16_tc_peak_wgmma", 4096),
          ("fp8", "fp8_tc_peak_wgmma", 8192)]  # architectural dense FLOP/clk/SM (H100 SXM 989.4 T = 132 x 1830 MHz x 4096)
    if tw is not None and len(tw):
        S.note_src(tw)
        g, _ = good(tw)
        g = g.copy()
        g["rate"] = g.flop / ((g.ms_event - floor_ms) * 1e-3)
        keys = ["n", "nacc", "wait_groups", "consumers", "fill"]
        for kind, name, arch in WG:
            d = g[g.kind == kind]
            nf = int(((tw.kind == kind) & ((tw.clock_bad != 0) | (tw.placement_bad != 0))).sum())
            if d.empty:
                S.add(name, np.nan, "FLOP/s", f"{kind}: no unflagged tc_peak launch in run {rw}", "not_measured", flagged=nf, rids=rw)
                continue
            med_cfg = d.groupby(keys).rate.median().sort_values(ascending=False)
            bk = med_cfg.index[0]
            dd = d[np.logical_and.reduce([d[k] == v for k, v in zip(keys, bk)])]
            med, p10, p90, n = q(dd.rate)
            fpc, mhz = float(dd.flop_per_clk_sm.median()), float(dd.sm_mhz_blocks.median())
            cfg = lambda k: f"N {k[0]} acc {k[1]} wait {k[2]} wg {k[3]} {k[4]}"
            runners = "; ".join(f"{cfg(k)}: {v / 1e12:.1f}T" for k, v in list(med_cfg.items())[1:3])
            S.add(name, med, "FLOP/s",
                  f"tc_peak: back-to-back wgmma.mma_async.{dd.instr.iloc[0]} (A, B in smem by descriptor, 128B swizzle, K-major; "
                  f"accumulators in registers), 1 CTA per SM on all {int(dd.blocks.iloc[0])} SMs; best of {len(med_cfg)} "
                  f"configurations by median: {cfg(bk)} (next: {runners}); median of {n} launches, 2*64*N*K per wgmma / "
                  f"(event time - {floor_ms * 1e3:.2f} us floor); {fpc:.0f} FLOP/clk/SM from %clock64 = {100 * fpc / arch:.1f} % "
                  f"of the architectural {arch} at {mhz:.0f} MHz (block clock); LOCK", p10=p10, p90=p90, n=n, flagged=nf, rids=rw)
        # Power-limited rate: the best configuration of each kind re-run with random finite operands; all of these
        # launches ran below the lock (400 W limit), so they are flagged and kept out of the peak above. Their median
        # rate is the sustained tensor-core rate with real data at this power limit (DESIGN §6, R1).
        for kind, name, arch in WG:
            rr_ = tw[(tw.kind == kind) & (tw.fill == "rand")]
            if rr_.empty:
                continue
            rate = rr_.flop / ((rr_.ms_event - floor_ms) * 1e-3)
            med, p10, p90, n = q(rate)
            S.add(name.replace("_peak_wgmma", "_power_limited"), med, "FLOP/s",
                  f"tc_peak with random finite operands (|x| < 2), best zero-operand configuration of {kind}: every launch "
                  f"ran below the lock: in-kernel clock {rr_.sm_mhz_blocks.min():.0f}–{rr_.sm_mhz_blocks.max():.0f} MHz "
                  f"(median {rr_.sm_mhz_blocks.median():.0f} MHz) at {rr_.flop_per_clk_sm.median():.0f} FLOP/clk/SM: the "
                  f"400 W power limit, not the instruction rate, bounds dense tensor work with real data; median of {n} "
                  f"launches (all clock-flagged, kept as such)", p10=p10, p90=p90, n=n, flagged=n, rids=rw)
        src["tcpeak"] = rw
    else:
        for _, name, _ in WG:
            S.add(name, np.nan, "FLOP/s", "no complete tc_peak peak run", "not_measured")
    # fp16/bf16/fp8_tc_peak = max(torch/cuBLAS, wgmma); the cuBLAS rows above are left as computed, then replaced in place.
    idx = {r[0]: i for i, r in enumerate(S.rows)}  # last row of each name
    for peak, cands in [("fp16_tc_peak", ["fp16_tc_peak_wgmma", "fp16acc_tc_peak_wgmma"]), ("bf16_tc_peak", ["bf16_tc_peak_wgmma"]),
                        ("fp8_tc_peak", ["fp8_tc_peak_wgmma"])]:
        if peak not in idx:
            continue
        pr = S.rows[idx[peak]]
        val = lambda r: r[1] if isinstance(r[1], (float, np.floating)) and np.isfinite(r[1]) else -np.inf
        wrs = [S.rows[idx[c]] for c in cands if c in idx and np.isfinite(val(S.rows[idx[c]]))]
        if not wrs:
            continue
        wb = max(wrs, key=val)
        cub = f"{pr[1] / 1e12:.1f} T" if np.isfinite(val(pr)) else "not measured"
        alts = ", ".join(f"{r[0]} {r[1] / 1e12:.1f} T" for r in wrs)
        if val(pr) >= val(wb):
            pr[3] = f"max(torch/cuBLAS {cub}, {alts}): torch/cuBLAS won; {pr[3]}"
        else:
            S.rows[idx[peak]] = [peak, wb[1], wb[2], f"max(torch/cuBLAS {cub}, {alts}): {wb[0]} won; {wb[3]}", wb[4], wb[5], wb[6],
                                 wb[7], wb[8], wb[9]]

    for name, v, unit in DATASHEET:
        S.add(name, v, unit, "H100 NVL datasheet (per GPU, dense = half the with-sparsity tensor-core figure, boost clock)", "datasheet")

    if S.meta is None:
        raise SystemExit("no source data")
    lock_txt = S.lock_note([r for r in src.values() if r])
    for r in S.rows:
        r[3] = r[3].replace("LOCK", lock_txt)

    crosschecks(S)

    # ---- write
    run_id = iso_now()
    cs = pd.concat(S.clock_samples)
    pre = [S.meta.gpu_name, int(S.meta.sm_count), int(cs.sm_clock_mhz.median()), int(cs.mem_clock_mhz.median()),
           S.meta.cuda_version, S.meta.driver_version, run_id, -1]

    def fmt(x):
        if isinstance(x, (float, np.floating)):
            return "" if not np.isfinite(x) else f"{x:.7g}"
        return x

    out = [pre + [fmt(x) for x in r] for r in S.rows]
    xc = [pre + [fmt(x) for x in r] for r in S.xc]
    print_table(S, floor_ms)
    if a.dry_run:
        print("(dry run: nothing written)")
        return
    cmd = " ".join(["harness/ceilings_summary.py"] + sys.argv[1:])
    append_rows(f"{RES}/roofline__runs.csv", REG_HDR, [[run_id, iso_now(), "", "started", "", cmd]])
    append_rows(f"{RES}/ceilings.csv", MANDATORY + OUT_COLS, out)
    if xc:
        append_rows(f"{CEIL}/crosscheck.csv", MANDATORY + XC_COLS, xc)
    append_rows(f"{RES}/roofline__runs.csv", REG_HDR, [[run_id, "", iso_now(), "complete", "", cmd]])
    print(f"{len(out)} rows -> {RES}/ceilings.csv, {len(xc)} -> {CEIL}/crosscheck.csv (run_id {run_id})")


# ---------------------------------------------------------------- ncu cross-checks
def load_ncu(step):
    """Long-format ncu rows of the latest run in results/ceilings/ncu/ncu_<step>.csv (value as float)."""
    p = f"{CEIL}/ncu/ncu_{step}.csv"
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p, keep_default_na=False, na_values=[""])
    if d.empty:
        return None
    if step == "tc":  # one run per config: keep the latest run of every config
        d = d[d.run_id == d.groupby("config").run_id.transform("max")]
    else:
        d = d[d.run_id == d.run_id.max()]
    d = d.copy()
    d["value"] = pd.to_numeric(d.value.astype(str).str.replace(",", ""), errors="coerce")
    return d


def crosschecks(S):
    def add(check, key, metric, measured, expected, note, rid):
        ratio = measured / expected if expected else np.nan
        S.xc.append([check, key, metric, measured, expected, ratio, note, rid])

    c = load_ncu("compute")
    if c is not None:
        for (kind, li), d in c.groupby(["kind", "launch_idx"]):
            v = d.groupby("metric").value.sum()
            inst = float(d.inst_total.iloc[0])
            threads = float(d.blocks.iloc[0]) * float(d.threads.iloc[0])
            if kind == "ffma" and "sm__sass_thread_inst_executed_op_ffma_pred_on.sum" in v:
                add("fp32_ffma_count", f"ffma ilp {d.ilp.iloc[0]}", "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
                    v["sm__sass_thread_inst_executed_op_ffma_pred_on.sum"], inst + threads,
                    "expected = timed-loop FFMAs + 1 set-up FFMA per thread", d.run_id.iloc[0])
            for m in ["sm__sass_thread_inst_executed_op_integer_pred_on.sum", "sm__inst_executed_pipe_fma.sum",
                      "sm__inst_executed_pipe_fmaheavy.sum", "sm__inst_executed_pipe_alu.sum"]:
                if m in v:
                    exp = inst if m.startswith("sm__sass_thread") else inst / 32
                    add(f"{kind}_counters", f"{kind} ilp {d.ilp.iloc[0]}", m, v[m], exp,
                        "expected = timed-loop instructions (thread-level) or / 32 (warp-level); informational", d.run_id.iloc[0])
    h = load_ncu("hbm")
    if h is not None:
        for op, d in h.groupby("op"):
            v = d.groupby("metric").value.sum()
            for m, col in [("dram__bytes_read.sum", "read_bytes"), ("dram__bytes_write.sum", "write_bytes")]:
                if m in v and float(d[col].iloc[0]) > 0:
                    add("hbm_bytes", op, m, v[m], float(d[col].iloc[0]),
                        "write: lines still dirty in L2 at kernel end are missing", d.run_id.iloc[0])
    ds = load_ncu("dsmem")
    if ds is not None:
        for (op, cs), d in ds.groupby(["op", "cs"]):
            v = d.groupby("metric").value.sum()
            for m in ["l1tex__m_l1tex2xbar_write_bytes_mem_dshared.sum", "l1tex__m_xbar2l1tex_read_bytes_mem_dshared.sum"]:
                if m in v:
                    add("dsmem_bytes", f"{op} cs{cs}", m, v[m], float(d.issuer_bytes.iloc[0]),
                        "expected = payload (collectives C1 / MB 11: each payload byte once per counter)", d.run_id.iloc[0])
    t = load_ncu("tc")
    if t is not None:
        for cfg, d in t.groupby("config"):
            v = d.groupby("metric").value.sum()
            names = "|".join(sorted(set(d.kernel_name.astype(str))))[:200]
            for m in ["sm__inst_executed_pipe_tensor_op_hmma.sum", "sm__inst_executed_pipe_tensor_op_gmma.sum"]:
                if m in v and v[m] > 0:
                    add("tc_inst", cfg, m, v[m], np.nan,
                        f"FLOP per instruction = {float(d.flops.iloc[0]) / v[m]:.0f}; kernels: {names}", d.run_id.iloc[0])
            if "gpu__time_duration.sum" in v:
                add("tc_time", cfg, "gpu__time_duration.sum", v["gpu__time_duration.sum"], float(d.ms_event.iloc[0]) * 1e6,
                    f"ncu kernel time vs cudaEvent time of the profiled GEMM (ns); kernels: {names}", d.run_id.iloc[0])

    tp = load_ncu("tcpeak")
    if tp is not None:
        for kind, d in tp.groupby("kind"):
            v = d.groupby("metric").value.sum()
            wg, rid = float(d.wgmma_total.iloc[0]), d.run_id.iloc[0]
            m = "sm__inst_executed_pipe_tensor_op_gmma.sum"
            if m in v:
                add("tcpeak_gmma_inst", kind, m, v[m], 4 * wg,
                    f"expected = 4 warps x warpgroup-level wgmma count (ratio 0.25 = counted once per warpgroup); "
                    f"FLOP per counted instruction {float(d.flop.iloc[0]) / v[m]:.0f}", rid)
            for pm in ["sm__pipe_tensor_op_gmma_cycles_active.avg.pct_of_peak_sustained_active",
                       "sm__pipe_tensor_op_gmma_cycles_active.avg.pct_of_peak_sustained_elapsed"]:
                if pm in v:
                    add("tcpeak_gmma_util", kind, pm, v[pm], np.nan, "gmma pipe utilisation (%); informational", rid)
            if "gpu__time_duration.sum" in v:
                add("tcpeak_time", kind, "gpu__time_duration.sum", v["gpu__time_duration.sum"], float(d.ms_event.iloc[0]) * 1e6,
                    "ncu kernel time vs cudaEvent time of the profiled launch (ns)", rid)


def print_table(S, floor_ms):
    print()
    print(f"{'name':32s} {'value':>12s} {'unit':10s} {'n':>3s} {'flag':>4s}  source")
    for r in S.rows:
        v = r[1]
        if r[2] in ("B/s", "FLOP/s", "intop/s") and isinstance(v, float) and np.isfinite(v):
            vs = f"{v / 1e12:9.3f} T"
        else:
            vs = f"{v:11.4g}" if isinstance(v, float) and np.isfinite(v) else "   —"
        print(f"{r[0]:32s} {vs:>12s} {r[2]:10s} {r[7]:>3} {r[8]:>4}  {r[4]}")
    print(f"\nevent floor {floor_ms * 1e3:.2f} us. Comparison with the GPU-2 references (MB 05 / 11, 1635 MHz):")
    val = {r[0]: r[1] for r in S.rows}
    for k, ref in GPU2_REF.items():
        v = val.get(k, np.nan)
        if isinstance(v, float) and np.isfinite(v):
            print(f"  {k:14s} GPU 3 {v / 1e12:6.2f} TB/s   GPU 2 {ref / 1e12:5.2f} TB/s   ratio {v / ref:5.3f}")
    v = val.get("dsmem_bw", np.nan)
    if isinstance(v, float) and np.isfinite(v):
        print(f"  dsmem_bw       GPU 3 {v / 1e12:6.2f} TB/s   GPU 2 {GPU2_DSMEM[0] / 1e12:.1f}–{GPU2_DSMEM[1] / 1e12:.1f} TB/s (ring, "
              f"CS 2–16; MB 11 GPU-wide ld 3.48/2.77/2.78/2.61, TMA put 3.98/3.25/2.92/2.31 for CS 2/4/8/16)")
    if S.xc:
        print("\nncu cross-checks:")
        for c in S.xc:
            r = c[5]
            rs = f"{r:7.4f}" if isinstance(r, float) and np.isfinite(r) else "      —"
            print(f"  {c[0]:16s} {c[1]:18s} {c[2]:55s} {c[3]:>16.6g}  ratio {rs}  {c[6][:90]}")


if __name__ == "__main__":
    main()
