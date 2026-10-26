#!/usr/bin/env python3
"""Cross-check of the hand-computed roofline points against Nsight Compute's roofline sections (DESIGN §3 step 5, R8).

    roofline/tools/run --profile --lock-sm-mhz 1635 -- python3 roofline/harness/sol_crosscheck.py \
        [--workload REGEX] [--roles dsmem non_dsmem] [--force] [--dry-run]
    roofline/tools/run --no-gpu -- python3 roofline/harness/sol_crosscheck.py --dry-run     # prints the commands

Configurations: for every runnable manifest and shape, the operator-scope marker of each selected role (default
dsmem and non_dsmem): `roofline_figures.load()['markers']` (= fastest P per variant), the fastest variant if a
role has several. Each one runs the variant's run command (warmup 2, iters 1) under the same ncu command line as
`run_manifest.do_ncu` (--clock-control none --cache-control all --kernel-name-base demangled, --launch-skip
pre + 2k --launch-count k from results/nsys/<cfg>/kernel_counts.json) with the sections

    SpeedOfLight_RooflineChart                    (FP32/FP64 CUDA-core work vs DRAM)
    SpeedOfLight_HierarchicalTensorRooflineChart  (tensor-core work vs DRAM, L2, L1)
    SpeedOfLight_HierarchicalSingleRooflineChart  (FP32 work vs L2, L1: same traffic metrics as the tensor one)
    SpeedOfLight_HierarchicalHalfRooflineChart    (FP16 CUDA-core work)

and their raw metrics requested explicitly with --metrics (so they are printed by --page raw whatever the
section output). ncu's coordinates are recomputed from the raw values exactly as the sections define them
(Nsight Compute 2025.2.1 section files, GH100 = CC_90 branches):

  achieved work [FLOP/s] = sum(ValuePerCycleMetrics) x CyclesPerSecondMetric
    FP32 (RooflineChart.section l. 223-243, l. 49-52):
        (smsp__..._op_fadd_pred_on + smsp__..._op_fmul_pred_on + 2 x smsp__..._op_ffma_pred_on).sum.per_cycle_elapsed
        x smsp__cycles_elapsed.avg.per_second            (`smsp__sass_thread_inst_executed_*`)
    FP64 (same file l. 278-298): dadd + dmul + 2 x dfma, same form
    FP16 CUDA core (HierarchicalHalf l. 78-91, 306-321): 2 x hadd + 2 x hmul + 4 x hfma, same form
    tensor (HierarchicalTensor GH100 item l. 11805-16777, e.g. src_fp16 l. 15537-15593):
        sm__ops_path_tensor_src_<type>.sum.per_cycle_elapsed x sm__cycles_elapsed.avg.per_second for
        <type> in bf16_dst_fp32, fp16, fp64, fp8_sparsity_off, fp8_sparsity_on, tf32_dst_fp32 (floating point;
        int8 / int1 kept apart). These are the section's aggregate per-source points; the op_* (hgmma/hmma/...)
        points are a breakdown of the same ops and are not added.
  achieved traffic [B/s]
    DRAM: dram__bytes.sum.per_second                                        (RooflineChart l. 244-272)
    L2:   derived__lts__lts2xbar_bytes = lts__lts2xbar_cycles_active.sum.per_second x 32       (Tensor l. 114-117)
    L1:   derived__l1tex__lsu_writeback_bytes_mem_lgds = l1tex__lsu_writeback_active_mem_lgds.sum.per_second x 128
                                                                            (Tensor l. 103-109, CC_90 branch)
  intensity = achieved work / achieved traffic.
Per launch, work = achieved work x gpu__time_duration and bytes = achieved traffic x gpu__time_duration; a
configuration sums them over its k kernels (DESIGN §2.2 aggregation): ncu throughput = sum work / sum time,
ncu intensity = sum work / sum bytes. The same totals from the plain `.sum` counters are kept as a check.
The point ncu places for the workload is the one of its precision (manifest `precision`): fp32_cuda -> FP32,
*_tensor -> tensor; int32 has no roof in these sections (ratios left empty).

Our point (roofline_figures.configs): throughput = manifest work / nsys median time; intensity_L = manifest work /
bytes_L of the points.csv ncu row (HBM = dram read + write, L2 = lts__t_bytes, L1 = l1tex__t_bytes + 128 B x
shared wavefronts). Ratios ncu / ours with |ratio - 1| > 10 % are flagged and given a likely reason from the
decomposition r_throughput = r_work x r_time and r_ai_L = r_work / r_bytes_L.

Output: one row per configuration appended to results/sol_crosscheck.csv (rl.MANDATORY columns first; a failed
ncu run is recorded with status = failed), raw captures results/ncu_sol/<cfg>__<stamp>.csv, the command logged in
results/logs/sol__<cfg>.log, the run registered in results/roofline__runs.csv. A configuration with a complete row
is skipped unless --force.
"""
import argparse
import json
import math
import os
import shlex
import sys

import rooflib as rl
import run_manifest as rm

OUT = os.path.join(rl.RES, "sol_crosscheck.csv")
RAW_DIR = os.path.join(rl.RES, "ncu_sol")
TOL = 0.10  # DESIGN §3 step 5
SECTIONS = ["SpeedOfLight_RooflineChart", "SpeedOfLight_HierarchicalTensorRooflineChart",
            "SpeedOfLight_HierarchicalSingleRooflineChart", "SpeedOfLight_HierarchicalHalfRooflineChart"]

FP_OPS = {  # op -> FLOP per thread instruction, as in the sections' MetricDefinitions
    "fp32": {"fadd": 1, "fmul": 1, "ffma": 2},
    "fp64": {"dadd": 1, "dmul": 1, "dfma": 2},
    "fp16": {"hadd": 2, "hmul": 2, "hfma": 4},
}
TENSOR_FP = ["bf16_dst_fp32", "fp16", "fp64", "fp8_sparsity_off", "fp8_sparsity_on", "tf32_dst_fp32"]
TENSOR_INT = ["int8", "int1"]
L2_BYTES_PER_CYCLE, L1_BYTES_PER_CYCLE = 32, 128


def _inst(op):
    return f"smsp__sass_thread_inst_executed_op_{op}_pred_on"


def _tens(src):
    return f"sm__ops_path_tensor_src_{src}"


METRICS = (
    ["gpu__time_duration.sum", "smsp__cycles_elapsed.avg.per_second", "sm__cycles_elapsed.avg.per_second",
     "sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained",
     "dram__bytes.sum", "dram__bytes.sum.per_second", "dram__bytes.sum.peak_sustained",
     "dram__cycles_elapsed.avg.per_second",
     "lts__lts2xbar_cycles_active.sum", "lts__lts2xbar_cycles_active.sum.per_second",
     "lts__lts2xbar_cycles_active.sum.peak_sustained", "lts__cycles_elapsed.avg.per_second",
     "l1tex__lsu_writeback_active_mem_lgds.sum", "l1tex__lsu_writeback_active_mem_lgds.sum.per_second",
     "l1tex__lsu_writeback_active_mem_lgds.sum.peak_sustained", "l1tex__cycles_elapsed.avg.per_second",
     "sm__inst_executed_pipe_tensor_op_hmma.sum", "sm__inst_executed_pipe_tensor_op_gmma.sum"]
    + [f"{_inst(op)}.sum{s}" for ops in FP_OPS.values() for op in ops for s in ("", ".per_cycle_elapsed")]
    + [f"{_tens(t)}.sum{s}" for t in TENSOR_FP + TENSOR_INT for s in ("", ".per_cycle_elapsed", ".peak_sustained")]
)

HDR = rl.MANDATORY + [
    "workload", "variant_id", "role", "shape_id", "cluster_size", "status", "kernels_n", "precision", "ncu_point",
    "work_formula", "nsys_time_ns", "ncu_time_ns", "points_ncu_time_ns", "ncu_sm_mhz",
    "ncu_work_fp32", "ncu_work_fp64", "ncu_work_fp16", "ncu_work_tensor", "ncu_tensor_int_ops",
    "ncu_work_point", "ncu_work_total", "ncu_work_point_counters",
    "ncu_dram_bytes", "ncu_l2_bytes", "ncu_l1_bytes", "our_hbm_bytes", "our_l2_bytes", "our_l1_bytes",
    "our_throughput", "ncu_throughput", "r_throughput",
    "our_ai_dram", "ncu_ai_dram", "r_ai_dram",
    "our_ai_l2", "ncu_ai_l2", "r_ai_l2",
    "our_ai_l1", "ncu_ai_l1", "r_ai_l1",
    "r_work", "r_work_total", "r_time", "r_dram_bytes", "r_l2_bytes", "r_l1_bytes",
    "ncu_peak_work", "ncu_peak_dram_bw", "ncu_peak_l2_bw", "ncu_peak_l1_bw",
    "flags", "reason", "raw_file", "clock_bad", "notes"]

POINT_OF = {"fp32_cuda": "fp32", "fp16_cuda": "fp16", "fp64_cuda": "fp64", "fp16_tensor": "tensor",
            "bf16_tensor": "tensor", "fp8_tensor": "tensor", "tf32_tensor": "tensor"}


# ---------------------------------------------------------------- units
_PFX = {"": 1.0, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12, "P": 1e15}
_TIME = {"second": 1.0, "s": 1.0, "msecond": 1e-3, "ms": 1e-3, "usecond": 1e-6, "us": 1e-6, "nsecond": 1e-9,
         "ns": 1e-9}
_BASES = ("byte", "cycle", "inst", "warp", "thread", "sector", "request")


def _num_scale(u):
    if u in _TIME:
        return _TIME[u]
    for b in _BASES:
        if u.endswith(b) and u[:-len(b)] in _PFX:
            return _PFX[u[:-len(b)]]
    return 1.0


def to_base(unit, value):
    """Value in SI base units (time in s, rates per s, Hz); unknown units are taken as base."""
    u = (unit or "").strip()
    v = rl.num(value)
    if not u or v != v:
        return v
    if u.endswith("Hz") and u[:-2] in _PFX:
        return v * _PFX[u[:-2]]
    if "/" in u:
        n, d = u.split("/", 1)
        return v * _num_scale(n) / (_TIME.get(d, 1.0))
    return v * _num_scale(u)


# ---------------------------------------------------------------- ncu coordinates
def coords(launches):
    """Section-defined work and traffic summed over the launches (+ plain-counter totals as a check)."""
    tot, missing = {}, set()

    def g(d, name):
        if name not in d:
            missing.add(name)
            return 0.0
        x = to_base(*d[name])
        return 0.0 if x != x else x

    add = lambda k, x: tot.__setitem__(k, tot.get(k, 0.0) + x)  # noqa: E731
    for _, d in launches:
        t = g(d, "gpu__time_duration.sum")                       # s
        f_smsp, f_sm = g(d, "smsp__cycles_elapsed.avg.per_second"), g(d, "sm__cycles_elapsed.avg.per_second")
        add("time", t)
        add("sm_hz_t", f_sm * t)
        for cat, ops in FP_OPS.items():
            add(cat, sum(w * g(d, f"{_inst(op)}.sum.per_cycle_elapsed") for op, w in ops.items()) * f_smsp * t)
            add(cat + "_cnt", sum(w * g(d, f"{_inst(op)}.sum") for op, w in ops.items()))
        tens = {s: g(d, f"{_tens(s)}.sum.per_cycle_elapsed") * f_sm * t for s in TENSOR_FP}
        add("tensor", sum(tens.values()))
        add("tensor_cnt", sum(g(d, f"{_tens(s)}.sum") for s in TENSOR_FP))
        add("tensor_int", sum(g(d, f"{_tens(s)}.sum.per_cycle_elapsed") for s in TENSOR_INT) * f_sm * t)
        add("tensor_inst", g(d, "sm__inst_executed_pipe_tensor_op_hmma.sum") + g(d, "sm__inst_executed_pipe_tensor_op_gmma.sum"))
        add("dram", g(d, "dram__bytes.sum.per_second") * t)
        add("dram_cnt", g(d, "dram__bytes.sum"))
        add("l2", g(d, "lts__lts2xbar_cycles_active.sum.per_second") * L2_BYTES_PER_CYCLE * t)
        add("l2_cnt", g(d, "lts__lts2xbar_cycles_active.sum") * L2_BYTES_PER_CYCLE)
        add("l1", g(d, "l1tex__lsu_writeback_active_mem_lgds.sum.per_second") * L1_BYTES_PER_CYCLE * t)
        add("l1_cnt", g(d, "l1tex__lsu_writeback_active_mem_lgds.sum") * L1_BYTES_PER_CYCLE)
        # ncu's roofs (PeakWork / PeakTraffic expressions), time-weighted over the launches
        add("pk_fp32", 2 * g(d, "sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained") * f_sm * t)
        top = max(TENSOR_FP, key=lambda s: tens[s])
        add("pk_tensor", g(d, f"{_tens(top)}.sum.peak_sustained") * f_sm * t if tens[top] > 0 else 0.0)
        add("pk_dram", g(d, "dram__bytes.sum.peak_sustained") * g(d, "dram__cycles_elapsed.avg.per_second") * t)
        add("pk_l2", g(d, "lts__lts2xbar_cycles_active.sum.peak_sustained") * L2_BYTES_PER_CYCLE
            * g(d, "lts__cycles_elapsed.avg.per_second") * t)
        add("pk_l1", g(d, "l1tex__lsu_writeback_active_mem_lgds.sum.peak_sustained") * L1_BYTES_PER_CYCLE
            * g(d, "l1tex__cycles_elapsed.avg.per_second") * t)
    return tot, sorted(missing)


# ---------------------------------------------------------------- comparison
def ratio(a, b):
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return float("nan")
    return a / b if b and b == b and a == a else float("nan")


def off(r):
    return r == r and abs(r - 1) > TOL


def reasons(r, point, precision, tot):
    """Likely causes of the flagged ratios from the decomposition into work, time and byte ratios."""
    out = []
    if point is None:
        return ["integer workload: the roofline sections have no INT roof (ncu FP work "
                f"{tot.get('fp32', 0) + tot.get('fp16', 0) + tot.get('tensor', 0):.3g})"]
    flagged = [k for k in ("r_throughput", "r_ai_dram", "r_ai_l2", "r_ai_l1") if off(r[k])]
    if not flagged:
        return out
    if off(r["r_work"]):
        other = r["ncu_work_total"] - r["ncu_work_point"]
        if r["ncu_work_total"] > 0 and other / r["ncu_work_total"] > TOL and not off(r["r_work_total"]):
            out.append(f"work split: {other / r['ncu_work_total']:.0%} of ncu FP work on other pipes "
                       f"(plotted by ncu as separate points); total ncu work / formula {r['r_work_total']:.2f}")
        elif point == "tensor" and r["r_work"] > 1:
            out.append("work: ncu counts executed tensor ops incl. padded / replicated MMA tiles "
                       "(e.g. m16n8k16 or m64nNk16 tiles larger than the useful rows); ours = formula")
        elif r["r_work"] > 1:
            out.append("work: ncu counts executed FP instructions incl. replicated per-CTA work and non-formula "
                       "FP ops (address/index math, reductions, guards); ours = formula")
        else:
            out.append("work: part of the formula's ops runs outside the counted instructions (MUFU "
                       "transcendentals, integer/conversion ops, the other of tensor/CUDA-core pipes) "
                       "or is not executed")
    if "r_throughput" in flagged and off(r["r_time"]):
        out.append(f"time: ncu kernel time (serialized replay, --cache-control all, {r['ncu_sm_mhz']:.0f} MHz "
                   f"measured by ncu) vs nsys median; nsys/ncu = {r['r_time']:.2f}")
    if "r_ai_dram" in flagged and off(r["r_dram_bytes"]):
        out.append(f"bytes: DRAM bytes of this run / points.csv ncu run = {r['r_dram_bytes']:.2f}")
    if "r_ai_l2" in flagged and off(r["r_l2_bytes"]):
        out.append("bytes: section L2 = lts__lts2xbar_cycles_active x 32 B (L2 -> SM return path); ours = "
                   f"lts__t_bytes (all L2 slice traffic incl. stores and DRAM fills); ratio {r['r_l2_bytes']:.2f}")
    if "r_ai_l1" in flagged and off(r["r_l1_bytes"]):
        out.append("bytes: section L1 = LSU writeback cycles (local/global/shared/dshared) x 128 B; ours = "
                   "l1tex__t_bytes + shared wavefronts x 128 B (TMA/bulk copies do not pass the LSU writeback); "
                   f"ratio {r['r_l1_bytes']:.2f}")
    if not out:
        out.append("product of several sub-10 % deviations (see r_work, r_time, r_*_bytes)")
    return out


# ---------------------------------------------------------------- configurations
def select(args):
    sys.path.insert(0, os.path.join(rl.ROOF, "analysis"))
    import roofline_figures as rf
    L = rf.load()
    B = L["markers"]
    B = B[(B.scope == "operator") & B.role.isin(args.roles)]
    if args.workload:
        B = B[B.workload.str.contains(args.workload, regex=True)]
    paths = {}
    for p in sorted(os.listdir(os.path.join(rl.ROOF, "workloads"))):
        if p.endswith(".yaml") and not p.startswith("_"):
            m = rm.load(os.path.join(rl.ROOF, "workloads", p))
            paths[m["name"]] = m
    out = []
    for (w, sid, role), g in B.groupby(["workload", "shape_id", "role"], sort=True):
        m = paths.get(w)
        if not m or m.get("runnable") is False:
            continue
        r = g.loc[g.time_ns.idxmin()]
        v = next((x for x in m["scope_operator"]["variants"] if x["id"] == r.variant_id), None)
        if v is None or sid not in m["scope_operator"]["shapes"]:
            print(f"[sol] {w} {r.variant_id} {sid}: not in the manifest any more, skipped")
            continue
        P = None if int(r.cluster_size) == -1 else int(r.cluster_size)
        out.append(({"m": m, "v": v, "sid": sid, "shape": m["scope_operator"]["shapes"][sid], "P": P}, r))
    return out


def done(c):
    want = [str(x) for x in rm.key(c)]
    rows = [x for x in rl.read_csv(OUT) if [x["workload"], x["variant_id"], x["shape_id"], x["cluster_size"]] == want]
    return bool(rows) and rows[-1]["status"] == "complete"


def ncu_cmd(c, kc, metrics, sections, raw):
    regex = "|".join(f"(?:{k})" for k in c["v"]["kernels"])
    cmd = rm.fmt(c["v"]["run"]["cmd"], c, warmup=rm.NCU_WARMUP, iters=1)
    full = (f"ncu --clock-control none --cache-control all --kernel-name-base demangled "
            f"--kernel-name regex:{shlex.quote(regex)} --launch-skip {kc['pre'] + rm.NCU_WARMUP * kc['k']} "
            f"--launch-count {kc['k']} " + " ".join(f"--section {s}" for s in sections) +
            f" --metrics {','.join(metrics)} --csv --page raw --print-units base --log-file {shlex.quote(raw)} "
            f"bash -c {shlex.quote(cmd)}")
    return cmd, full


def f(x):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else x


def run_one(c, r, args, run_id, metrics):
    name = rm.cfg_name(c)
    kc_path = os.path.join(rl.RES, "nsys", name, "kernel_counts.json")
    val = rm.latest_validation(c)
    problem = ("not validated (latest: %s)" % (val or "none")) if val != "PASS" else \
        ("no timing run (kernel counts unknown)" if not os.path.exists(kc_path) else "")
    wt = os.path.join(rl.BUILD, "src", rm.worktree_key(c["v"]))
    if not problem and not os.path.isdir(wt):
        problem = f"worktree {wt} missing (run_manifest --stage build)"
    if problem:
        print(f"[sol] {name}: {problem}, skipped")
        return
    kc = json.load(open(kc_path))
    stamp = rl.iso_now().replace(":", "")
    raw = os.path.join(RAW_DIR, f"{name}__{stamp}.csv")
    cmd, full = ncu_cmd(c, kc, metrics, args.sections, raw + ".part")
    print(f"[sol] {name} ({r.role}): {cmd}")
    if args.dry_run:
        print("      " + full)
        return
    os.makedirs(RAW_DIR, exist_ok=True)
    c0 = rl.clock_sample()
    rc, out = rm.sh(full, wt, rm.env_for(c, "run", warmup=rm.NCU_WARMUP, iters=1),
                    os.path.join(rl.RES, "logs", f"sol__{name}.log"), args.timeout)
    if os.path.exists(raw + ".part"):
        os.rename(raw + ".part", raw)
    c1 = rl.clock_sample()
    launches = rl.parse(raw) if os.path.exists(raw) else []
    prec = c["m"].get("precision", "")
    k = rm.key(c)
    row = {"workload": k[0], "variant_id": k[1], "role": r.role, "shape_id": k[2], "cluster_size": k[3],
           "kernels_n": len(launches), "precision": prec, "work_formula": r.work, "nsys_time_ns": r.time_ns,
           "points_ncu_time_ns": f(r.ncu_time_ns), "raw_file": os.path.basename(raw) if os.path.exists(raw) else "",
           "clock_bad": int(c0["bad"] or c1["bad"])}
    put = lambda d: rl.append_rows(OUT, HDR, [rl.mandatory(run_id, 0, c0) + [d.get(h, "") for h in HDR[len(rl.MANDATORY):]]])  # noqa: E731
    if rc != 0 or len(launches) != kc["k"]:
        tail = out.strip().splitlines()[-1][-300:] if out.strip() else ""
        put(dict(row, status="failed", reason=f"rc={rc}, {len(launches)} launches profiled, expected {kc['k']}",
                 notes=tail))
        print(f"[sol] {name}: FAILED rc={rc}, {len(launches)} launches (expected {kc['k']})")
        return
    tot, missing = coords(launches)
    point = POINT_OF.get(prec)
    T = tot["time"]
    total = tot["fp32"] + tot["fp64"] + tot["fp16"] + tot["tensor"]
    wp = tot[point] if point else float("nan")
    ncu_thr = wp / T if T else float("nan")
    ai = {lvl: (wp / tot[lvl] if tot[lvl] else float("nan")) for lvl in ("dram", "l2", "l1")}
    R = {"r_throughput": ratio(ncu_thr, r.throughput), "r_ai_dram": ratio(ai["dram"], r.ai_hbm),
         "r_ai_l2": ratio(ai["l2"], r.ai_l2), "r_ai_l1": ratio(ai["l1"], r.ai_l1),
         "r_work": ratio(wp, r.work), "r_work_total": ratio(total, r.work), "r_time": ratio(r.time_ns * 1e-9, T),
         "r_dram_bytes": ratio(tot["dram"], r.hbm_bytes), "r_l2_bytes": ratio(tot["l2"], r.l2_bytes),
         "r_l1_bytes": ratio(tot["l1"], r.l1_bytes), "ncu_work_point": wp if point else 0.0,
         "ncu_work_total": total, "ncu_sm_mhz": tot["sm_hz_t"] / T / 1e6 if T else float("nan")}
    flags = [x[2:] for x in ("r_throughput", "r_ai_dram", "r_ai_l2", "r_ai_l1") if off(R[x])]
    why = reasons(R, point, prec, tot)
    pk = {"tensor": tot["pk_tensor"], "fp32": tot["pk_fp32"]}.get(point, float("nan"))
    dev = max((abs(ratio(tot[x], tot[x + "_cnt"]) - 1) for x in ("fp32", "fp16", "tensor", "dram", "l2", "l1")
               if tot[x + "_cnt"] > 0), default=0.0)
    notes = [f"section-rate vs .sum counters max deviation {dev:.2%}",
             "ncu run kernels: " + "|".join(sorted({rm.short(d.get("Kernel Name", ("", ""))[1]) for _, d in launches}))]
    if tot["tensor_inst"] > 0:
        notes.append(f"tensor ops per hmma+gmma instruction {tot['tensor'] / tot['tensor_inst']:.0f}")
    if missing:
        notes.append("metrics missing from the capture (taken as 0): " + ",".join(missing))
    row.update({k_: f(v) for k_, v in R.items()})
    row.update({
        "status": "complete", "ncu_point": point or "none", "ncu_time_ns": T * 1e9,
        "ncu_work_fp32": tot["fp32"], "ncu_work_fp64": tot["fp64"], "ncu_work_fp16": tot["fp16"],
        "ncu_work_tensor": tot["tensor"], "ncu_tensor_int_ops": tot["tensor_int"],
        "ncu_work_point_counters": f(tot[point + "_cnt"]) if point else "",
        "ncu_dram_bytes": tot["dram"], "ncu_l2_bytes": tot["l2"], "ncu_l1_bytes": tot["l1"],
        "our_hbm_bytes": f(r.hbm_bytes), "our_l2_bytes": f(r.l2_bytes), "our_l1_bytes": f(r.l1_bytes),
        "our_throughput": f(r.throughput), "ncu_throughput": f(ncu_thr),
        "our_ai_dram": f(r.ai_hbm), "ncu_ai_dram": f(ai["dram"]), "our_ai_l2": f(r.ai_l2), "ncu_ai_l2": f(ai["l2"]),
        "our_ai_l1": f(r.ai_l1), "ncu_ai_l1": f(ai["l1"]),
        "ncu_peak_work": f(pk / T if T else float("nan")), "ncu_peak_dram_bw": tot["pk_dram"] / T,
        "ncu_peak_l2_bw": tot["pk_l2"] / T, "ncu_peak_l1_bw": tot["pk_l1"] / T,
        "flags": "|".join(flags) if point else "n/a", "reason": "; ".join(why), "notes": "; ".join(notes)})
    put(row)
    fmt = lambda x: f"{x:.3f}" if x == x else "-"  # noqa: E731
    print(f"[sol] {name}: ncu/ours throughput {fmt(R['r_throughput'])}, AI dram {fmt(R['r_ai_dram'])}, "
          f"L2 {fmt(R['r_ai_l2'])}, L1 {fmt(R['r_ai_l1'])} (work {fmt(R['r_work'])}, time {fmt(R['r_time'])})"
          + (f"  FLAG {','.join(flags) or 'n/a'}: {'; '.join(why)}" if flags or not point else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workload", help="regex on the workload name")
    ap.add_argument("--roles", nargs="+", default=["dsmem", "non_dsmem"])
    ap.add_argument("--sections", nargs="+", default=SECTIONS)
    ap.add_argument("--timeout", type=int, default=3600, help="seconds per ncu command")
    ap.add_argument("--force", action="store_true", help="re-run configurations with a complete row")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.dry_run and not os.environ.get("H100MB_GPU_UUID"):
        sys.exit("needs the GPU: roofline/tools/run --profile --lock-sm-mhz 1635 -- python3 "
                 "roofline/harness/sol_crosscheck.py (or --dry-run)")
    cs = select(args)
    metrics = rl.available(METRICS, rm.ncu_query(args))
    todo = [(c, r) for c, r in cs if args.force or not done(c)]
    print(f"[sol] {len(cs)} marker configurations ({len(cs) - len(todo)} already complete), sections "
          f"{', '.join(args.sections)}, {len(metrics)} explicit metrics")
    reg = None if args.dry_run else rl.Registry(sys.argv)
    run_id = reg.run_id if reg else "dry-run"
    try:
        for c, r in todo:
            run_one(c, r, args, run_id, metrics)
    except BaseException:
        if reg:
            reg.complete("failed")
        raise
    if reg:
        reg.complete()


if __name__ == "__main__":
    main()
