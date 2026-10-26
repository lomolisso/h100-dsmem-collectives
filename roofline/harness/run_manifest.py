#!/usr/bin/env python3
"""Generic manifest runner of the roofline study: build -> validate -> time -> ncu -> results/points.csv.

    roofline/tools/run --no-gpu -- python3 roofline/harness/run_manifest.py <manifest>... --stage build
    roofline/tools/run --profile --lock-sm-mhz 1635 -- python3 roofline/harness/run_manifest.py <manifest>... \
        --stage validate|time|ncu|all [--variant ID] [--shape ID] [--P N] [--force] [--dry-run]

Configurations: one per (variant, shape, P), where P runs over the variant's `cluster_size` list (a scalar or
null gives one configuration). A variant's optional `scope: layer` (default operator) marks the layer-scope
version of a variant (DESIGN §1, §3.4: its kernels = the operator region + the rest of the layer). Commands are Python format strings over the shape keys plus {P}, {warmup},
{iters}, {worktree}, {pyenv}, {repo}, {harness}.

Build. One git worktree per (submodule, patch) at the submodule's pinned commit:
`roofline/build/src/<repo>__<patch stem | pin>/`, the patch applied there (third_party/ stays clean). Variants
with the same (repo, patch) share it; the build command runs once per distinct (worktree, command). A stamp file
records the built command; `--force` rebuilds.

Validate. `validate.cmd` (default: the run command with warmup 0, iters 1) must exit 0 and its output must
match every regex in `validate.expect`. Results go to `results/validation.csv`; `time`/`ncu` refuse a
configuration whose latest validation is not PASS. `skip: {shape_id: reason}` marks configurations that do
not exist (recorded as SKIP with the reason, never run).

Time. The run command (warmup W, iters I) runs under `nsys profile --trace=cuda`; every iteration of every
wrapper is preceded by the L2-flush kernel `roofline_l2_flush_kernel` (CREDIT: patch
`credit__roofline_hooks.patch`; Python wrappers: `flush.py`), which delimits the iterations in the kernel trace.
Per iteration: time_ns = sum of the durations of the kernels matching the variant's `kernels` regexes,
wall_ns = first start to last end, kernels_n. Iterations W..W+I-1 become rows rep = 0..I-1 of points.csv.
Clocks and throttle reasons are sampled before and after; a command whose samples are bad is re-run (up to 2
times) and its rows are kept with clock_bad = 1 (the notebook discards and counts them).

ncu. The run command with warmup 2, iters 1 under `ncu --clock-control none --cache-control all
--kernel-name-base demangled --kernel-name regex:<kernels> --launch-skip <pre + 2k> --launch-count k` (k kernels per iteration and pre
matching kernels before the first flush, both from the latest timing run); metrics filtered through
`ncu --query-metrics`; raw capture kept under results/ncu_raw/, every metric of every launch appended to
results/ncu_metrics.csv; one points.csv row with rep = -1 holds the byte counts summed over the k kernels.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import time

import yaml

import rooflib as rl

POINTS = os.path.join(rl.RES, "points.csv")
POINTS_HDR = rl.MANDATORY + ["workload", "variant_id", "role", "scope", "shape_id", "rep", "time_ns", "wall_ns",
                             "flops", "int_ops", "hbm_bytes", "l2_bytes", "l1_bytes", "dsmem_bytes", "kernels_n",
                             "source", "cluster_size", "clock_bad", "method", "multicast_bytes", "flops_ncu",
                             "kernel_names", "notes"]
VALID = os.path.join(rl.RES, "validation.csv")
VALID_HDR = rl.MANDATORY + ["workload", "variant_id", "shape_id", "cluster_size", "status", "returncode", "seconds",
                            "reason", "cmd", "output_tail"]
STAGES = os.path.join(rl.RES, "runner_stages.csv")
STAGES_HDR = ["run_id", "iso", "workload", "variant_id", "shape_id", "cluster_size", "stage", "status", "detail"]
NCU_METRICS = os.path.join(rl.RES, "ncu_metrics.csv")
NCU_HDR = rl.MANDATORY + ["workload", "variant_id", "shape_id", "cluster_size", "launch", "kernel", "metric", "unit",
                          "value"]
FLUSH_KERNEL = "roofline_l2_flush_kernel"
NCU_WARMUP = 2

METRICS = [
    "gpu__time_duration.sum",
    "dram__bytes_read.sum", "dram__bytes_write.sum",
    "lts__t_bytes.sum",
    "l1tex__t_bytes.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum", "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
    "l1tex__m_l1tex2xbar_write_bytes_mem_dshared.sum", "l1tex__m_xbar2l1tex_read_bytes_mem_dshared.sum",
    "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld_dest_multicast.sum",
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum", "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "sm__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "sm__sass_thread_inst_executed_op_hfma_pred_on.sum", "sm__sass_thread_inst_executed_op_hadd_pred_on.sum",
    "sm__sass_thread_inst_executed_op_hmul_pred_on.sum",
    "sm__sass_thread_inst_executed_op_integer_pred_on.sum",
    "sm__inst_executed_pipe_tensor_op_hmma.sum", "sm__inst_executed_pipe_tensor_op_gmma.sum",
    "sm__cycles_elapsed.avg", "launch__grid_size", "launch__cluster_dim_x", "launch__block_size",
]


# ---------------------------------------------------------------- manifests and configurations
def load(path):
    m = yaml.safe_load(open(path))
    m["_path"] = path
    return m


def configs(m, sel):
    so = m["scope_operator"]
    for v in so["variants"]:
        if sel.variant and v["id"] not in sel.variant:
            continue
        cs = v.get("cluster_size")
        ps = cs if isinstance(cs, list) else [cs]
        for sid, shape in so["shapes"].items():
            if sel.shape and sid not in sel.shape:
                continue
            for p in ps:
                if sel.P and p not in sel.P:
                    continue
                yield {"m": m, "v": v, "sid": sid, "shape": shape, "P": p}


def key(c):
    return [c["m"]["name"], c["v"]["id"], c["sid"], "" if c["P"] is None else c["P"]]


def cfg_name(c):
    return "__".join(str(x) for x in key(c) if x != "")


def worktree_key(v):
    repo = os.path.basename(v["build"]["cwd"].rstrip("/"))
    patch = v["build"].get("patch")
    return f"{repo}__{os.path.splitext(os.path.basename(patch))[0] if patch else 'pin'}"


def fmt(s, c, **kw):
    wt = os.path.join(rl.BUILD, "src", worktree_key(c["v"]))
    vals = dict(c["shape"], P=c["P"], worktree=wt, pyenv=os.path.join(rl.BUILD, "pyenv", worktree_key(c["v"])),
                repo=rl.REPO, harness=rl.HARNESS, shape_id=c["sid"], warmup=kw.get("warmup", 0),
                iters=kw.get("iters", 1))
    return str(s).format(**vals)


def env_for(c, section, **kw):
    e = dict(os.environ)
    for k, v in (c["v"].get(section, {}) or {}).get("env", {}).items():
        e[k] = fmt(v, c, **kw)
    extra = [fmt(p, c, **kw) for p in (c["v"].get(section, {}) or {}).get("pythonpath", [])]
    e["PYTHONPATH"] = ":".join(extra + [rl.HARNESS] + ([e["PYTHONPATH"]] if e.get("PYTHONPATH") else []))
    return e


def scope_of(c):
    """The manifest scope block whose formulas apply to this configuration (scope_layer for `scope: layer` variants)."""
    return c["m"].get("scope_layer") or {} if c["v"].get("scope") == "layer" else c["m"]["scope_operator"]


def evalf(expr, c):
    return float(eval(str(expr), {"__builtins__": {}}, dict(c["shape"], P=c["P"] or 1))) if expr else ""


def stage_done(c, stage):
    rows = [r for r in rl.read_csv(STAGES) if [r["workload"], r["variant_id"], r["shape_id"], r["cluster_size"]]
            == [str(x) for x in key(c)] and r["stage"] == stage]
    return bool(rows) and rows[-1]["status"] == "complete"


def note_stage(run_id, c, stage, status, detail=""):
    rl.append_rows(STAGES, STAGES_HDR, [[run_id, rl.iso_now()] + key(c) + [stage, status, detail]])


def sh(cmd, cwd, env, log, timeout):
    """Run a shell command, tee output to `log`; returns (returncode, output)."""
    os.makedirs(os.path.dirname(log), exist_ok=True)
    t0 = time.time()
    try:
        p = subprocess.run(["bash", "-c", cmd], cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
        out, rc = p.stdout + p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        out, rc = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or ""), -9
        out += f"\n[runner] TIMEOUT after {timeout} s"
    with open(log, "a") as f:
        f.write(f"\n===== {rl.iso_now()} cwd={cwd} ({time.time() - t0:.1f} s, rc={rc})\n$ {cmd}\n{out}")
    return rc, out


# ---------------------------------------------------------------- stages
def ensure_worktree(v, dry):
    wk = worktree_key(v)
    wt = os.path.join(rl.BUILD, "src", wk)
    sub = os.path.join(rl.REPO, v["build"]["cwd"])
    if os.path.isdir(wt):
        return wt
    pin = subprocess.run(["git", "-C", rl.REPO, "ls-tree", "HEAD", v["build"]["cwd"]], capture_output=True,
                         text=True).stdout.split()[2]
    cmds = [["git", "-C", sub, "worktree", "add", "--detach", wt, pin]]
    if v["build"].get("patch"):
        cmds.append(["git", "-C", wt, "apply", os.path.join(rl.ROOF, v["build"]["patch"])])
    for cmd in cmds:
        print("+", shlex.join(cmd))
        if not dry:
            subprocess.run(cmd, check=True)
    return wt


def do_build(cs, args, run_id):
    done = set()
    for c in cs:
        v = c["v"]
        wt = ensure_worktree(v, args.dry_run)
        cmd = fmt(v["build"]["cmd"], c)
        if (wt, cmd) in done:
            continue
        done.add((wt, cmd))
        stamp = os.path.join(wt, ".roofline_build_" + hashlib.sha1(cmd.encode()).hexdigest()[:12])
        if os.path.exists(stamp) and not args.force:
            print(f"[build] up to date: {os.path.basename(wt)}: {cmd}")
            continue
        if v["build"].get("gpu") and not os.environ.get("H100MB_GPU_UUID"):
            sys.exit(f"[build] {v['id']}: build.gpu is set, run this stage with a GPU (tools/run without --no-gpu)")
        print(f"[build] {os.path.basename(wt)}: {cmd}")
        if args.dry_run:
            continue
        rc, out = sh(cmd, wt, env_for(c, "build"), os.path.join(rl.BUILD, "logs", os.path.basename(wt) + ".log"),
                     args.timeout * 6)
        note_stage(run_id, c, "build", "complete" if rc == 0 else "failed", f"rc={rc}")
        if rc != 0:
            print(out[-3000:])
            sys.exit(f"[build] FAILED: {os.path.basename(wt)} (log roofline/build/logs/{os.path.basename(wt)}.log)")
        open(stamp, "w").write(cmd + "\n")


def latest_validation(c):
    rows = [r for r in rl.read_csv(VALID) if [r["workload"], r["variant_id"], r["shape_id"], r["cluster_size"]]
            == [str(x) for x in key(c)]]
    return rows[-1]["status"] if rows else ""


def do_validate(c, args, run_id):
    v = c["v"]
    if latest_validation(c) in ("PASS", "SKIP") and not args.force:
        print(f"[validate] {cfg_name(c)}: {latest_validation(c)} already, skipped")
        return
    sk = v.get("skip") or {}
    reason = sk.get(f"{c['sid']}/P{c['P']}") or sk.get(c["sid"])
    wt = os.path.join(rl.BUILD, "src", worktree_key(v))
    val = v.get("validate") or {}
    cmd = fmt(val.get("cmd") or v["run"]["cmd"], c, warmup=0, iters=1)
    if args.dry_run:
        print(f"[validate] {cfg_name(c)}: {'SKIP ' + reason if reason else cmd}")
        return
    clk = rl.clock_sample()
    if reason:
        status, rc, dt, out = "SKIP", "", 0.0, ""
    else:
        print(f"[validate] {cfg_name(c)}: {cmd}")
        if args.dry_run:
            return
        t0 = time.time()
        rc, out = sh(cmd, wt, env_for(c, "validate" if val.get("env") else "run", warmup=0, iters=1),
                     os.path.join(rl.RES, "logs", f"validate__{cfg_name(c)}.log"), args.timeout)
        dt = time.time() - t0
        miss = [x for x in val.get("expect", []) if not re.search(fmt(x, c), out, re.M)]
        status = "PASS" if rc == 0 and not miss else "FAIL"
        reason = f"rc={rc}" + (f"; expect not matched: {miss}" if miss else "")
    tail = " | ".join(out.strip().splitlines()[-6:])[-1500:]
    rl.append_rows(VALID, VALID_HDR, [rl.mandatory(run_id, 0, clk) + key(c) + [status, rc, f"{dt:.1f}", reason, cmd,
                                                                               tail]])
    note_stage(run_id, c, "validate", "complete" if status in ("PASS", "SKIP") else "failed", status)
    print(f"[validate] {cfg_name(c)}: {status} {reason}")


def kernel_re(v):
    return re.compile("|".join(f"(?:{k})" for k in v["kernels"]))


def nsys_trace(rep_base, cmd, cwd, env, log, timeout):
    """Run `cmd` under nsys; returns (rc, [(start, dur, name)]) sorted by start."""
    rep = rep_base + ".nsys-rep"
    full = (f"nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=false "
            f"--output {shlex.quote(rep_base)} bash -c {shlex.quote(cmd)}")
    rc, out = sh(full, cwd, env, log, timeout)
    if rc != 0 or not os.path.exists(rep):
        return rc or 1, []
    rc2, _ = sh(f"nsys stats --force-export=true --report cuda_gpu_trace --format csv --output {shlex.quote(rep_base)} "
                f"{shlex.quote(rep)}", cwd, env, log, timeout)
    trace = rep_base + "_cuda_gpu_trace.csv"
    if rc2 != 0 or not os.path.exists(trace):
        return rc2 or 1, []
    ks = []
    for r in rl.read_csv(trace):
        name = r.get("Name", "")
        if name.startswith("[CUDA"):
            continue
        ks.append((int(r["Start (ns)"]), int(r["Duration (ns)"]), name))
    return 0, sorted(ks)


def group(ks, kre):
    """(pre, groups): matched kernels before the first flush, and per flush-delimited iteration the matched kernels."""
    pre, groups, cur = 0, [], None
    for k in ks:
        if FLUSH_KERNEL in k[2]:
            cur = []
            groups.append(cur)
        elif kre.search(k[2]):
            if cur is None:
                pre += 1
            else:
                cur.append(k)
    return pre, [g for g in groups if g]


def short(name):
    return re.sub(r"\(.*$", "", re.sub(r"^void ", "", name))[:80]


def do_time(c, args, run_id):
    v = c["v"]
    if latest_validation(c) != "PASS":
        print(f"[time] {cfg_name(c)}: not validated (latest: {latest_validation(c) or 'none'}), skipped")
        return
    if stage_done(c, "time") and not args.force:
        print(f"[time] {cfg_name(c)}: complete, skipped")
        return
    W, I = args.warmup, args.reps
    wt = os.path.join(rl.BUILD, "src", worktree_key(v))
    cmd = fmt(v["run"]["cmd"], c, warmup=W, iters=I)
    print(f"[time] {cfg_name(c)}: {cmd}")
    if args.dry_run:
        return
    raw = os.path.join(rl.RES, "nsys", cfg_name(c))
    os.makedirs(raw, exist_ok=True)
    kre = kernel_re(v)
    for attempt in range(3):
        c0 = rl.clock_sample()
        stamp = rl.iso_now().replace(":", "")
        rc, ks = nsys_trace(os.path.join(raw, stamp), cmd, wt, env_for(c, "run", warmup=W, iters=I),
                            os.path.join(rl.RES, "logs", f"time__{cfg_name(c)}.log"), args.timeout)
        c1 = rl.clock_sample()
        bad = c0["bad"] or c1["bad"]
        if rc != 0:
            note_stage(run_id, c, "time", "failed", f"rc={rc}")
            print(f"[time] {cfg_name(c)}: FAILED rc={rc}")
            return
        pre, groups = group(ks, kre)
        if len(groups) < W + I:
            note_stage(run_id, c, "time", "failed", f"{len(groups)} flush-delimited iterations < {W + I}")
            print(f"[time] {cfg_name(c)}: only {len(groups)} iterations with matching kernels (expected >= {W + I})")
            return
        sel = groups[W:W + I]
        kn = statistics.mode(len(g) for g in sel)
        names = "|".join(sorted({short(k[2]) for g in sel for k in g}))
        rows = []
        for rep, g in enumerate(sel):
            t = sum(k[1] for k in g)
            wall = max(k[0] + k[1] for k in g) - min(k[0] for k in g)
            rows.append(rl.mandatory(run_id, rep, c0) + [c["m"]["name"], v["id"], v["role"], v.get("scope", "operator"), c["sid"], rep, t,
                        wall, evalf(scope_of(c).get("flops"), c),
                        evalf(scope_of(c).get("int_ops"), c), "", "", "", "", len(g), "measured",
                        "" if c["P"] is None else c["P"], int(bad), f"nsys:{stamp}", "", "", names,
                        f"attempt {attempt}; clocks {c0['sm']}/{c1['sm']} MHz reasons {c0['reasons']:#x}/{c1['reasons']:#x}"])
        rl.append_rows(POINTS, POINTS_HDR, rows)
        json.dump({"pre": pre, "k": kn, "names": names, "stamp": stamp},
                  open(os.path.join(raw, "kernel_counts.json"), "w"))
        med = statistics.median(r[POINTS_HDR.index("time_ns")] for r in rows)
        print(f"[time] {cfg_name(c)}: {I} reps, median {med / 1e3:.2f} us, {kn} kernel(s)/iter, pre={pre}, "
              f"clock_bad={int(bad)}")
        if not bad:
            break
    note_stage(run_id, c, "time", "complete", f"clock_bad={int(bad)}")


def ncu_query(args):
    q = os.path.join(rl.RES, "ncu_query_metrics.txt")
    if not os.path.exists(q) and not args.dry_run:
        with open(q + ".part", "w") as f:
            subprocess.run(["ncu", "--query-metrics"], stdout=f, check=True)
        os.rename(q + ".part", q)
    return q


def do_ncu(c, args, run_id, metrics):
    v = c["v"]
    if latest_validation(c) != "PASS":
        print(f"[ncu] {cfg_name(c)}: not validated, skipped")
        return
    if stage_done(c, "ncu") and not args.force:
        print(f"[ncu] {cfg_name(c)}: complete, skipped")
        return
    kc_path = os.path.join(rl.RES, "nsys", cfg_name(c), "kernel_counts.json")
    if not os.path.exists(kc_path):
        print(f"[ncu] {cfg_name(c)}: no timing run yet (kernel counts unknown), skipped")
        return
    kc = json.load(open(kc_path))
    wt = os.path.join(rl.BUILD, "src", worktree_key(v))
    cmd = fmt(v["run"]["cmd"], c, warmup=NCU_WARMUP, iters=1)
    regex = "|".join(f"(?:{k})" for k in v["kernels"])
    raw_dir = os.path.join(rl.RES, "ncu_raw")
    os.makedirs(raw_dir, exist_ok=True)
    raw = os.path.join(raw_dir, f"{cfg_name(c)}__{rl.iso_now().replace(':', '')}.csv")
    full = (f"ncu --clock-control none --cache-control all --kernel-name-base demangled --kernel-name regex:{shlex.quote(regex)} "
            f"--launch-skip {kc['pre'] + NCU_WARMUP * kc['k']} --launch-count {kc['k']} --metrics {','.join(metrics)} "
            f"--csv --page raw --print-units base --log-file {shlex.quote(raw + '.part')} bash -c {shlex.quote(cmd)}")
    print(f"[ncu] {cfg_name(c)}: {cmd}")
    if args.dry_run:
        return
    c0 = rl.clock_sample()
    rc, out = sh(full, wt, env_for(c, "run", warmup=NCU_WARMUP, iters=1),
                 os.path.join(rl.RES, "logs", f"ncu__{cfg_name(c)}.log"), args.timeout * 4)
    os.rename(raw + ".part", raw) if os.path.exists(raw + ".part") else None
    launches = rl.parse(raw) if os.path.exists(raw) else []
    if rc != 0 or len(launches) != kc["k"]:
        note_stage(run_id, c, "ncu", "failed", f"rc={rc} launches={len(launches)} expected={kc['k']}")
        print(f"[ncu] {cfg_name(c)}: FAILED rc={rc}, {len(launches)} launches profiled (expected {kc['k']})")
        return
    rows, tot = [], {}
    for n, d in launches:
        kname = d.get("Kernel Name", ("", ""))[1]
        for mname in metrics:
            if mname in d:
                u, val = d[mname]
                rows.append(rl.mandatory(run_id, n, c0) + key(c) + [n, short(kname), mname, u, val])
                tot[mname] = tot.get(mname, 0.0) + rl.num(val)
    rl.append_rows(NCU_METRICS, NCU_HDR, rows)
    g = lambda *ms: sum(tot.get(x, 0.0) for x in ms)
    l1 = g("l1tex__t_bytes.sum") + 128 * g("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
                                           "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum")
    flops_ncu = 2 * g("sm__sass_thread_inst_executed_op_ffma_pred_on.sum") + g(
        "sm__sass_thread_inst_executed_op_fadd_pred_on.sum", "sm__sass_thread_inst_executed_op_fmul_pred_on.sum")
    names = "|".join(sorted({short(d.get("Kernel Name", ("", ""))[1]) for _, d in launches}))
    c1 = rl.clock_sample()
    rl.append_rows(POINTS, POINTS_HDR, [rl.mandatory(run_id, -1, c0) + [
        c["m"]["name"], v["id"], v["role"], v.get("scope", "operator"), c["sid"], -1, g("gpu__time_duration.sum"), "",
        evalf(scope_of(c).get("flops"), c), evalf(scope_of(c).get("int_ops"), c),
        g("dram__bytes_read.sum", "dram__bytes_write.sum"), g("lts__t_bytes.sum"), l1,
        g("l1tex__m_l1tex2xbar_write_bytes_mem_dshared.sum"), len(launches), "measured",
        "" if c["P"] is None else c["P"], int(c0["bad"] or c1["bad"]), f"ncu:{os.path.basename(raw)}",
        g("l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld_dest_multicast.sum"), flops_ncu, names,
        f"ncu --cache-control all; launch-skip {kc['pre'] + NCU_WARMUP * kc['k']}"]])
    note_stage(run_id, c, "ncu", "complete", f"{len(launches)} launches")
    print(f"[ncu] {cfg_name(c)}: {len(launches)} kernel(s), HBM {g('dram__bytes_read.sum', 'dram__bytes_write.sum') / 1e6:.2f} MB, "
          f"DSMEM {g('l1tex__m_l1tex2xbar_write_bytes_mem_dshared.sum') / 1e6:.3f} MB, "
          f"time {g('gpu__time_duration.sum') / 1e3:.2f} us")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifests", nargs="+")
    ap.add_argument("--stage", choices=["build", "validate", "time", "ncu", "all"], default="all")
    ap.add_argument("--variant", nargs="*")
    ap.add_argument("--shape", nargs="*")
    ap.add_argument("--P", nargs="*", type=int)
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=900, help="seconds per command (builds 6x, ncu 4x)")
    ap.add_argument("--force", action="store_true", help="redo stages already complete")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    paths = [p for x in args.manifests for p in (sorted(glob.glob(x)) if any(ch in x for ch in "*?[") else [x])]
    ms = [load(p) for p in paths if not os.path.basename(p).startswith("_")]
    ms = [m for m in ms if m.get("runnable") is not False]
    stages = ["build", "validate", "time", "ncu"] if args.stage == "all" else [args.stage]
    if any(s != "build" for s in stages) and not os.environ.get("H100MB_GPU_UUID") and not args.dry_run:
        sys.exit("stages validate/time/ncu need the GPU: roofline/tools/run --profile --lock-sm-mhz <LOCK> -- ...")
    reg = None if args.dry_run else rl.Registry(sys.argv)
    run_id = reg.run_id if reg else "dry-run"
    metrics = None
    for m in ms:
        meas = m.get("measure") or {}
        a = argparse.Namespace(**vars(args))
        a.reps = args.reps or meas.get("reps", 100)
        a.warmup = args.warmup if args.warmup is not None else meas.get("warmup", 20)
        cs = list(configs(m, args))
        for s in stages:
            if s == "build":
                do_build(cs, a, run_id)
                continue
            if s == "ncu" and metrics is None:
                metrics = rl.available(METRICS, ncu_query(a))
            for c in cs:
                {"validate": do_validate, "time": do_time}.get(s, lambda c, a, r: do_ncu(c, a, r, metrics))(c, a, run_id)
    if reg:
        reg.complete()


if __name__ == "__main__":
    main()
