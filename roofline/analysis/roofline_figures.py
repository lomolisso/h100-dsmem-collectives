"""Data loading, aggregation, figures and tables of the roofline notebook (`roofline.ipynb`).

Inputs: `../workloads/*.yaml` (manifests; work and compulsory bytes are evaluated from them, so a formula fix
needs no re-run), `../results/ceilings.csv`, `../results/points.csv`, `../results/validation.csv`.
One configuration = (workload, variant_id, scope, shape_id, cluster_size). Per configuration:
- time: median / p10 / p90 of the latest timing run's reps (rows rep >= 0), launches with clock_bad = 1 dropped
  and counted;
- bytes per level, ncu FLOP count and ncu kernel time: the latest counter row (rep = -1);
- work (FLOP or integer ops) and compulsory HBM bytes from the manifest at that shape.
The marker of a variant with several cluster sizes is its fastest P per shape (`best_p`)."""
import glob
import os

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOF = os.path.dirname(HERE)
RES = os.path.join(ROOF, "results")
KEY = ["workload", "variant_id", "scope", "shape_id", "cluster_size"]


def manifests():
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOF, "workloads", "*.yaml"))):
        if os.path.basename(p).startswith("_"):
            continue
        m = yaml.safe_load(open(p))
        out[m["name"]] = m
    return out


def _eval(expr, shape, p):
    if not expr:
        return np.nan
    return float(eval(str(expr), {"__builtins__": {}}, dict(shape, P=p if p == p else 1)))


def ceilings():
    """Latest value per ceiling name -> DataFrame indexed by name (value, unit, method, source, p10, p90)."""
    path = os.path.join(RES, "ceilings.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["value", "unit", "method", "source"])
    c = pd.read_csv(path)
    return c.groupby("name").tail(1).set_index("name")


def points():
    p = pd.read_csv(os.path.join(RES, "points.csv"), low_memory=False)
    p["cluster_size"] = p["cluster_size"].fillna(-1).astype(int)
    p["scope"] = p["scope"].fillna("operator")
    return p


def configs(M=None, P=None):
    """One row per configuration: timing statistics, counters, manifest work and compulsory bytes."""
    M = M or manifests()
    P = points() if P is None else P
    t = P[P.rep >= 0].copy()
    t["run"] = t["method"]
    last = t.groupby(KEY)["run"].transform("max")
    t = t[t.run == last]
    good = t[t.clock_bad == 0]
    agg = good.groupby(KEY).agg(time_ns=("time_ns", "median"), time_p10=("time_ns", lambda x: x.quantile(.1)),
                                time_p90=("time_ns", lambda x: x.quantile(.9)), wall_ns=("wall_ns", "median"),
                                reps=("time_ns", "size"), kernels_n=("kernels_n", "median"), role=("role", "first"),
                                kernel_names=("kernel_names", "first"), timing_run=("run", "first"))
    agg["discarded"] = t.groupby(KEY)["clock_bad"].sum().reindex(agg.index).fillna(0).astype(int)
    cnt = P[P.rep == -1].groupby(KEY).tail(1).set_index(KEY)[
        ["hbm_bytes", "l2_bytes", "l1_bytes", "dsmem_bytes", "multicast_bytes", "flops_ncu", "time_ns", "role"]]
    cnt = cnt.rename(columns={"time_ns": "ncu_time_ns", "role": "role_c"})
    df = agg.join(cnt, how="outer").reset_index()
    df["role"] = df["role"].fillna(df.pop("role_c"))
    work, minb, prec = [], [], []
    for r in df.itertuples():
        m = M.get(r.workload)
        so = m["scope_operator"] if m else {}
        shape = so.get("shapes", {}).get(r.shape_id, {})
        p = r.cluster_size if r.cluster_size > 0 else 1
        w = _eval(so.get("flops"), shape, p) if so.get("flops") else _eval(so.get("int_ops"), shape, p)
        work.append(w)
        minb.append(_eval(so.get("min_hbm_bytes"), shape, p))
        prec.append(m.get("precision") if m else "")
    df["work"], df["min_hbm_bytes"], df["precision"] = work, minb, prec
    df["throughput"] = df.work / (df.time_ns * 1e-9)
    for lvl in ["hbm", "l2", "l1", "dsmem"]:
        b = df[f"{lvl}_bytes"].where(df[f"{lvl}_bytes"] > 0)
        df[f"ai_{lvl}"] = df.work / b
    df["ai_compulsory"] = df.work / df.min_hbm_bytes
    return df


def best_p(df):
    """Per (workload, variant, scope, shape): the configuration with the lowest median time (marker of the variant)."""
    d = df.dropna(subset=["time_ns"])
    idx = d.groupby(["workload", "variant_id", "scope", "shape_id"])["time_ns"].idxmin()
    return d.loc[idx].reset_index(drop=True)


def validation():
    v = pd.read_csv(os.path.join(RES, "validation.csv"))
    v["cluster_size"] = v["cluster_size"].fillna(-1)
    return v.groupby(["workload", "variant_id", "shape_id", "cluster_size"]).tail(1)


# ---------------------------------------------------------------- figures and tables
import matplotlib.pyplot as plt  # noqa: E402

BOOST_MHZ = 1785  # H100 NVL maximum SM clock (datasheet peaks are quoted at this clock)
COMPUTE = [("fp8_tc_peak", "FP8 tensor"), ("bf16_tc_peak", "BF16 tensor"), ("fp16_tc_peak", "FP16 tensor"),
           ("fp32_peak", "FP32 CUDA core"), ("int32_peak", "INT32 (IMAD)")]
MEMORY = [("l1_smem_bw", "SMEM"), ("l2_bw", "L2"), ("dsmem_bw", "DSMEM"), ("hbm_read_bw", "HBM")]
DATASHEET = {"fp8_tc_peak": "fp8_tc_peak_datasheet", "bf16_tc_peak": "bf16_tc_peak_datasheet",
             "fp16_tc_peak": "fp16_tc_peak_datasheet", "fp32_peak": "fp32_peak_datasheet",
             "hbm_read_bw": "hbm_bw_datasheet"}


def _show(fig, save=None):
    """Embed the figure as a PNG in a notebook (any matplotlib backend; copied from collectives_figures._show);
    with `save`, write it to that path instead."""
    import io
    if save:
        fig.savefig(save, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return
    try:
        from IPython import get_ipython
        from IPython.display import Image, display
    except ImportError:
        get_ipython = lambda: None  # noqa: E731
    if get_ipython() is None:
        plt.show()
        return
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    display(Image(data=buf.getvalue()))


def lock_mhz(C):
    return int(C.loc["hbm_read_bw", "sm_clock_mhz"]) if "hbm_read_bw" in C.index else 1635


def table_ceilings(C=None):
    """Measured ceilings next to the datasheet value, scaled to the locked SM clock for compute ceilings."""
    C = ceilings() if C is None else C
    lock = lock_mhz(C)
    rows = []
    for name, label in COMPUTE + MEMORY:
        if name not in C.index:
            continue
        v, unit = float(C.loc[name, "value"]), C.loc[name, "unit"]
        ds = DATASHEET.get(name)
        dv = float(C.loc[ds, "value"]) if ds in C.index else np.nan
        dlock = dv * lock / BOOST_MHZ if unit.endswith("/s") and "FLOP" in unit else dv
        rows.append({"ceiling": label, "measured": v / 1e12, "unit": "T" + unit,
                     "datasheet": dv / 1e12, "datasheet at lock": dlock / 1e12, "measured / datasheet at lock": v / dlock,
                     "ridge vs HBM (FLOP/B)": v / float(C.loc["hbm_read_bw", "value"]) if "FLOP" in unit or "intop" in unit else np.nan})
    t = pd.DataFrame(rows).set_index("ceiling")
    return t.style.format({"measured": "{:.1f}", "datasheet": "{:.1f}", "datasheet at lock": "{:.1f}",
                           "measured / datasheet at lock": "{:.0%}", "ridge vs HBM (FLOP/B)": "{:.1f}"}, na_rep="—")


def table_dsmem(C=None):
    """GPU-wide DSMEM bandwidth per cluster size and direction (all clusters active, 1 CTA per SM)."""
    C = ceilings() if C is None else C
    rows = {}
    for cs in (2, 4, 8, 16):
        rows[cs] = {d: float(C.loc[f"dsmem_bw_{k}_cs{cs}", "value"]) / 1e12 for d, k in
                    (("pull (ld.shared::cluster.v4), TB/s", "ld"), ("push (bulk copy), TB/s", "push"))
                    if f"dsmem_bw_{k}_cs{cs}" in C.index}
    t = pd.DataFrame(rows).T
    t.index.name = "cluster size"
    return t.style.format("{:.2f}")


def fig_ceilings(C=None, save=None):
    """Roofline of the device alone: compute roofs over the HBM slope, plus the other memory levels' slopes."""
    C = ceilings() if C is None else C
    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.logspace(-2, 4, 400)
    hbm = float(C.loc["hbm_read_bw", "value"])
    colors = {"fp8_tc_peak": "C3", "fp16_tc_peak": "C1", "fp32_peak": "C0", "int32_peak": "C2"}
    for name, label in COMPUTE:
        if name not in C.index or name not in colors:
            continue
        v = float(C.loc[name, "value"])
        ax.plot(x, np.minimum(v, hbm * x), color=colors[name], lw=2, label=f"{label} {v / 1e12:.0f} T/s")
        ds = DATASHEET.get(name)
        if ds in C.index:
            ax.axhline(float(C.loc[ds, "value"]), color=colors[name], ls=":", lw=1)
    for name, label in MEMORY[:-1]:
        if name in C.index:
            bw = float(C.loc[name, "value"])
            y = bw * x
            ax.plot(x[y < 1.2e15], y[y < 1.2e15], color="0.6", lw=1, ls="--")
            xi = 3e12 / bw
            ax.text(xi, bw * xi * 1.15, f"{label} {bw / 1e12:.1f} TB/s", color="0.4", fontsize=8, rotation=37)
    ax.text(0.012, hbm * 0.012 * 1.3, f"HBM {hbm / 1e12:.2f} TB/s", fontsize=8, rotation=37)
    ax.set(xscale="log", yscale="log", xlabel="operational intensity (FLOP or int op per HBM byte)",
           ylabel="throughput (FLOP/s or int op/s)", ylim=(1e10, 3e15), xlim=(1e-2, 1e4))
    ax.legend(loc="lower right", fontsize=8, title=f"measured at {lock_mhz(C)} MHz; dotted: datasheet", title_fontsize=8)
    ax.grid(True, which="major", alpha=.3)
    _show(fig, save)


def table_setup(C=None, P=None):
    C = ceilings() if C is None else C
    r = C.iloc[0]
    rows = {"device": f"{r.gpu_name}, {int(r.sm_count)} SMs (GPU 3)", "driver / CUDA": f"{r.driver_version} / {r.cuda_version}",
            "SM clock": f"locked at {lock_mhz(C)} MHz for every timed and ncu run; memory clock {int(r.mem_clock_mhz)} MHz",
            "ceilings run": str(C["run_id"].iloc[-1])}
    if P is not None and len(P):
        rows["points rows"] = f"{len(P)} ({(P.rep >= 0).sum()} timed reps, {(P.rep == -1).sum()} counter rows)"
    return pd.DataFrame({"": rows}).style
