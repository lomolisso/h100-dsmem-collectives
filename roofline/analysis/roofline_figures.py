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
        so = (m.get("scope_layer") or {} if r.scope == "layer" else m["scope_operator"]) if m else {}
        so = dict(so, shapes=m["scope_operator"]["shapes"]) if m else so
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


# Architectural dense peak per SM per clock (GH100): FP32 128 FFMA = 256 FLOP; tensor cores 4096 FLOP (FP16/BF16)
# and 8192 (FP8); INT32 IMAD issues at half the FFMA rate. HBM is not clock-scaled.
PER_CLK_SM = {"fp32_peak": 256, "int32_peak": 128, "fp16_tc_peak": 4096, "bf16_tc_peak": 4096, "fp8_tc_peak": 8192}


def table_ceilings(C=None):
    """Measured ceilings next to the datasheet value and the architectural peak at the locked clock
    (FLOP/clk/SM x 132 SMs x lock). The H100 NVL datasheet quotes FP32 at ~1775 MHz but tensor cores at 1545 MHz
    (835.5 T = 132 x 4096 x 1545 MHz), so it is not scaled."""
    C = ceilings() if C is None else C
    lock = lock_mhz(C)
    sms = int(C.loc["hbm_read_bw", "sm_count"]) if "hbm_read_bw" in C.index else 132
    rows = []
    for name, label in COMPUTE + MEMORY:
        if name not in C.index:
            continue
        v, unit = float(C.loc[name, "value"]), C.loc[name, "unit"]
        ds = DATASHEET.get(name)
        dv = float(C.loc[ds, "value"]) if ds in C.index else np.nan
        arch = PER_CLK_SM[name] * sms * lock * 1e6 if name in PER_CLK_SM else (dv if name == "hbm_read_bw" else np.nan)
        rows.append({"ceiling": label, "measured": v / 1e12, "unit": "T" + unit, "datasheet": dv / 1e12,
                     "peak at lock": arch / 1e12, "measured / peak at lock": v / arch,
                     "ridge vs HBM (op/B)": v / float(C.loc["hbm_read_bw", "value"]) if name in PER_CLK_SM else np.nan})
    t = pd.DataFrame(rows).set_index("ceiling")
    return t.style.format({"measured": "{:.1f}", "datasheet": "{:.1f}", "peak at lock": "{:.1f}",
                           "measured / peak at lock": "{:.0%}", "ridge vs HBM (op/B)": "{:.1f}"}, na_rep="—")


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
        ax.plot(x, np.minimum(v, hbm * x), color=colors[name], lw=2, label=f"{label} {v / 1e12:.0f} T op/s")
        pl = name.replace("_peak", "_power_limited")
        if pl in C.index:
            vp = float(C.loc[pl, "value"])
            ax.plot(x, np.minimum(vp, hbm * x), color=colors[name], lw=1.2, ls="--")
    for (name, label), ls in zip(MEMORY[:-1], [":", "-.", "--"]):
        if name in C.index:
            bw = float(C.loc[name, "value"])
            y = bw * x
            ax.plot(x[y < 1.2e15], y[y < 1.2e15], color="0.55", lw=1, ls=ls, label=f"{label} {bw / 1e12:.1f} TB/s")
    ax.plot([], [], color="0.2", lw=2, label=f"HBM {hbm / 1e12:.2f} TB/s (slope of every roof)")
    ax.set(xscale="log", yscale="log", xlabel="operational intensity (FLOP or int op per HBM byte)",
           ylabel="throughput (FLOP/s or int op/s)", ylim=(1e10, 3e15), xlim=(1e-2, 1e4))
    ax.legend(loc="lower right", fontsize=8, title=f"measured at {lock_mhz(C)} MHz; dashed: power-limited (real data)",
              title_fontsize=8)
    ax.grid(True, which="major", alpha=.3)
    _show(fig, save)


def table_setup(C=None, P=None):
    C = ceilings() if C is None else C
    r = C.iloc[0]
    rows = {"device": f"{r.gpu_name}, {int(r.sm_count)} SMs (GPU 3)", "driver / CUDA": f"{r.driver_version} / {r.cuda_version}",
            "SM clock": f"locked at {lock_mhz(C)} MHz for every timed and ncu run; memory clock {int(r.mem_clock_mhz)} MHz",
            "ceilings run": str(C["run_id"].iloc[-1])}
    rows["toolchain"] = ("image h100-roofline:cuda12.9.1-83734cedf58e (CUDA 12.9.1, nvcc 12.9.86, Nsight Compute 2025.2.1, "
                         "Nsight Systems 2025.1.3, PyTorch 2.11.0+cu129, Triton 3.6.0, flashinfer 0.7.2.post1)")
    rows["timing"] = ("kernel durations from nsys (20 warm-up + 100 iterations, L2 flushed before each); bytes and "
                      "instruction counts from ncu (--clock-control none --cache-control all)")
    if P is not None and len(P):
        rows["points rows"] = f"{len(P)} ({(P.rep >= 0).sum()} timed reps, {(P.rep == -1).sum()} counter rows)"
        rows["clock-flagged timed reps"] = f"{int((P[P.rep >= 0].clock_bad == 1).sum())}"
    try:
        r = pd.read_csv(os.path.join(RES, "roofline__runs.csv"))
        st = r.groupby("run_id").status.agg(set)
        rows["registered runs"] = (f"{len(st)} ({sum('complete' in x for x in st)} complete); first "
                                   f"{r.run_id.min()}, last {r.run_id.max()}")
    except FileNotFoundError:
        pass
    return pd.DataFrame({"": rows}).style


# ---------------------------------------------------------------- workload pairs
ROLE_STYLE = {"non_dsmem": dict(color="0.45", marker="o"), "control": dict(color="C1", marker="s"),
              "dsmem": dict(color="C0", marker="^"), "reference": dict(color="k", marker="x")}
ROLE_ORDER = ["non_dsmem", "control", "dsmem"]
PREC_CEIL = {"fp32_cuda": "fp32_peak", "fp16_tensor": "fp16_tc_peak", "bf16_tensor": "bf16_tc_peak",
             "fp8_tensor": "fp8_tc_peak", "int32": "int32_peak"}


def load():
    """Everything the notebook needs: manifests, ceilings, points, per-configuration table, markers."""
    M, C, P = manifests(), ceilings(), points()
    D = configs(M, P)
    return {"M": M, "C": C, "P": P, "D": D, "markers": best_p(D[D.scope == "operator"])}


def short_name(w):
    return w.replace("credit_", "").replace("_backward", " bwd").replace("_", " ")


def pairs(B, workloads=None):
    """One row per (workload, shape): the operator-scope markers of each role side by side."""
    B = B[B.scope == "operator"]
    if workloads is not None:
        B = B[B.workload.isin(workloads)]
    rows = []
    for (w, s), g in B.groupby(["workload", "shape_id"], sort=False):
        r = {"workload": w, "shape": s}
        for role in ROLE_ORDER + ["reference"]:
            h = g[g.role == role]
            if len(h):
                h = h.loc[h.time_ns.idxmin()]
                for k in ["time_ns", "hbm_bytes", "dsmem_bytes", "throughput", "ai_hbm", "cluster_size", "variant_id",
                          "work", "min_hbm_bytes"]:
                    r[f"{role}.{k}"] = h[k]
        rows.append(r)
    return pd.DataFrame(rows)


def table_pairs(D_or_load, workloads=None, elems=None):
    L = D_or_load
    T = pairs(L["markers"], workloads)
    out = pd.DataFrame({
        "workload": T.workload.map(short_name), "shape": T["shape"],
        "non-DSMEM µs": T["non_dsmem.time_ns"] / 1e3,
        "control µs": T.get("control.time_ns", np.nan) / 1e3,
        "DSMEM µs": T["dsmem.time_ns"] / 1e3,
        "best P": T["dsmem.cluster_size"].astype("Int64"),
        "speedup vs non-DSMEM": T["non_dsmem.time_ns"] / T["dsmem.time_ns"],
        "HBM bytes DSMEM / non-DSMEM": T["dsmem.hbm_bytes"] / T["non_dsmem.hbm_bytes"],
        "HBM bytes DSMEM / compulsory": T["dsmem.hbm_bytes"] / T["dsmem.min_hbm_bytes"],
        "HBM TB/s non-DSMEM": T["non_dsmem.hbm_bytes"] / T["non_dsmem.time_ns"] / 1e3,
        "HBM TB/s DSMEM": T["dsmem.hbm_bytes"] / T["dsmem.time_ns"] / 1e3,
        "DSMEM / HBM bytes": T["dsmem.dsmem_bytes"] / T["dsmem.hbm_bytes"],
    })
    return out.set_index(["workload", "shape"]).style.format(
        {c: "{:.1f}" for c in ["non-DSMEM µs", "control µs", "DSMEM µs"]} |
        {c: "{:.2f}" for c in ["speedup vs non-DSMEM", "HBM bytes DSMEM / non-DSMEM", "HBM bytes DSMEM / compulsory",
                               "HBM TB/s non-DSMEM", "HBM TB/s DSMEM"]} | {"DSMEM / HBM bytes": "{:.3f}"}, na_rep="—")


ABBR = {}


def abbr(w):
    """Short label of a workload for crowded plots (first letters of its words, e.g. 'layernorm bwd' -> 'LB')."""
    if w not in ABBR:
        ABBR[w] = "".join(t[0] for t in short_name(w).split()).upper()
    return ABBR[w]


def fig_roofline(L, workloads, shapes=None, title=None, save=None, one_panel=False):
    """Classic roofline (x = work per HBM byte), one panel per shape: one marker per role and arrows
    non-DSMEM -> control -> DSMEM per workload; the compute roof of the workloads' precision."""
    C, B = L["C"], L["markers"]
    B = B[(B.scope == "operator") & B.workload.isin(workloads)]
    shapes = shapes or list(dict.fromkeys(B.shape_id))
    B = B[B.shape_id.isin(shapes)]
    panels = [shapes] if one_panel else [[s_] for s_ in shapes]
    fig, axs = plt.subplots(1, len(panels), figsize=(6.5 if one_panel else 4.2 * len(panels), 4.4), sharey=True,
                            squeeze=False)
    hbm = float(C.loc["hbm_read_bw", "value"])
    lo, hi = np.nanmin(B.ai_hbm) / 2.5, np.nanmax(B.ai_hbm) * 40
    ylo, yhi = np.nanmin(B.throughput) / 3, None
    x = np.logspace(np.log10(lo), np.log10(hi), 200)
    for ax, sh in zip(axs[0], panels):
        S = B[B.shape_id.isin(sh)]
        for p in sorted(set(S.precision.dropna())):
            roof = float(C.loc[PREC_CEIL[p], "value"])
            ax.plot(x, np.minimum(roof, hbm * x), color="0.2", lw=1.3)
            pl = PREC_CEIL[p].replace("_peak", "_power_limited")
            if pl in C.index:
                ax.plot(x, np.minimum(float(C.loc[pl, "value"]), hbm * x), color="0.2", lw=1, ls="--")
            ax.text(hi / 1.3, roof * 1.12, f"{p.replace('_', ' ')} {roof / 1e12:.0f} T/s", fontsize=7, ha="right")
            yhi = max(yhi or 0, roof * 2)
        for (w, s_), g in S.groupby(["workload", "shape_id"]):
            g = g.set_index("role")
            chain = [r for r in ROLE_ORDER if r in g.index]
            for a_, b_ in zip(chain, chain[1:]):
                ax.annotate("", xy=(g.loc[b_, "ai_hbm"], g.loc[b_, "throughput"]),
                            xytext=(g.loc[a_, "ai_hbm"], g.loc[a_, "throughput"]),
                            arrowprops=dict(arrowstyle="->", color="0.6", lw=.8))
            if "dsmem" in g.index:
                ax.annotate(s_ if one_panel else abbr(w), (g.loc["dsmem", "ai_hbm"], g.loc["dsmem", "throughput"]), xytext=(3, -8),
                            textcoords="offset points", fontsize=7, color="C0")
        for role, st in ROLE_STYLE.items():
            h = S[S.role == role]
            if len(h):
                ax.scatter(h.ai_hbm, h.throughput, s=26, zorder=3, label=role.replace("_", "-"), **st)
        ax.set(xscale="log", yscale="log", xlim=(lo, hi), title=None if one_panel else sh[0],
               xlabel="work per HBM byte")
        ax.grid(True, which="major", alpha=.3)
    axs[0][0].set(ylabel="throughput (work/s)", ylim=(ylo, max(yhi or 0, np.nanmax(B.throughput) * 2)))
    axs[0][0].legend(fontsize=7, loc="upper left")
    key = "" if one_panel else ", ".join(f"{abbr(w)} = {short_name(w)}" for w in sorted(set(B.workload)))
    fig.suptitle((title + "\n" if title else "") + key, fontsize=8)
    fig.tight_layout()
    _show(fig, save)


def fig_time_vs_bytes(L, workloads, title=None, save=None):
    """Per (workload, shape): DSMEM / non-DSMEM time ratio against HBM-byte ratio. On the diagonal the DSMEM version
    keeps the non-DSMEM version's achieved HBM bandwidth; above it, it loses bandwidth (time falls less than bytes)."""
    T = pairs(L["markers"], workloads)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    shapes = list(dict.fromkeys(T["shape"]))
    for i, s in enumerate(shapes):
        t = T[T["shape"] == s]
        ax.scatter(t["dsmem.hbm_bytes"] / t["non_dsmem.hbm_bytes"], t["dsmem.time_ns"] / t["non_dsmem.time_ns"],
                   color=f"C{i}", label=s, s=30, zorder=3)
        for _, r in t.iterrows():
            ax.annotate(abbr(r.workload), (r["dsmem.hbm_bytes"] / r["non_dsmem.hbm_bytes"],
                                           r["dsmem.time_ns"] / r["non_dsmem.time_ns"]),
                        xytext=(3, 2), textcoords="offset points", fontsize=7, color="0.4")
    v = np.linspace(0.2, 2.6, 10)
    ax.plot(v, v, color="0.6", ls="--", lw=1)
    ax.axhline(1, color="0.8", lw=1)
    ax.set(xlabel="HBM bytes, DSMEM / non-DSMEM", ylabel="time, DSMEM / non-DSMEM", title=title,
           xlim=(0.3, 1.1), ylim=(0.5, 2.0))
    ax.set_yscale("log")
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 1, 1.2, 1.5, 2])
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
    ax.legend(fontsize=8, title="shape", title_fontsize=8)
    ax.grid(True, alpha=.3)
    _show(fig, save)


def table_flop_check(L, workloads):
    """ncu-counted work of the marker configuration of every role / manifest formula (CUDA-core FP32 counters:
    2·FFMA + FADD + FMUL); a ratio outside 0.9–1.1 needs the explanation given in the text."""
    B = L["markers"]
    B = B[(B.scope == "operator") & B.workload.isin(workloads)]
    t = B.assign(ratio=B.flops_ncu / B.work).pivot_table(index=["workload", "shape_id"], columns="role", values="ratio")
    t.index = t.index.set_levels([t.index.levels[0].map(short_name), t.index.levels[1]])
    t = t[[r for r in ROLE_ORDER + ["reference"] if r in t.columns]]
    return t.style.format("{:.2f}", na_rep="—")


LEVELS = [("hbm", "hbm_read_bw", "HBM"), ("l2", "l2_bw", "L2"), ("l1", "l1_smem_bw", "L1/SMEM"),
          ("dsmem", "dsmem_bw", "DSMEM")]


def fig_hierarchical(L, workload, shape, save=None):
    """Hierarchical roofline of one workload at one shape: the same throughput against the intensity w.r.t. each
    memory level, with that level's measured bandwidth and the compute roof of the workload's precision. A marker far
    right of a level's ridge is not limited by that level."""
    C, B = L["C"], L["markers"]
    B = B[(B.scope == "operator") & (B.workload == workload) & (B.shape_id == shape)]
    roof = float(C.loc[PREC_CEIL[B.precision.iloc[0]], "value"])
    fig, axs = plt.subplots(1, len(LEVELS), figsize=(14, 3.6), sharey=True)
    for ax, (lvl, ceil, label) in zip(axs, LEVELS):
        bw = float(C.loc[ceil, "value"])
        x = np.logspace(-2, 6, 300)
        ax.plot(x, np.minimum(roof, bw * x), color="0.2", lw=1.3)
        for role, st in ROLE_STYLE.items():
            h = B[B.role == role]
            if len(h):
                ax.scatter(h[f"ai_{lvl}"].fillna(1e6), h.throughput, s=30, zorder=3, label=role.replace("_", "-"), **st)
        ax.set(xscale="log", yscale="log", xlim=(1e-1, 1e6), title=f"{label} ({bw / 1e12:.1f} TB/s)",
               xlabel=f"work per {label} byte")
        ax.grid(True, which="major", alpha=.3)
    axs[0].set(ylabel="throughput (work/s)", ylim=(np.nanmin(B.throughput) / 10, roof * 3))
    axs[0].legend(fontsize=7)
    fig.suptitle(f"{short_name(workload)}, {shape} (a level with no bytes is drawn at the right edge)", fontsize=9)
    fig.tight_layout()
    _show(fig, save)


# ---------------------------------------------------------------- all workloads (R8)
MECHANISM = {"clusterfusion_rmsnorm": "more SMs per row", "credit": "fewer HBM bytes (row read once)", "dak_gemm_multicast": "TMA multicast",
             "deepgemm_fp8_gemm_multicast": "TMA multicast", "hpcops_topk_cluster8": "more SMs per row",
             "clusterfusion": "kernel fusion"}
MECH_COLOR = {"fewer HBM bytes (row read once)": "C0", "TMA multicast": "C1", "more SMs per row": "C2",
              "kernel fusion": "C3"}
OVERVIEW_SHAPES = {"dak_gemm_multicast": "llama_qo_t1024", "deepgemm_fp8_gemm_multicast": "m4096_n4096_k7168",
                   "hpcops_topk_cluster8": "m8_n240k", "clusterfusion_llama2_7b_decoder": "s4k",
                   "clusterfusion_deepseek_v2_lite_mla": "s4k", "clusterfusion_rmsnorm": "s0"}


def mechanism(w):
    for k, v in MECHANISM.items():
        if w.startswith(k):
            return v
    return "other"


def reported(M):
    """Rows of the `reported` blocks of runnable: false manifests (list per variant; throughput null = not reported)."""
    rows = []
    for w, m in M.items():
        for vid, items in (m.get("reported") or {}).items():
            for it in items if isinstance(items, list) else [items]:
                if it and it.get("throughput"):
                    rows.append({"workload": w, "variant_id": vid, "shape_id": it.get("shape"),
                                 "throughput": float(it["throughput"]), "ai_hbm": float(it["flops"]) / float(it["hbm_bytes"]),
                                 "source": it.get("source", "")})
    return pd.DataFrame(rows)


def fig_overview(L, save=None):
    """Classic roofline of every workload at one shape (CREDIT: n64k): non-DSMEM (grey) -> DSMEM (coloured by how the
    DSMEM version differs). Right: the FA4 numbers reported for B200 by the paper (not comparable with the H100 roofs)."""
    C, B, M = L["C"], L["markers"], L["M"]
    B = B[B.scope == "operator"]
    sel = [(w, OVERVIEW_SHAPES.get(w, "n64k")) for w in M if M[w].get("runnable") is not False]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [3.2, 1]})
    hbm = float(C.loc["hbm_read_bw", "value"])
    x = np.logspace(-2, 4, 300)
    for name, label in [("fp8_tc_peak", "FP8 tensor"), ("fp16_tc_peak", "FP16/BF16 tensor"), ("fp32_peak", "FP32"),
                        ("int32_peak", "INT32")]:
        v = float(C.loc[name, "value"])
        ax.plot(x, np.minimum(v, hbm * x), color="0.25", lw=1.1)
        ax.text(9e3, v * 1.12, f"{label} {v / 1e12:.0f} T/s", fontsize=7, ha="right")
        pl = name.replace("_peak", "_power_limited")
        if pl in C.index:
            ax.plot(x, np.minimum(float(C.loc[pl, "value"]), hbm * x), color="0.25", lw=.8, ls="--")
    ax.text(0.012, hbm * 0.012 * 1.4, f"HBM {hbm / 1e12:.2f} TB/s", fontsize=7, rotation=33)
    seen = set()
    for w, s in sel:
        g = B[(B.workload == w) & (B.shape_id == s)].set_index("role")
        if not {"non_dsmem", "dsmem"} <= set(g.index):
            continue
        a, b = g.loc["non_dsmem"], g.loc["dsmem"]
        mech = mechanism(w)
        ax.annotate("", xy=(b.ai_hbm, b.throughput), xytext=(a.ai_hbm, a.throughput),
                    arrowprops=dict(arrowstyle="->", color="0.55", lw=.8))
        ax.scatter(a.ai_hbm, a.throughput, color="0.5", s=22, zorder=3, label="non-DSMEM" if "n" not in seen else None)
        seen.add("n")
        ax.scatter(b.ai_hbm, b.throughput, color=MECH_COLOR.get(mech, "k"), marker="^", s=40, zorder=4,
                   label=f"DSMEM: {mech}" if mech not in seen else None)
        seen.add(mech)
        ax.annotate(abbr(w), (b.ai_hbm, b.throughput), xytext=(4, -9), textcoords="offset points", fontsize=7)
    ax.set(xscale="log", yscale="log", xlim=(1e-2, 1e4), ylim=(1e10, 4e15), xlabel="work per HBM byte",
           ylabel="throughput (work/s)", title="H100 NVL, 1635 MHz, measured (operator scope)")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, which="major", alpha=.3)
    R = reported(M)
    for i, r in enumerate(R.itertuples()):
        ax2.scatter(r.ai_hbm, r.throughput, marker="*", s=140, color="C4")
        lab = {"s16k_full": "non-causal", "s16k_causal": "causal"}.get(r.shape_id, r.shape_id)
        ax2.annotate(f"FA4 bwd {r.variant_id.replace('_', '-')}, {lab}\n{r.throughput / 1e12:.0f} TFLOP/s", (r.ai_hbm, r.throughput),
                     xytext=(-60, 10 - 30 * i),
                     textcoords="offset points", fontsize=7)
    ax2.set(xscale="log", yscale="log", xlim=(1e2, 1e5), ylim=(1e14, 1e16), xlabel="FLOP per HBM byte (compulsory)",
            title="reported (B200, FA4 paper)")
    ax2.grid(True, which="major", alpha=.3)
    import textwrap
    key = ", ".join(f"{abbr(w)} = {short_name(w)} ({s})" for w, s in sel)
    fig.suptitle(textwrap.fill(key, 150), fontsize=7)
    fig.tight_layout()
    _show(fig, save)


def all_pairs(L):
    T = pairs(L["markers"])
    T = T.dropna(subset=["non_dsmem.time_ns", "dsmem.time_ns"]).copy()
    T["speedup"] = T["non_dsmem.time_ns"] / T["dsmem.time_ns"]
    T["hbm_ratio"] = T["dsmem.hbm_bytes"] / T["non_dsmem.hbm_bytes"]
    T["dsmem_per_hbm"] = T["dsmem.dsmem_bytes"] / T["dsmem.hbm_bytes"]
    T["mechanism"] = T.workload.map(mechanism)
    return T


def fig_speedup_vs_bytes(L, save=None):
    """Every measured pair (operator scope, every shape): DSMEM / non-DSMEM time against HBM-byte ratio, coloured by
    how the DSMEM version differs. Diagonal: time proportional to HBM bytes."""
    T = all_pairs(L)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for mech, g in T.groupby("mechanism"):
        ax.scatter(g.hbm_ratio, 1 / g.speedup, color=MECH_COLOR.get(mech, "k"), s=30, label=mech, zorder=3)
    v = np.linspace(0.3, 1.2, 10)
    ax.plot(v, v, color="0.6", ls="--", lw=1)
    ax.axhline(1, color="0.75", lw=1)
    ax.set(xlabel="HBM bytes, DSMEM / non-DSMEM", ylabel="time, DSMEM / non-DSMEM", xlim=(0.3, 1.15), yscale="log",
           ylim=(0.5, 2.0))
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 1, 1.25, 1.5, 2])
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter("%.2f"))
    ax.yaxis.set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
    ax.legend(fontsize=8)
    ax.grid(True, alpha=.3)
    _show(fig, save)


def table_summary(L):
    T = all_pairs(L)
    g = T.groupby("workload", sort=False)
    out = pd.DataFrame({
        "how DSMEM is used": g.mechanism.first(),
        "shapes": g.size(),
        "speedup min": g.speedup.min(), "speedup max": g.speedup.max(),
        "HBM bytes ratio min": g.hbm_ratio.min(), "HBM bytes ratio max": g.hbm_ratio.max(),
        "DSMEM bytes per HBM byte (max)": g.dsmem_per_hbm.max(),
    })
    out.index = out.index.map(short_name)
    return out.style.format({c: "{:.2f}" for c in out.columns if c not in ("how DSMEM is used", "shapes")} |
                            {"DSMEM bytes per HBM byte (max)": "{:.3f}"})


def sol_check(L=None):
    """Latest complete row per configuration of results/sol_crosscheck.csv (harness/sol_crosscheck.py: ncu's
    SpeedOfLight roofline sections vs our point)."""
    s = pd.read_csv(os.path.join(RES, "sol_crosscheck.csv"))
    s = s[s.status == "complete"]
    return s.groupby(["workload", "variant_id", "shape_id", "cluster_size"], dropna=False).tail(1)


def table_sol_check(L=None):
    """ncu / ours for throughput and DRAM intensity (the 10 % criterion of DESIGN §3.5), with the decomposition
    throughput = work x time and DRAM intensity = work / bytes."""
    s = sol_check(L)
    s = s[s.ncu_point != "none"]
    t = s.assign(workload=s.workload.map(short_name)).set_index(["workload", "shape_id", "role"])[
        ["r_throughput", "r_ai_dram", "r_work", "r_time", "r_dram_bytes"]]
    t.columns = ["throughput", "DRAM intensity", "work", "time (nsys / ncu)", "DRAM bytes"]
    return t.sort_index().style.format("{:.2f}").map(
        lambda v: "color: #b00" if isinstance(v, float) and abs(v - 1) > 0.10 else "", subset=["throughput", "DRAM intensity"])


# ---------------------------------------------------------------- revision after review (R8): leaner figures
# How each workload uses the cluster / DSMEM (from the kernel code; docs/WORKLOADS.md, kernels/*.cu docstrings).
DSMEM_USE = {
    "credit": ("P = 2, 4, 8 CTAs per row", "each CTA keeps 1/P of the row in its shared memory; the row statistics (1–3 floats "
               "per pass) are all-reduced by pushing them into every peer's shared memory (st.shared::cluster) + cluster barrier",
               "re-reading the row from HBM/L2 once per pass"),
    "dak_gemm_multicast": ("2 CTAs per output tile", "TMA multicast: one L2 read of each 8 KB weight tile lands in both CTAs' shared "
                           "memory; the peer frees the buffer slot with a remote mbarrier arrive", "each CTA loading the tile from L2"),
    "deepgemm_fp8_gemm_multicast": ("2 CTAs", "TMA multicast of the A or B tile into both CTAs", "each CTA loading the tile from L2"),
    "hpcops_topk_cluster8": ("8 CTAs per row", "each CTA scans 1/8 of the row; histograms (2048 bins) and candidate counts are "
                             "combined through remote shared-memory accesses + cluster barriers", "one CTA scanning the whole row"),
    "clusterfusion_llama2_7b_decoder": ("4 CTAs per head (32 clusters)", "each CTA computes 1/4 of a head's GEMVs and attention; "
                                        "partial q|k|v (768 B) and attention outputs (256 B) are all-reduced by bulk-copy pushes, "
                                        "norm / softmax scalars by remote atomics", "18 kernels handing results over in global memory"),
    "clusterfusion_deepseek_v2_lite_mla": ("4 CTAs per head (16 clusters)", "as Llama, plus an all-gather of the latent query",
                                           "10 kernels handing results over in global memory"),
    "clusterfusion_rmsnorm": ("2 CTAs per row", "one remote atomicAdd of the partial sum of squares + 2 cluster barriers",
                              "one CTA per row"),
    "fa4_bwd_2cta_ds_exchange": ("2 CTAs (B200)", "half of dS (16 KB) bulk-copied into the peer every iteration",
                                 "each CTA computing / storing the full dS tile"),
}


def dsmem_use(w):
    for k, v in DSMEM_USE.items():
        if w.startswith(k):
            return v
    return ("", "", "")


def table_dsmem_use(L):
    """How every workload uses the cluster, and how much it sends over the SM-to-SM network (DSMEM version, overview
    shape)."""
    B = L["markers"]
    rows = []
    for w in L["M"]:
        cl, what, repl = dsmem_use(w)
        s = OVERVIEW_SHAPES.get(w, "n64k")
        g = B[(B.workload == w) & (B.shape_id == s) & (B.role == "dsmem") & (B.scope == "operator")]
        r = {"workload": short_name(w), "cluster": cl, "what goes through DSMEM / the cluster": what, "replaces": repl}
        if len(g):
            g = g.iloc[0]
            r["DSMEM KB per run"] = g.dsmem_bytes / 1e3
            r["multicast MB per run"] = g.multicast_bytes / 1e6 if g.multicast_bytes == g.multicast_bytes else np.nan
            r["HBM MB per run"] = g.hbm_bytes / 1e6
        rows.append(r)
    t = pd.DataFrame(rows).set_index("workload")
    return t.style.format({"DSMEM KB per run": "{:,.1f}", "multicast MB per run": "{:,.1f}", "HBM MB per run": "{:,.1f}"},
                          na_rep="—").set_properties(subset=["what goes through DSMEM / the cluster"],
                                                     **{"text-align": "left", "min-width": "380px"})


def fig_fabric_use(L, save=None):
    """Achieved bandwidth of the DSMEM version of every workload (overview shape): HBM, SM-to-SM (DSMEM) and TMA
    multicast, against the measured HBM and DSMEM ceilings."""
    C, B = L["C"], L["markers"]
    rows = []
    for w in L["M"]:
        s = OVERVIEW_SHAPES.get(w, "n64k")
        g = B[(B.workload == w) & (B.shape_id == s) & (B.role == "dsmem") & (B.scope == "operator")]
        if len(g):
            g = g.iloc[0]
            t = g.time_ns * 1e-9
            rows.append((short_name(w), g.hbm_bytes / t, g.dsmem_bytes / t, (g.multicast_bytes or 0) / t))
    rows.sort(key=lambda r: r[2])
    fig, ax = plt.subplots(figsize=(7.5, 5))
    y = np.arange(len(rows))
    ax.barh(y + 0.25, [r[1] / 1e12 for r in rows], 0.25, color="0.6", label="HBM")
    ax.barh(y, [max(r[2], 1) / 1e12 for r in rows], 0.25, color="C0", label="DSMEM (SM to SM)")
    ax.barh(y - 0.25, [max(r[3], 1) / 1e12 for r in rows], 0.25, color="C1", label="TMA multicast (L2 to several SMs)")
    hb, db = float(C.loc["hbm_read_bw", "value"]) / 1e12, float(C.loc["dsmem_bw", "value"]) / 1e12
    ax.axvline(hb, color="0.4", ls="--", lw=1)
    ax.axvline(db, color="C0", ls="--", lw=1)
    ax.text(hb / 1.1, len(rows) - 0.3, f"HBM ceiling {hb:.1f}", color="0.4", fontsize=7, ha="right")
    ax.text(db * 1.1, len(rows) - 0.3, f"DSMEM {db:.1f}", color="C0", fontsize=7)
    ax.set(yticks=y, yticklabels=[r[0] for r in rows], xscale="log", xlim=(1e-4, 30), ylim=(-0.6, len(rows) + 0.2),
           xlabel="achieved bandwidth (TB/s)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    ax.grid(True, axis="x", alpha=.3)
    _show(fig, save)


GROUPS = [("Row read once (CREDIT, 64K rows)", "fewer HBM bytes (row read once)"), ("TMA multicast", "TMA multicast"),
          ("Kernel fusion (decode)", "kernel fusion"), ("More SMs per row", "more SMs per row")]


def fig_roofline_groups(L, save=None):
    """Classic roofline, one panel per way of using DSMEM, one shape per workload: grey circle = non-DSMEM,
    coloured triangle = DSMEM version, arrow between them. Solid roofs: measured peak; dashed: power-limited."""
    C, B, M = L["C"], L["markers"], L["M"]
    B = B[B.scope == "operator"]
    hbm = float(C.loc["hbm_read_bw", "value"])
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.5))
    for ax, (title, mech) in zip(axs.flat, GROUPS):
        ws = [w for w in M if M[w].get("runnable") is not False and mechanism(w) == mech]
        pts = []
        for i, w in enumerate(ws):
            g = B[(B.workload == w) & (B.shape_id == OVERVIEW_SHAPES.get(w, "n64k"))].set_index("role")
            if not {"non_dsmem", "dsmem"} <= set(g.index):
                continue
            a, b = g.loc["non_dsmem"], g.loc["dsmem"]
            ax.annotate("", xy=(b.ai_hbm, b.throughput), xytext=(a.ai_hbm, a.throughput),
                        arrowprops=dict(arrowstyle="->", color=f"C{i}", lw=1.2))
            ax.scatter(a.ai_hbm, a.throughput, color="0.55", s=25, zorder=3)
            ax.scatter(b.ai_hbm, b.throughput, color=f"C{i}", marker="^", s=45, zorder=4, label=short_name(w))
            pts += [(a.ai_hbm, a.throughput), (b.ai_hbm, b.throughput)]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        lo, hi = min(xs) / 4, max(max(xs) * 8, 30)
        x = np.logspace(np.log10(lo), np.log10(hi), 200)
        for p in sorted({M[w]["precision"] for w in ws}):
            roof = float(C.loc[PREC_CEIL[p], "value"])
            ax.plot(x, np.minimum(roof, hbm * x), color="0.2", lw=1.2)
            ax.text(hi / 1.15, roof * 1.06, {"fp32_cuda": "FP32", "fp16_tensor": "FP16 tensor", "bf16_tensor": "BF16 tensor",
                                             "fp8_tensor": "FP8 tensor", "int32": "INT32"}[p], fontsize=7, ha="right")
            pl = PREC_CEIL[p].replace("_peak", "_power_limited")
            if pl in C.index:
                ax.plot(x, np.minimum(float(C.loc[pl, "value"]), hbm * x), color="0.2", lw=.8, ls="--")
        ymax = max(max(ys) * 3, min(hbm * hi, max(float(C.loc[PREC_CEIL[M[w]["precision"]], "value"]) for w in ws) * 1.5))
        ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(min(ys) / 4, ymax), title=title,
               xlabel="work per HBM byte", ylabel="throughput (work/s)")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, which="major", alpha=.3)
    fig.tight_layout()
    _show(fig, save)


def table_credit_p(L):
    """CREDIT: speedup over the one-CTA read kernel for every cluster size P measured (and the one-CTA staged control)."""
    D = L["D"]
    D = D[D.workload.str.startswith("credit") & (D.scope == "operator")]
    rows = []
    for (w, s), g in D.groupby(["workload", "shape_id"]):
        base = g[g.variant_id == "block_read"].time_ns
        if base.empty:
            continue
        base = float(base.iloc[0])
        r = {"workload": short_name(w), "row": s, "one-CTA read (µs)": base / 1e3}
        st = g[g.variant_id == "block_smem"].time_ns
        r["one-CTA staged"] = base / float(st.iloc[0]) if len(st) else np.nan
        for p in (2, 4, 8):
            h = g[(g.variant_id == "cluster_smem") & (g.cluster_size == p)].time_ns
            r[f"cluster P={p}"] = base / float(h.iloc[0]) if len(h) else np.nan
        rows.append(r)
    t = pd.DataFrame(rows)
    t["row"] = pd.Categorical(t.row, ["n4k", "n16k", "n64k"])
    t = t.sort_values(["workload", "row"]).set_index(["workload", "row"])
    sp = ["one-CTA staged", "cluster P=2", "cluster P=4", "cluster P=8"]
    return (t.style.format({"one-CTA read (µs)": "{:.0f}"} | {c: "{:.2f}×" for c in sp}, na_rep="—")
            .background_gradient(cmap="RdBu", subset=sp, vmin=0.4, vmax=1.6))


def fig_multicast(L, save=None):
    """TMA multicast GEMMs: DSMEM / non-DSMEM ratios of time, HBM bytes and L2 bytes per shape."""
    B = L["markers"]
    B = B[(B.scope == "operator") & B.workload.isin(["dak_gemm_multicast", "deepgemm_fp8_gemm_multicast"])]
    rows = []
    for (w, s), g in B.groupby(["workload", "shape_id"]):
        g = g.set_index("role")
        if {"non_dsmem", "dsmem"} <= set(g.index):
            a, b = g.loc["non_dsmem"], g.loc["dsmem"]
            rows.append((("DAK " if w.startswith("dak") else "DeepGEMM ") + s, b.time_ns / a.time_ns,
                         b.hbm_bytes / a.hbm_bytes, b.l2_bytes / a.l2_bytes))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    y = np.arange(len(rows))
    for j, (lab, mk, c) in enumerate([("time", "o", "k"), ("HBM bytes", "s", "0.6"), ("L2 bytes", "^", "C1")]):
        ax.scatter([r[1 + j] for r in rows], y, marker=mk, color=c, s=40, label=lab, zorder=3)
    ax.axvline(1, color="0.6", lw=1)
    ax.set(yticks=y, yticklabels=[r[0] for r in rows], xlabel="multicast / no multicast", xlim=(0.85, 1.3))
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
    ax.grid(True, axis="x", alpha=.3)
    _show(fig, save)


def fig_decode_time(L, save=None):
    """Decode attention block: kernel time of the unfused sequence, its first-start-to-last-end time when launched
    eagerly (includes launch gaps), the fused DSMEM kernel (clusters of 4) and the fused cluster-1 control."""
    D = L["D"]
    D = D[D.scope == "operator"]
    cases = [("clusterfusion_llama2_7b_decoder", s, f"Llama-2-7B\n{s[1:]} tokens") for s in ("s1k", "s4k", "s16k")]
    cases += [("clusterfusion_deepseek_v2_lite_mla", "s4k", "DeepSeek-V2-Lite MLA\n4k tokens")]
    bars = [("unfused: kernel time", "unfused", "time_ns", "0.6"), ("unfused: incl. launch gaps", "unfused", "wall_ns", "0.85"),
            ("fused, cluster of 4 (DSMEM)", "fused_c4", "time_ns", "C3"), ("fused, cluster of 1", "fused_c1", "time_ns", "C1")]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(cases))
    for j, (lab, v, col, c) in enumerate(bars):
        vals = []
        for w, s, _ in cases:
            h = D[(D.workload == w) & (D.shape_id == s) & (D.variant_id == v)][col]
            vals.append(float(h.iloc[0]) / 1e3 if len(h) else np.nan)
        ax.bar(x + (j - 1.5) * 0.2, vals, 0.2, color=c, label=lab)
    ax.set(xticks=x, xticklabels=[c[2] for c in cases], ylabel="µs per decode step")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=.3)
    _show(fig, save)


def fig_decode_bytes(L, save=None):
    """Where the HBM bytes of one Llama-2-7B attention-block decode step come from (model: fp16 weights, K and V
    cache, intermediates the unfused sequence writes and reads), against the measured HBM bytes."""
    D = L["D"]
    D = D[(D.workload == "clusterfusion_llama2_7b_decoder") & (D.scope == "operator")]
    h = 4096
    labels, comp = [], []
    for s, S in (("s1k", 1024), ("s4k", 4096), ("s16k", 16384)):
        w_b = 2 * 4 * h * h
        kv = 2 * 2 * S * h
        inter = 2 * (2 * h + 2 * 3 * h + 2 * 2 * h + 2 * h)  # normed x, qkv, roped q/k, attention out: written + read
        labels.append(f"{S} tokens")
        comp.append((w_b, kv, inter, [float(D[(D.shape_id == s) & (D.variant_id == v)].hbm_bytes.iloc[0])
                                      for v in ("unfused", "fused_c4")]))
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    y = np.arange(len(labels))
    ax.barh(y, [c[0] / 1e6 for c in comp], color="0.55", label="weights (QKV, O)")
    ax.barh(y, [c[1] / 1e6 for c in comp], left=[c[0] / 1e6 for c in comp], color="C0", label="K and V cache")
    ax.barh(y, [c[2] / 1e6 for c in comp], left=[(c[0] + c[1]) / 1e6 for c in comp], color="C3",
            label="intermediates (what fusion removes)")
    ax.scatter([c[3][0] / 1e6 for c in comp], y + 0.25, marker="|", s=200, color="k", label="measured, unfused")
    ax.scatter([c[3][1] / 1e6 for c in comp], y - 0.25, marker="|", s=200, color="C3", label="measured, fused")
    ax.set(yticks=y, yticklabels=labels, xlabel="HBM MB per decode step", xlim=(0, 470))
    ax.text(5, 2.45, f"intermediates: {comp[0][2] / 1e3:.0f} KB per step ({100 * comp[0][2] / (comp[0][0] + comp[0][1]):.2f} % "
            "of the bytes at 1k tokens), too small to see", fontsize=7, color="C3")
    ax.set_ylim(-0.5, 2.8)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    _show(fig, save)


def fig_reuse_model(L, save=None):
    """Model: operational intensity of the Llama-2-7B attention block (fp16) versus the number of sequences decoded
    together (batch), for its two parts. Only reuse of the bytes read from HBM raises the intensity."""
    C = L["C"]
    hbm = float(C.loc["hbm_read_bw", "value"])
    b = np.logspace(0, 3, 100)
    h, S = 4096, 4096
    P = 4 * h * h                                  # weights of QKV + O
    gemv = b * 2 * P / (2 * P)                     # each weight (2 B) used for 2 FLOP per sequence
    attn = np.ones_like(b)                         # each sequence has its own K/V: 4 S h FLOP per 4 S h bytes
    block = b * (2 * P + 4 * S * h) / (2 * P + b * 4 * S * h)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(b, gemv, color="0.4", label="QKV / O projections (weights shared by the batch)")
    ax.plot(b, attn, color="C0", label="attention over the KV cache (not shared)")
    ax.plot(b, block, color="C3", lw=2, label="whole attention block, 4k-token context")
    for name, lab in [("fp16_tc_peak", "FP16 tensor ridge"), ("fp32_peak", "FP32 ridge")]:
        r = float(C.loc[name, "value"]) / hbm
        ax.axhline(r, color="0.7", ls="--", lw=1)
        ax.text(1.1, r * 1.08, f"{lab} ({r:.0f} FLOP/B)", fontsize=7, color="0.4")
    ax.scatter([1], [1.0], color="k", zorder=4, label="measured (batch 1, both versions)")
    ax.set(xscale="log", yscale="log", xlabel="sequences decoded together (batch)", ylabel="FLOP per HBM byte",
           ylim=(0.5, 500))
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax.grid(True, which="major", alpha=.3)
    _show(fig, save)
