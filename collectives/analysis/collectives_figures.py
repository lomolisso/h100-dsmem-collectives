"""Data preparation, figures and tables for dsmem_collectives.ipynb (EXPERIMENTS.md §2, §6).

The notebook only calls the functions below. Inputs are the local result files of the performance sweep
(../results): stamps_<op>_s{1,2}.csv (one row per launch, rank and stamp), model.csv (per-rank traffic model),
ncu.csv (Nsight Compute, mean of 5 launches per configuration) and collectives__runs.csv (run registry).
prepare() streams the large stamp files once and caches compact tables in ../results/derived/ (pickles); load()
returns them. Both sweeps are pooled: 2 x 200 timed launches per configuration.
"""
import csv
import glob
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parent.parent / "results"
DER = RES / "derived"
OPS = ["allreduce", "allgather", "reducescatter", "alltoall"]
PAYLOADS = [256, 1024, 4096, 16384, 65536]
FIG4 = {"allreduce": "allreduce__ring__nccl", "allgather": "allgather__recursive_doubling__clusterfusion",
        "reducescatter": "reducescatter__recursive_halving__mpich", "alltoall": "alltoall__bruck__mpich"}
MANDATORY = ["gpu_name", "sm_count", "sm_clock_mhz", "mem_clock_mhz", "cuda_version", "driver_version", "run_id",
             "launch_idx"]


def payload_bytes(op, n, m, b=4):
    """Payload per CTA P = nccl-tests size: allreduce M·B, allgather/reducescatter/alltoall N·M·B (recomputed: allgather
    stamp rows written before this definition carry M·B in their payload_bytes column)."""
    return m * b if op == "allreduce" else n * m * b


def bus_factor(op, n):
    return 2 * (n - 1) / n if op == "allreduce" else (n - 1) / n


# ---------------------------------------------------------------------------------------------- preparation --
def _flush(cur, out):
    """Reduce the stamps of one (configuration, sweep) block."""
    key, rid, meta, reps = cur
    f, n, m, sweep = key
    op, alg, payload = meta["op"], meta["alg"], meta["payload"]
    lr_all = defaultdict(list)
    launches = []
    for rep, ranks in reps.items():
        if len(ranks) != n:
            continue
        ent = {r: v["entry"] for r, v in ranks.items()}
        ext = {r: v["exit"] for r, v in ranks.items()}
        L = max(ext.values()) - min(ent.values())
        lr = {r: ext[r] - ent[r] for r in ranks}
        bad = any(v["bad"] for v in ranks.values())
        launches.append((f, op, alg, n, m, payload, sweep, rep, L, max(lr.values()) - min(lr.values()), bad))
        if not bad:
            for r, x in lr.items():
                lr_all[r].append(x)
                if n == 16 and payload in (4096, 16384):
                    out["inj"].append((f, op, alg, n, m, payload, sweep, rep, r, x))
    out["launch"] += launches
    for r, xs in lr_all.items():
        out["rank"].append((f, op, alg, n, m, payload, sweep, r, reps_smid(reps, r), float(np.median(xs))))
    if FIG4.get(op) == f and n == 8 and payload == 4096 and sweep == "s1":
        good = sorted((L, rep) for (*_, rep, L, _, bad) in launches if not bad)
        rep = good[len(good) // 2][1]
        for r, v in reps[rep].items():
            for st, t in v["all"].items():
                out["timeline"].append((f, op, alg, n, m, payload, rep, r, v["smid"], st, t))
    out["meta"].append((f, op, alg, n, m, payload, sweep, rid, meta["gpu"], meta["sm"], meta["cuda"], meta["drv"],
                        float(np.median(meta["clk"])) if meta["clk"] else np.nan, meta["mem"]))


def reps_smid(reps, r):
    for ranks in reps.values():
        if r in ranks:
            return ranks[r]["smid"]
    return -1


def prepare(force=False):
    """Stream results/stamps_*_s{1,2}.csv once and cache the reduced tables in results/derived/."""
    files = sorted(glob.glob(str(RES / "stamps_*_s[0-9].csv")))
    DER.mkdir(exist_ok=True)
    stamp = DER / "launch.pkl"
    if not force and stamp.exists() and all(os.path.getmtime(stamp) > os.path.getmtime(p) for p in files):
        return
    out = {k: [] for k in ("launch", "rank", "inj", "timeline", "meta")}
    for path in files:
        sweep = re.search(r"_(s\d+)\.csv$", path).group(1)
        latest = {}  # pass 1: latest run_id per (file, N, M) — a re-run supersedes earlier rows
        with open(path, newline="") as fh:
            rd = csv.reader(fh)
            h = {k: i for i, k in enumerate(next(rd))}
            for row in rd:
                k = (row[h["file"]], row[h["N"]], row[h["M"]])
                if row[h["run_id"]] > latest.get(k, ""):
                    latest[k] = row[h["run_id"]]
        cur = None
        with open(path, newline="") as fh:
            rd = csv.reader(fh)
            h = {k: i for i, k in enumerate(next(rd))}
            for row in rd:
                rid = row[h["run_id"]]
                if rid != latest[(row[h["file"]], row[h["N"]], row[h["M"]])]:
                    continue
                key = (row[h["file"]], int(row[h["N"]]), int(row[h["M"]]), sweep)
                if cur is None or cur[0] != key or cur[1] != rid:
                    if cur is not None:
                        _flush(cur, out)  # rows of one launch block are contiguous (one process at a time)
                    cur = (key, rid, {"op": row[h["op"]], "alg": row[h["alg"]], "payload": payload_bytes(row[h["op"]], int(row[h["N"]]), int(row[h["M"]])),
                                      "gpu": row[h["gpu_name"]], "sm": row[h["sm_count"]], "cuda": row[h["cuda_version"]],
                                      "drv": row[h["driver_version"]], "clk": [], "mem": row[h["mem_clock_mhz"]]},
                           defaultdict(dict))
                rep, rank = int(row[h["rep"]]), int(row[h["rank"]])
                v = cur[3][rep].setdefault(rank, {"entry": 0, "exit": 0, "bad": False, "smid": int(row[h["smid"]]),
                                                  "all": {}})
                st, t = row[h["stamp"]], int(row[h["t_ns"]])
                v["all"][st] = t
                if st in ("entry", "exit"):
                    v[st] = t
                    if st == "entry" and rank == 0:
                        cur[2]["clk"].append(int(row[h["sm_clock_mhz"]]))
                v["bad"] |= row[h["clock_bad"]] == "1"
            if cur is not None:
                _flush(cur, out)
    cols = ["file", "op", "alg", "N", "M", "payload", "sweep"]
    pd.DataFrame(out["launch"], columns=cols + ["rep", "L", "skew", "clock_bad"]).to_pickle(DER / "launch.pkl.tmp")
    pd.DataFrame(out["rank"], columns=cols + ["rank", "smid", "Lr_med"]).to_pickle(DER / "rank.pkl")
    pd.DataFrame(out["inj"], columns=cols + ["rep", "rank", "Lr"]).to_pickle(DER / "inj.pkl")
    pd.DataFrame(out["timeline"], columns=["file", "op", "alg", "N", "M", "payload", "rep", "rank", "smid", "stamp",
                                           "t"]).to_pickle(DER / "timeline.pkl")
    pd.DataFrame(out["meta"], columns=cols + ["run_id", "gpu", "sm_count", "cuda", "driver", "sm_clock_med",
                                              "mem_clock"]).to_pickle(DER / "meta.pkl")
    os.replace(DER / "launch.pkl.tmp", stamp)  # written last: marks the cache complete


def load():
    prepare()
    D = {k: pd.read_pickle(DER / f"{k}.pkl") for k in ("launch", "rank", "inj", "timeline", "meta")}
    D["model"] = pd.read_csv(RES / "model.csv")
    D["model"] = D["model"][D["model"].run_id == D["model"].run_id.max()]
    D["ncu"] = pd.read_csv(RES / "ncu.csv", low_memory=False)
    D["ncu"] = (D["ncu"].sort_values("run_id")  # latest ncu run per configuration (the pass ran in several runs)
                .drop_duplicates(["file", "N", "M", "kernel", "metric"], keep="last"))
    L = D["launch"][~D["launch"].clock_bad]
    g = L.groupby(["file", "op", "alg", "N", "M", "payload"]).L
    cfg = g.agg(L_med="median", L_p10=lambda x: x.quantile(.1), L_p90=lambda x: x.quantile(.9),
                launches="size").reset_index()
    cfg["algbw"] = cfg.payload / cfg.L_med                       # bytes/ns = GB/s
    cfg["busbw"] = cfg.algbw * [bus_factor(o, n) for o, n in zip(cfg.op, cfg.N)]
    top = cfg.groupby(["file", "N"]).payload.transform("max")
    cfg["is_max"] = (cfg.payload == top) & ~cfg.payload.isin(PAYLOADS)
    D["cfg"] = cfg[cfg.payload.isin(PAYLOADS) | cfg.is_max].reset_index(drop=True)  # the sweep's payload grid
    return D


# ---------------------------------------------------------------------------------------------------- style --
def _show(fig):
    """Embed the figure as a PNG in a notebook (works with any matplotlib backend); plt.show() elsewhere."""
    try:
        from IPython import get_ipython
        from IPython.display import Image, display
    except ImportError:
        get_ipython = lambda: None  # noqa: E731
    if get_ipython() is None:
        plt.show()
        return
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90)
    plt.close(fig)
    display(Image(data=buf.getvalue()))


COLORS = list(plt.get_cmap("tab10").colors) + ["black", "goldenrod"]  # 12 distinct colours


def _style(algs):
    marks = "osD^v<>pPhX*"
    return {a: (COLORS[i % len(COLORS)], marks[i % len(marks)]) for i, a in enumerate(sorted(algs))}


def _bytes(v, _=None):
    return f"{v / 1024:g} KiB" if v >= 1024 else f"{v:g} B"


def _axes(ax, x):
    """Plain tick labels on log axes: x = 'N' (2, 4, 8, 16) or 'payload' (256 B … 64 KiB …)."""
    if x == "N":
        ax.set_xticks([2, 4, 8, 16])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    else:
        ax.set_xticks(PAYLOADS + [262144])
        ax.xaxis.set_major_formatter(FuncFormatter(_bytes))
    ax.xaxis.set_minor_formatter(NullFormatter())
    if ax.get_yscale() == "log":
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())


def _bw_optimal(D):
    """Algorithms whose modelled traffic is minimal and balanced: every rank sends exactly the lower bound
    (allreduce 2(N-1)/N·M·B, others (N-1)·M·B), evaluated at N = 8."""
    m = D["model"]
    out = set()
    for (f, n, mm), g in m.groupby(["file", "N", "M"]):
        if n != 8:
            continue
        op = f.split("__")[0]
        bound = 2 * (n - 1) / n * mm * 4 if op == "allreduce" else (n - 1) * mm * 4
        if abs(g.bytes_sent.max() - bound) < 1e-6 * bound:
            out.add(f.split("__")[1] + "|" + op)
    return out


# --------------------------------------------------------------------------------------------------- tables --
def _sweep_runs():
    runs = pd.read_csv(RES / "collectives__runs.csv")
    done = set(runs[runs.status == "complete"].run_id)
    s = runs[(runs.status == "started") & runs.cmdline.str.contains("--mode perf", regex=False) &
             runs.cmdline.str.contains("--set perf", regex=False) & runs.run_id.isin(done)]
    return sorted(s.run_id)


def table_setup(D):
    meta = D["meta"]
    keys = D["cfg"][["file", "N", "M"]]
    L = D["launch"].merge(keys, on=["file", "N", "M"])  # launches of the configurations shown
    rows = [("GPU", f"{meta.gpu.iloc[0]} (GPU 3, {meta.sm_count.iloc[0]} SMs)"),
            ("Driver / CUDA runtime", f"{meta.driver.iloc[0]} / {meta.cuda.iloc[0]}"),
            ("Toolchain", "nvcc 12.9.86, -arch=sm_90a -O3; Nsight Compute 2025.2.1"),
            ("SM clock", f"locked at 1635 MHz (median sample {meta.sm_clock_med.median():.0f} MHz); memory "
                         f"{meta.mem_clock.iloc[0]} MHz"),
            ("Launch", "one cluster of N CTAs, 256 threads per CTA, dynamic shared memory up to 227 KiB per CTA"),
            ("Configurations", f"{D['cfg'].shape[0]} (file × N × payload), float"),
            ("Timed launches", f"{len(L)} (2 sweeps × 200 per configuration, randomised order, 20 warm-ups); "
                               f"{int(L.clock_bad.sum())} discarded for a clock below the lock"),
            ("Sweep run_ids", ", ".join(_sweep_runs())),
            ("Timer", "%globaltimer, 32 ns resolution")]
    return pd.DataFrame(rows, columns=["", "value"]).set_index("")


def table_latency_4k(D):
    c = D["cfg"][D["cfg"].payload == 4096]
    t = c.pivot_table(index=["op", "alg"], columns="N", values="L_med") / 1000
    return t.round(2).rename_axis(columns="N →  median L (µs) at 4 KiB per CTA")


def table_best(D, metric="latency"):
    c = D["cfg"][D["cfg"].payload.isin(PAYLOADS)]
    key = "L_med" if metric == "latency" else "busbw"
    idx = c.groupby(["op", "N", "payload"])[key].idxmin() if metric == "latency" else \
        c.groupby(["op", "N", "payload"])[key].idxmax()
    b = c.loc[idx]
    val = (b.L_med / 1000).round(2).astype(str) + " µs" if metric == "latency" else b.busbw.round(1).astype(str) + " GB/s"
    b = b.assign(cell=b.alg + " (" + val + ")")
    t = b.pivot_table(index=["op", "N"], columns="payload", values="cell", aggfunc="first")
    t.columns = [f"{p // 1024} KiB" if p >= 1024 else f"{p} B" for p in t.columns]
    return t.reindex(OPS, level=0)


# -------------------------------------------------------------------------------------------------- figures --
def _panels(title):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    fig.suptitle(title, fontsize=12)
    return fig, dict(zip(OPS, axes.flat))


def fig_latency_vs_n(D, payload=4096):
    c = D["cfg"][D["cfg"].payload == payload]
    fig, ax = _panels(f"Collective latency vs cluster size, payload {payload // 1024} KiB per CTA "
                      "(median, band = p10–p90)")
    for op in OPS:
        d = c[c.op == op]
        sty = _style(d.alg.unique())
        for alg, g in d.groupby("alg"):
            g = g.sort_values("N")
            col, mk = sty[alg]
            ax[op].plot(g.N, g.L_med / 1000, marker=mk, color=col, label=alg, lw=1.3)
            ax[op].fill_between(g.N, g.L_p10 / 1000, g.L_p90 / 1000, color=col, alpha=.12)
        ax[op].set(title=op, xscale="log", yscale="log", xlabel="cluster size N", ylabel="L (µs)")
        _axes(ax[op], "N")
        ax[op].grid(alpha=.3, which="both")
        ax[op].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    _show(fig)


def fig_latency_vs_payload(D, n=8):
    c = D["cfg"][D["cfg"].N == n]
    fig, ax = _panels(f"Collective latency vs payload per CTA, N = {n} (open marker = largest payload that fits)")
    for op in OPS:
        d = c[c.op == op]
        sty = _style(d.alg.unique())
        for alg, g in d.groupby("alg"):
            g = g.sort_values("payload")
            col, mk = sty[alg]
            ax[op].plot(g.payload, g.L_med / 1000, marker=mk, color=col, label=alg, lw=1.3)
            mx = g[g.is_max]
            ax[op].plot(mx.payload, mx.L_med / 1000, marker=mk, color=col, mfc="white", ms=8, ls="none")
        ax[op].set(title=op, xscale="log", yscale="log", xlabel="payload per CTA", ylabel="L (µs)")
        _axes(ax[op], "payload")
        ax[op].grid(alpha=.3, which="both")
        ax[op].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    _show(fig)


def fig_busbw_vs_payload(D, n=8):
    c = D["cfg"][D["cfg"].N == n]
    opt = _bw_optimal(D)
    fig, ax = _panels(f"Bus bandwidth vs payload per CTA, N = {n} (solid = traffic-optimal algorithm, "
                      "dashed = moves more than the minimum or unevenly)")
    for op in OPS:
        d = c[c.op == op]
        sty = _style(d.alg.unique())
        for alg, g in d.groupby("alg"):
            g = g.sort_values("payload")
            col, mk = sty[alg]
            ax[op].plot(g.payload, g.busbw, marker=mk, color=col, label=alg, lw=1.6 if f"{alg}|{op}" in opt else 1,
                        ls="-" if f"{alg}|{op}" in opt else "--")
        ax[op].set(title=op, xscale="log", xlabel="payload per CTA", ylabel="bus bandwidth (GB/s)")
        _axes(ax[op], "payload")
        ax[op].grid(alpha=.3, which="both")
        ax[op].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    _show(fig)


def fig_rank_timeline(D):
    t = D["timeline"]
    fig, ax = _panels("One median launch per op, N = 8, 4 KiB per CTA: entry (|), end of each round (•), exit (▶) "
                      "per rank")
    for op in OPS:
        d = t[t.op == op]
        if d.empty:
            continue
        t0 = d[d.stamp == "entry"].t.min()
        for r, g in d.groupby("rank"):
            s = dict(zip(g.stamp, g.t - t0))
            rounds = sorted((int(k[5:]), v) for k, v in s.items() if k.startswith("round"))
            xs = [s["entry"]] + [v for _, v in rounds] + [s["exit"]]
            ax[op].plot(xs, [r] * len(xs), color="0.6", lw=1, zorder=1)
            ax[op].scatter([s["entry"]], [r], marker="|", s=120, color="k", zorder=2)
            ax[op].scatter([v for _, v in rounds], [r] * len(rounds), s=14, color="tab:blue", zorder=2)
            ax[op].scatter([s["exit"]], [r], marker=">", s=40, color="tab:red", zorder=2)
            ax[op].text(s["exit"], r, f"  SM {g.smid.iloc[0]}", va="center", fontsize=7)
        ax[op].set(title=f"{op}: {d.alg.iloc[0]}", xlabel="ns since the first rank's entry", ylabel="rank",
                   yticks=range(8))
        ax[op].invert_yaxis()
        ax[op].grid(alpha=.3, axis="x")
    fig.tight_layout()
    _show(fig)


def fig_injection_bw(D, n=16, payload=16384):
    inj = D["inj"]
    m = D["model"][["file", "N", "M", "rank", "bytes_sent"]]
    d = inj[inj.N == n].merge(m, on=["file", "N", "M", "rank"])
    d["bw"] = d.bytes_sent / d.Lr                                 # bytes/ns = GB/s
    fig, ax = _panels(f"Per-SM injection bandwidth = bytes a rank sends / its own latency L_r, N = {n} "
                      f"(all ranks × launches)")
    for op in OPS:
        p = payload if (d[d.op == op].payload == payload).any() else d[d.op == op].payload.max()
        dd = d[(d.op == op) & (d.payload == p) & (d.bytes_sent > 0)]
        algs = sorted(dd.alg.unique(), key=lambda a: dd[dd.alg == a].bw.median())
        ax[op].boxplot([dd[dd.alg == a].bw for a in algs], showfliers=False, vert=False)
        ax[op].set_yticks(range(1, len(algs) + 1))
        ax[op].set_yticklabels(algs, fontsize=8)
        ax[op].set(title=f"{op}, {_bytes(p)} per CTA" + ("" if p == payload else f" ({_bytes(payload)} does not fit)"),
                   xlabel="GB/s per rank")
        ax[op].grid(alpha=.3, axis="x")
    fig.tight_layout()
    _show(fig)


def traffic_table(D):
    """Measured kernel-wide DSMEM bytes (ncu, minus the same-N baseline) and the model, per configuration."""
    nc = D["ncu"]
    met = "l1tex__m_l1tex2xbar_write_bytes_mem_dshared.sum"
    v = nc[nc.metric == met].copy()
    v["value"] = pd.to_numeric(v.value)
    base = v[v.kernel == "baseline_kernel"].groupby("N").value.max()
    meas = v[v.kernel == "collective_kernel"][["file", "op", "alg", "N", "M", "value"]]
    meas["measured"] = meas.value - meas.N.map(base).fillna(0)
    tot = D["model"].groupby(["file", "N", "M"])[["bytes_sent", "bytes_self"]].sum().reset_index()
    j = meas.merge(tot, on=["file", "N", "M"])
    j["ratio"] = j.measured / j.bytes_sent
    return j


def fig_traffic_vs_model(D):
    j = traffic_table(D)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.8))
    sty = {op: c for op, c in zip(OPS, ["tab:blue", "tab:orange", "tab:green", "tab:red"])}
    for op, g in j.groupby("op"):
        ax[0].scatter(g.bytes_sent, g.measured, s=12, color=sty[op], label=op, alpha=.7)
    lim = [j[["bytes_sent", "measured"]].min().min() * 0.7, j[["bytes_sent", "measured"]].max().max() * 1.4]
    ax[0].plot(lim, lim, "k--", lw=.8)
    ax[0].set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="model: kernel-wide DSMEM bytes",
              ylabel="measured (Nsight Compute) − baseline", title="All configurations")
    ax[0].legend()
    ax[0].grid(alpha=.3, which="both")
    j["label"] = j.op + ": " + j.alg
    r = j.groupby("label").ratio.agg(["min", "max"]).sort_values("max")
    ax[1].hlines(range(len(r)), r["min"], r["max"], color="0.4")
    ax[1].scatter(r["min"], range(len(r)), s=12, color="k")
    ax[1].scatter(r["max"], range(len(r)), s=12, color="k")
    ax[1].axvline(1, color="k", ls="--", lw=.8)
    ax[1].axvspan(0.9, 1.1, color="g", alpha=.08)
    ax[1].set_yticks(range(len(r)))
    ax[1].set_yticklabels(r.index, fontsize=8)
    ax[1].set(xlabel="measured / model (range over N and payload)", title="Ratio per algorithm (shaded ±10 %)")
    ax[1].grid(alpha=.3, axis="x")
    fig.tight_layout()
    _show(fig)


# ---------------------------------------------------------------------------------------- summary CSV files --
def write_summaries(D):
    """results/summary_latency.csv and results/summary_bandwidth.csv (committed with the notebook)."""
    meta = D["meta"]
    mrow = meta.iloc[0]
    rid = "+".join(_sweep_runs())  # the driver run_ids of the sweeps (per-configuration ids: results/derived)
    ranks = D["rank"].groupby(["file", "N", "M"]).Lr_med.max().rename("Lr_med_max").reset_index()
    inj = D["model"].groupby(["file", "N", "M"]).bytes_sent.max().rename("sent_max").reset_index()
    c = D["cfg"].merge(ranks, on=["file", "N", "M"], how="left").merge(inj, on=["file", "N", "M"], how="left")
    pre = {"gpu_name": mrow.gpu, "sm_count": mrow.sm_count, "sm_clock_mhz": int(round(meta.sm_clock_med.median())),
           "mem_clock_mhz": mrow.mem_clock, "cuda_version": mrow.cuda, "driver_version": mrow.driver,
           "run_id": rid, "launch_idx": -1}
    lat = c[["file", "op", "alg", "N", "M", "payload", "launches", "L_med", "L_p10", "L_p90", "Lr_med_max"]].rename(
        columns={"payload": "payload_bytes", "L_med": "L_med_ns", "L_p10": "L_p10_ns", "L_p90": "L_p90_ns",
                 "Lr_med_max": "Lr_med_max_ns"})
    L = D["launch"][~D["launch"].clock_bad].merge(c[["file", "N", "M", "payload", "op"]], on=["file", "N", "M", "payload", "op"])
    L["busbw"] = L.payload / L.L * [bus_factor(o, n) for o, n in zip(L.op, L.N)]
    q = L.groupby(["file", "N", "M"]).busbw.quantile([.1, .9]).unstack().rename(columns={.1: "busbw_p10", .9: "busbw_p90"})
    bw = c.merge(q.reset_index(), on=["file", "N", "M"])
    bw["inj_max_GBps"] = bw.sent_max / bw.L_med
    bw = bw[["file", "op", "alg", "N", "M", "payload", "launches", "algbw", "busbw", "busbw_p10", "busbw_p90",
             "inj_max_GBps"]].rename(columns={"payload": "payload_bytes", "algbw": "algbw_med_GBps",
                                              "busbw": "busbw_med_GBps", "busbw_p10": "busbw_p10_GBps",
                                              "busbw_p90": "busbw_p90_GBps"})
    for df, name in ((lat, "summary_latency.csv"), (bw, "summary_bandwidth.csv")):
        out = pd.concat([pd.DataFrame([pre] * len(df), index=df.index), df], axis=1)
        out.sort_values(["op", "alg", "N", "payload_bytes"]).to_csv(RES / name, index=False, float_format="%.6g")
    return lat.shape[0]
