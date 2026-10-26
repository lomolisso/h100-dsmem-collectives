"""Data preparation, network constants, cost model, figures and tables for dsmem_collectives.ipynb.

The notebook only calls the functions below. Inputs are the local result files of the performance sweep
(../results): stamps_<op>_s{1,2}.csv (one row per launch, rank and stamp), model.csv (per-rank traffic model),
ncu.csv (Nsight Compute, mean of 5 launches per configuration) and collectives__runs.csv (run registry).
prepare() streams the large stamp files once and caches compact tables in ../results/derived/ (pickles); load()
returns them. Both sweeps are pooled: 2 x 200 timed launches per configuration. fabric() reads the SM-to-SM network
measurements of h100-microbench (07, 09, 11, 12; local summary CSVs).
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


# ------------------------------------------------------------------------------------- the SM-to-SM network --
MB = Path(os.path.expanduser("~/work/fa2026/iacoma/h100-microbench/experiments"))
CLK = 1.635  # SM cycles per ns at the 1635 MHz lock


def _latest(rel):
    d = pd.read_csv(MB / rel, low_memory=False)
    return d[d.run_id == d.run_id.max()]


def fabric():
    """Network constants measured by h100-microbench 07, 09, 11, 12 (local summary CSVs, latest run of each)."""
    s9 = _latest("09__dsmem_latency_vs_cluster_size/results/09__dsmem_latency__summary.csv")
    s11 = _latest("11__dsmem_bandwidth_vs_cluster_size/results/11__dsmem_bw__summary.csv")
    s12 = _latest("12__dsmem_atomics_and_async_stores/results/12__atomics__summary.csv")
    s7 = _latest("07__synchronization_primitives/results/07__sync__summary_v2.csv")

    def v9(level, cfg="cs 8"):
        return float(s9[(s9.level == level) & (s9.config == cfg)]["median"].iloc[0])

    def v12(op, kind):
        r = s12[(s12.target == "dsmem") & (s12.cs.astype(str) == "2") & (s12.op == op) & (s12.kind == kind)]
        return float(r.latency_cycles.iloc[0])

    def v7(cfg):
        return float(s7[(s7.level == "k2/cluster_sync/single") & (s7.config == cfg)]["median"].iloc[0])

    lat = [  # (primitive, ns, source)
        ("local shared-memory load", 23.0 / CLK, "09 k3: 23 cycles"),
        ("remote shared-memory load (DSMEM)", v9("k1/ld_cluster/stride1/Default/remote") / CLK, "09 k1, CS 8"),
        ("hand-off: bulk copy / st.async + mbarrier (one way)", v9("k2/st_async_mbarrier/r01/one_way_ns"), "09 k2"),
        ("hand-off: plain store + flag poll (one way)", v9("k2/st_plain/r01/one_way_ns"), "09 k2"),
        ("cluster barrier, idle (CS 8, 128-512 threads)", v7("cs 8, threads 128") / CLK, "07 k2: 360-511 cycles"),
        ("cluster barrier after stores (release fence ≈ 520 cycles)", 545 / CLK, "07: release behind a store 520-550"),
        ("remote atomic add.u32", v12("add.u32", "atom") / CLK, "12, CS 2"),
        ("remote atomic add.f32 (compare-and-swap loop)", v12("add.f32", "atom") / CLK, "12, CS 2"),
        ("issue of one bulk-copy command (≤ 1 KiB)", 65 / CLK, "11: issue-bound at ≈ 65 cycles"),
    ]
    ring = s11[s11.level.str.match(r"ring_(ld|st|st_async|tma_put)_cs\d+_single$")].copy()
    ring["op"] = ring.level.str.extract(r"ring_(.+)_cs")[0]
    ring["N"] = ring.level.str.extract(r"_cs(\d+)_")[0].astype(int)
    ring["Bpc"] = ring.value.astype(float)
    return {"lat": pd.DataFrame(lat, columns=["primitive", "ns", "source"]),
            "ring": ring[["op", "N", "Bpc"]].sort_values(["op", "N"]),
            "send_limit_Bpc": 21.33, "recv_limit_Bpc": 32.0}


def table_fabric(F):
    t = F["lat"].copy()
    t["cycles"] = (t.ns * CLK).round(0).astype(int)
    t["ns"] = t.ns.round(0).astype(int)
    return t[["primitive", "cycles", "ns", "source"]].set_index("primitive")


# ---------------------------------------------------------------------------------- synchronisation structure --
def _l2(n):
    return int(np.log2(n))


SYNC = {  # file -> N -> (cluster barriers inside the timed region, mbarrier hand-off rounds); README / docstrings
    "allreduce__two_shot_push__trtllm": lambda n: (0, 2),
    "allreduce__2d_mesh__generic": lambda n: (0, 3 if n >= 4 else 2),
    "allreduce__one_shot_pull__trtllm": lambda n: (2, 0),
    "allreduce__direct_replicated_push__credit": lambda n: (2, 0),
    "allreduce__pull_reduce_scatter_push_allgather__hpcops": lambda n: (2, 0),
    "allreduce__atomic_all_push__generic": lambda n: (2, 0),
    "allreduce__ring__nccl": lambda n: (2 * (n - 1), 2 * (n - 1)),
    "allreduce__rabenseifner__mpich": lambda n: (2 * _l2(n), 2 * _l2(n)),
    "allreduce__recursive_doubling__clusterfusion": lambda n: (_l2(n), _l2(n)),
    "allreduce__reduce_bcast_binomial__mpich": lambda n: (2 * _l2(n), 2 * _l2(n)),
    "allreduce__shift_accumulate_bulk__clusterfusion": lambda n: (n - 2, n - 2),
    "allreduce__shift_accumulate_atomic__clusterfusion": lambda n: (2 * max(1, n - 2), 0),
    "allgather__direct_replicated_push__credit": lambda n: (1, 0),
    "allgather__direct_pull__generic": lambda n: (2, 0),
    "allgather__recursive_doubling__clusterfusion": lambda n: (_l2(n), _l2(n)),
    "allgather__bruck__mpich": lambda n: (int(np.ceil(np.log2(n))), int(np.ceil(np.log2(n)))),
    "allgather__ring__nccl": lambda n: (n - 1, n - 1),
    "allgather__direct_shift__clusterfusion": lambda n: (2 * (n - 1), n - 1),
    "allgather__gather_bcast_binomial__mpich": lambda n: (2 * _l2(n), 2 * _l2(n)),
    "reducescatter__direct_push__generic": lambda n: (0, 1),
    "reducescatter__direct_pull__generic": lambda n: (2, 0),
    "reducescatter__recursive_halving__mpich": lambda n: (_l2(n), _l2(n)),
    "reducescatter__ring__nccl": lambda n: (n - 1, n - 1),
    "alltoall__direct_push__generic": lambda n: (0, 1),
    "alltoall__pairwise_exchange__fa4": lambda n: (0, 1),
    "alltoall__direct_pull__generic": lambda n: (2, 0),
    "alltoall__pairwise_xor__mpich": lambda n: (n - 1, n - 1),
    "alltoall__shift__generic": lambda n: (n - 1, n - 1),
    "alltoall__bruck__mpich": lambda n: (2 * _l2(n), _l2(n)),
}
STYLES = ["mbarrier only", "one exchange + cluster barriers", "cluster barrier every round"]
STYLE_COLOR = dict(zip(STYLES, ["tab:blue", "tab:orange", "tab:red"]))


def sync_style(f):
    nb, nm = SYNC[f](8)
    if nb == 0:
        return STYLES[0]
    return STYLES[1] if nm == 0 and nb <= 2 and "shift_accumulate" not in f else STYLES[2]


def sync_fit(D, payload=256):
    """Least-squares fit L = t0 + n_barrier·t_barrier + n_handoff·t_handoff over all non-atomic configurations at
    `payload` (atomics excluded: their cost is the compare-and-swap loop, not synchronisation)."""
    c = D["cfg"][(D["cfg"].payload == payload) & ~D["cfg"].alg.str.contains("atomic")].copy()
    c[["nb", "nm"]] = [SYNC[f](n) for f, n in zip(c.file, c.N)]
    X = np.c_[np.ones(len(c)), c.nb, c.nm]
    coef, *_ = np.linalg.lstsq(X, c.L_med.values, rcond=None)
    c["pred"] = X @ coef
    c["style"] = [sync_style(f) for f in c.file]
    r2 = 1 - ((c.L_med - c.pred) ** 2).sum() / ((c.L_med - c.L_med.mean()) ** 2).sum()
    return {"t0": coef[0], "t_barrier": coef[1], "t_handoff": coef[2], "r2": r2, "n": len(c), "data": c,
            "median_abs_err": float(np.median(np.abs(c.pred - c.L_med) / c.L_med))}


def table_sync_fit(D, F):
    s = sync_fit(D)
    lat = F["lat"].set_index("primitive").ns
    rows = [("fixed cost per collective", s["t0"], "—"),
            ("one cluster barrier", s["t_barrier"],
             f"{lat['cluster barrier, idle (CS 8, 128-512 threads)']:.0f} idle, "
             f"{lat['cluster barrier after stores (release fence ≈ 520 cycles)']:.0f} after stores"),
            ("one mbarrier hand-off", s["t_handoff"],
             f"{lat['hand-off: bulk copy / st.async + mbarrier (one way)']:.0f}")]
    t = pd.DataFrame(rows, columns=["cost", "fitted on the collectives (ns)", "measured on the network (ns)"])
    t["fitted on the collectives (ns)"] = t["fitted on the collectives (ns)"].round(0).astype(int)
    return t.set_index("cost")


# ---------------------------------------------------------------------------------------------- plot helpers --
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False})


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
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    display(Image(data=buf.getvalue()))


def _bytes(v, _=None):
    return f"{v / 1024:g} KiB" if v >= 1024 else f"{v:g} B"


def _logx(ax, kind):
    ax.set_xscale("log")
    if kind == "N":
        ax.set_xticks([2, 4, 8, 16])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    else:
        ax.set_xticks([256, 1024, 4096, 16384, 65536, 262144])
        ax.xaxis.set_major_formatter(FuncFormatter(_bytes))
    ax.xaxis.set_minor_formatter(NullFormatter())


def _logy(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


SHORT = {"pull_reduce_scatter_push_allgather": "pull-RS/push-AG", "direct_replicated_push": "replicated push",
         "shift_accumulate_bulk": "shift-accumulate", "shift_accumulate_atomic": "shift-acc. atomic",
         "reduce_bcast_binomial": "reduce+bcast tree", "gather_bcast_binomial": "gather+bcast tree",
         "recursive_doubling": "recursive doubling", "recursive_halving": "recursive halving",
         "two_shot_push": "two-shot push", "one_shot_pull": "one-shot pull", "atomic_all_push": "atomic push",
         "pairwise_exchange": "pairwise exchange", "pairwise_xor": "pairwise XOR", "direct_push": "direct push",
         "direct_pull": "direct pull", "direct_shift": "direct shift", "2d_mesh": "2D mesh"}


def _short(a):
    return SHORT.get(a, a)


def _best_per_style(d, key="L_med", best=min, axis="N"):
    """Name of the best algorithm of each synchronisation style in d (by the median of `key` over d), among the
    algorithms present at every value of `axis` in d (a fair comparison)."""
    full = d.groupby("alg")[axis].nunique()
    d = d[d.alg.isin(full[full == full.max()].index)]
    out = {}
    for st in STYLES:
        g = d[[sync_style(f) == st for f in d.file]]
        if len(g):
            s = g.groupby("alg")[key].median()
            out[st] = s.idxmin() if best is min else s.idxmax()
    return out


def _label_ends(ax, items, log):
    """Write each line's name at its right end, nudging labels apart vertically so they never overlap."""
    items = sorted(items, key=lambda t: t[1])
    lo, hi = ax.get_ylim()
    gap = (np.log10(hi) - np.log10(lo)) * 0.07 if log else (hi - lo) * 0.06
    placed = []
    for x, y, text, col in items:
        yy = np.log10(y) if log else y
        if placed and yy < placed[-1] + gap:
            yy = placed[-1] + gap
        placed.append(yy)
        ax.annotate(text, (x, 10 ** yy if log else yy), xytext=(5, 0), textcoords="offset points", fontsize=8,
                    color=col, va="center")


# ------------------------------------------------------------------------------------------------- figures --
def fig_fabric_costs(F):
    t = F["lat"].iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.barh(t.primitive, t.ns, color="0.55", height=0.6)
    for y, v in enumerate(t.ns):
        ax.text(v + 5, y, f"{v:.0f}", va="center", fontsize=9)
    ax.set_xlabel("time (ns, SM clock 1635 MHz)")
    ax.grid(axis="y", visible=False)
    _show(fig)


def fig_fabric_bandwidth(F):
    r = F["ring"]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    names = {"tma_put": "bulk copy (TMA)", "st_async": "st.async", "st": "remote store", "ld": "remote load"}
    for op, col in zip(["tma_put", "st_async", "st", "ld"], ["tab:blue", "tab:green", "tab:purple", "tab:gray"]):
        g = r[r.op == op]
        ax.plot(g.N, g.Bpc * CLK, marker="o", color=col, label=names[op])
    ax.axhline(F["send_limit_Bpc"] * CLK, color="k", lw=1, ls="--")
    ax.text(2, F["send_limit_Bpc"] * CLK + 0.6, "per-SM send limit 21.33 B/cycle", fontsize=9)
    ax.axhline(F["recv_limit_Bpc"] * CLK, color="k", lw=1, ls=":")
    ax.text(2, F["recv_limit_Bpc"] * CLK + 0.6, "per-SM receive limit ≈ 32 B/cycle", fontsize=9)
    _logx(ax, "N")
    ax.set(xlabel="cluster size N (all N SMs send to their ring neighbour at once)", ylabel="GB/s per SM",
           ylim=(0, 58))
    ax.legend(loc="lower left", ncol=2)
    _show(fig)


def fig_sync_fit(D):
    s = sync_fit(D)
    c = s["data"]
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for st in STYLES:
        g = c[c["style"] == st]
        ax.scatter(g.pred / 1000, g.L_med / 1000, s=22, color=STYLE_COLOR[st], label=st, alpha=0.8)
    lim = [0.12, 25]
    ax.plot(lim, lim, color="k", lw=0.8)
    _logy(ax)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set(xlim=lim, ylim=lim, xlabel="predicted: t0 + barriers·t_barrier + hand-offs·t_handoff (µs)",
           ylabel="measured median latency (µs)")
    ax.set_title(f"256 B per CTA, all N; R² = {s['r2']:.2f}", fontsize=10)
    ax.legend(loc="upper left")
    _show(fig)


def fig_latency_vs_n(D, payload=4096):
    c = D["cfg"][D["cfg"].payload == payload]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
    for ax, op in zip(axes, OPS):
        d = c[c.op == op]
        best = _best_per_style(d)
        labels = []
        for alg, g in d.groupby("alg"):
            g = g.sort_values("N")
            if alg in best.values():
                st = sync_style(g.file.iloc[0])
                ax.plot(g.N, g.L_med / 1000, marker="o", color=STYLE_COLOR[st], lw=2, zorder=3)
                labels.append((g.N.iloc[-1], g.L_med.iloc[-1] / 1000, _short(alg), STYLE_COLOR[st]))
            else:
                ax.plot(g.N, g.L_med / 1000, color="0.75", lw=1, zorder=1)
        _logx(ax, "N")
        _logy(ax)
        _label_ends(ax, labels, log=True)
        ax.set(title=op, xlabel="cluster size N", xlim=(1.8, 60))
    axes[0].set_ylabel(f"median latency (µs), {_bytes(payload)} per CTA")
    handles = [plt.Line2D([], [], color=STYLE_COLOR[s], lw=2) for s in STYLES] + [plt.Line2D([], [], color="0.75")]
    fig.legend(handles, [f"best with {s}" for s in STYLES] + ["other algorithms"], loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.08))
    _show(fig)


def fig_busbw(D, n=8):
    c = D["cfg"][D["cfg"].N == n]
    lim = 21.33 * CLK
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
    for ax, op in zip(axes, OPS):
        d = c[c.op == op]
        grid = d[d.payload.isin(PAYLOADS)]
        full = grid.groupby("alg").payload.nunique()  # algorithms present at every payload; best = highest at 64 KiB
        best = _best_per_style(grid[grid.alg.isin(full[full == full.max()].index) & (grid.payload == 65536)],
                               key="busbw", best=max, axis="payload")
        labels = []
        for alg, g in d.groupby("alg"):
            g = g.sort_values("payload")
            if alg in best.values():
                st = sync_style(g.file.iloc[0])
                ax.plot(g.payload, g.busbw, marker="o", color=STYLE_COLOR[st], lw=2, zorder=3)
                labels.append((g.payload.iloc[-1], g.busbw.iloc[-1], _short(alg), STYLE_COLOR[st]))
            else:
                ax.plot(g.payload, g.busbw, color="0.75", lw=1, zorder=1)
        ax.axhline(lim, color="k", ls="--", lw=1)
        _logx(ax, "payload")
        ax.set_xticks([256, 4096, 65536])
        ax.set(title=op, xlabel="payload per CTA", xlim=(200, 3e6), ylim=(0, 38))
        _label_ends(ax, labels, log=False)
    axes[0].text(260, lim + 0.8, "per-SM send limit", fontsize=8)
    axes[0].set_ylabel(f"bus bandwidth (GB/s), N = {n}")
    handles = [plt.Line2D([], [], color=STYLE_COLOR[s], lw=2) for s in STYLES] + [plt.Line2D([], [], color="0.75")]
    fig.legend(handles, [f"best with {s}" for s in STYLES] + ["other algorithms"], loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.08))
    _show(fig)


def fig_allreduce_ring(D, n=8):
    c = D["cfg"][(D["cfg"].op == "allreduce") & (D["cfg"].N == n)]
    pick = [("two_shot_push", "tab:blue"), ("one_shot_pull", "tab:orange"), ("rabenseifner", "tab:red"),
            ("ring", "tab:brown")]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for alg, col in pick:
        g = c[c.alg == alg].sort_values("payload")
        ax.plot(g.payload, g.L_med / 1000, marker="o", color=col, label=alg)
        ax.plot(g.payload.iloc[-1], g.L_med.iloc[-1] / 1000, marker="o", mfc="white", ms=9, color=col)
    _logx(ax, "payload")
    _logy(ax)
    ax.set(xlabel="payload per CTA (open marker: largest that fits in shared memory)",
           ylabel=f"median latency (µs), allreduce, N = {n}")
    ax.legend()
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


def fig_traffic_ratio(D):
    j = traffic_table(D)
    j["label"] = j.op + ": " + j.alg
    r = j.groupby("label").ratio.agg(["median", "min", "max"]).sort_values("median")
    fig, ax = plt.subplots(figsize=(7, 7))
    y = np.arange(len(r))
    col = ["tab:red" if m > 1.1 else "0.5" for m in r["median"]]
    ax.hlines(y, r["min"], r["max"], color=col, lw=1.5)
    ax.scatter(r["median"], y, color=col, s=25, zorder=3)
    ax.axvline(1, color="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(r.index, fontsize=8)
    ax.set(xlabel="measured / modelled DSMEM bytes (dot: median, line: range over N and payload)")
    ax.grid(axis="y", visible=False)
    _show(fig)


# --------------------------------------------------------------------------------------------------- tables --
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
            ("Configurations", f"{D['cfg'].shape[0]} (algorithm × N × payload), float"),
            ("Timed launches", f"{len(L)} (2 sweeps × 200 per configuration, randomised order, 20 warm-ups); "
                               f"{int(L.clock_bad.sum())} discarded for a clock below the lock"),
            ("Timer", "%globaltimer, 32 ns resolution")]
    return pd.DataFrame(rows, columns=["", "value"]).set_index("")


def table_best(D, metric="latency"):
    c = D["cfg"][D["cfg"].payload.isin(PAYLOADS)]
    key = "L_med" if metric == "latency" else "busbw"
    idx = c.groupby(["op", "N", "payload"])[key].idxmin() if metric == "latency" else \
        c.groupby(["op", "N", "payload"])[key].idxmax()
    b = c.loc[idx]
    val = (b.L_med / 1000).round(2).astype(str) + " µs" if metric == "latency" else b.busbw.round(1).astype(str) + " GB/s"
    b = b.assign(cell=b.alg + " (" + val + ")")
    t = b.pivot_table(index=["op", "N"], columns="payload", values="cell", aggfunc="first")
    t.columns = [_bytes(p) for p in t.columns]
    return t.reindex(OPS, level=0)


def numbers(D, F):
    """Every number quoted in the notebook text, computed here."""
    c = D["cfg"]
    s = sync_fit(D)
    out = {"fit": s}
    smax = D["model"].groupby(["file", "N", "M"]).bytes_sent.max().rename("smax").reset_index()
    cc = c.merge(smax, on=["file", "N", "M"])
    lim = F["send_limit_Bpc"] * CLK
    out["send_limit_GBs"] = lim
    out["peak"] = {n: {op: cc[(cc.op == op) & (cc.N == n)].sort_values("busbw").iloc[-1][["alg", "payload", "busbw"]]
                       .to_dict() for op in OPS} for n in (8, 16)}
    a = c[(c.op == "allreduce") & (c.N == 8)].set_index(["alg", "payload"]).L_med
    out["ar8"] = a
    j = traffic_table(D)
    out["traffic_within10"] = int(((j.ratio - 1).abs() <= 0.1).sum())
    out["traffic_n"] = len(j)
    out["atomic"] = c[c.alg == "atomic_all_push"].pivot_table(index="N", columns="payload", values="L_med")
    return out


def _sweep_runs():
    runs = pd.read_csv(RES / "collectives__runs.csv")
    done = set(runs[runs.status == "complete"].run_id)
    s = runs[(runs.status == "started") & runs.cmdline.str.contains("--mode perf", regex=False) &
             runs.cmdline.str.contains("--set perf", regex=False) & runs.run_id.isin(done)]
    return sorted(s.run_id)


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
