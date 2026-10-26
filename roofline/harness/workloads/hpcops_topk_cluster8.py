"""Wrapper of manifest hpcops_topk_cluster8: exact top-2048 indices of m rows of n fp32 logits (Tencent hpc-ops,
src/topk/topk_filtered.cu), through the public op hpc.topk(logits, ke, out, num_valid_rows, top_k, counters,
workspace) = torch.ops.hpc.topk_filtered, as tests/test_topk.py test_cluster_uniform_rows calls it (randn rows,
ke = n for every row, num_valid_rows = m, counters zeroed and workspace filled with 0xA5 once).

Variants (--variant; the worktree on PYTHONPATH decides which build is imported, the manifest's `expect` checks it):
  cluster8   pin build, row capacity m      -> topk_filtered_cluster8_exact_kernel (8-CTA cluster per row)  (dsmem)
  nocluster  use_cluster=false build, cap m -> row-local streaming kernel (n <= 131072) or the 8-way KV split
             (n = 244650): hpc-ops' own dispatch without the cluster route (= SM major < 9)              (non_dsmem)
  pad17      pin build, row capacity 17 (ke.numel() = 17, logits/out with 17 rows, num_valid_rows = m): the
             dispatcher sees 17 rows, so use_cluster is false; 4-way KV split at n = 244650              (reference)
Dispatch depends only on (ke.numel(), n) (choose_topk2048_plan); `expected_kernel` replicates it for the shapes of
the manifest and --validate checks it against the kernel the profiler sees (PLAN line).

setup allocates every tensor once. step = one hpc.topk call with the preallocated counters and workspace: without
them topk_filtered_entry would torch::zeros the counters and torch::empty the workspace per call (an extra fill
kernel). The kernels leave the counters zero (checked in --validate after repeated calls), the workspace needs no
initialisation, and every call rewrites all k output slots of the m live rows, so no per-step reset is needed. No
synchronisation in step (the op does none). Validation (`check`): 3 calls (repeated launches reuse the counters),
then tests/test_topk.py's `_validate` criterion per live row on the CPU against torch.topk: k unique in-range
indices, index set equal to torch.topk's or, under ties, equal sorted selected values; padding rows untouched
(-1); counters zero; plus the kernel audit (kernel_audit.py)."""
import os
import re

import torch

import wrapper
from kernel_audit import audit

MANIFEST = "hpcops_topk_cluster8"
VARIANTS = ("cluster8", "nocluster", "pad17")
PAD_ROWS = 17      # first row capacity that leaves the cluster route (kClusterMaxRows = 16)
SM_COUNT = 132     # H100 NVL; get_sm_count() in the dispatcher
REPEATS = 3        # tests/test_topk.py test_cluster_dispatch_boundary_repeated_launches


def expected_kernel(cap, n, k, use_cluster_enabled):
    """(kernel family, template arguments) that choose_topk2048_plan / topk_filtered_async launch
    (topk_filtered.cu:2860-2956) for a contiguous [cap, n] fp32 input, k = 2048; bool arguments as 0/1, None = not
    checked (routes the manifest's shapes never reach)."""
    assert k == 2048
    vec = 4 if n % 4 == 0 else 2 if n % 2 == 0 else 1          # vector_width(row_stride), row_stride = n
    grid = 3 * SM_COUNT                                         # grid_num_blocks(): BLOCKS_PER_SM = 3
    if use_cluster_enabled and cap > 0 and ((cap <= 8 and n >= 32768) or (cap <= 16 and n >= 65536)):
        return "topk_filtered_cluster8_exact_kernel", (vec, 2048, 8, 1024)
    row_local_max = 65536 if cap > 2 * SM_COUNT else 262144 if cap >= 64 else 196608 if cap >= 32 else 131072
    if n <= row_local_max:
        if n <= 16384 and vec == 4:
            return "topk_filtered_row_local_register_kernel", None
        if cap > grid:
            return "topk_filtered_row_local_streaming_kernel", None
        return "topk_filtered_row_local_streaming_kernel", (vec, 2048, 0, 1 if cap <= SM_COUNT else 2, 0)
    if n >= 128 * 1024 + 1 and cap <= 192:                     # choose_splits
        splits = 8 if cap <= max(16, SM_COUNT // 8) else 4 if cap <= grid // 4 else 2
        return "topk_filtered_split_kernel", (vec, 2048, splits, 64, 48)
    return "topk_filtered_kernel", None                         # persistent routes: not used by the manifest


def template_args(name, family):
    """Template arguments of `family<...>` in a demangled name, as ints. Demanglers differ (cu++filt / nsys:
    `<(unsigned int)4, (int)2048, (bool)0, ...>`, GNU style: `<4u, 2048, false, ...>`), so both are normalised."""
    m = re.search(re.escape(family) + r"<([^>]*)>", name)
    if not m:
        return None
    out = []
    for a in m.group(1).split(","):
        a = re.sub(r"^\((unsigned int|int|bool)\)", "", a.strip()).rstrip("u")
        out.append({"false": 0, "true": 1}.get(a, int(a) if a.lstrip("-").isdigit() else a))
    return tuple(out)


def setup(args):
    assert args.variant in VARIANTS, f"--variant must be one of {VARIANTS}"
    m, n, k = args.m, args.n, args.k
    assert k == 2048 and 0 < m <= 16, "the manifest's shapes: top_k = 2048, m <= 16 (cluster-8 envelope)"
    import hpc
    print(f"setup: hpc from {os.path.dirname(hpc.__file__)}")
    print(f"setup: hpc {hpc.__version__}, module sm{hpc.__loaded_arch__}")
    cap = PAD_ROWS if args.variant == "pad17" else m
    torch.manual_seed(20 + m)  # test_cluster_uniform_rows
    st = {"variant": args.variant, "hpc": hpc, "m": m, "n": n, "k": k, "cap": cap,
          "logits": torch.randn((cap, n), dtype=torch.float32, device="cuda"),
          "ke": torch.full((cap,), n, dtype=torch.int32, device="cuda"),
          "out": torch.full((cap, k), -1, dtype=torch.int32, device="cuda"),
          "nvr": torch.tensor([m], dtype=torch.int32, device="cuda")}
    cnt_bytes, ws_bytes = hpc.topk_workspace_size(cap, n)       # from the loaded build's own plan
    st["cnt"] = torch.zeros(cnt_bytes, dtype=torch.uint8, device="cuda")
    st["ws"] = torch.full((ws_bytes,), 0xA5, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    print(f"setup variant={args.variant} m={m} n={n} k={k} row_capacity={cap} counters={cnt_bytes} B "
          f"workspace={ws_bytes / 2**20:.1f} MiB")
    return st


def step(st):
    st["hpc"].topk(st["logits"], st["ke"], st["out"], st["nvr"], st["k"], st["cnt"], st["ws"])


def kernels_of_one_step(st):
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        step(st)
        torch.cuda.synchronize()
    return [e.name for e in prof.events() if str(getattr(e, "device_type", "")).endswith("CUDA")
            and not re.match(r"(Memcpy|Memset)", e.name)]


def validate_rows(logits, idx, m, n, k):
    """tests/test_topk.py _validate for uniform rows (ke = n). Returns (ok, tie_rows, message)."""
    ties, lg, ix = 0, logits[:m].cpu(), idx[:m].cpu()
    for r in range(m):
        valid = lg[r]
        ref = valid.topk(k, dim=-1)[1]
        cu = ix[r, :k]
        cu = cu[cu >= 0]
        if cu.numel() != k:
            return False, ties, f"row {r}: {cu.numel()} valid indices, expected {k}"
        if not bool((cu < n).all()):
            return False, ties, f"row {r}: out-of-range index"
        if cu.unique().numel() != k:
            return False, ties, f"row {r}: duplicate indices"
        if set(cu.tolist()) == set(ref.tolist()):
            continue
        ties += 1
        if not torch.equal(valid[cu].sort(descending=True)[0], valid[ref].sort(descending=True)[0]):
            return False, ties, f"row {r}: top-k values differ from torch.topk"
    return True, ties, "all rows ok"


def check(st):
    m, n, k, cap, v = st["m"], st["n"], st["k"], st["cap"], st["variant"]
    fam, targs = expected_kernel(cap, n, k, use_cluster_enabled=(v != "nocluster"))
    names = kernels_of_one_step(st)
    got = [x for x in names if "roofline_l2_flush_kernel" not in x]
    plan_ok = len(got) == 1 and fam in got[0] and (targs is None or template_args(got[0], fam) == targs)
    print(f"PLAN variant={v} row_capacity={cap} n={n} expected={fam}<{targs}> launched={len(got)} "
          f"kernel={got[0][:200] if got else 'none'} {'ok' if plan_ok else 'MISMATCH'}")
    st["out"].fill_(-1)
    for _ in range(REPEATS):
        step(st)
    torch.cuda.synchronize()
    ok, ties, msg = validate_rows(st["logits"], st["out"], m, n, k)
    pad_ok = bool((st["out"][m:] == -1).all()) if cap > m else True
    cnt_ok = not bool(st["cnt"].any())
    good = ok and pad_ok and cnt_ok and plan_ok
    print(f"CHECK variant={v} m={m} n={n} k={k} repeats={REPEATS} rows_with_tie_resolution={ties} {msg}; "
          f"padding_rows_untouched={pad_ok} counters_zero={cnt_ok} {'ok' if good else 'FAIL'}")
    aud = audit(MANIFEST, v, lambda: step(st))
    return good and aud


if __name__ == "__main__":
    wrapper.main(setup, step, check, extra={"m": int, "n": int, "k": int})
