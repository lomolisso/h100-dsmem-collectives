"""Wrapper of manifest dak_gemm_multicast: C[tokens, m] = X[tokens, k] . W[k, m], fp16 (DAK, arXiv 2604.26074).

Variants (--variant):
  multicast          gemv_multicast<...>          cluster 2, rank 0 TMA-multicasts the weight tiles   (role dsmem)
  multicast_prefill  gemv_multicast_prefill<...>  same tiles, per-CTA TMA, no cluster                  (role non_dsmem)
  cublas             torch.mm(X, W_t.t()) = nn.Linear (upstream benchmark_linear.py --type vanila)    (role reference)

setup builds DAK's LinearMulticast layer (third_party/DirectAccessKernel/app/python/layers/linearLayerMulticast.py:
weight moved to its k x m device layout, TMA descriptors), the input and output descriptors exactly as
LinearMulticast.forward does, and a preallocated output. step launches the variant's kernel with the arguments
forward() would pass (multicast: num_copy_blocks = 132 = NUM_SMS, the value gemv_multicast needs; prefill: 128,
hard-coded in forward(); sms_per_row = 1) but without forward()'s per-call torch.zeros of C (an extra fill kernel
outside the GEMM; the kernel writes every C tile with TMA stores, so C needs no initialisation). No synchronisation
in step. Imports from the worktree need `{worktree}` (package `app`) and `{worktree}/python` (`offload.runtime`) on
PYTHONPATH (manifest run.pythonpath)."""
import math

import torch

import wrapper
from flush import flush_l2

NUM_SMS = 132          # include/task/config.cuh, DAK_SM_ARCH == 90
PREFILL_BLOCKS = 128   # linearLayerMulticast.py:122
CHECK_REPS = 5
VARIANTS = ("multicast", "multicast_prefill", "cublas")


def tolerance(k):
    """Per-tile relative Frobenius bound: fp16 accumulator, one rounding (relative <= 2^-11) per k16 WGMMA step."""
    return 4.0 * math.sqrt(k / 16) * 2.0 ** -11


def setup(args):
    assert args.variant in VARIANTS, f"--variant must be one of {VARIANTS}"
    T, M, K = args.tokens, args.m, args.k
    assert T % 512 == 0, "tokens must be a multiple of 512 (both CTAs of a multicast cluster share the m-tile)"
    assert M % 128 == 0 and K % 64 == 0, "m must be a multiple of 128 and k of 64 (TILE_M_MULTICAST x 2, TILE_K_MULTICAST)"
    dev, dt = "cuda", torch.float16
    x = torch.randn(T, K, device=dev, dtype=dt)
    w_mk = ((torch.rand(M, K, device=dev, dtype=torch.float32) * 2 - 1) / math.sqrt(K)).to(dt)  # nn.Linear layout, kaiming-uniform range
    st = {"args": args, "x": x, "out": torch.empty(T, M, device=dev, dtype=dt), "T": T, "M": M, "K": K}
    if args.variant == "cublas":
        st["w_mk"] = w_mk
        st["w_km"] = w_mk.t()  # view; reference uses the same values
        return st

    from offload import runtime
    from app.python.utils import build_tma_wgmma_k, build_tma_wgmma_mn, get_smem_size
    from app.python.config import TILE_K_MULTICAST, TILE_M_MULTICAST, TILE_N_MULTICAST
    from app.python.layers.linearLayerMulticast import LinearMulticast

    assert runtime.NUM_SMS == NUM_SMS, f"extension built with NUM_SMS={runtime.NUM_SMS}, expected {NUM_SMS}"
    assert torch.cuda.get_device_properties(0).multi_processor_count == NUM_SMS, "device SM count != NUM_SMS"
    prefill = args.variant == "multicast_prefill"
    smem = runtime.set_smem_size(get_smem_size(multicast=True), 4 if prefill else 2)  # GEMV_MULTICAST(_PREFILL) opcodes
    layer = LinearMulticast(K, M, smem_size=smem, num_copy_blocks=NUM_SMS, device=dev, dtype=dt, skip_init=True)
    layer.weight.data.copy_(w_mk)
    layer.config(NUM_SMS, 1)                 # num_copy_blocks = 132, sms_per_row = 1
    layer.move_weight_to_device("cuda")      # weight -> [K, M] contiguous
    layer.prefill = prefill
    layer.build_tma_desc()
    del w_mk
    _, x_desc = build_tma_wgmma_k(x.view(-1, K), TILE_K_MULTICAST, TILE_N_MULTICAST)
    _, c_desc = build_tma_wgmma_mn(st["out"].view(-1, M), TILE_M_MULTICAST, TILE_N_MULTICAST, debug=True)
    fn = runtime.gemv_multicast_prefill if prefill else runtime.gemv_multicast
    blocks = PREFILL_BLOCKS if prefill else layer.num_copy_blocks
    st.update(runtime=runtime, layer=layer, w_km=layer.weight.data, x_desc=x_desc, c_desc=c_desc,
              launch=lambda: fn(layer.w_desc, x_desc, c_desc, M, T, K, blocks, smem, layer.sms_per_row))
    torch.cuda.synchronize()
    print(f"setup variant={args.variant} tokens={T} m={M} k={K} smem={smem} B blocks={blocks} prefill={prefill}")
    return st


def step(st):
    if st["args"].variant == "cublas":
        torch.mm(st["x"], st["w_mk"].t(), out=st["out"])
    else:
        st["launch"]()


def tile_err(c, ref):
    """Max over 256(tokens) x 64(m) output tiles of ||c - ref||_F / ||ref||_F (c, ref fp32 [T, M])."""
    T, M = ref.shape
    d = (c - ref).view(T // 256, 256, M // 64, 64).pow(2).sum(dim=(1, 3))
    r = ref.view(T // 256, 256, M // 64, 64).pow(2).sum(dim=(1, 3))
    return (d / r).sqrt().max().item()


def check(st):
    T, M, K = st["T"], st["M"], st["K"]
    ref = st["x"].float() @ st["w_km"].float()
    tol = tolerance(K)
    ok = True
    outs = []
    for i in range(CHECK_REPS):
        st["out"].fill_(float("nan"))
        flush_l2()
        step(st)
        c = st["out"].float()
        outs.append(("step", i, c.clone()))
    if st["args"].variant != "cublas":
        outs.append(("forward", 0, st["layer"].forward(st["x"]).float()))  # upstream entry point, same arguments
    for how, i, c in outs:
        fin = bool(torch.isfinite(c).all().item())
        e = tile_err(torch.nan_to_num(c, nan=1e4), ref)
        rel = ((c - ref).norm() / ref.norm()).item() if fin else float("nan")
        mx = (c - ref).abs().max().item() if fin else float("nan")
        good = fin and e <= tol
        ok &= good
        print(f"check {how}[{i}]: finite={fin} max_tile_rel={e:.3e} rel_fro={rel:.3e} max_abs={mx:.3e} "
              f"(|ref|max={ref.abs().max().item():.3f}) tol={tol:.3e} {'ok' if good else 'MISMATCH'}")
    if len(outs) > 1:
        same = all(torch.equal(outs[0][2], o[2]) for o in outs[1:CHECK_REPS])
        print(f"check: {CHECK_REPS} step repetitions bit-identical: {same}")
    return ok


if __name__ == "__main__":
    wrapper.main(setup, step, check, extra={"tokens": int, "m": int, "k": int})
