"""Wrapper of manifest clusterfusion_llama2_7b_decoder: one Llama-2-7B decode step (one token, fp16) through the
attention block, ClusterFusion (NeurIPS 2025) fused kernel vs the unfused flashinfer/torch sequence.

Variants (--variant):
  unfused   flashinfer.fused_add_rmsnorm -> F.linear QKV (cuBLAS) -> torch NEOX RoPE -> KV append ->
            flashinfer.single_decode_with_kv_cache(use_tensor_cores=False) -> F.linear O-proj      (role non_dsmem)
  fused_c1  clusterfusion.llama_decoder_layer_sglang, worktree built with patches/clusterfusion__cluster_size_1.patch
            (CLUSTER_SIZE 1 + RMSNorm-prologue fix + residual_out fix + TMA-buffer WAR fix)       (role control)
  fused_c4  clusterfusion.llama_decoder_layer_sglang, cluster of 4 CTAs per head, worktree built with
            patches/clusterfusion__residual_out_c4_fix.patch (residual_out fix + TMA-buffer WAR fix
            + dsm.cuh cluster_reduce hardening)                                                   (role dsmem)
--scope layer (default operator) appends the rest of the decoder layer to every variant, in the sglang
decomposition that llama_decoder_layer_sglang is written for: flashinfer.fused_add_rmsnorm(o, residual, w_post)
(residual add + post-attention RMSNorm) -> F.linear merged gate|up (4096 -> 2*ffn) -> flashinfer.silu_and_mul ->
F.linear down (ffn -> 4096). Same maths as chat/llama/model.py TransformerBlock (h = x + attn(norm(x)); out = h +
w2(silu(w1 norm(h)) * w3 norm(h))) with w1|w3 merged; the FFN residual add is the next layer's first
fused_add_rmsnorm, i.e. the region's first step, so every add is counted once per layer.

Faithful to tests/test_llama.py (random weights N(0, 0.1^2) at the real shapes, NEOX RoPE with random angles, cos/sin
fp32 (1, 128)) except the KV cache: the test concatenates the cache with the new k/v (torch.cat copies the whole
cache, 2*S*hidden*2 B per call, an artifact of the test); here the cache has S + 1 rows, the unfused sequence writes
the new token's k (after RoPE) and v into row S (two small copy kernels, the decode step's KV append) and attends
over S + 1 rows; the fused kernel gets rows 0..S-1 and folds the new token in on chip (it returns k, v instead of
appending), so both attend over the same S + 1 tokens.

Residual: upstream's kernel wrote residual = half(input + residual) back into `residual` from every one of the 32 head
clusters; a CTA reading after another cluster's write normalised 2*input + residual (wrong q/k/v for the whole head,
nondeterministic; confirmed on GPU in R4). Both fused builds carry the residual_out patch: the sum goes to a new
tensor returned in place of `residual` (as upstream's kernel_batch_sglang.cuh does), `residual` is only read. Both also
carry a block.sync() at the top of the Q/K/V GEMV and flash-decoding loops (upstream refilled TMA buffer id % 2 while
warps 1-3 could still read tile id - 2; seen at cluster 4 only, see the patch headers). In-place
updates left (no re-initialisation needed; neither kernel has value-dependent control flow):
flashinfer.fused_add_rmsnorm writes residual += hidden and hidden = rmsnorm(residual) (unfused; and the layer tail of
every variant, on the residual the region returned). Values stay bounded over 120 iterations (rmsnorm output scale =
|w| ~ 0.1); the work per iteration is unchanged. The fused dispatcher allocates o, k, v with torch::full (3
FillFunctor kernels, excluded from the region: zero-init of 3 x 8 KB) and residual_output with torch::empty_like (no
kernel), re-encodes 4 TMA descriptors and calls cudaDeviceSynchronize before and after the
kernel (host overhead only; kernel durations are measured). Norm kernels: FLASHINFER_USE_CUDA_NORM=1 selects
flashinfer's CUDA norm kernels (FusedAddRMSNormKernel; flashinfer 0.7.2 would otherwise use CuTe-DSL kernels, not
the ones ClusterFusion compares with); enable_pdl=False everywhere (programmatic dependent launch would let a kernel
start under the previous one, e.g. the L2 flush, and blur kernel durations)."""
import math
import os
import sys
import types

os.environ.setdefault("FLASHINFER_USE_CUDA_NORM", "1")  # read by flashinfer.norm at import time


def _pop_scope():
    """--scope operator|layer is consumed here (wrapper.parse only knows the shape keys); the manifest's
    `scope: layer` variants (<id>_layer) pass `--variant <id> --scope layer`."""
    scope = "operator"
    if "--scope" in sys.argv:
        i = sys.argv.index("--scope")
        scope = sys.argv[i + 1]
        del sys.argv[i:i + 2]
    assert scope in ("operator", "layer"), f"--scope must be operator or layer, got {scope}"
    return scope


SCOPE = _pop_scope()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import flashinfer  # noqa: E402

import wrapper  # noqa: E402
from kernel_audit import audit  # noqa: E402

MANIFEST = "clusterfusion_llama2_7b_decoder"
VARIANTS = ("unfused", "fused_c1", "fused_c4")
EPS = 1e-6
TRIALS = 20          # fused launches checked on identical inputs (each with a fresh copy of residual)
# manifest validate.tolerance: o (attention-block output) max abs 2e-2 and MAE 1e-3 (upstream test's figures) and
# relative L2 1e-2 (o has std ~1e-2 with these weights, the absolute bounds alone are weak); new-token k (after RoPE)
# and v, and the layer output: relative L2 1e-2 and max abs <= 2e-2 * max|ref|
TOL_O = {"max_abs": 2e-2, "mae": 1e-3, "rel_l2": 1e-2}
TOL_REL = {"rel_l2": 1e-2, "rel_max": 2e-2}


def normal(shape, std=0.1):
    # generated on the device by one distribution kernel (no kernel that matches the manifest regexes)
    return torch.empty(shape, dtype=torch.float16, device="cuda").normal_(0.0, std)


def rope_neox(x, cos, sin):
    """tests/test_llama.py::apply_neox_style_rotary_pos_emb for one tensor; x (1, H, D) fp16, cos/sin (1, 1, D)
    fp32 -> fp32."""
    h = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., h:], x[..., :h]), dim=-1) * sin


def setup(args):
    assert args.variant in VARIANTS, f"--variant must be one of {VARIANTS}"
    # compile-time constants of include/H100/llama/config.h (HIDDEN_DIM, HEAD_NUM, HEAD_DIM)
    assert (args.hidden, args.heads, args.head_dim) == (4096, 32, 128), "ClusterFusion is compiled for 4096/32/128"
    H, D, S, hid, ffn = args.heads, args.head_dim, args.seq_len, args.hidden, args.ffn
    st = types.SimpleNamespace(args=args, variant=args.variant, layer=SCOPE == "layer", H=H, D=D, S=S, hid=hid, ffn=ffn)
    st.inp = normal((1, hid))
    st.res = normal((1, hid))
    st.w_rms = normal((hid,))
    st.wqkv = normal((3 * hid, hid))          # nn.Linear layout [out, in]; rows q | k | v, head-major
    st.wo = normal((hid, hid))
    st.kc = normal((S + 1, H, D))             # NHD; rows 0..S-1 = cache, row S = append slot of the new token
    st.vc = normal((S + 1, H, D))
    st.kc_S = st.kc[:S].view(S, hid)          # the fused kernel's k_cache / v_cache: (S, hidden), same memory
    st.vc_S = st.vc[:S].view(S, hid)
    g = torch.Generator().manual_seed(args.seed)
    ang = torch.rand((1, D // 2), generator=g, dtype=torch.float32) * (2 * math.pi)
    st.cos = torch.cat([torch.cos(ang), torch.cos(ang)], dim=-1).cuda()   # (1, D) fp32, host-computed (H2D copy)
    st.sin = torch.cat([torch.sin(ang), torch.sin(ang)], dim=-1).cuda()
    st.cos3, st.sin3 = st.cos.view(1, 1, D), st.sin.view(1, 1, D)
    if st.layer:
        st.w_post = normal((hid,))
        st.wgu = normal((2 * ffn, hid))       # gate rows | up rows (flashinfer.silu_and_mul: silu(first half) * second)
        st.wd = normal((hid, ffn))
    st.cf = None
    if st.variant != "unfused":
        import clusterfusion
        st.cf = clusterfusion
        print(f"setup: clusterfusion from {os.path.dirname(clusterfusion.__file__)}")
    st.attn = attn_unfused if st.variant == "unfused" else attn_fused
    torch.cuda.synchronize()
    print(f"setup variant={st.variant} scope={SCOPE} hidden={hid} heads={H} head_dim={D} seq_len={S} ffn={ffn}")
    return st


def attn_unfused(st, hidden, residual):
    """Returns (o, residual, k_new, v_new); hidden and residual are updated in place (fused_add_rmsnorm)."""
    flashinfer.fused_add_rmsnorm(hidden, residual, st.w_rms, EPS, enable_pdl=False)
    qkv = F.linear(hidden, st.wqkv).view(3, st.H, st.D)
    q = rope_neox(qkv[0:1], st.cos3, st.sin3).to(torch.float16)
    st.kc[st.S:].copy_(rope_neox(qkv[1:2], st.cos3, st.sin3))     # KV append (fp32 -> fp16 cast in the copy)
    st.vc[st.S:].copy_(qkv[2:3])
    o = flashinfer.single_decode_with_kv_cache(q.view(st.H, st.D), st.kc, st.vc, "NHD", "NONE",
                                               use_tensor_cores=False)
    return F.linear(o.view(1, st.hid), st.wo), residual, st.kc[st.S], st.vc[st.S]


def attn_fused(st, hidden, residual):
    o, residual, k, v = st.cf.llama_decoder_layer_sglang(hidden, residual, st.wqkv, st.wo, st.kc_S, st.vc_S, st.w_rms,
                                                          EPS, st.cos, st.sin)
    return o, residual, k.view(st.H, st.D), v.view(st.H, st.D)


def ffn_tail(st, o, residual):
    flashinfer.fused_add_rmsnorm(o, residual, st.w_post, EPS, enable_pdl=False)   # residual = h; o = rmsnorm(h) w
    act = flashinfer.silu_and_mul(F.linear(o, st.wgu), enable_pdl=False)
    return F.linear(act, st.wd)


def step(st):
    o, r, _, _ = st.attn(st, st.inp, st.res)
    if st.layer:
        ffn_tail(st, o, r)


# ---------------------------------------------------------------- validation
def reference_fp32(st, inp, res):
    """fp32 reference of the same maths (no fp16 intermediates): returns o, k_new, v_new (and the layer output)."""
    x = inp.float() + res.float()
    xn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * st.w_rms.float()
    qkv = (xn @ st.wqkv.float().t()).view(3, st.H, st.D)
    q = rope_neox(qkv[0:1], st.cos3, st.sin3)[0]
    k = rope_neox(qkv[1:2], st.cos3, st.sin3)[0]
    v = qkv[2]
    K = torch.cat([st.kc[:st.S].float(), k[None]], 0)
    V = torch.cat([st.vc[:st.S].float(), v[None]], 0)
    p = torch.softmax(torch.einsum("hd,shd->hs", q, K) / math.sqrt(st.D), dim=-1)
    a = torch.einsum("hs,shd->hd", p, V)
    o = a.reshape(1, st.hid) @ st.wo.float().t()
    out = None
    if st.layer:
        h = x + o
        hn = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + EPS) * st.w_post.float()
        gu = hn @ st.wgu.float().t()
        out = (F.silu(gu[:, :st.ffn]) * gu[:, st.ffn:]) @ st.wd.float().t()
    return o, k, v, out


def errs(x, ref):
    d = x.float() - ref.float()
    rm = ref.float().abs().max().item()
    return {"max_abs": d.abs().max().item(), "mae": d.abs().mean().item(),
            "rel_l2": (d.norm() / ref.float().norm()).item(), "rel_max": d.abs().max().item() / max(rm, 1e-30),
            "ref_max": rm}


def ok_o(e):
    return e["max_abs"] <= TOL_O["max_abs"] and e["mae"] <= TOL_O["mae"] and e["rel_l2"] <= TOL_O["rel_l2"]


def ok_rel(e):
    return e["rel_l2"] <= TOL_REL["rel_l2"] and e["rel_max"] <= TOL_REL["rel_max"]


def fmt(name, e):
    return (f"{name}: max_abs={e['max_abs']:.3e} mae={e['mae']:.3e} rel_l2={e['rel_l2']:.3e} "
            f"rel_max={e['rel_max']:.3e} (max|ref|={e['ref_max']:.3e})")


def run_variant(st, inp, res, attn=None):
    """One run of the variant (operator, plus the FFN tail for --scope layer) on the given input tensors; `attn`
    overrides the variant's attention block (the reference of the fused variants is attn_unfused)."""
    o, r, k, v = (attn or st.attn)(st, inp, res)
    o, k, v = o.clone(), k.clone().view(st.H, st.D), v.clone().view(st.H, st.D)
    out = ffn_tail(st, o.clone(), r.clone()) if st.layer else None
    return o, r, k, v, out


def check(st):
    inp0, res0 = st.inp.clone(), st.res.clone()
    expected_res = (inp0.float() + res0.float()).half()
    ro, rk, rv, rout = reference_fp32(st, inp0, res0)
    # the unfused sequence (attn_unfused, + the same ffn_tail for --scope layer) on identical inputs = the manifest
    # reference for the fused variants. (Until 2026-10-26 this called run_variant without `attn`, i.e. st.attn, so a
    # fused variant was compared with one extra launch of itself: accuracy was never checked, only determinism.)
    uo, ur, uk, uv, uout = run_variant(st, inp0.clone(), res0.clone(), attn=attn_unfused)
    print("reference: unfused sequence (flashinfer + torch) on identical inputs; fp32 torch for the unfused variant")
    fused = st.variant != "unfused"
    reference = "unfused" if fused else "fp32"
    if not fused:
        trials = [(uo, ur, uk, uv, uout, res0)]
        ref = (ro, rk, rv, rout)
        print("unfused vs fp32:")
    else:
        trials = []
        for _ in range(TRIALS):
            res_in = res0.clone()
            trials.append(run_variant(st, inp0, res_in) + (res_in,))   # the fused kernel only reads input, residual
        ref = (uo, uk, uv, uout)
        print(f"{st.variant} vs unfused ({TRIALS} launches on identical inputs, fresh residual copy each):")
    nfail, nres, ndet, no_bits, o_spread = 0, 0, 0, 0, 0.0
    o0, r0, k0, v0 = trials[0][0], trials[0][1], trials[0][2], trials[0][3]
    for t, (o, r, k, v, out, res_in) in enumerate(trials):
        # residual: returned residual == half(input + residual) bit-exactly; for the fused variants the input
        # residual must be unchanged (residual_out patch: the kernel only reads it)
        bad_res = int((r != expected_res).sum().item())
        if fused:
            bad_res += int((res_in != res0).sum().item())
        nres += bad_res > 0
        # determinism vs trial 0: k, v and the returned residual must be bit-identical (per-head maths, fixed
        # reduction order); o is summed over the 32 heads with fp16 atomicAdd in arbitrary order, so it may differ
        # in the last bits between launches: reported, gated only by the tolerance against the reference
        det = torch.equal(k, k0) and torch.equal(v, v0) and torch.equal(r, r0)
        ndet += not det
        no_bits += torch.equal(o, o0)
        o_spread = max(o_spread, (o.float() - o0.float()).abs().max().item())
        eo, ek, ev = errs(o, ref[0]), errs(k, ref[1]), errs(v, ref[2])
        good = ok_o(eo) and ok_rel(ek) and ok_rel(ev) and bad_res == 0 and det
        line = f"  trial {t}: {fmt('o', eo)} | {fmt('k_new', ek)} | {fmt('v_new', ev)}"
        if st.layer:
            el = errs(out, ref[3])
            good = good and ok_rel(el)
            line += f" | {fmt('layer_out', el)}"
        line += (f" | residual mismatches {bad_res} | k/v/residual bit-identical to trial 0: {det} | "
                 f"{'ok' if good else 'FAIL'}")
        if t < 3 or not good:
            print(line)
        nfail += not good
    if fused:
        e32 = errs(o0, ro)
        print(f"  (info) {st.variant} trial 0 vs fp32: {fmt('o', e32)}; unfused vs fp32: {fmt('o', errs(uo, ro))}")
        print(f"  (info) o bit-identical to trial 0 in {no_bits}/{len(trials)} launches, max |o_t - o_0| = "
              f"{o_spread:.3e} (fp16 atomicAdd order of the 32 heads)")
    print(f"CHECK variant={st.variant} scope={SCOPE} reference={reference} trials={len(trials)} failed={nfail} "
          f"residual_bad_trials={nres} nondeterministic_kv_trials={ndet}")
    # kernel audit: the kernels of one step vs the manifest regexes (fused dispatchers' zero-fills are excluded)
    excluded = ("FillFunctor",) if fused else ()
    vid = st.variant + ("_layer" if st.layer else "")      # the manifest's `scope: layer` variant ids
    aud = audit(MANIFEST, vid, lambda: step(st), excluded=excluded)
    return nfail == 0 and aud


if __name__ == "__main__":
    wrapper.main(setup, step, check, extra={"hidden": int, "heads": int, "head_dim": int, "seq_len": int, "ffn": int})
