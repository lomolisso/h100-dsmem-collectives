"""Wrapper of manifest clusterfusion_deepseek_v2_lite_mla: one DeepSeek-V2-Lite MLA decode step (one token, fp16,
weight-absorbed MLA, 16 heads), ClusterFusion (NeurIPS 2025, App. B.1) fused kernel vs an unfused torch/flashinfer
sequence written here (no upstream test or reference exists).

What the fused kernel computes (third_party/ClusterFusion/include/H100/deepseek/kernel.cuh, `kernel.cuh:<line>`;
dispatcher deepseek_kernel_dispatch.cu; compile-time constants config.h: D = 2048, H = 16, nope 128, rope 64,
lora l = 512, SEQ_LEN S = 4096; eps 1e-6 hard-coded, l.45). Weights are [in, out] row-major (TMA maps, dispatch l.51-216):
  x_hat = RMSNorm(x) * w_in                                      l.81-126 (x: [1, D]; no residual)
  q_nope_h = x_hat @ Wq_nope[:, h*128:(h+1)*128]   Wq_nope [D, H*128]   l.128-164
  q_pe_h   = x_hat @ Wq_pe[:, h*64:(h+1)*64]       Wq_pe   [D, H*64]    l.166-206
  c_kv     = x_hat @ Wkv                           Wkv     [D, 512]     l.208-246  (every head cluster recomputes it)
  k_pe     = x_hat @ Wkpe                          Wkpe    [D, 64]      l.248-288  (idem)
  NEOX RoPE on q_pe, k_pe with cos[64], sin[64] fp32                    l.299-316
  c_kv_new = RMSNorm(c_kv) * w_ckv                                       l.318-350
  q_lat_h  = q_nope_h @ W_UK[h],  W_UK[h][n, j] = Wuk[n, h*512 + j]  Wuk [128, H*512]   l.352-397
  scores_t = q_lat_h . ckv_cache[t, 0:512] / sqrt(192)   (192 = nope + rope; l.45)        l.399-516
  a_lat_h  = softmax(scores) @ ckv_cache[:, 0:512]       (MQA: K = V = the latent cache)
             over cache rows 0..S-2 plus the new token: row S-1 is replaced by c_kv_new in registers (l.472-475,
             508-511); the cache is never written (the new latent is not appended)
  a_h      = a_lat_h @ W_UV[h],  W_UV[h][j, n] = Wuv[j, h*128 + n]   Wuv [512, H*128]  l.601-645
  o        = sum_h a_h @ Wo[h*128:(h+1)*128, :]    Wo [H*128, D]; fp16 atomicAdd over heads into a zeroed o
                                                                         l.647-697, dispatch l.23
Returned: o [1, D] only. The rope part never enters the attention: the cache TMA boxes cover columns 0..511 of the
576-wide rows (l.408-409, 417-418) and the score uses q_lat only (reg_input = local_qkv[0..512), l.403), so q_pe,
k_pe and their RoPE are computed and discarded (the k_pe RoPE also reads a wrong partner, l.305 misses
`+ KV_LORA_RANK`; unobservable). The fused kernel therefore implements absorbed MLA *without* the q_pe . k_pe score
term and without the KV append; the unfused baseline below computes exactly that, and does the same discarded work.

Variants (--variant):
  unfused   flashinfer.norm.rmsnorm(x) -> torch.mm q (x_hat @ [Wq_nope | Wq_pe], one cuBLAS GEMV, N = 3072) ->
            torch.mm kv_a (x_hat @ [Wkv | Wkpe], N = 576) -> flashinfer rmsnorm(c_kv) written into the latent-cache
            row S-1 (KV append) -> flashinfer RoPE (NEOX, cos/sin cache) on q_pe in place and k_pe into the k_pe-cache
            row S-1 (one kernel) -> torch.bmm absorption q_nope_h @ W_UK[h] -> flashinfer.single_decode_with_kv_cache
            (q [16, 512], K = V = latent cache [S, 1, 512], sm_scale 1/sqrt(192), CUDA cores, partition-kv + merge)
            -> torch.bmm a_lat_h @ W_UV[h] -> torch.mm O-proj                                     (role non_dsmem)
  fused_c1  clusterfusion.deepseek_decoder_layer, worktree with patches/clusterfusion__deepseek_cluster_size_1.patch
            (CLUSTER_SIZE 1 + three fixes: upstream's cluster-1 build is wrong, see the patch)   (role control)
  fused_c4  clusterfusion.deepseek_decoder_layer as shipped (pin), 16 clusters x 4 CTAs, DSMEM reduces/gather (dsmem)

Baseline choices. The attention is flashinfer's single-request decode kernel at head_dim 512 with 16 query heads on
one KV head (group size 16), K and V both the latent cache: the exact score the fused kernel computes. flashinfer's
MLA kernels (BatchMLAPagedAttentionWrapper) are not used: they always add q_pe . k_pe_cache to the score, so they
would compute a different function (with q_pe = 0 they would match, at the cost of reading the 512 KB k_pe cache and
12.5 % more QK work, which the fused kernel does not do). Weights: the baseline keeps its own copies in the layouts
cuBLAS wants (q and kv_a weights concatenated column-wise as DeepSeek's q_proj / kv_a_proj_with_mqa; W_UK, W_UV as
contiguous [H, 128, 512] / [H, 512, 128]); same values as the fused kernel's tensors. Caches: the fused kernel reads
one [S, 576] cache (latent | k_pe per row); the baseline holds the same rows as two contiguous caches [S, 512] and
[S, 64] (flashinfer/sglang MLA layout) and appends the new token at row S-1 (the row the fused kernel replaces), so
both attend over the same S tokens = S-1 cached + the new one. The q_pe RoPE result is unused by both (it would only
enter the rope term the fused kernel omits); the baseline computes it anyway so that both variants do the same work.

Random inputs at the real shapes (no checkpoint): x ~ N(0, 1); norm weights 1 + N(0, 0.1^2); projection weights
N(0, 1/fan_in) so that every activation has unit scale (q_lat ~ N(0, 1), scores std ~1.6); cache ~ N(0, 1); RoPE
angles uniform. The fused dispatcher zero-fills o with torch::full (one FillFunctor kernel, excluded from the region),
re-encodes 9 TMA descriptors and calls cudaDeviceSynchronize three times per call (host overhead only; kernel
durations are measured). Neither variant updates its inputs except the baseline's cache row S-1 (same value every
iteration). FLASHINFER_USE_CUDA_NORM=1 (CUDA norm kernels, as R4); enable_pdl=False for the norms.

  --warm-jit   compile (no GPU needed) the flashinfer JIT modules this wrapper uses into $FLASHINFER_WORKSPACE_BASE
               with FLASHINFER_CUDA_ARCH_LIST=9.0a: CUDA norm, RoPE, single decode fp16 head_dim 512 (the manifest's
               unfused build command)."""
import math
import os
import sys
import types

os.environ.setdefault("FLASHINFER_USE_CUDA_NORM", "1")  # read by flashinfer.norm at import time

MANIFEST = "clusterfusion_deepseek_v2_lite_mla"
VARIANTS = ("unfused", "fused_c1", "fused_c4")
COMPILED = {"hidden": 2048, "heads": 16, "nope": 128, "rope": 64, "lora": 512, "seq_len": 4096}   # config.h
EPS = 1e-6
SM_SCALE = 1.0 / math.sqrt(128 + 64)   # kernel.cuh l.45: __frsqrt_rn(HEAD_DIM), HEAD_DIM = nope + rope
TRIALS = 20          # fused launches checked on identical inputs (race candidates: see manifest notes)
# manifest validate.tolerance (same criteria as clusterfusion_llama2_7b_decoder): o rel. L2 <= 1e-2 and max abs
# <= 2e-2 * max|ref|; a CPU emulation of the fused kernel's fp16 P.V accumulation (R7) predicts a_lat rel. L2 2.2e-3
# (cluster 4) and 4.8e-3 (cluster 1, 2048 tokens per thread). Appended cache rows of the baseline: same criteria.
TOL = {"rel_l2": 1e-2, "rel_max": 2e-2}


def warm_jit():
    os.environ.setdefault("FLASHINFER_CUDA_ARCH_LIST", "9.0a")
    import torch
    from flashinfer.norm import get_norm_module
    from flashinfer.rope import get_rope_module
    from flashinfer.decode import get_single_decode_module
    from flashinfer.utils import PosEncodingMode
    get_norm_module()
    print("[warm-jit] flashinfer norm module ready", flush=True)
    get_rope_module()
    print("[warm-jit] flashinfer rope module ready", flush=True)
    # same argument tuple as single_decode_with_kv_cache(use_tensor_cores=False), fp16, head_dim 512, no RoPE
    get_single_decode_module(torch.float16, torch.float16, torch.float16, 512, 512, PosEncodingMode["NONE"].value,
                             False, False)
    print("[warm-jit] flashinfer single_decode module (fp16, head_dim 512) ready", flush=True)


if __name__ == "__main__" and "--warm-jit" in sys.argv:
    warm_jit()
    sys.exit(0)

import torch  # noqa: E402
import flashinfer  # noqa: E402
from flashinfer.rope import _apply_rope_pos_ids_cos_sin_cache  # noqa: E402

import wrapper  # noqa: E402
from kernel_audit import audit  # noqa: E402


def normal(shape, std=1.0, mean=0.0):
    # generated on the device by one distribution kernel (setup only)
    return torch.empty(shape, dtype=torch.float16, device="cuda").normal_(mean, std)


def setup(args):
    assert args.variant in VARIANTS, f"--variant must be one of {VARIANTS}"
    got = {k: getattr(args, k) for k in COMPILED}
    assert got == COMPILED, f"ClusterFusion's DeepSeek kernel is compiled for {COMPILED} (config.h), got {got}"
    D, H, N, R, L, S = (args.hidden, args.heads, args.nope, args.rope, args.lora, args.seq_len)
    st = types.SimpleNamespace(args=args, variant=args.variant, D=D, H=H, N=N, R=R, L=L, S=S)
    st.x = normal((1, D))
    st.w_in = normal((D,), 0.1, 1.0)
    st.w_ckv = normal((L,), 0.1, 1.0)
    # the fused kernel's weights, [in, out] row-major (dispatcher TMA maps)
    st.wq_nope = normal((D, H * N), D ** -0.5)
    st.wq_pe = normal((D, H * R), D ** -0.5)
    st.wkv = normal((D, L), D ** -0.5)
    st.wkpe = normal((D, R), D ** -0.5)
    st.wuk = normal((N, H * L), N ** -0.5)        # W_UK[h][n, j] = wuk[n, h*L + j]
    st.wuv = normal((L, H * N), L ** -0.5)        # W_UV[h][j, n] = wuv[j, h*N + n]
    st.wo = normal((H * N, D), (H * N) ** -0.5)   # shared by both variants (read-only)
    st.cache = normal((S, L + R))                 # fused kernel's cache: row = c_kv (512) | k_pe (64)
    g = torch.Generator().manual_seed(args.seed)
    ang = torch.rand((R // 2,), generator=g, dtype=torch.float32) * (2 * math.pi)
    c, s = torch.cos(ang), torch.sin(ang)
    st.cos64 = torch.cat([c, c]).cuda()           # fused kernel: cos[64], sin[64] fp32 (NEOX, halves duplicated)
    st.sin64 = torch.cat([s, s]).cuda()
    st.c32, st.s32 = c.cuda(), s.cuda()
    # baseline copies (same values, cuBLAS / flashinfer layouts)
    st.wq_cat = torch.cat([st.wq_nope, st.wq_pe], dim=1).contiguous()          # [D, H*N | H*R]
    st.wkva = torch.cat([st.wkv, st.wkpe], dim=1).contiguous()                 # [D, L | R]
    st.wuk_b = st.wuk.view(N, H, L).permute(1, 0, 2).contiguous()              # [H, N, L]
    st.wuv_b = st.wuv.view(L, H, N).permute(1, 0, 2).contiguous()              # [H, L, N]
    st.ckv = st.cache[:, :L].contiguous()                                      # [S, L] latent cache
    st.kpe = st.cache[:, L:].contiguous()                                      # [S, R] k_pe cache
    st.ckv3 = st.ckv.view(S, 1, L)                                             # NHD, one KV head
    st.cos_sin = torch.cat([c, s]).view(1, R).cuda()                           # flashinfer: [pos, cos | sin]
    st.pos = torch.zeros(1, dtype=torch.int64, device="cuda")
    st.cf = None
    if st.variant != "unfused":
        import clusterfusion
        st.cf = clusterfusion
        print(f"setup: clusterfusion from {os.path.dirname(clusterfusion.__file__)}")
    torch.cuda.synchronize()
    print(f"setup variant={st.variant} hidden={D} heads={H} nope={N} rope={R} lora={L} seq_len={S} "
          f"(attention over {S - 1} cached tokens + the new one)")
    return st


def mla_unfused(st):
    H, N, R, L, S = st.H, st.N, st.R, st.L, st.S
    xn = flashinfer.norm.rmsnorm(st.x, st.w_in, EPS, enable_pdl=False)
    q = torch.mm(xn, st.wq_cat)                    # [1, H*N + H*R] = q_nope | q_pe
    kv = torch.mm(xn, st.wkva)                     # [1, L + R] = c_kv | k_pe
    flashinfer.norm.rmsnorm(kv[:, :L], st.w_ckv, EPS, out=st.ckv[S - 1:S], enable_pdl=False)   # norm + append
    q_pe = q[:, H * N:].view(1, H, R)
    _apply_rope_pos_ids_cos_sin_cache(q_pe, kv[:, L:].view(1, 1, R), q_pe, st.kpe[S - 1:S].view(1, 1, R),
                                      st.cos_sin, st.pos, False)                # NEOX; k_pe -> cache row S-1
    q_lat = torch.bmm(q[:, :H * N].view(H, 1, N), st.wuk_b)                     # [H, 1, L]
    a_lat = flashinfer.single_decode_with_kv_cache(q_lat.view(H, L), st.ckv3, st.ckv3, "NHD", "NONE",
                                                   use_tensor_cores=False, sm_scale=SM_SCALE)
    a = torch.bmm(a_lat.view(H, 1, L), st.wuv_b)                                # [H, 1, N]
    return torch.mm(a.view(1, H * N), st.wo)


def mla_fused(st):
    return st.cf.deepseek_decoder_layer(st.x, st.wq_nope, st.wq_pe, st.wuk, st.wkv, st.wkpe, st.wuv, st.wo, st.cache,
                                        st.w_in, st.w_ckv, st.cos64, st.sin64)


def step(st):
    (mla_unfused if st.variant == "unfused" else mla_fused)(st)


# ---------------------------------------------------------------- validation
def rope_neox(x, c, s):
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


def reference_fp32(st):
    """fp32 torch of the same maths from the same fp16 tensors: o, the new latent c_kv_new and k_pe (after RoPE)."""
    H, N, L, S = st.H, st.N, st.L, st.S
    x = st.x.float()
    xn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * st.w_in.float()
    q = xn @ st.wq_cat.float()
    kv = xn @ st.wkva.float()
    c = kv[:, :L]
    cn = c * torch.rsqrt(c.pow(2).mean(-1, keepdim=True) + EPS) * st.w_ckv.float()
    kpe = rope_neox(kv[:, L:], st.c32, st.s32)
    q_lat = torch.einsum("hn,hnl->hl", q[0, :H * N].view(H, N), st.wuk_b.float())
    K = torch.cat([st.cache[:S - 1, :L].float(), cn], 0)                       # S-1 cached rows + the new token
    p = torch.softmax((q_lat @ K.t()) * SM_SCALE, dim=-1)
    a = torch.einsum("hl,hln->hn", p @ K, st.wuv_b.float())
    return a.reshape(1, H * N) @ st.wo.float(), cn, kpe


def errs(x, ref):
    d = x.float() - ref.float()
    rm = ref.float().abs().max().item()
    return {"max_abs": d.abs().max().item(), "rel_l2": (d.norm() / ref.float().norm()).item(),
            "rel_max": d.abs().max().item() / max(rm, 1e-30), "ref_max": rm,
            "finite": bool(torch.isfinite(x.float()).all().item())}


def ok(e):
    return e["finite"] and e["rel_l2"] <= TOL["rel_l2"] and e["rel_max"] <= TOL["rel_max"]


def fmt(name, e):
    return (f"{name}: rel_l2={e['rel_l2']:.3e} rel_max={e['rel_max']:.3e} max_abs={e['max_abs']:.3e} "
            f"(max|ref|={e['ref_max']:.3e}{'' if e['finite'] else ', NON-FINITE'})")


def check(st):
    o32, cn32, kpe32 = reference_fp32(st)
    S = st.S
    uo = mla_unfused(st).clone()
    print("reference: unfused sequence (flashinfer + torch) on identical inputs for the fused variants; "
          "fp32 torch for the unfused variant")
    nfail, ntr = 0, 1
    if st.variant == "unfused":
        eo, ec, ek = errs(uo, o32), errs(st.ckv[S - 1:S], cn32), errs(st.kpe[S - 1:S], kpe32)
        good = ok(eo) and ok(ec) and ok(ek)
        print(f"unfused vs fp32: {fmt('o', eo)} | {fmt('ckv_row', ec)} | {fmt('kpe_row', ek)} | "
              f"{'ok' if good else 'FAIL'}")
        nfail += not good
    else:
        x0, cache0 = st.x.clone(), st.cache.clone()
        print(f"{st.variant} vs unfused ({TRIALS} launches on identical inputs):")
        outs = [mla_fused(st).clone() for _ in range(TRIALS)]
        ntr = len(outs)
        nbits, spread = 0, 0.0
        for t, o in enumerate(outs):
            e = errs(o, uo)
            good = ok(e)
            nfail += not good
            nbits += torch.equal(o, outs[0])
            spread = max(spread, (o.float() - outs[0].float()).abs().max().item())
            if t < 3 or not good:
                print(f"  trial {t}: {fmt('o', e)} | {'ok' if good else 'FAIL'}")
        ro = torch.equal(st.x, x0) and torch.equal(st.cache, cache0)
        print(f"  inputs unchanged by the fused launches (x, cache bit-identical): {ro}")
        nfail += not ro
        print(f"  (info) {st.variant} trial 0 vs fp32: {fmt('o', errs(outs[0], o32))}; "
              f"unfused vs fp32: {fmt('o', errs(uo, o32))}")
        print(f"  (info) o bit-identical to trial 0 in {nbits}/{ntr} launches, max |o_t - o_0| = {spread:.3e} "
              f"(fp16 atomicAdd order of the 16 heads)")
    print(f"CHECK variant={st.variant} trials={ntr} failed={nfail} ")
    excluded = ("FillFunctor",) if st.variant != "unfused" else ()   # fused dispatcher's torch::full zero-fill of o
    aud = audit(MANIFEST, st.variant, lambda: step(st), excluded=excluded)
    return nfail == 0 and aud


if __name__ == "__main__":
    wrapper.main(setup, step, check, extra={k: int for k in COMPILED})
