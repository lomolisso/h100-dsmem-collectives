"""Wrapper of manifest deepgemm_fp8_gemm_multicast: D = A . B^T, fp8 e4m3 with 1x128 (A) / 128x128 (B) fp32 scales,
bf16 output (DeepGEMM, SM90 1D2D kernel).

Variants (--variant):
  cluster2  deep_gemm.fp8_gemm_nt / m_grouped_fp8_gemm_nt_contiguous, shipped heuristic (cluster 2, TMA multicast) (dsmem)
  cluster1  the same calls with patches/deepgemm__force_cluster_1.patch and DG_FORCE_CLUSTER_1=1 (no multicast) (non_dsmem)
  cublaslt  deep_gemm.cublaslt_gemm_nt on the same fp8 values, per-tensor (no block scales); grouped: one call per group
            (reference)

Shape options: --m (rows per group), --n, --k, --groups (1 = dense fp8_gemm_nt, > 1 = m-grouped contiguous).

setup generates the inputs as third_party/DeepGEMM/tests/generators.py does (bf16 randn, A cast per token with
cast_fp8_fp4_with_major, B per 128x128 block with use_block_cast_for_fp8=True, ue8m0 off as on SM90; grouped B with
grouped_cast_fp8_fp4_with_major), with two deviations: every group has exactly --m rows (the tests draw 0.7-1.3 x m at
random), and SFA is converted once to the MN-major TMA-aligned layout that the SM90 kernel needs (fp8_gemm_nt would
otherwise launch DeepGEMM's transpose_fp32 kernel on SFA in every call; with the converted SFA the call launches only
the GEMM kernel). step is one GEMM call: host-side heuristic + TMA descriptor encoding, one kernel launch on the current
stream, no synchronisation; the first call JIT-compiles the kernel (nvcc, cached under DG_JIT_CACHE_DIR), which the
warm-up iterations absorb. check runs step once and compares with the tests' references via
deep_gemm.testing.calc_diff (< 1e-3, QuantConfig.max_diff() for fp8 x fp8). The package is imported from the worktree on
PYTHONPATH (manifest run.pythonpath); tests/ of the same worktree is appended to sys.path for generators.py."""
import os
import sys

import torch

import wrapper

VARIANTS = ("cluster1", "cluster2", "cublaslt")
MAX_DIFF = 1e-3   # tests/generators.py QuantConfig.max_diff() for fp8 x fp8
GRAN = 128        # scale granularity along K (and along N for B)


def import_deep_gemm(variant):
    force = os.environ.get("DG_FORCE_CLUSTER_1") == "1"
    if variant == "cluster1":
        assert force, "cluster1 needs DG_FORCE_CLUSTER_1=1 (manifest run.env) and the patched worktree"
    else:
        assert not force, f"{variant} must run without DG_FORCE_CLUSTER_1"
    import deep_gemm
    root = os.path.dirname(os.path.dirname(os.path.abspath(deep_gemm.__file__)))
    sys.path.append(os.path.join(root, "tests"))
    print(f"deep_gemm {deep_gemm.__version__} from {root}", flush=True)
    return deep_gemm


def setup(args):
    assert args.variant in VARIANTS, f"--variant must be one of {VARIANTS}"
    dg = import_deep_gemm(args.variant)
    from generators import MajorTypeAB, cast_fp8_fp4_with_major, grouped_cast_fp8_fp4_with_major
    M, N, K, G = args.m, args.n, args.k, args.groups
    assert K % GRAN == 0 and G >= 1
    K_MAJOR = MajorTypeAB.KMajor
    st = {"args": args, "dg": dg, "M": M, "N": N, "K": K, "G": G}

    if G == 1:
        a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)
        a8 = cast_fp8_fp4_with_major(a, K_MAJOR, GRAN, False, False)
        b8 = cast_fp8_fp4_with_major(b, K_MAJOR, GRAN, False, False, use_block_cast_for_fp8=True)
        d = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    else:
        align = dg.get_theoretical_mk_alignment_for_contiguous_layout()   # 128 on SM90, as test_fp8_fp4.py
        dg.set_mk_alignment_for_contiguous_layout(align)
        assert M % align == 0, f"rows per group must be a multiple of the contiguous-layout alignment {align}"
        a = torch.randn((G * M, K), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((G, N, K), device="cuda", dtype=torch.bfloat16)
        a8 = cast_fp8_fp4_with_major(a, K_MAJOR, GRAN, False, False)
        b8 = grouped_cast_fp8_fp4_with_major(b, K_MAJOR, GRAN, False, False, use_block_cast_for_fp8=True)
        d = torch.empty((G * M, N), device="cuda", dtype=torch.bfloat16)
        st["m_indices"] = torch.arange(G, device="cuda", dtype=torch.int32).repeat_interleave(M)

    # SFA -> MN-major, TMA-aligned (what fp8_gemm_nt's transform_sf_into_required_layout does on SM90, done once here)
    st["a8"] = (a8[0], dg.get_mn_major_tma_aligned_tensor(a8[1]))
    st["b8"], st["d"] = b8, d
    if args.validate:
        st["a"], st["b"] = a, b
    else:
        del a, b
    torch.cuda.synchronize()
    return st


def step(st):
    dg, v = st["dg"], st["args"].variant
    a8, b8, d = st["a8"], st["b8"], st["d"]
    if v == "cublaslt":
        if st["G"] == 1:
            dg.cublaslt_gemm_nt(a8[0], b8[0], d)
        else:
            M = st["M"]
            for g in range(st["G"]):
                dg.cublaslt_gemm_nt(a8[0][g * M:(g + 1) * M], b8[0][g], d[g * M:(g + 1) * M])
    elif st["G"] == 1:
        dg.fp8_gemm_nt(a8, b8, d, disable_ue8m0_cast=True)
    else:
        dg.m_grouped_fp8_gemm_nt_contiguous(a8, b8, d, st["m_indices"], disable_ue8m0_cast=True)


def reference(st):
    """The tests' references (generators.py); cuBLASLt: fp32 product of the same fp8 values (it applies no scales)."""
    M, G = st["M"], st["G"]
    if st["args"].variant == "cublaslt":
        a8, b8 = st["a8"][0].float(), st["b8"][0].float()
        if G == 1:
            return (a8 @ b8.t()).to(torch.bfloat16)
        return torch.cat([(a8[g * M:(g + 1) * M] @ b8[g].t()).to(torch.bfloat16) for g in range(G)])
    a, b = st["a"], st["b"]
    if G == 1:
        return (a.float() @ b.float().t()).to(torch.bfloat16)          # generate_normal
    return torch.cat([a[g * M:(g + 1) * M] @ b[g].t() for g in range(G)])  # generate_m_grouped_contiguous (bf16)


def check(st):
    from deep_gemm.testing import calc_diff
    st["d"].fill_(float("nan"))
    step(st)
    torch.cuda.synchronize()
    ref = reference(st)
    d = st["d"]
    finite = bool(torch.isfinite(d).all())
    diff = float(calc_diff(d, ref))
    print(f"variant={st['args'].variant} m={st['M']} n={st['N']} k={st['K']} groups={st['G']} "
          f"finite={finite} calc_diff={diff:.3e} (tolerance {MAX_DIFF:g}) "
          f"max_abs_err={float((d.float() - ref.float()).abs().max()):.4g} max_abs_ref={float(ref.float().abs().max()):.4g}",
          flush=True)
    return finite and diff < MAX_DIFF


if __name__ == "__main__":
    wrapper.main(setup, step, check, extra={"m": int, "n": int, "k": int, "groups": int})
