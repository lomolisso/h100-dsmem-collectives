"""Wrapper of manifest clusterfusion_rmsnorm: y = x * rsqrt(mean(x^2) + 1e-6) * w per row, fp16, 64 rows x 8192
(compile-time BATCH_SIZE / HIDDEN_DIM of ClusterFusion include/H100/norm/config.h), as tests/test_norm.py.

Variants (--variant):
  flashinfer  flashinfer.norm.rmsnorm -> flashinfer::norm::RMSNormKernel, one CTA (1024 threads) per row (role non_dsmem)
  cluster2    clusterfusion.rmsnorm   -> RMSNormKernel, cluster of 2 CTAs (512 threads) per row; the row's sum of
              squares is AllReduced by one remote atomicAdd into the peer's shared memory          (role dsmem)
No cluster-1 control: with CLUSTER_SIZE 1 the shift-by-1 atomicAdd would add a CTA's sum into itself (2x).
FLASHINFER_USE_CUDA_NORM=1 selects flashinfer's CUDA kernel (the one tests/test_norm.py compared with; flashinfer
0.7.2 would otherwise run its CuTe-DSL kernel); enable_pdl=False (no programmatic dependent launch: the kernel must
not start under the L2 flush). The clusterfusion dispatcher allocates the output with torch::full (one FillFunctor
kernel, excluded from the region) and calls cudaDeviceSynchronize before and after the kernel (host overhead only).
Inputs are read-only (no in-place updates). Validation: flashinfer vs an fp32 torch reference, cluster2 vs flashinfer
(manifest reference) with the fp32 errors printed, max abs 1e-2; plus the kernel audit (kernel_audit.py)."""
import os

os.environ.setdefault("FLASHINFER_USE_CUDA_NORM", "1")  # read by flashinfer.norm at import time

import torch  # noqa: E402
import flashinfer  # noqa: E402

import wrapper  # noqa: E402
from kernel_audit import audit  # noqa: E402

MANIFEST = "clusterfusion_rmsnorm"
VARIANTS = ("flashinfer", "cluster2")
EPS = 1e-6           # hard-coded in ClusterFusion's RMSNormKernel; flashinfer's default
TOL_MAX_ABS = 1e-2   # manifest validate.tolerance


def setup(args):
    assert args.variant in VARIANTS, f"--variant must be one of {VARIANTS}"
    assert (args.rows, args.hidden) == (64, 8192), "ClusterFusion's RMSNormKernel is compiled for 64 x 8192"
    st = {"variant": args.variant,
          "x": torch.empty((args.rows, args.hidden), dtype=torch.float16, device="cuda").normal_(),  # test: randn
          "w": torch.empty((args.hidden,), dtype=torch.float16, device="cuda").normal_()}
    st["out"] = torch.empty_like(st["x"])
    if args.variant == "cluster2":
        import clusterfusion
        st["cf"] = clusterfusion
        print(f"setup: clusterfusion from {os.path.dirname(clusterfusion.__file__)}")
    torch.cuda.synchronize()
    print(f"setup variant={args.variant} rows={args.rows} hidden={args.hidden}")
    return st


def run(st):
    if st["variant"] == "flashinfer":
        return flashinfer.norm.rmsnorm(st["x"], st["w"], EPS, out=st["out"], enable_pdl=False)
    return st["cf"].rmsnorm(st["x"], st["w"])


def step(st):
    run(st)


def errs(y, ref):
    d = (y.float() - ref.float()).abs()
    return d.max().item(), d.mean().item()


def check(st):
    x, w = st["x"], st["w"]
    xf = x.float()
    y32 = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * w.float()
    y_fi = flashinfer.norm.rmsnorm(x, w, EPS, enable_pdl=False).clone()
    y = run(st).clone()
    m32, a32 = errs(y, y32)
    if st["variant"] == "flashinfer":
        mx, mae = m32, a32
        print(f"flashinfer vs fp32 torch: max_abs={mx:.3e} mae={mae:.3e} (max|ref|={y32.abs().max().item():.3f})")
    else:
        mx, mae = errs(y, y_fi)
        print(f"cluster2 vs flashinfer: max_abs={mx:.3e} mae={mae:.3e}; (info) vs fp32: max_abs={m32:.3e} "
              f"mae={a32:.3e} (max|ref|={y32.abs().max().item():.3f})")
    good = mx <= TOL_MAX_ABS
    print(f"CHECK variant={st['variant']} max_abs={mx:.3e} tol={TOL_MAX_ABS} {'ok' if good else 'FAIL'}")
    excluded = ("FillFunctor",) if st["variant"] == "cluster2" else ()
    aud = audit(MANIFEST, st["variant"], lambda: step(st), excluded=excluded)
    return good and aud


if __name__ == "__main__":
    wrapper.main(setup, step, check, extra={"rows": int, "hidden": int})
