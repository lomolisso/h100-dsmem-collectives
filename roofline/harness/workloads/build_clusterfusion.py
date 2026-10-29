#!/usr/bin/env python3
"""Build ClusterFusion's extension `clusterfusion._clusterfusion` in place, without a visible GPU.

    cd roofline/build/src/ClusterFusion__<pin | patch stem>     # the runner's per-variant worktree
    MAX_JOBS=16 python3 roofline/harness/workloads/build_clusterfusion.py [--no-flashinfer-jit]

Why a wrapper: upstream `setup.py` picks the target from `torch.cuda.get_device_properties(0)` at import time
(`_get_arch()`: sm_90 -> sources of include/H100/, `-gencode arch=compute_90a,code=sm_90a`, `-DCOMPILE_SM90`),
so `python setup.py build_ext` fails in a `h100 run --no-gpu` container. This script stubs the three torch.cuda
queries to report an sm_90 device (the H100 NVL of the study) and then runs the unmodified `setup.py build_ext
--inplace` (no `pip install -e .`: `pyproject.toml` would need build isolation, and containers are `--rm`). The
result is `clusterfusion/_clusterfusion*.so` inside the worktree; the manifests put the worktree on PYTHONPATH.
torch.utils.cpp_extension takes the architecture from TORCH_CUDA_ARCH_LIST (image: 9.0a) and never queries the
device when it is set; the stubs only affect this build process.

Unless --no-flashinfer-jit: also compiles the flashinfer JIT modules the ClusterFusion wrappers use (CUDA norm
kernels, single-request decode fp16/head_dim 128/no RoPE, silu_and_mul) into $FLASHINFER_WORKSPACE_BASE
(roofline/build/cache/flashinfer, set by roofline/tools/run) with FLASHINFER_CUDA_ARCH_LIST=9.0a, so that no JIT
compilation happens on the GPU. Failure of this optional step is reported, not fatal (the modules then compile on
first use)."""
import argparse
import os
import runpy
import sys
import types


def stub_sm90(torch):
    torch.cuda.is_available = lambda: True
    torch.cuda.device_count = lambda: 1
    torch.cuda.get_device_capability = lambda *a, **k: (9, 0)
    torch.cuda.get_device_properties = lambda *a, **k: types.SimpleNamespace(
        major=9, minor=0, name="NVIDIA H100 NVL (build-time stub)", multi_processor_count=132)


def build_extension():
    import torch
    stub_sm90(torch)
    os.environ.setdefault("MAX_JOBS", "16")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    if not os.path.exists("setup.py") or not os.path.isdir("include/H100"):
        sys.exit(f"build_clusterfusion: run me in a ClusterFusion worktree (cwd={os.getcwd()})")
    print(f"[build_clusterfusion] cwd={os.getcwd()} MAX_JOBS={os.environ['MAX_JOBS']} "
          f"TORCH_CUDA_ARCH_LIST={os.environ['TORCH_CUDA_ARCH_LIST']} torch={torch.__version__}", flush=True)
    argv = sys.argv
    sys.argv = ["setup.py", "build_ext", "--inplace"]
    try:
        runpy.run_path("setup.py", run_name="__main__")
    finally:
        sys.argv = argv
    so = [f for f in os.listdir("clusterfusion") if f.startswith("_clusterfusion") and f.endswith(".so")]
    if not so:
        sys.exit("build_clusterfusion: no clusterfusion/_clusterfusion*.so produced")
    print(f"[build_clusterfusion] built clusterfusion/{so[0]}", flush=True)


def warm_flashinfer_jit():
    """Compile and load (not run) the flashinfer JIT modules used by the wrappers; FLASHINFER_CUDA_ARCH_LIST
    fixes the target (cache directory .../90a, the same one a GPU run on sm_90 uses)."""
    os.environ.setdefault("FLASHINFER_CUDA_ARCH_LIST", "9.0a")
    os.environ["FLASHINFER_USE_CUDA_NORM"] = "1"
    try:
        import torch
        from flashinfer.norm import get_norm_module
        from flashinfer.activation import get_act_and_mul_module
        from flashinfer.decode import get_single_decode_module
        from flashinfer.utils import PosEncodingMode
        get_norm_module()
        print("[build_clusterfusion] flashinfer JIT: norm module ready", flush=True)
        get_act_and_mul_module("silu")
        print("[build_clusterfusion] flashinfer JIT: silu_and_mul module ready", flush=True)
        # same argument tuple as flashinfer.decode.single_decode_with_kv_cache(use_tensor_cores=False) for
        # q/k/v fp16, head_dim 128, pos_encoding_mode "NONE", no sliding window, no logits soft cap
        get_single_decode_module(torch.float16, torch.float16, torch.float16, 128, 128,
                                 PosEncodingMode["NONE"].value, False, False)
        print("[build_clusterfusion] flashinfer JIT: single_decode module ready", flush=True)
    except Exception as e:  # optional step
        print(f"[build_clusterfusion] WARNING: flashinfer JIT warm-up failed ({type(e).__name__}: {e}); "
              "the modules will compile on first use", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-flashinfer-jit", action="store_true")
    a = ap.parse_args()
    build_extension()
    if not a.no_flashinfer_jit:
        warm_flashinfer_jit()


if __name__ == "__main__":
    main()
