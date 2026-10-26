#!/usr/bin/env python3
"""Build Tencent hpc-ops' operator module `hpc/_C_sm90.abi3.so` in place, without a visible GPU.

    cd roofline/build/src/hpc-ops__<pin | patch stem>          # the runner's per-variant worktree
    MAX_JOBS=16 SM_ARCH=90 python3 roofline/harness/workloads/build_hpcops.py

Upstream `make sm90` = `SM_ARCH=90 python3 -m build --wheel --no-isolation`, i.e. setup.py's CMakeBuild: one CMake
tree per architecture under build/sm<arch>/ (cmake configure with -DHPC_TARGET_ARCH=90 -> gencode sm_90a, then
`cmake --build . --config Release -j64`), the module copied into the package directory `hpc/`. This script runs the
unmodified setup.py as `build_ext --inplace` (the module lands in the worktree's own `hpc/`, which the manifests put
on PYTHONPATH; no wheel, no `pip install`: containers are `--rm`) with two changes made from outside:
  * the hard-coded `-j64` of setup.py:51 is replaced by `-j$MAX_JOBS` (default 16: the study's compile-job limit);
  * SM_ARCH defaults to 90, so setup.py never calls nvidia-smi.
setup.py also writes hpc/version.py from `git rev-parse` of the worktree (a git worktree's .git file works). The
build compiles every sm90 source of src/ (64 .cu/.cc files: CMakeLists.txt globs them all), not only top-k."""
import os
import runpy
import subprocess
import sys


def main():
    jobs = os.environ.setdefault("MAX_JOBS", "16")
    os.environ.setdefault("SM_ARCH", "90")
    if not (os.path.exists("setup.py") and os.path.exists("src/topk/topk_filtered.cu")):
        sys.exit(f"build_hpcops: run me in an hpc-ops worktree (cwd={os.getcwd()})")
    real = subprocess.check_call

    def check_call(cmd, *a, **k):
        if isinstance(cmd, list):
            cmd = [f"-j{jobs}" if c == "-j64" else c for c in cmd]
        print("[build_hpcops] +", " ".join(map(str, cmd)), flush=True)
        return real(cmd, *a, **k)

    subprocess.check_call = check_call
    print(f"[build_hpcops] cwd={os.getcwd()} SM_ARCH={os.environ['SM_ARCH']} MAX_JOBS={jobs}", flush=True)
    argv = sys.argv
    sys.argv = ["setup.py", "build_ext", "--inplace"]
    try:
        runpy.run_path("setup.py", run_name="__main__")
    finally:
        sys.argv = argv
        subprocess.check_call = real
    so = os.path.join("hpc", f"_C_sm{os.environ['SM_ARCH']}.abi3.so")
    if not os.path.exists(so):
        sys.exit(f"build_hpcops: {so} not produced")
    # the top-k kernels compiled into the module (demangled), for the manifest's `kernels` regexes
    sym = subprocess.run(f"cuobjdump -symbols {so} | grep -o '_Z[0-9A-Za-z_]*topk_filtered[0-9A-Za-z_]*' | "
                         "sort -u | cu++filt", shell=True, capture_output=True, text=True).stdout.split("\n")
    names = sorted({s.split("<")[0].split("(")[0] for s in sym if s.strip()})
    print(f"[build_hpcops] built {so} ({os.path.getsize(so) / 2**20:.1f} MiB); top-k kernel families: {names}",
          flush=True)


if __name__ == "__main__":
    main()
