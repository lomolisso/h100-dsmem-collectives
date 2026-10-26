"""Tensor-core ceilings from kernel durations (nsys), the same timer as the workload points (run_manifest.py).

Runs `ceilings_tc.py --size N` under `nsys profile --trace=cuda` for every N in --sizes (inside a GPU container at
the locked clock, from roofline/), and takes from each trace the GEMM kernels of every configuration: the four
configurations (fp16, bf16, fp8, fp8_fastacc) run in this order, separated by ceilings_tc.py's idle --cool-s gap, so
a gap > 5 s in the kernel trace starts the next configuration; within a configuration the GEMM is the kernel name
with the longest total duration (the 256 x 256 correctness check runs a different, small kernel). Rate = 2·N³ /
kernel duration for every launch; throttled launches only lower it, so the best launch is a valid ceiling at the
locked clock. Rows (append-only, mandatory columns first) -> results/ceilings/tc_kernel.csv; ceilings_summary.py
takes the best launch over all sizes. The cudaEvent-based rows of ceilings_tc.py stay as they are (results/ceilings/
tc.csv): for GEMMs of tens of microseconds they include launch costs the kernel-duration points do not have."""
import argparse
import csv
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rooflib as rl  # noqa: E402

CONFIGS = ["fp16", "bf16", "fp8", "fp8_fastacc"]
HDR = rl.MANDATORY + ["config", "size", "kernel", "duration_ns", "flops", "flops_per_s", "nsys_rep"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sizes", default="2048,4096,8192")
    a = ap.parse_args()
    reg = rl.Registry(sys.argv)
    out_dir = os.path.join(rl.RES, "ceilings", "nsys")
    os.makedirs(out_dir, exist_ok=True)
    for n in [int(x) for x in a.sizes.split(",")]:
        clk = rl.clock_sample()
        base = os.path.join(out_dir, f"tc_{n}_{rl.iso_now().replace(':', '')}")
        cmd = ["nsys", "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none", "--output", base,
               "python3", "harness/ceilings_tc.py", "--size", str(n)]
        p = subprocess.run(cmd, cwd=rl.ROOF, capture_output=True, text=True)
        print(p.stdout[-1500:])
        subprocess.run(["nsys", "stats", "--force-export=true", "--report", "cuda_gpu_trace", "--format", "csv",
                        "--output", base, base + ".nsys-rep"], check=True, capture_output=True)
        ks = sorted((int(r["Start (ns)"]), int(r["Duration (ns)"]), r["Name"])
                    for r in csv.DictReader(open(base + "_cuda_gpu_trace.csv")) if not r["Name"].startswith("[CUDA"))
        groups, cur, last = [], [], None
        for k in ks:
            if last is not None and k[0] - last > 5e9:
                groups.append(cur)
                cur = []
            cur.append(k)
            last = k[0] + k[1]
        groups.append(cur)
        groups = [g for g in groups if len(g) >= 10][-len(CONFIGS):]
        rows, flops = [], 2 * n ** 3
        for cfg, g in zip(CONFIGS, groups):
            tot = {}
            for _, d, name in g:
                tot[name] = tot.get(name, 0) + d
            gemm = max(tot, key=tot.get)
            for i, (_, d, name) in enumerate(k for k in g if k[2] == gemm):
                rows.append(rl.mandatory(reg.run_id, i, clk) + [cfg, n, name[:120], d, flops, flops / (d * 1e-9),
                                                                 os.path.basename(base)])
            best = max(r[-2] for r in rows if r[len(rl.MANDATORY)] == cfg)
            print(f"size {n} {cfg:12s} best {best / 1e12:7.1f} TFLOP/s ({gemm[:60]})")
        rl.append_rows(os.path.join(rl.RES, "ceilings", "tc_kernel.csv"), HDR, rows)
    reg.complete()


if __name__ == "__main__":
    main()
