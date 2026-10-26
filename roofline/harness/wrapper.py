"""Common CLI and loop of the per-manifest wrappers under roofline/harness/workloads/.

A wrapper defines `setup(args) -> state` (allocate inputs at the shape, build the callable), `step(state)` (exactly
one iteration of the variant: the kernels of `kernels` in the manifest, nothing else), and `check(state)` (run the
variant once and compare with the reference; raise or return False on a mismatch). `main(setup, step, check)`
parses `--variant --warmup --iters --validate` plus `--<shape key> <value>` options and runs:

    validate: check(state) -> exit 0 / 1, prints max errors
    otherwise: (warmup + iters) x [flush_l2(); step(state)], then one synchronize

The flush kernel delimits iterations in the nsys trace (run_manifest.py); no host timing is done here (time comes
from nsys / ncu kernel durations). Steps must not synchronize unless the artifact does so itself (say so in the
manifest notes)."""
import argparse
import sys

import torch

from flush import flush_l2


def parse(extra=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    for k, t in (extra or {}).items():
        ap.add_argument(f"--{k}", type=t, required=True)
    return ap.parse_args()


def main(setup, step, check, extra=None):
    args = parse(extra)
    torch.manual_seed(args.seed)
    state = setup(args)
    if args.validate:
        ok = check(state)
        torch.cuda.synchronize()
        print("VALIDATION", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)
    for _ in range(args.warmup + args.iters):
        flush_l2()
        step(state)
    torch.cuda.synchronize()
    print(f"DONE variant={args.variant} iters={args.warmup + args.iters}")
