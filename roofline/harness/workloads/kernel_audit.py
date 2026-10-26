"""Kernel audit for --validate runs of the Python wrappers: runs one `step` under torch.profiler (CUPTI activity
trace, no hardware counters) and checks every kernel it launches against the manifest's `kernels` regexes of the
variant (for layer scope: the `scope: layer` variant `<id>_layer`), the same regexes run_manifest.py applies to the
nsys trace. A kernel that matches no regex and is not declared as deliberately excluded is UNMATCHED and fails the
validation, so the timed sum can never silently miss a kernel of the region (ADDING_A_WORKLOAD.md checklist 4).
Printed lines: `KERNEL <matched|excluded|UNMATCHED> <name>`, then `KERNELS_PER_ITER <n matched>`, which is the
kernels_n the timing run must report."""
import os
import re

import torch
import yaml

WORKLOADS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "workloads")


def regexes(manifest, variant_id):
    m = yaml.safe_load(open(os.path.join(WORKLOADS, manifest + ".yaml")))
    return list(next(x for x in m["scope_operator"]["variants"] if x["id"] == variant_id)["kernels"])


def audit(manifest, variant_id, fn, excluded=()):
    """Run fn() once under the profiler; returns True if every kernel is matched or excluded."""
    from torch.profiler import ProfilerActivity, profile
    kre = re.compile("|".join(f"(?:{k})" for k in regexes(manifest, variant_id)))
    xre = re.compile("|".join(f"(?:{k})" for k in excluded)) if excluded else None
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = []
    for e in prof.events():
        if str(getattr(e, "device_type", "")).endswith("CUDA") and not re.match(r"(Memcpy|Memset)", e.name):
            names.append(e.name)
    matched, ok = 0, True
    for n in names:
        if "roofline_l2_flush_kernel" in n:
            tag = "excluded"
        elif kre.search(n):
            tag, matched = "matched", matched + 1
        elif xre is not None and xre.search(n):
            tag = "excluded"
        else:
            tag, ok = "UNMATCHED", False
        print(f"KERNEL {tag} {n[:220]}")
    if not names:
        print("KERNEL audit: the profiler recorded no kernels (CUPTI unavailable?) -> audit FAILED")
        ok = False
    print(f"KERNELS_PER_ITER {matched} (manifest variant {variant_id}; {len(names)} kernels launched by one step)")
    return ok
