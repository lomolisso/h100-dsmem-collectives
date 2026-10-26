"""R0 smoke test of the roofline image on the GPU: device, a 4096^3 fp16 matmul checked against fp32,
flashinfer import + one JIT-compiled op checked against torch, and the versions DESIGN.md §6 records.
Prints one JSON object; exits non-zero if a check fails."""
import json
import subprocess
import sys

import torch

out = {}
smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,clocks.sm,clocks.mem,serial",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
out["nvidia_smi"] = smi
out["device"] = torch.cuda.get_device_name()
out["torch"] = torch.__version__
out["torch_cuda"] = torch.version.cuda
out["cudnn"] = torch.backends.cudnn.version()

a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
c = a @ b
ref = a.float() @ b.float()
out["matmul_max_rel_err"] = ((c.float() - ref).abs().max() / ref.abs().max()).item()
start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
for _ in range(3):
    a @ b
start.record()
for _ in range(10):
    a @ b
end.record()
torch.cuda.synchronize()
out["matmul_tflops_rough"] = 10 * 2 * 4096**3 / (start.elapsed_time(end) * 1e-3) / 1e12

import flashinfer  # noqa: E402
import triton  # noqa: E402

out["flashinfer"] = flashinfer.__version__
out["triton"] = triton.__version__
x = torch.randn(64, 8192, device="cuda", dtype=torch.float16)
w = torch.randn(8192, device="cuda", dtype=torch.float16)
y = flashinfer.norm.rmsnorm(x, w, eps=1e-6)
xf = x.float()
y_ref = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()
out["flashinfer_rmsnorm_max_abs_err"] = (y.float() - y_ref).abs().max().item()

ok = (out["device"].startswith("NVIDIA H100 NVL") and out["matmul_max_rel_err"] < 1e-2
      and out["flashinfer_rmsnorm_max_abs_err"] < 5e-2)
out["ok"] = ok
print(json.dumps(out, indent=1))
sys.exit(0 if ok else 1)
