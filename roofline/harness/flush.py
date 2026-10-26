"""L2 flush for Python wrappers: a Triton kernel named `roofline_l2_flush_kernel` that reads a clean, zeroed
buffer (default 256 MB, > 4x the 60 MiB L2), so the next kernel finds neither useful nor dirty lines in L2. Same
name and semantics as the CREDIT hook (patches/credit__roofline_hooks.patch): the runner uses it as the iteration
delimiter in the nsys trace (run_manifest.py)."""
import torch
import triton
import triton.language as tl

_BLOCK = 4096
_state = {}


@triton.jit
def roofline_l2_flush_kernel(buf, sink, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(buf + offs, mask=offs < n, other=0, cache_modifier=".cg")
    acc = tl.sum(v, axis=0)
    # never true for the zeroed buffer; keeps the loads alive
    tl.store(sink, acc, mask=acc == 0x5EED1E55)


def flush_l2(mb=256):
    st = _state.get(mb)
    if st is None:
        n = mb * (1 << 20) // 4
        st = _state[mb] = (torch.zeros(n, dtype=torch.int32, device="cuda"), torch.zeros(1, dtype=torch.int32,
                                                                                         device="cuda"), n)
        torch.cuda.synchronize()
    buf, sink, n = st
    roofline_l2_flush_kernel[(triton.cdiv(n, _BLOCK),)](buf, sink, n, BLOCK=_BLOCK)
