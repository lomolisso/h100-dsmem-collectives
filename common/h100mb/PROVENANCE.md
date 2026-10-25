# Provenance of the vendored h100-microbench harness

Copied on 2026-10-25 by collectives task C1 (first study to vendor; roofline R1 only verifies this copy),
following `docs/REUSE.md` §1/§4.

- Source repository: `~/work/fa2026/iacoma/h100-microbench` (MB below)
- MB `HEAD` at copy time: `14875a5cc720cb69532e4fcebada0ebfd4c9005b`
- Last MB commit touching `experiments/harness/`: `4d73449aa30e16e67111cfc48b89814db63f4d3f` (2026-10-24)

| File here | Source | sha256 of the source | Local edits |
|---|---|---|---|
| `host.hpp` | `MB/experiments/harness/host.hpp` | `880732f22a22340ecef39823ec82d162b96c8d2abb9bd1b7c6a523a95fb33db1` | none |
| `harness.cuh` | `MB/experiments/harness/harness.cuh` | `b971b3c2b4be58d58f0d92169fc193b61055bd37f3a800eab90d2a34f6c58a27` | none |
| `Makefile.inc` | `MB/experiments/harness/Makefile.inc` | `217bcc0e9048090d19463ea0534f66bb9a8553c6fa4478aa162919f46f26a540` | path fix only (below) |

`Makefile.inc` path fix (REUSE §4): the variable `H100MB_INC := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))`
(this directory) replaces `../harness` in `-I` and in the header prerequisites of the `bin/%` rule. Everything
else is byte-identical.

Pieces of `MB/experiments/07__synchronization_primitives/src/sync_common.hpp` used by a study are copied
into that study's `harness/` (with a comment naming the source), not here.

To pick up a later MB fix: re-copy the file, update this table (commit, checksum), and say so in the commit
message.
