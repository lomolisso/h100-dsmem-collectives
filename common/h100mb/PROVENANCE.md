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

## roofline R1 copies (into roofline/harness/ceilings/, not vendored here)

Copied on 2026-10-26 by roofline task R1 from MB `HEAD` `8f8fc7bec69f07db49f3a5a95bbdfb08de5d0ca8`. These pieces live in
the study's harness (docs/REUSE.md §1: study-specific code goes into `<study>/harness/`); the vendored files above are
unchanged. Every copied block carries a comment naming its source; kernels are verbatim.

| Copy (roofline/harness/…) | Source (MB/experiments/…) | sha256 of the source | What was taken / changed |
|---|---|---|---|
| `ceilings/src/ceil_common.hpp` | `05__memory_hierarchy_bandwidth/src/bw_common.hpp` | `c6ef31118e06df8c3b25c85f24f6c2315adb179091e76fb589f5b3bbabe140c1` | BlockRec, V4, xor4, xor_into, CacheOp, ld_v4/st_v4, PlacementCheck, clean_read_kernel, CleanL2 verbatim; dropped the `gpc_map.hpp` include, `kPartitionOfGpc`, `partition_of_sm` (GPU-2 partition labels) |
| `ceilings/src/ceil_common.hpp` | `harness/harness.cuh` (newer than the vendored copy) | `b667a33bd67600f198518987a20f73de9205e07b3b78617bc75b180465dabe6b` | `ld_dsmem_v4_u32` verbatim |
| `ceilings/src/ceil_common.hpp`, `ceilings/src/dsmem_bw.cu` | `09__dsmem_latency_vs_cluster_size/src/common09.hpp` | `a80dd3ecd1703f816eae77bdea113ac6d807e5c3bbf6b3e16e0b1e7daddeda04` | `record_place`, `Placement`, `DevBuf` verbatim; dropped `gpc_of`, `Grid09`, `Discards` |
| `ceilings/src/event_floor.cu` | `01__timers_and_harness/src/timers.cu` | `5e8c227dd38a59a0515c6b74028c6e1c72711519e6cbe6e1e9d24507abb77b1f` | K5 (`k5_empty`, `run_k5`: single / batch / empty_pair); output file and a `clock_bad` column changed, `--quick` = 100 repetitions |
| `ceilings/src/bw_hbm.cu` | `05__memory_hierarchy_bandwidth/src/bw_dram.cu` | `72fa85adcc6d65953758f25e271fe158dcac082a9b30f71dacec9a6d076dedb4` | `read_step`, `write_step`, `k4_body`, `k4` verbatim (`c_part` never filled → 0); full grid only, ops read / write / mix11 (reported as "copy"), .cg, ILP 4; dropped GPC labels, per-block CSV, active-SM sweep, cs, other ILPs, mix, diag |
| `ceilings/src/bw_l2.cu` | `05__memory_hierarchy_bandwidth/src/bw_l2.cu` | `26a5fb90f4f43fd02c4663a459df205a65f695734a09f391d7bbb862b56fae56` | `k3_direct`, `StreamArgs`, `warp_range`, `access_step`, `stream_body`, `k3_stream` verbatim (`c_part`/`c_gpc` never filled → 0, `reader_gpcs` 0xFF = MB "mixed"); dropped home labelling, lists, link / first-pass / store-copy / ncu modes |
| `ceilings/src/bw_smem.cu` | `05__memory_hierarchy_bandwidth/src/bw_smem_l1.cu` | `bbc881765713c06ad5cb5964877edbdb69f0f07d0d9f1819b28cef86afdc5d72` | `k12` with `Words`, `consume`, `ld_one`, `st_one`, `slot_off` verbatim; only smem / ld / 16 B / stride / all_sms instantiated |
| `ceilings/src/dsmem_bw.cu` | `11__dsmem_bandwidth_vs_cluster_size/src/{common11.hpp,dsmem_bw.cu}`, `harness/harness.cuh` | `64e8038403a11cf43b2cfb30e20ee42c8c11a0a2cab2f1cff0024edc9496a4b3`, `d4352a6b8824cc5fe19fb3acea8c5f14d5df290ceacab03d352964adeae66790` | common11 patterns/plan verbatim (without `Discards11`); `k_stream` and its stream helpers verbatim; `st_dsmem_v4_u32`, `st_async_v4_b32`, `bulk_red_s2s_add_u32` verbatim; `kFlagOff` dropped (K4 only); host = run_point launch + step S, ld/tma_put only |
| `ceilings_ncu.sh` | `05__memory_hierarchy_bandwidth/src/run.sh` | `0a031f0a9296dd2828e66cf69ea3e8877f295b8642f6698960e7dbbd3e8edcbc` | `available()` filter and the ncu -o / -i --csv --page raw --print-units base pattern |
| `ceilings_ncu_post.py` | `05__memory_hierarchy_bandwidth/src/post_ncu.py` | `8c9256f5657ddfe8aff0ab03c68e5459a3e97856aef98d1641e68110ca3894a0` | manifest join by launch order; long output format, latest manifest run only, `--aggregate` |
| `ceilings_summary.py` | `05__memory_hierarchy_bandwidth/src/common05.py` | `c443ba89a9872ccd91565b11c5006c8d464a563cbe681961dadb630fc5bcebd2` | methods of `latest_run`, `event_floor_ms`, `net_tbps`, `k12_slope`, `hbm_write_factor` re-implemented (not copied verbatim) |
