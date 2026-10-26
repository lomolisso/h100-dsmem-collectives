#!/usr/bin/env bash
# roofline R1: every ceiling measurement on the GPU, in order. From the repo root (GPU 3, locked SM clock):
#   roofline/tools/run --profile --lock-sm-mhz 1635 -- bash roofline/harness/ceilings/run_all.sh [--quick] [step ...]
# Build first, without the GPU:
#   roofline/tools/run --no-gpu -- make -C roofline/harness/ceilings
# Summary afterwards, without the GPU (writes results/ceilings.csv):
#   roofline/tools/run --no-gpu -- python3 roofline/harness/ceilings_summary.py
#
# Runs from roofline/ (hb::Run registry results/roofline__runs.csv, clock log results/roofline__clock_log.csv; raw CSVs
# results/ceilings/*.csv). Every binary invocation has its own run_id.
# Steps (default: all, in this order):
#   floor    event_floor floor          cudaEvent floor (empty pair), MB 01 K5                       ≈ 5 s
#   hbm      bw_hbm k4                  read / write / copy, 2 GiB, CleanL2 before each launch       ≈ 15 s
#   l2       bw_l2 k3                   k3_direct and k3_stream at 16 MiB                            ≈ 10 s
#   smem     bw_smem k12                24 configurations × 12+ launches                             ≈ 15 s
#   dsmem    dsmem_bw S                 {ld, tma_put} × CS {2,4,8,16} × passes {4,8,16,32}           ≈ 30 s
#   compute  compute_peak peak          ffma / imad × ILP {1,2,4,8}                                  ≈ 15 s
#   tc       ceilings_tc.py             fp16, bf16, fp8, fp8_fastacc 8192³ GEMMs (torch import ≈ 10 s) ≈ 1 min
#   tcpeak   tc_peak peak               wgmma instruction-throughput peaks: fp16 / fp16acc / bf16 / fp8 × N {128, 256}
#                                       × accumulators {1, 2} × wait_group {0, 1} × warpgroups {1, 2, 3} (78 configs × 6
#                                       launches, 1 s idle before each launch against the power limit) + 4 rand-fill ≈ 9 min
#   ncu      ceilings_ncu.sh            compute, hbm, dsmem, 4 × tc, 4 × tc_peak under ncu           ≈ 5–10 min
# (each step also pays one `nvidia-smi -q` before and after; times are estimates, not measured yet)
# Clock checks (event_floor clockcheck) at the start and the end.
set -euo pipefail
cd "$(dirname "$0")/../.."   # roofline/

Q=""
if [[ "${1:-}" == "--quick" ]]; then Q="--quick"; shift; fi
STEPS="${*:-floor hbm l2 smem dsmem compute tc tcpeak ncu}"
want() { [[ " $STEPS " == *" $1 "* ]]; }
B=harness/ceilings/bin
for b in event_floor bw_hbm bw_l2 bw_smem dsmem_bw compute_peak tc_peak; do
    [[ -x $B/$b ]] || { echo "ERROR: $B/$b missing — build first: roofline/tools/run --no-gpu -- make -C roofline/harness/ceilings" >&2; exit 1; }
done
if [[ -z "${H100MB_LOCK_SM_MHZ:-}" && -z "$Q" ]]; then
    echo "ERROR: no SM clock lock (H100MB_LOCK_SM_MHZ unset): run under tools/run --profile --lock-sm-mhz 1635" >&2
    exit 1
fi
mkdir -p results/ceilings/ncu
echo "== roofline R1 ceilings: steps [$STEPS] $Q, lock ${H100MB_LOCK_SM_MHZ:-none} MHz, GPU ${H100MB_GPU_UUID:-?}, $(date -Is)"

$B/event_floor clockcheck --label ceilings_start $Q
if want floor;   then echo "== event floor";            $B/event_floor floor $Q; fi
if want hbm;     then echo "== HBM (k4)";               $B/bw_hbm k4 $Q; fi
if want l2;      then echo "== L2 (k3_direct, k3_stream)"; $B/bw_l2 k3 $Q; fi
if want smem;    then echo "== SMEM (k12 slope)";       $B/bw_smem k12 $Q; fi
if want dsmem;   then echo "== DSMEM (k_stream, full)"; $B/dsmem_bw S $Q; fi
if want compute; then echo "== FFMA / IMAD peaks";      $B/compute_peak peak $Q; fi
if want tc;      then echo "== tensor cores (torch)";   python3 harness/ceilings_tc.py $Q; fi
if want tcpeak;  then echo "== tensor cores (wgmma)";   $B/tc_peak peak $Q; fi
if want ncu;     then echo "== ncu cross-checks";       bash harness/ceilings_ncu.sh; fi
$B/event_floor clockcheck --label ceilings_end $Q
echo "== done $(date -Is). Summary (no GPU): roofline/tools/run --no-gpu -- python3 roofline/harness/ceilings_summary.py"
