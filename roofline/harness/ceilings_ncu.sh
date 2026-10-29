#!/usr/bin/env bash
# roofline R1: Nsight Compute cross-checks of the ceiling kernels (one profiled launch per configuration).
# Called by harness/ceilings/run_all.sh inside
#   roofline/tools/run --profile --lock-sm-mhz 1635 -- bash roofline/harness/ceilings/run_all.sh
# (or alone: ... -- bash roofline/harness/ceilings_ncu.sh). Needs the binaries (make -C roofline/harness/ceilings) and,
# for the GEMMs, torch (roofline image).
#
# Template: MB/experiments/05__memory_hierarchy_bandwidth/src/run.sh (available() filter against ncu --query-metrics,
# ncu -o + ncu -i --csv --page raw --print-units base, join to a launch manifest written by the profiled program).
# Flags (DESIGN §3 / docs/REUSE.md §4): --clock-control none (keeps the h100 run lock), --cache-control all, kernel
# replay; never --print-metric-name. One report per profiled program, named with a time stamp (never overwritten).
#   compute   bin/compute_peak ncu   -k regex:_peak     sm__sass_thread_inst_executed_op_ffma_pred_on.sum (FFMA count of
#             ffma_peak; expected = manifest inst_total + 1 set-up FFMA per thread), integer / pipe counters for imad_peak
#   hbm       bin/bw_hbm ncu         -k regex:k4        dram__bytes_read.sum / dram__bytes_write.sum (expected: payload;
#             write lines still dirty in L2 at kernel end are missing from dram__bytes_write: MB 05 hbm_write_factor)
#   dsmem     bin/dsmem_bw ncu       -k regex:k_stream  l1tex__m_l1tex2xbar_write_bytes_mem_dshared.sum and
#             l1tex__m_xbar2l1tex_read_bytes_mem_dshared.sum (+ _op_ld / _op_tma_st): expected = payload bytes in each
#             (collectives C1, EXPERIMENTS §1; MB 11)
#   tc        python3 harness/ceilings_tc.py --ncu <config>, --profile-from-start off (only the GEMM between
#             torch.cuda.profiler.start/stop is profiled), all its kernels summed: sm__inst_executed_pipe_tensor_op_hmma.sum
#             and the gmma (wgmma) variant if listed
#   tcpeak    bin/tc_peak ncu        -k regex:tc_peak   sm__inst_executed_pipe_tensor_op_gmma.sum (expected: 4 warps × the
#             manifest's warpgroup-level wgmma_total, if ncu counts warp-level instructions) and the gmma pipe utilisation
#             (sm__pipe_tensor_op_gmma_cycles_active.avg.pct_of_peak_sustained_active), one launch per kind (fp16, fp16acc,
#             bf16, fp8) at N 256, 1 accumulator, wait_group 1, 2 warpgroups
# → results/ceilings/ncu/<step>_<stamp>.ncu-rep, .raw.csv, ncu_<step>.csv (long format, harness/ceilings_ncu_post.py),
#   query_metrics_<stamp>.txt.
# Usage: harness/ceilings_ncu.sh [step ...]   steps: compute hbm dsmem tc tcpeak (default: all)
set -euo pipefail
cd "$(dirname "$0")/.."   # roofline/
STEPS="${*:-compute hbm dsmem tc tcpeak}"
want() { [[ " $STEPS " == *" $1 "* ]]; }
B=harness/ceilings/bin
OUT=results/ceilings/ncu
mkdir -p "$OUT"
STAMP=$(date +%Y%m%dT%H%M%S)
Q=$OUT/query_metrics_$STAMP.txt
ncu --query-metrics > "$Q" 2>&1 || true

available() {  # keep the metrics whose base name the device lists (MB 05 run.sh)
    local out="launch__grid_size,launch__block_size,launch__cluster_dim_x,profiler__replayer_passes"
    for m in $1; do
        if grep -q "^${m%%.*} " "$Q"; then out="$out,$m"; else echo "  $m: NOT LISTED (dropped)" >&2; fi
    done
    echo "$out"
}
COMMON="gpu__time_duration.sum sm__cycles_elapsed.avg.per_second sm__cycles_elapsed.max"

# prof <step> <metrics> <manifest> <post flags> -- <ncu flags...> -- <command...>
prof() {
    local step=$1 metrics=$2 manifest=$3 post=$4; shift 4
    local nflags=()
    while [[ $1 != -- ]]; do nflags+=("$1"); shift; done
    shift
    local rep=$OUT/${step}_$STAMP
    echo "== ncu $step: $*"
    ncu --clock-control none --cache-control all "${nflags[@]}" --metrics "$metrics" -o "$rep" "$@"
    ncu -i "$rep.ncu-rep" --csv --page raw --print-units base > "$rep.raw.csv"
    python3 harness/ceilings_ncu_post.py --ncu-csv "$rep.raw.csv" --manifest "$manifest" --metrics "$metrics" \
        --report "$rep.ncu-rep" --out "$OUT/ncu_${step%%_*}.csv" $post
}

if want compute; then
    M=$(available "$COMMON sm__sass_thread_inst_executed_op_ffma_pred_on.sum sm__sass_thread_inst_executed_op_integer_pred_on.sum
sm__sass_thread_inst_executed.sum sm__inst_executed.sum sm__inst_executed_pipe_fma.sum sm__inst_executed_pipe_fmaheavy.sum
sm__inst_executed_pipe_fmalite.sum sm__inst_executed_pipe_alu.sum sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active
sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active")
    prof compute "$M" $OUT/manifest_compute_peak.csv "" -k regex:_peak -- $B/compute_peak ncu
fi

if want hbm; then
    M=$(available "$COMMON dram__bytes_read.sum dram__bytes_write.sum dram__throughput.avg.pct_of_peak_sustained_elapsed")
    prof hbm "$M" $OUT/manifest_bw_hbm.csv "" -k regex:k4 -- $B/bw_hbm ncu
fi

if want dsmem; then
    M=$(available "$COMMON l1tex__m_l1tex2xbar_write_bytes_mem_dshared.sum l1tex__m_xbar2l1tex_read_bytes_mem_dshared.sum
l1tex__m_l1tex2xbar_write_bytes_mem_dshared_op_ld.sum l1tex__m_xbar2l1tex_read_bytes_mem_dshared_op_ld.sum
l1tex__m_l1tex2xbar_write_bytes_mem_dshared_op_tma_st.sum l1tex__m_xbar2l1tex_read_bytes_mem_dshared_op_tma_st.sum
sm__sass_inst_executed_op_dshared_ld.sum sm__sass_inst_executed_op_dshared_tma_st.sum
l1tex__m_l1tex2xbar_write_bytes_mem_dshared.avg.peak_sustained l1tex__m_xbar2l1tex_read_bytes_mem_dshared.avg.peak_sustained")
    prof dsmem "$M" $OUT/manifest_dsmem_bw.csv "" -k regex:k_stream -- $B/dsmem_bw ncu
fi

if want tc; then
    M=$(available "$COMMON sm__inst_executed_pipe_tensor_op_hmma.sum sm__inst_executed_pipe_tensor_op_gmma.sum
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed sm__pipe_tensor_op_gmma_cycles_active.avg.pct_of_peak_sustained_elapsed
dram__bytes_read.sum")
    for cfg in fp16 bf16 fp8 fp8_fastacc; do
        prof "tc_$cfg" "$M" $OUT/manifest_tc.csv --aggregate --profile-from-start off -- python3 harness/ceilings_tc.py --ncu "$cfg"
    done
fi
if want tcpeak; then
    M=$(available "$COMMON sm__inst_executed_pipe_tensor_op_gmma.sum sm__inst_executed.sum
sm__pipe_tensor_op_gmma_cycles_active.avg.pct_of_peak_sustained_active sm__pipe_tensor_op_gmma_cycles_active.avg.pct_of_peak_sustained_elapsed")
    prof tcpeak "$M" $OUT/manifest_tc_peak.csv "" -k regex:tc_peak -- $B/tc_peak ncu
fi
echo "ncu cross-checks done ($STAMP)"
