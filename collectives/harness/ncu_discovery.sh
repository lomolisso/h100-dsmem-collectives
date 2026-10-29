#!/usr/bin/env bash
# ncu_discovery.sh — C1 step 8: which DSMEM metrics exist on this device and which byte metric carries the payload
# of a push (bulk copy / plain remote store) and of a pull (remote loads). Run from the repo root inside
#   h100 run --profile --lock-sm-mhz <LOCK> -- bash collectives/harness/ncu_discovery.sh
# 1. ncu --query-metrics → results/ncu_query_metrics.txt (full list) and the dshared|cluster|cgas subset on stdout.
# 2. Candidate metrics of docs/EXPERIMENTS.md §1 (+ lts__t_sectors_srcunit_tex.sum, launch__cluster_*) filtered
#    through that list (the available() pattern of h100-microbench 05/run.sh), then one profiled launch per
#    configuration below (N = 4, M = 256 float: 1 KiB per rank and message): the bulk push alltoall__direct_push,
#    the plain-store push allgather__direct_replicated_push, the pull allgather__direct_pull, and the same-grid
#    all-exit baseline. ncu flags per docs/ENVIRONMENT.md §4 (--clock-control none keeps our lock; no
#    --print-metric-name with --page raw). Raw CSVs: results/ncu_discovery__<cfg>.csv (git-ignored).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
Q=results/ncu_query_metrics.txt
ncu --query-metrics > "$Q" 2>&1
echo "== metrics matching dshared|cluster|cgas ($(grep -c . "$Q") lines in $Q)"
grep -E -i "dshared|cluster|cgas" "$Q" | awk '{print $1}' | sort -u

CANDIDATES=(
  l1tex__m_l1tex2xbar_write_bytes_mem_dshared.sum
  l1tex__m_l1tex2xbar_write_bytes_mem_dshared_op_st.sum
  l1tex__m_l1tex2xbar_write_bytes_mem_dshared_op_atom.sum
  l1tex__m_l1tex2xbar_write_bytes_mem_dshared_op_redas.sum
  l1tex__m_l1tex2xbar_write_bytes_mem_dshared_op_tma_st.sum
  l1tex__m_l1tex2xbar_write_bytes_mem_dshared_op_ld.sum
  l1tex__m_xbar2l1tex_read_bytes_mem_dshared.sum
  l1tex__m_xbar2l1tex_read_bytes_mem_dshared_op_ld.sum
  l1tex__m_xbar2l1tex_read_bytes_mem_dshared_op_st.sum
  l1tex__m_xbar2l1tex_read_bytes_mem_dshared_op_tma_st.sum
  l1tex__t_requests_pipe_lsu_mem_dshared.sum
  l1tex__t_sectors_pipe_lsu_mem_dshared.sum
  l1tex__t_bytes_pipe_lsu_mem_dshared.sum
  l1tex__t_requests_pipe_lsu_mem_dshared_op_ld.sum
  l1tex__t_requests_pipe_lsu_mem_dshared_op_st.sum
  l1tex__t_bytes_pipe_lsu_mem_dshared_op_ld.sum
  l1tex__t_bytes_pipe_lsu_mem_dshared_op_st.sum
  sm__sass_inst_executed_op_dshared_ld.sum
  sm__sass_inst_executed_op_dshared_st.sum
  sm__sass_inst_executed_op_dshared_atom.sum
  sm__sass_inst_executed_op_dshared_redas.sum
  sm__sass_inst_executed_op_dshared_stas.sum
  sm__sass_inst_executed_op_dshared_tma_st.sum
  sm__sass_inst_executed_op_dshared_tma_red.sum
  l1tex__m_l1tex2xbar_throughput_pipe_lsu_mem_dshared.avg.pct_of_peak_sustained_elapsed
  l1tex__m_xbar2l1tex_throughput_pipe_lsu_mem_dshared.avg.pct_of_peak_sustained_elapsed
  lts__t_sectors_srcunit_tex.sum
  launch__cluster_dim_x
  launch__cluster_max_active
  gpc__cgas_launched.sum
  sm__ctas_launched.sum
  sm__cycles_elapsed.avg
)
avail=()
for m in "${CANDIDATES[@]}"; do
  base=${m%%.*}
  if grep -q -E "^${base}[[:space:]]" "$Q"; then avail+=("$m"); else echo "NOT LISTED: $m"; fi
done
METRICS=$(IFS=,; echo "${avail[*]}")
echo "== profiling with ${#avail[@]} metrics"

prof() {  # <cfg> <extra args...>
  local cfg=$1; shift
  local tag=${cfg}$( [[ " $* " == *" --baseline "* ]] && echo "__baseline" )
  ncu --clock-control none --cache-control all --kernel-name regex:"(collective|baseline)_kernel" \
      --metrics "$METRICS" --csv --page raw --print-units base \
      "build/$cfg" --mode ncu --reps 1 "$@" > "results/ncu_discovery__${tag}.csv"
  echo "-- $tag"
  python3 - "results/ncu_discovery__${tag}.csv" <<'EOF'
import csv, sys
rows = [r for r in csv.reader(open(sys.argv[1])) if r]
i = next(k for k, r in enumerate(rows) if r and r[0] == "ID")
hdr, unit = rows[i], rows[i + 1]
vals = [r for r in rows[i + 2:] if len(r) == len(hdr)]  # the app's own stdout lines are mixed in
for v in vals:
    for h, u, x in zip(hdr, unit, v):
        if "__" in h or h.startswith(("l1tex", "sm__", "lts", "gpc", "launch")):
            print(f"   {h:85s} {x:>14s} {u}")
EOF
}
prof alltoall__direct_push__generic__N4_M256_float
prof allgather__direct_replicated_push__credit__N4_M256_float
prof allgather__direct_pull__generic__N4_M256_float
prof allgather__direct_pull__generic__N4_M256_float --baseline
