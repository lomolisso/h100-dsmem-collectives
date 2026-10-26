# roofline R1 ceiling microbenchmarks. Build (from the repo root, no GPU):
#   ~/work/fa2026/iacoma/h100-microbench/tools/run --no-gpu -- make -C roofline/harness/ceilings
#   (or roofline/tools/run --no-gpu -- make -C roofline/harness/ceilings)
# → bin/<name> for every src/<name>.cu, and results/sass/<name>.sass (cuobjdump -sass, `make sass`); both git-ignored
#   (.gitignore here). Rules, flags (-arch=sm_90a -O3 -lineinfo -std=c++17 -Xptxas -v -Xptxas -warn-spills) and the
#   NVML/CUDA stub link come from the vendored common/h100mb/Makefile.inc.
# Run everything on the GPU: roofline/tools/run --profile --lock-sm-mhz 1635 -- bash roofline/harness/ceilings/run_all.sh
BINS := event_floor bw_hbm bw_l2 bw_smem dsmem_bw compute_peak
include ../../../common/h100mb/Makefile.inc

# local shared header (Makefile.inc's bin/% rule only lists the vendored headers)
$(addprefix $(BIN_DIR)/,$(BINS)): src/ceil_common.hpp
