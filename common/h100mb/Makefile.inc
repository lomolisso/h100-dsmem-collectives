# Include from experiments/<NN__name>/Makefile:
#   EXP   := 04__memory_hierarchy_latency
#   BINS  := lat_smem lat_footprint lat_fine        (one .cu per binary under src/)
#   include ../harness/Makefile.inc
# Vendored copy (h100-dsmem-collectives/common/h100mb): headers resolved through H100MB_INC (see PROVENANCE.md).
H100MB_INC := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
NVCC      ?= nvcc
ARCH      ?= sm_90a
NVCCFLAGS ?= -arch=$(ARCH) -O3 -lineinfo -std=c++17 -Xptxas -v -Xptxas -warn-spills -I$(H100MB_INC)
# Link libcuda/libnvidia-ml through the toolkit stubs (the host has only the .so.1 runtime libraries);
# at run time the driver's real .so.1 is loaded by soname.
CUDA_HOME ?= $(abspath $(dir $(shell command -v $(NVCC)))..)
LDFLAGS   ?= -L$(CUDA_HOME)/lib64/stubs -lcuda -lnvidia-ml
BIN_DIR   := bin
SASS_DIR  := results/sass

.PHONY: all run sass clean
all: $(addprefix $(BIN_DIR)/,$(BINS)) sass

$(BIN_DIR)/%: src/%.cu $(H100MB_INC)/harness.cuh $(H100MB_INC)/host.hpp | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# SASS of every binary is a deliverable (shown in the notebook).
sass: $(addprefix $(BIN_DIR)/,$(BINS)) | $(SASS_DIR)
	@for b in $(BINS); do cuobjdump -sass $(BIN_DIR)/$$b > $(SASS_DIR)/$$b.sass; done

$(SASS_DIR):
	mkdir -p $(SASS_DIR)

# Experiments define the `run` recipe themselves (sweeps, ncu invocations) after including this file.

clean:
	rm -rf $(BIN_DIR)
