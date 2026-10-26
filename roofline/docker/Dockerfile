# Roofline-study image: the h100-microbench toolchain image plus PyTorch, flashinfer and the build
# dependencies of the artifacts in third_party/. Build with `roofline/tools/build_image.sh build`; never
# `docker run` it directly — use `roofline/tools/run`, which goes through the microbench tools/run
# (GPU preflight, lock, clock lock, device restriction).
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Packages on top of the microbench venv, fully pinned (see requirements.txt for the intent).
COPY requirements.lock /opt/roofline-requirements.lock
RUN /opt/venv/bin/pip install --no-cache-dir -r /opt/roofline-requirements.lock \
    && /opt/venv/bin/pip check

# CUTLASS 4.4.2 headers for DAK (its setup.sh installs CUTLASS through conda); taken from the copy vendored
# by hpc-ops at its pinned commit (named build context `cutlass`).
COPY --from=cutlass include /opt/cutlass/include

RUN { echo "# roofline additions"; /opt/venv/bin/pip freeze; } >> /opt/env-manifest.txt

# JIT caches and PYTHONPATH depend on the checkout path and are set by roofline/tools/run. The CUDA driver
# stub lets extensions link -lcuda in --no-gpu builds.
ENV CUTLASS_INCLUDE=/opt/cutlass/include \
    LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
    TORCH_CUDA_ARCH_LIST=9.0a
