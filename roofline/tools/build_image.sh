#!/usr/bin/env bash
# Roofline image management (derived from the h100-microbench image).
#   roofline/tools/build_image.sh tag     print the image name for the current Dockerfile + lock
#   roofline/tools/build_image.sh lock    resolve docker/requirements.txt -> docker/requirements.lock
#   roofline/tools/build_image.sh build   build and tag the image; remove only our superseded roofline images
#
# Tag = content hash of the Dockerfile, the lock, the base image and the hpc-ops commit (CUTLASS headers).
# The image is labelled roofline.owner=$USER and h100mb.owner is blanked: the derived image would otherwise
# inherit the microbench label, and `tools/env build` in h100-microbench removes every image carrying it.
# Never prunes.
set -euo pipefail
MB=$HOME/work/fa2026/iacoma/h100-microbench
source "$MB/tools/common.sh"            # containerd_free_gb, MIN_BUILD_FREE_GB, say, die
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
DOCKER_DIR=$REPO/roofline/docker
CUTLASS_SRC=$REPO/third_party/hpc-ops/3rd/cutlass

BASE_IMAGE=h100-microbench:cuda12.9.1-26d4f436f8c5   # pinned microbench toolchain image
ROOF_REPO=h100-roofline

image_name() {
    local hpc tag
    hpc=$(git -C "$REPO/third_party/hpc-ops" rev-parse HEAD 2>/dev/null) || die "third_party/hpc-ops not initialised"
    tag=$( { cat "$DOCKER_DIR/Dockerfile" "$DOCKER_DIR/requirements.lock"; echo "$BASE_IMAGE $hpc"; } | sha256sum | cut -c1-12)
    echo "$ROOF_REPO:cuda12.9.1-$tag"
}

cmd_lock() {
    # Resolve in the base image on top of its venv; the venv lives on NFS (torch + CUDA wheels exceed the
    # container's 8 GB RAM /tmp). Only packages that are new or changed w.r.t. /opt/venv are frozen.
    local env=$REPO/roofline/build/lockenv
    [[ -e "$env" ]] && die "$env exists (left from an earlier lock); remove it first"
    mkdir -p "$REPO/roofline/build"
    (cd "$REPO" && H100MB_IMAGE="$BASE_IMAGE" "$MB/tools/run" --no-gpu -- bash -c "
        set -e
        python3 -m venv --system-site-packages '$env'
        TMPDIR='$REPO/roofline/build' '$env/bin/pip' install -q --no-cache-dir -r '$DOCKER_DIR/requirements.txt'
        { echo '# Fully pinned additions to the h100-microbench venv (pip freeze --local, resolved $(date +%F), Python 3.12).'
          echo '# Intent: requirements.txt; regenerate with: roofline/tools/build_image.sh lock'
          echo '--extra-index-url https://download.pytorch.org/whl/cu129'
          '$env/bin/pip' freeze --local; } > '$DOCKER_DIR/requirements.lock'")
    echo "wrote $DOCKER_DIR/requirements.lock (now: roofline/tools/build_image.sh build)"
    echo "the resolver venv $env can be removed (build tree, not results)"
}

cmd_build() {
    local free img cur old
    [[ -f "$DOCKER_DIR/requirements.lock" ]] || die "no requirements.lock: run '$0 lock' first"
    docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || die "base image $BASE_IMAGE not present"
    [[ -d "$CUTLASS_SRC/include/cute" ]] || die "$CUTLASS_SRC missing (git submodule update --init)"
    free=$(containerd_free_gb)
    (( free >= MIN_BUILD_FREE_GB )) || die "container store has ${free} GB free; need >= ${MIN_BUILD_FREE_GB} GB"
    img=$(image_name)
    docker build -f "$DOCKER_DIR/Dockerfile" --build-arg BASE_IMAGE="$BASE_IMAGE" \
        --build-context cutlass="$CUTLASS_SRC" \
        --label h100mb.owner= --label roofline.owner="$USER" --label roofline.base="$BASE_IMAGE" \
        -t "$img" "$DOCKER_DIR"
    cur=$(docker image inspect --format '{{.Id}}' "$img")
    for old in $(docker images -q --no-trunc --filter "label=roofline.owner=$USER" | sort -u); do
        [[ "$old" == "$cur" ]] && continue
        if docker rmi "$old" >/dev/null 2>&1; then echo "removed old roofline image ${old:7:12}"; fi
    done
    echo "built $img"
}

case ${1:-} in
    tag) image_name ;;
    lock) cmd_lock ;;
    build) cmd_build ;;
    *) sed -n '2,10p' "$0"; exit 2 ;;
esac
