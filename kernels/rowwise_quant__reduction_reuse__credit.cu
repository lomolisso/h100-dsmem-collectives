/*
 * Row-wise int8 quantisation (CREDIT reduction-reuse workload, clustered variant).
 * Source: third_party/CREDIT/src/workloads/rowwise_quant.cu (cluster kernel + device helpers; host benchmark stripped).
 *
 * 1. Computation (row-wise, M rows x N cols, fp32):
 *    amax = max |x|, scale = amax/127, q = clamp(round(x/scale), -127, 127)       (one stage: max |x|)
 *
 * 2. Split: one cluster per row (gridDim = rows * P). CTA b of the P-CTA cluster stages its column slice
 *    [b*slice_elems, (b+1)*slice_elems) once in shared memory (owner-local staging, no global re-reads)
 *    and computes per-slice partial statistics with warp shuffles + a block reduction.
 *
 * 3. Collective: every row statistic is an AllReduce(sum or max) of 1 float(s) across the P CTAs,
 *    done 1 time(s) per row with the direct replicated push (cluster_reduce_max): thread t < P writes
 *    the block partial into slot[b] of CTA t via cluster.map_shared_rank(), cluster.sync(), each CTA
 *    reduces its P local slots (warp shuffle), cluster.sync(). P*(P-1) remote scalar stores and 2 cluster
 *    barriers per stage; no remote reads. The final elementwise pass reuses the staged slice.
 */
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstddef>
#include <cstdint>

namespace cg = cooperative_groups;

constexpr int kBlockThreads = 256;
constexpr int kMaxClusterSize = 8;
constexpr float kQMax = 127.0f;

// ======================================================================
// verbatim from third_party/CREDIT/src/workloads/rowwise_quant.cu (reduction helpers)
// ======================================================================
template <int Width>
__device__ __forceinline__ float subwarp_reduce_max(float value) {
    for (int offset = Width / 2; offset > 0; offset >>= 1) {
        const float other = __shfl_xor_sync(0xffffffffu, value, offset, Width);
        value = fmaxf(value, other);
    }
    return value;
}

template <int BlockThreads>
__device__ __forceinline__ float block_reduce_max(float value) {
    constexpr int Warps = BlockThreads / 32;
    __shared__ float warp_values[Warps];
    __shared__ float broadcast;

    value = subwarp_reduce_max<32>(value);
    if ((threadIdx.x & 31) == 0) {
        warp_values[threadIdx.x >> 5] = value;
    }
    __syncthreads();

    float reduced = 0.0f;
    if (threadIdx.x < Warps) {
        reduced = warp_values[threadIdx.x];
    }
    if (threadIdx.x < 32) {
        reduced = subwarp_reduce_max<32>(reduced);
    }
    if (threadIdx.x == 0) {
        broadcast = reduced;
    }
    __syncthreads();
    return broadcast;
}

template <int BlockThreads, int MaxClusterSize>
__device__ __forceinline__ float cluster_reduce_max(cg::cluster_group cluster,
                                                     float block_value) {
    __shared__ float cluster_values[MaxClusterSize];
    __shared__ float broadcast;
    const int block_rank = cluster.block_rank();
    const int blocks_per_cluster = cluster.dim_blocks().x;

    if (threadIdx.x < blocks_per_cluster) {
        float* remote_value =
            // [DSMEM] generic DSMEM pointer: slot[my rank] inside CTA threadIdx.x's smem (replicated push)
            cluster.map_shared_rank(&cluster_values[block_rank], threadIdx.x);
        *remote_value = block_value;
    }
    // [DSMEM] barrier.cluster (release/acquire): pushed partials visible / slots reusable
    cluster.sync();

    float reduced = 0.0f;
    if (threadIdx.x < blocks_per_cluster) {
        reduced = cluster_values[threadIdx.x];
    }
    if (threadIdx.x < 32) {
        reduced = subwarp_reduce_max<32>(reduced);
    }
    if (threadIdx.x == 0) {
        broadcast = reduced;
    }
    __syncthreads();
    // [DSMEM] barrier.cluster (release/acquire): pushed partials visible / slots reusable
    cluster.sync();
    return broadcast;
}

__host__ __device__ __forceinline__ std::int8_t quantize_to_i8(float value,
                                                               float inv_scale) {
    float q = nearbyintf(value * inv_scale);
    q = fminf(kQMax, fmaxf(-kQMax, q));
    return static_cast<std::int8_t>(q);
}

// ======================================================================
// verbatim from third_party/CREDIT/src/workloads/rowwise_quant.cu (cluster kernel)
// ======================================================================
template <int BlockThreads, int MaxClusterSize>
__global__ void cluster_quant_staged_kernel(const float* __restrict__ input,
                                            std::int8_t* __restrict__ output,
                                            float* __restrict__ scales,
                                            int rows,
                                            int cols,
                                            int slice_elems) {
    extern __shared__ float tile[];
    cg::cluster_group cluster = cg::this_cluster();
    const int blocks_per_cluster = cluster.dim_blocks().x;
    const int block_rank = cluster.block_rank();
    const int row = blockIdx.x / blocks_per_cluster;
    const bool row_valid = row < rows;
    const int slice_start = block_rank * slice_elems;
    const int local_cols =
        row_valid ? max(0, min(slice_elems, cols - slice_start)) : 0;
    const std::size_t row_base = static_cast<std::size_t>(row) * cols;

    for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
        tile[col] = input[row_base + slice_start + col];
    }
    __syncthreads();

    float local_max = 0.0f;
    for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
        local_max = fmaxf(local_max, fabsf(tile[col]));
    }
    const float block_max = block_reduce_max<BlockThreads>(local_max);
    const float max_abs =
        // [DSMEM] AllReduce over the cluster: direct replicated push + local reduce
        cluster_reduce_max<BlockThreads, MaxClusterSize>(cluster, block_max);
    const float inv_scale = max_abs > 0.0f ? kQMax / max_abs : 0.0f;
    if (row_valid && block_rank == 0 && threadIdx.x == 0) {
        scales[row] = max_abs / kQMax;
    }
    if (!row_valid) {
        return;
    }
    for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
        output[row_base + slice_start + col] =
            quantize_to_i8(tile[col], inv_scale);
    }
    // [DSMEM] barrier.cluster (release/acquire): pushed partials visible / slots reusable
    cluster.sync();
}
