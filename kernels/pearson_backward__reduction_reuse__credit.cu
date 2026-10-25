/*
 * Pearson-correlation backward (dx, dy) (CREDIT reduction-reuse workload, clustered variant).
 * Source: third_party/CREDIT/src/workloads/pearson_backward.cu (cluster kernel + device helpers; host benchmark stripped).
 *
 * 1. Computation (row-wise, M rows x N cols, fp32):
 *    stats {sx, sy, sxx, syy, sxy}; mean_x, mean_y, var_x, var_y, cov; inv_x = rsqrt(var_x+eps), inv_y likewise
 *    dx = g*((y-mean_y)*inv_x*inv_y - (x-mean_x)*cov*inv_x^3*inv_y), dy symmetric            (one 5-scalar stage)
 *
 * 2. Split: one cluster per row (gridDim = rows * P). CTA b of the P-CTA cluster stages its column slice
 *    [b*slice_elems, (b+1)*slice_elems) once in shared memory (owner-local staging, no global re-reads)
 *    and computes per-slice partial statistics with warp shuffles + a block reduction.
 *
 * 3. Collective: every row statistic is an AllReduce(sum) of 5 float(s) across the P CTAs,
 *    done 1 time(s) per row with the direct replicated push (cluster_reduce_stats): thread t < P writes
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
constexpr float kEps = 1e-5f;

struct Stats {
    float sx;
    float sy;
    float sxx;
    float syy;
    float sxy;
};

// ======================================================================
// verbatim from third_party/CREDIT/src/workloads/pearson_backward.cu (reduction helpers)
// ======================================================================
template <int Width>
__device__ __forceinline__ float subwarp_reduce_sum(float value) {
    for (int offset = Width / 2; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset, Width);
    }
    return value;
}

template <int BlockThreads>
__device__ __forceinline__ Stats block_reduce_stats(Stats value) {
    constexpr int Warps = BlockThreads / 32;
    __shared__ float warp_sx[Warps];
    __shared__ float warp_sy[Warps];
    __shared__ float warp_sxx[Warps];
    __shared__ float warp_syy[Warps];
    __shared__ float warp_sxy[Warps];
    __shared__ Stats broadcast;

    value.sx = subwarp_reduce_sum<32>(value.sx);
    value.sy = subwarp_reduce_sum<32>(value.sy);
    value.sxx = subwarp_reduce_sum<32>(value.sxx);
    value.syy = subwarp_reduce_sum<32>(value.syy);
    value.sxy = subwarp_reduce_sum<32>(value.sxy);
    if ((threadIdx.x & 31) == 0) {
        const int warp = threadIdx.x >> 5;
        warp_sx[warp] = value.sx;
        warp_sy[warp] = value.sy;
        warp_sxx[warp] = value.sxx;
        warp_syy[warp] = value.syy;
        warp_sxy[warp] = value.sxy;
    }
    __syncthreads();

    Stats reduced{0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (threadIdx.x < Warps) {
        reduced.sx = warp_sx[threadIdx.x];
        reduced.sy = warp_sy[threadIdx.x];
        reduced.sxx = warp_sxx[threadIdx.x];
        reduced.syy = warp_syy[threadIdx.x];
        reduced.sxy = warp_sxy[threadIdx.x];
    }
    if (threadIdx.x < 32) {
        reduced.sx = subwarp_reduce_sum<32>(reduced.sx);
        reduced.sy = subwarp_reduce_sum<32>(reduced.sy);
        reduced.sxx = subwarp_reduce_sum<32>(reduced.sxx);
        reduced.syy = subwarp_reduce_sum<32>(reduced.syy);
        reduced.sxy = subwarp_reduce_sum<32>(reduced.sxy);
    }
    if (threadIdx.x == 0) {
        broadcast = reduced;
    }
    __syncthreads();
    return broadcast;
}

template <int BlockThreads, int MaxClusterSize>
__device__ __forceinline__ Stats cluster_reduce_stats(cg::cluster_group cluster,
                                                       Stats block_value) {
    __shared__ float cluster_sx[MaxClusterSize];
    __shared__ float cluster_sy[MaxClusterSize];
    __shared__ float cluster_sxx[MaxClusterSize];
    __shared__ float cluster_syy[MaxClusterSize];
    __shared__ float cluster_sxy[MaxClusterSize];
    __shared__ Stats broadcast;
    const int block_rank = cluster.block_rank();
    const int blocks_per_cluster = cluster.dim_blocks().x;

    if (threadIdx.x < blocks_per_cluster) {
        float* remote_sx =
            // [DSMEM] generic DSMEM pointer: slot[my rank] inside CTA threadIdx.x's smem (replicated push)
            cluster.map_shared_rank(&cluster_sx[block_rank], threadIdx.x);
        float* remote_sy =
            // [DSMEM] generic DSMEM pointer: slot[my rank] inside CTA threadIdx.x's smem (replicated push)
            cluster.map_shared_rank(&cluster_sy[block_rank], threadIdx.x);
        float* remote_sxx =
            // [DSMEM] generic DSMEM pointer: slot[my rank] inside CTA threadIdx.x's smem (replicated push)
            cluster.map_shared_rank(&cluster_sxx[block_rank], threadIdx.x);
        float* remote_syy =
            // [DSMEM] generic DSMEM pointer: slot[my rank] inside CTA threadIdx.x's smem (replicated push)
            cluster.map_shared_rank(&cluster_syy[block_rank], threadIdx.x);
        float* remote_sxy =
            // [DSMEM] generic DSMEM pointer: slot[my rank] inside CTA threadIdx.x's smem (replicated push)
            cluster.map_shared_rank(&cluster_sxy[block_rank], threadIdx.x);
        *remote_sx = block_value.sx;
        *remote_sy = block_value.sy;
        *remote_sxx = block_value.sxx;
        *remote_syy = block_value.syy;
        *remote_sxy = block_value.sxy;
    }
    // [DSMEM] barrier.cluster (release/acquire): pushed partials visible / slots reusable
    cluster.sync();

    Stats reduced{0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (threadIdx.x < blocks_per_cluster) {
        reduced.sx = cluster_sx[threadIdx.x];
        reduced.sy = cluster_sy[threadIdx.x];
        reduced.sxx = cluster_sxx[threadIdx.x];
        reduced.syy = cluster_syy[threadIdx.x];
        reduced.sxy = cluster_sxy[threadIdx.x];
    }
    if (threadIdx.x < 32) {
        reduced.sx = subwarp_reduce_sum<32>(reduced.sx);
        reduced.sy = subwarp_reduce_sum<32>(reduced.sy);
        reduced.sxx = subwarp_reduce_sum<32>(reduced.sxx);
        reduced.syy = subwarp_reduce_sum<32>(reduced.syy);
        reduced.sxy = subwarp_reduce_sum<32>(reduced.sxy);
    }
    if (threadIdx.x == 0) {
        broadcast = reduced;
    }
    __syncthreads();
    // [DSMEM] barrier.cluster (release/acquire): pushed partials visible / slots reusable
    cluster.sync();
    return broadcast;
}

// ======================================================================
// verbatim from third_party/CREDIT/src/workloads/pearson_backward.cu (cluster kernel)
// ======================================================================
template <int BlockThreads, int MaxClusterSize>
__global__ void cluster_bwd_staged_kernel(const float* __restrict__ x,
                                          const float* __restrict__ y,
                                          const float* __restrict__ grad,
                                          float* __restrict__ dx,
                                          float* __restrict__ dy,
                                          int rows,
                                          int cols,
                                          float eps,
                                          int slice_elems) {
    extern __shared__ float tile[];
    float* tile_x = tile;
    float* tile_y = tile + slice_elems;
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
        tile_x[col] = x[row_base + slice_start + col];
        tile_y[col] = y[row_base + slice_start + col];
    }
    __syncthreads();

    Stats stats{0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
        const float xv = tile_x[col];
        const float yv = tile_y[col];
        stats.sx += xv;
        stats.sy += yv;
        stats.sxx += xv * xv;
        stats.syy += yv * yv;
        stats.sxy += xv * yv;
    }
    const Stats block_stats = block_reduce_stats<BlockThreads>(stats);
    const Stats row_stats =
        // [DSMEM] AllReduce over the cluster: direct replicated push + local reduce
        cluster_reduce_stats<BlockThreads, MaxClusterSize>(cluster, block_stats);
    const float inv_cols = 1.0f / static_cast<float>(cols);
    const float mean_x = row_stats.sx * inv_cols;
    const float mean_y = row_stats.sy * inv_cols;
    const float var_x = fmaxf(row_stats.sxx - row_stats.sx * mean_x, 0.0f);
    const float var_y = fmaxf(row_stats.syy - row_stats.sy * mean_y, 0.0f);
    const float cov = row_stats.sxy - row_stats.sx * mean_y;
    const float inv_x = rsqrtf(var_x + eps);
    const float inv_y = rsqrtf(var_y + eps);
    const float inv_xy = inv_x * inv_y;
    const float coeff_x = cov * inv_x * inv_x * inv_x * inv_y;
    const float coeff_y = cov * inv_x * inv_y * inv_y * inv_y;
    const float grad_row = row_valid ? grad[row] : 0.0f;

    if (!row_valid) {
        return;
    }
    for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
        const int global_col = slice_start + col;
        const float xc = tile_x[col] - mean_x;
        const float yc = tile_y[col] - mean_y;
        dx[row_base + global_col] = grad_row * (yc * inv_xy - xc * coeff_x);
        dy[row_base + global_col] = grad_row * (xc * inv_xy - yc * coeff_y);
    }
}
