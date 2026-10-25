/*
 * Weighted-variance backward (CREDIT reduction-reuse workload, clustered variant).
 * Source: third_party/CREDIT/src/workloads/weighted_var_backward.cu (cluster kernel + device helpers; host benchmark stripped).
 *
 * 1. Computation (row-wise, M rows x N cols, fp32):
 *    sw = sum(w), mean = sum(w*x)/sw, var = sum(w*(x-mean)^2)/sw, inv_std = rsqrt(var+eps)
 *    dx = inv_std*(dy - w/sw*sum(dy) - w*(x-mean)/var/sw*sum(dy*(x-mean)))   (three stages: {sw, sum wx}, {sum w(x-mean)^2}, {sum dy, sum dy(x-mean)})
 *
 * 2. Split: one cluster per row (gridDim = rows * P). CTA b of the P-CTA cluster stages its column slice
 *    [b*slice_elems, (b+1)*slice_elems) once in shared memory (owner-local staging, no global re-reads)
 *    and computes per-slice partial statistics with warp shuffles + a block reduction.
 *
 * 3. Collective: every row statistic is an AllReduce(sum) of 2 float(s) across the P CTAs,
 *    done 3 time(s) per row with the direct replicated push (cluster_reduce_pair): thread t < P writes
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
constexpr float kEps = 1.0e-6f;

struct Pair {
    float sum;
    float sumsq;
};

// ======================================================================
// verbatim from third_party/CREDIT/src/workloads/weighted_var_backward.cu (reduction helpers)
// ======================================================================
template <int Width>
__device__ __forceinline__ float subwarp_reduce_sum(float value) {
    for (int offset = Width / 2; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset, Width);
    }
    return value;
}

template <int BlockThreads>
__device__ __forceinline__ Pair block_reduce_pair(Pair value) {
    constexpr int Warps = BlockThreads / 32;
    __shared__ float warp_sum[Warps];
    __shared__ float warp_sumsq[Warps];
    __shared__ Pair broadcast;

    value.sum = subwarp_reduce_sum<32>(value.sum);
    value.sumsq = subwarp_reduce_sum<32>(value.sumsq);
    if ((threadIdx.x & 31) == 0) {
        const int warp = threadIdx.x >> 5;
        warp_sum[warp] = value.sum;
        warp_sumsq[warp] = value.sumsq;
    }
    __syncthreads();

    Pair reduced{0.0f, 0.0f};
    if (threadIdx.x < Warps) {
        reduced.sum = warp_sum[threadIdx.x];
        reduced.sumsq = warp_sumsq[threadIdx.x];
    }
    if (threadIdx.x < 32) {
        reduced.sum = subwarp_reduce_sum<32>(reduced.sum);
        reduced.sumsq = subwarp_reduce_sum<32>(reduced.sumsq);
    }
    if (threadIdx.x == 0) {
        broadcast = reduced;
    }
    __syncthreads();
    return broadcast;
}

template <int BlockThreads, int MaxClusterSize>
__device__ __forceinline__ Pair cluster_reduce_pair(cg::cluster_group cluster,
                                                     Pair block_value) {
    __shared__ float cluster_sum[MaxClusterSize];
    __shared__ float cluster_sumsq[MaxClusterSize];
    __shared__ Pair broadcast;
    const int block_rank = cluster.block_rank();
    const int blocks_per_cluster = cluster.dim_blocks().x;

    if (threadIdx.x < blocks_per_cluster) {
        float* remote_sum =
            // [DSMEM] generic DSMEM pointer: slot[my rank] inside CTA threadIdx.x's smem (replicated push)
            cluster.map_shared_rank(&cluster_sum[block_rank], threadIdx.x);
        float* remote_sumsq =
            // [DSMEM] generic DSMEM pointer: slot[my rank] inside CTA threadIdx.x's smem (replicated push)
            cluster.map_shared_rank(&cluster_sumsq[block_rank], threadIdx.x);
        *remote_sum = block_value.sum;
        *remote_sumsq = block_value.sumsq;
    }
    // [DSMEM] barrier.cluster (release/acquire): pushed partials visible / slots reusable
    cluster.sync();

    Pair reduced{0.0f, 0.0f};
    if (threadIdx.x < blocks_per_cluster) {
        reduced.sum = cluster_sum[threadIdx.x];
        reduced.sumsq = cluster_sumsq[threadIdx.x];
    }
    if (threadIdx.x < 32) {
        reduced.sum = subwarp_reduce_sum<32>(reduced.sum);
        reduced.sumsq = subwarp_reduce_sum<32>(reduced.sumsq);
    }
    if (threadIdx.x == 0) {
        broadcast = reduced;
    }
    __syncthreads();
    // [DSMEM] barrier.cluster (release/acquire): pushed partials visible / slots reusable
    cluster.sync();
    return broadcast;
}

__host__ __device__ __forceinline__ float backward_value(float value,
                                                         float weight,
                                                         float grad,
                                                         float mean,
                                                         float inv_sum_w,
                                                         float inv_std,
                                                         float sum_dy,
                                                         float sum_dy_centered) {
    const float centered = value - mean;
    const float inv_var = inv_std * inv_std;
    return inv_std *
           (grad - weight * inv_sum_w * sum_dy -
            weight * centered * inv_var * inv_sum_w * sum_dy_centered);
}

// ======================================================================
// verbatim from third_party/CREDIT/src/workloads/weighted_var_backward.cu (cluster kernel)
// ======================================================================
template <int BlockThreads, int MaxClusterSize>
__global__ void cluster_bwd_staged_kernel(const float* __restrict__ input,
                                          const float* __restrict__ weights,
                                          const float* __restrict__ dy,
                                          float* __restrict__ output,
                                          int rows,
                                          int cols,
                                          int slice_elems) {
    extern __shared__ float tile[];
    float* tile_x = tile;
    float* tile_w = tile + slice_elems;
    float* tile_dy = tile + 2 * slice_elems;
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
        const std::size_t idx = row_base + slice_start + col;
        tile_x[col] = input[idx];
        tile_w[col] = weights[idx];
        tile_dy[col] = dy[idx];
    }
    __syncthreads();

    Pair local{0.0f, 0.0f};
    for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
        const float value = tile_x[col];
        const float weight = tile_w[col];
        local.sum += weight;
        local.sumsq += weight * value;
    }
    const Pair block_weighted = block_reduce_pair<BlockThreads>(local);
    const Pair row_weighted =
        // [DSMEM] AllReduce over the cluster: direct replicated push + local reduce
        cluster_reduce_pair<BlockThreads, MaxClusterSize>(cluster,
                                                          block_weighted);
    const float inv_sum_w = 1.0f / (row_weighted.sum + kEps);
    const float mean = row_weighted.sumsq * inv_sum_w;

    Pair local_var{0.0f, 0.0f};
    for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
        const float diff = tile_x[col] - mean;
        local_var.sum += tile_w[col] * diff * diff;
    }
    const Pair block_var = block_reduce_pair<BlockThreads>(local_var);
    const Pair row_var_pair =
        // [DSMEM] AllReduce over the cluster: direct replicated push + local reduce
        cluster_reduce_pair<BlockThreads, MaxClusterSize>(cluster, block_var);
    const float var = fmaxf(row_var_pair.sum * inv_sum_w, 0.0f);
    const float inv_std = rsqrtf(var + kEps);

    Pair local_grad{0.0f, 0.0f};
    for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
        const float grad = tile_dy[col];
        local_grad.sum += grad;
        local_grad.sumsq += grad * (tile_x[col] - mean);
    }
    const Pair block_grad = block_reduce_pair<BlockThreads>(local_grad);
    const Pair row_grad =
        // [DSMEM] AllReduce over the cluster: direct replicated push + local reduce
        cluster_reduce_pair<BlockThreads, MaxClusterSize>(cluster, block_grad);

    if (!row_valid) {
        return;
    }
    for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
        output[row_base + slice_start + col] =
            backward_value(tile_x[col],
                           tile_w[col],
                           tile_dy[col],
                           mean,
                           inv_sum_w,
                           inv_std,
                           row_grad.sum,
                           row_grad.sumsq);
    }
    // [DSMEM] barrier.cluster (release/acquire): pushed partials visible / slots reusable
    cluster.sync();
}
