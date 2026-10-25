/*
 * Softmax-logits backward (CREDIT reduction-reuse workload, clustered variant).
 * Source: third_party/CREDIT/src/workloads/softmax_logits_backward.cu (cluster kernel + device helpers; host benchmark stripped).
 *
 * 1. Computation (row-wise, M rows x N cols, fp32):
 *    m = max(logits), Z = sum exp(logits-m), p = exp(logits-m)/Z, dot = sum p*dy,  dx = p*(dy - dot)
 *    (three stages: max, sum, dot)
 *
 * 2. Split: one cluster per row (gridDim = rows * P). CTA b of the P-CTA cluster stages its column slice
 *    [b*slice_elems, (b+1)*slice_elems) once in shared memory (owner-local staging, no global re-reads)
 *    and computes per-slice partial statistics with warp shuffles + a block reduction.
 *
 * 3. Collective: every row statistic is an AllReduce(sum or max) of 1 float(s) across the P CTAs,
 *    done 3 time(s) per row with the direct replicated push (cluster_reduce_max / cluster_reduce_sum): thread t < P writes
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

// ======================================================================
// verbatim from third_party/CREDIT/src/workloads/softmax_logits_backward.cu (reduction helpers)
// ======================================================================
template <int Width>
__device__ __forceinline__ float subwarp_reduce_sum(float value) {
    for (int offset = Width / 2; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset, Width);
    }
    return value;
}

template <int Width>
__device__ __forceinline__ float subwarp_reduce_max(float value) {
    for (int offset = Width / 2; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset, Width));
    }
    return value;
}

template <int BlockThreads>
__device__ __forceinline__ float block_reduce_sum(float value) {
    constexpr int Warps = BlockThreads / 32;
    __shared__ float warp_values[Warps];
    __shared__ float broadcast;

    value = subwarp_reduce_sum<32>(value);
    if ((threadIdx.x & 31) == 0) {
        warp_values[threadIdx.x >> 5] = value;
    }
    __syncthreads();

    float reduced = 0.0f;
    if (threadIdx.x < Warps) {
        reduced = warp_values[threadIdx.x];
    }
    if (threadIdx.x < 32) {
        reduced = subwarp_reduce_sum<32>(reduced);
    }
    if (threadIdx.x == 0) {
        broadcast = reduced;
    }
    __syncthreads();
    return broadcast;
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

    float reduced = -CUDART_INF_F;
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
__device__ __forceinline__ float cluster_reduce_sum(cg::cluster_group cluster,
                                                     float block_value) {
    __shared__ float slots[MaxClusterSize];
    __shared__ float broadcast;
    const int block_rank = cluster.block_rank();
    const int blocks_per_cluster = cluster.dim_blocks().x;

    if (threadIdx.x < blocks_per_cluster) {
        // [DSMEM] generic DSMEM pointer: slot[my rank] inside CTA threadIdx.x's smem (replicated push)
        float* remote = cluster.map_shared_rank(&slots[block_rank], threadIdx.x);
        *remote = block_value;
    }
    // [DSMEM] barrier.cluster (release/acquire): pushed partials visible / slots reusable
    cluster.sync();

    float reduced = 0.0f;
    if (threadIdx.x < blocks_per_cluster) {
        reduced = slots[threadIdx.x];
    }
    if (threadIdx.x < 32) {
        reduced = subwarp_reduce_sum<32>(reduced);
    }
    if (threadIdx.x == 0) {
        broadcast = reduced;
    }
    __syncthreads();
    // [DSMEM] barrier.cluster (release/acquire): pushed partials visible / slots reusable
    cluster.sync();
    return broadcast;
}

template <int BlockThreads, int MaxClusterSize>
__device__ __forceinline__ float cluster_reduce_max(cg::cluster_group cluster,
                                                     float block_value) {
    __shared__ float slots[MaxClusterSize];
    __shared__ float broadcast;
    const int block_rank = cluster.block_rank();
    const int blocks_per_cluster = cluster.dim_blocks().x;

    if (threadIdx.x < blocks_per_cluster) {
        // [DSMEM] generic DSMEM pointer: slot[my rank] inside CTA threadIdx.x's smem (replicated push)
        float* remote = cluster.map_shared_rank(&slots[block_rank], threadIdx.x);
        *remote = block_value;
    }
    // [DSMEM] barrier.cluster (release/acquire): pushed partials visible / slots reusable
    cluster.sync();

    float reduced = -CUDART_INF_F;
    if (threadIdx.x < blocks_per_cluster) {
        reduced = slots[threadIdx.x];
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

// ======================================================================
// verbatim from third_party/CREDIT/src/workloads/softmax_logits_backward.cu (cluster kernel)
// ======================================================================
template <int BlockThreads, int MaxClusterSize>
__global__ void cluster_softmax_logits_bwd_staged_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ dy,
    float* __restrict__ dx,
    int rows,
    int cols,
    int slice_elems) {
    extern __shared__ float tile[];
    float* tile_logits = tile;
    float* tile_dy = tile + slice_elems;

    cg::cluster_group cluster = cg::this_cluster();
    const int blocks_per_cluster = cluster.dim_blocks().x;
    const int block_rank = cluster.block_rank();
    const int row = blockIdx.x / blocks_per_cluster;
    const bool row_valid = row < rows;
    const int slice_start = block_rank * slice_elems;
    const int remaining = cols - slice_start;
    const int local_cols =
        (row_valid && remaining > 0)
            ? (remaining < slice_elems ? remaining : slice_elems)
            : 0;
    const std::size_t row_base = static_cast<std::size_t>(row) * cols;

    for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
        const int global_col = slice_start + col;
        tile_logits[col] = logits[row_base + global_col];
        tile_dy[col] = dy[row_base + global_col];
    }
    __syncthreads();

    float local_max = -CUDART_INF_F;
    for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
        local_max = fmaxf(local_max, tile_logits[col]);
    }
    const float block_max = block_reduce_max<BlockThreads>(local_max);
    const float row_max =
        // [DSMEM] AllReduce over the cluster: direct replicated push + local reduce
        cluster_reduce_max<BlockThreads, MaxClusterSize>(cluster, block_max);

    float local_sum = 0.0f;
    for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
        local_sum += expf(tile_logits[col] - row_max);
    }
    const float block_sum = block_reduce_sum<BlockThreads>(local_sum);
    const float row_sum =
        // [DSMEM] AllReduce over the cluster: direct replicated push + local reduce
        cluster_reduce_sum<BlockThreads, MaxClusterSize>(cluster, block_sum);

    float local_dot = 0.0f;
    const float inv_sum = 1.0f / row_sum;
    for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
        const float p = expf(tile_logits[col] - row_max) * inv_sum;
        local_dot += p * tile_dy[col];
    }
    const float block_dot = block_reduce_sum<BlockThreads>(local_dot);
    const float row_dot =
        // [DSMEM] AllReduce over the cluster: direct replicated push + local reduce
        cluster_reduce_sum<BlockThreads, MaxClusterSize>(cluster, block_dot);

    if (!row_valid) {
        return;
    }
    for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
        const int global_col = slice_start + col;
        const float p = expf(tile_logits[col] - row_max) * inv_sum;
        dx[row_base + global_col] = p * (tile_dy[col] - row_dot);
    }
}
