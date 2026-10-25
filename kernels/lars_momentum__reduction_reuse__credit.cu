/*
 * LARS momentum update (CREDIT reduction-reuse workload, clustered variant).
 * Source: third_party/CREDIT/src/workloads/lars_momentum.cu (cluster kernel + device helpers; host benchmark stripped).
 *
 * 1. Computation (row-wise, M rows x N cols, fp32):
 *    u = mu*m + g + wd*w;  trust = eta*||w|| / (||u|| + eps);  w' = w - lr*trust*u;  m' = u
 *    (one stage: {sum w^2, sum u^2})
 *
 * 2. Split: one cluster per row (gridDim = rows * P). CTA b of the P-CTA cluster stages its column slice
 *    [b*slice_elems, (b+1)*slice_elems) once in shared memory (owner-local staging, no global re-reads)
 *    and computes per-slice partial statistics with warp shuffles + a block reduction.
 *
 * 3. Collective: every row statistic is an AllReduce(sum) of 2 float(s) across the P CTAs,
 *    done 1 time(s) per row with the direct replicated push (cluster_reduce_pair): thread t < P writes
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
constexpr float kEps = 1e-6f;
constexpr float kMomentumCoeff = 0.9f;
constexpr float kWeightDecay = 0.01f;
constexpr float kLearningRate = 1.0e-3f;
constexpr float kTrustCoeff = 0.02f;

struct Pair {
    float a;
    float b;
};

// ======================================================================
// verbatim from third_party/CREDIT/src/workloads/lars_momentum.cu (reduction helpers)
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
    __shared__ float warp_a[Warps];
    __shared__ float warp_b[Warps];
    __shared__ Pair broadcast;

    value.a = subwarp_reduce_sum<32>(value.a);
    value.b = subwarp_reduce_sum<32>(value.b);
    if ((threadIdx.x & 31) == 0) {
        const int warp = threadIdx.x >> 5;
        warp_a[warp] = value.a;
        warp_b[warp] = value.b;
    }
    __syncthreads();

    Pair reduced{0.0f, 0.0f};
    if (threadIdx.x < Warps) {
        reduced.a = warp_a[threadIdx.x];
        reduced.b = warp_b[threadIdx.x];
    }
    if (threadIdx.x < 32) {
        reduced.a = subwarp_reduce_sum<32>(reduced.a);
        reduced.b = subwarp_reduce_sum<32>(reduced.b);
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
    __shared__ float cluster_a[MaxClusterSize];
    __shared__ float cluster_b[MaxClusterSize];
    __shared__ Pair broadcast;
    const int block_rank = cluster.block_rank();
    const int blocks_per_cluster = cluster.dim_blocks().x;

    if (threadIdx.x < blocks_per_cluster) {
        float* remote_a =
            // [DSMEM] generic DSMEM pointer: slot[my rank] inside CTA threadIdx.x's smem (replicated push)
            cluster.map_shared_rank(&cluster_a[block_rank], threadIdx.x);
        float* remote_b =
            // [DSMEM] generic DSMEM pointer: slot[my rank] inside CTA threadIdx.x's smem (replicated push)
            cluster.map_shared_rank(&cluster_b[block_rank], threadIdx.x);
        *remote_a = block_value.a;
        *remote_b = block_value.b;
    }
    // [DSMEM] barrier.cluster (release/acquire): pushed partials visible / slots reusable
    cluster.sync();

    Pair reduced{0.0f, 0.0f};
    if (threadIdx.x < blocks_per_cluster) {
        reduced.a = cluster_a[threadIdx.x];
        reduced.b = cluster_b[threadIdx.x];
    }
    if (threadIdx.x < 32) {
        reduced.a = subwarp_reduce_sum<32>(reduced.a);
        reduced.b = subwarp_reduce_sum<32>(reduced.b);
    }
    if (threadIdx.x == 0) {
        broadcast = reduced;
    }
    __syncthreads();
    // [DSMEM] barrier.cluster (release/acquire): pushed partials visible / slots reusable
    cluster.sync();
    return broadcast;
}

__host__ __device__ __forceinline__ float update_value(float weight,
                                                       float grad,
                                                       float momentum) {
    return kMomentumCoeff * momentum + grad + kWeightDecay * weight;
}

// ======================================================================
// verbatim from third_party/CREDIT/src/workloads/lars_momentum.cu (cluster kernel)
// ======================================================================
template <int BlockThreads, int MaxClusterSize>
__global__ void cluster_lars_staged_kernel(const float* __restrict__ weight,
                                           const float* __restrict__ grad,
                                           const float* __restrict__ momentum,
                                           float* __restrict__ weight_out,
                                           float* __restrict__ momentum_out,
                                           int rows,
                                           int cols,
                                           int slice_elems) {
    extern __shared__ float tile[];
    float* tile_w = tile;
    float* tile_g = tile + slice_elems;
    float* tile_m = tile + 2 * slice_elems;
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
        tile_w[col] = weight[row_base + slice_start + col];
        tile_g[col] = grad[row_base + slice_start + col];
        tile_m[col] = momentum[row_base + slice_start + col];
    }
    __syncthreads();

    Pair stats{0.0f, 0.0f};
    for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
        const float update = update_value(tile_w[col], tile_g[col], tile_m[col]);
        stats.a += tile_w[col] * tile_w[col];
        stats.b += update * update;
    }
    const Pair block_stats = block_reduce_pair<BlockThreads>(stats);
    const Pair row_stats =
        // [DSMEM] AllReduce over the cluster: direct replicated push + local reduce
        cluster_reduce_pair<BlockThreads, MaxClusterSize>(cluster, block_stats);
    const float trust =
        kTrustCoeff * sqrtf(row_stats.a) / (sqrtf(row_stats.b) + kEps);
    const float step = kLearningRate * trust;

    if (!row_valid) {
        return;
    }
    for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
        const int global_col = slice_start + col;
        const float update = update_value(tile_w[col], tile_g[col], tile_m[col]);
        weight_out[row_base + global_col] = tile_w[col] - step * update;
        momentum_out[row_base + global_col] = update;
    }
}
