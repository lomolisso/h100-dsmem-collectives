/*
 * RMSNorm kernel, one cluster per batch row (ClusterFusion, H100 path).
 * Source: third_party/ClusterFusion/include/H100/norm/kernel.cuh (+ config.h).
 *
 * 1. Computation: y = x * rsqrt(sum_d x_d^2 / D + eps) * w, x in R^{1xD}, D = HIDDEN_DIM = 8192, BATCH_SIZE rows.
 * 2. Split: cluster = one row; CTA b owns dims [b*D/N, (b+1)*D/N) (N = CLUSTER_SIZE = 2) and computes the
 *    partial sum of squares with warp shuffles + a smem reduction.
 * 3. Collective: AllReduce(sum) of one float. Thread 0 maps &cluster_local_sum of CTA (b+1) mod N with
 *    cluster.map_shared_rank(), cluster.sync(), remote atomicAdd of its partial, cluster.sync(). With N = 2
 *    this single shift-by-1 round gives both CTAs the full sum; each CTA then normalises its own slice.
 */

// ======================================================================
// verbatim from third_party/ClusterFusion/include/H100/norm/config.h
// ======================================================================

#define BATCH_SIZE 64
#define HIDDEN_DIM 8192   
#define CLUSTER_SIZE 2 // 2 4 8 16
#define DIM_PER_BLOCK (HIDDEN_DIM / CLUSTER_SIZE)
#define NUM_PER_THREAD 8
#define BLOCK_SIZE (DIM_PER_BLOCK / NUM_PER_THREAD)
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)

// ======================================================================
// verbatim from third_party/ClusterFusion/include/H100/norm/kernel.cuh
// ======================================================================
#include "cuda_runtime.h"                
#include "cooperative_groups.h"
#include "cuda_fp16.h"
namespace cg = cooperative_groups;


__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) RMSNormKernel(
    half* output, // 1 * hidden_dim
    half* input,  // 1 * hidden_dim
    half* weight// hidden_dim
) {
    cg::grid_group grid             = cg::this_grid();
    cg::cluster_group cluster       = cg::this_cluster();
    cg::thread_block block          = cg::this_thread_block();
    const uint32_t batch_id         = grid.cluster_rank();
    const uint32_t cluster_block_id = cluster.block_rank();
    const uint32_t tid              = block.thread_rank();
    const uint32_t lane_id = tid % WARP_SIZE; 
    const uint32_t warp_id = tid / WARP_SIZE;

    // Init shared memory
    __shared__ float reduction[NUM_WARPS];
    __shared__ float cluster_local_sum;

    // Init registers
    float local_sum = 0.0, eps = 1e-6, rms_rcp = 0.0;
    half __align__(16) reg_input[NUM_PER_THREAD], reg_weight[NUM_PER_THREAD];
    float* dst_shmem;

    // Precompute some indices
    uint cluster_block_st_idx = cluster_block_id * DIM_PER_BLOCK;

    // RMSNorm
    for (int d = tid * 8; d < DIM_PER_BLOCK; d+=BLOCK_SIZE * 8) { 
        *(uint4*)(&reg_input[0]) = *(uint4*)(&input[batch_id * HIDDEN_DIM + cluster_block_st_idx + d]);
        #pragma unroll
        for (int di = 0; di < 8; di++)
            local_sum += __half2float(reg_input[di]) * __half2float(reg_input[di]);
    }
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, mask);
    }
    if (lane_id == 0){
        reduction[warp_id] = local_sum;
    }
    block.sync(); 
    if (tid < NUM_WARPS) 
        local_sum = reduction[tid];
    #pragma unroll
    for (int mask = NUM_WARPS / 2; mask > 0; mask >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, mask);
    } 
    if (tid == 0)
        cluster_local_sum = local_sum;
    block.sync();
    // ClusterReduce
    if (tid == 0) {
        int dst_cta = (cluster_block_id + 1) % cluster.num_blocks();
        // [DSMEM] generic DSMEM pointer into the peer CTA's shared memory
        dst_shmem = cluster.map_shared_rank(&cluster_local_sum, dst_cta);  
    }
    // [DSMEM] barrier.cluster: all threads of all CTAs in the cluster
    cluster.sync();
    if (tid == 0) {
        // [DSMEM] remote atomic add into the peer's smem scalar (sum ClusterReduce)
        atomicAdd(dst_shmem, local_sum);
    }
    // [DSMEM] barrier.cluster: all threads of all CTAs in the cluster
    cluster.sync();
    rms_rcp = __frsqrt_rn(cluster_local_sum / HIDDEN_DIM + eps);
    for (int d = tid * 8; d < DIM_PER_BLOCK; d+=BLOCK_SIZE * 8) { 
        *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[cluster_block_st_idx + d]);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            reg_input[i] = __float2half(__half2float(reg_input[i]) * rms_rcp * __half2float(reg_weight[i]));
        }
        *(uint4*)(&output[batch_id * HIDDEN_DIM + cluster_block_st_idx + d]) = *(uint4*)(&reg_input[0]);
    }
}
