/*
 * Fused decoder-layer kernel for Llama-2-7B decoding (ClusterFusion, H100 path).
 * Source: third_party/ClusterFusion/include/H100/llama/kernel.cuh (+ dsm.cuh, config.h), SplitToken dataflow.
 *
 * 1. Computation (one token, one head per cluster; H = HEAD_DIM = 128, D = HIDDEN_DIM = 4096, S = SEQ_LEN):
 *    - RMSNorm:      x_hat = x * rsqrt(sum(x^2)/D + eps) * w_rms                      x in R^{1xD}
 *    - QKV GEMV:     q,k,v = x_hat @ W_{q,k,v}[:, head]                             W in R^{Dx3H} per head
 *    - RoPE on q, k (GPT-J interleaved pairs; NEOX variant behind a macro)
 *    - write k, v of the current token to the KV cache outputs
 *    - Flash-decoding: s = q @ K_cache^T * H^{-1/2} (log2e scaled), online softmax (running max m, sum l),
 *                     a = sum_j exp(s_j - m) V_j / l                                 K,V cache in R^{SxH}
 *    - O-proj:       out[d-slice] += a @ W_o[head, :]                                 W_o in R^{HxD}
 *
 * 2. Split across the N = CLUSTER_SIZE (4) CTAs of the cluster (cluster = one head, grid = HEAD_NUM clusters):
 *    - RMSNorm / QKV: CTA b owns input dims [b*D/N, (b+1)*D/N): partial sum of squares and partial
 *      q,k,v (3H = 384 halves) over its K-slice (split-K GEMV; weights streamed by TMA in 64-row tiles).
 *    - Attention: CTA b owns KV tokens [b*S/N, (b+1)*S/N): local (m_b, l_b, a_b) over its tokens
 *      (K/V tiles streamed by TMA). CTA 0 also folds in the current token's k, v.
 *    - O-proj: CTA b produces output dims [b*D/N, (b+1)*D/N) for this head; heads are summed in
 *      global memory with atomicAdd (cross-cluster dependency, not DSMEM).
 *
 * 3. Collectives that resolve the inter-CTA dependencies (all DSMEM, see dsm.cuh and the loops below):
 *    a. AllReduce(sum) of the RMS sum of squares: scalar, shift-and-accumulate with remote atomicAdd
 *       through cluster.map_shared_rank(), cluster.sync() per round.           -> every CTA has rsqrt
 *    b. AllReduce(sum) of the partial q|k|v (3H halves = 768 B): cluster_reduce<LINEAR>, bulk
 *       cp.async.bulk push into the peer's scratch + mbarrier complete_tx, N-2 rounds.
 *    c. AllReduce(max) of the softmax running max m_b: scalar, remote fmaxf store (shift schedule).
 *    d. AllReduce(sum) of the rescaled running sum l_b: scalar, remote atomicAdd.
 *    e. AllReduce(sum) of the rescaled partial attention output a_b (H halves = 256 B):
 *       cluster_reduce<ATTN>, bulk push + mbarrier.
 *    After (e) every CTA holds the full head output and computes its own slice of the O-projection.
 *    Variants kernel_sglang.cuh (paged KV) and kernel_batch_sglang.cuh (batched) use the same collectives.
 */
#include <cuda/barrier>
#include <cudaTypedefs.h>
#include <math_constants.h>

// ======================================================================
// verbatim from third_party/ClusterFusion/include/H100/llama/config.h
// ======================================================================

#define HEAD_DIM 128    
#define HEAD_NUM 32     
#define FFN_DIM 12288   
#define HIDDEN_DIM 4096 
// #define SEQ_LEN 4096

#define NUM_WARPS 4
#define WARP_SIZE 32
#define BLOCK_SIZE (NUM_WARPS * WARP_SIZE) 
#define CLUSTER_SIZE 4 // 2 4
#define NUM_PER_THREAD 8
#define NUM_ROW_PER_WARP (HEAD_DIM / NUM_WARPS) 
#define NUM_THREAD_PER_ROW (WARP_SIZE / NUM_ROW_PER_WARP) 
#define NUM_PER_ROW (NUM_PER_THREAD * NUM_THREAD_PER_ROW) 
#define DIM_PER_BLOCK (HIDDEN_DIM / CLUSTER_SIZE)
// #define KV_DIM_PER_BLOCK (SEQ_LEN / CLUSTER_SIZE) 
#define FFN_DIM_PER_CLUSTER (FFN_DIM / HEAD_NUM) 
#define MAX_SMEM_DIM FFN_DIM_PER_CLUSTER

#define TMA_LOAD_ONCE 64 // 8 16 32 64 128 256
#define TMA_LOAD_ONCE_MAX 256
#define TMA_LOAD_ONCE_NUM (TMA_LOAD_ONCE * HEAD_DIM)
#define TMA_LOAD_ONCE_SIZE (TMA_LOAD_ONCE_NUM * sizeof(half))
#define TMA_LOAD_ONCE_ATTN (TMA_LOAD_ONCE / 2)
#define TMA_LOAD_ONCE_NUM_ATTN ((TMA_LOAD_ONCE * HEAD_DIM) / 2)
#define TMA_LOAD_ONCE_SIZE_ATTN (TMA_LOAD_ONCE_NUM_ATTN * sizeof(half))
#define TMA_LOAD_ONCE_NUM_FFN (TMA_LOAD_ONCE * TMA_LOAD_ONCE_MAX)
#define TMA_LOAD_ONCE_NUM_FFN_TOTAL (TMA_LOAD_ONCE * FFN_DIM_PER_CLUSTER)
#define TMA_LOAD_ONCE_SIZE_FFN (TMA_LOAD_ONCE_NUM_FFN_TOTAL * sizeof(half))

#define NUM_THREAD_PER_ROW_2 (HEAD_DIM / NUM_PER_THREAD) 
#define NUM_ROW_PER_WARP_2 (TMA_LOAD_ONCE_ATTN / NUM_WARPS) 
#define DIM_BLOCK_REDUCE (2 * BLOCK_SIZE / NUM_THREAD_PER_ROW_2) 
#define DEC_TILE (NUM_ROW_PER_WARP_2 / (WARP_SIZE / NUM_THREAD_PER_ROW_2))
#define NUM_ROW_PER_WARP_3 (TMA_LOAD_ONCE / NUM_WARPS) 
#define NUM_THREAD_PER_ROW_3 (WARP_SIZE / NUM_ROW_PER_WARP_3) 
#define NUM_PER_ROW_3 (NUM_PER_THREAD * NUM_THREAD_PER_ROW_3) 
// ======================================================================
// verbatim from third_party/ClusterFusion/include/dsm.cuh
// ======================================================================
#ifndef DSM_CUH
#define DSM_CUH

#include "cuda_runtime.h"                
#include "cooperative_groups.h"
#include "cuda_fp16.h"
#include <cuda.h>    

namespace cg = cooperative_groups;

enum class Stage {
    LINEAR,
    ATTN,
    FFN,
    LINEAR_DEEPSEEK,
    QUK_DEEPSEEK,
    ATTN_DEEPSEEK
};

template <int cluster_size, Stage stage>
__device__ __forceinline__ void __cluster_dims__(cluster_size, 1, 1) cluster_reduce(
    const uint32_t size, const uint32_t tid, const uint32_t tile_size, 
    const uint32_t cluster_block_id, const uint32_t src_addr, const uint32_t dst_addr, 
    uint32_t barrier, uint32_t neighbor_dst_bar, half* src, half* dst
) {
    cg::cluster_group cluster = cg::this_cluster();
    uint32_t dst_cta, neighbor_dst_addr;
    half2 buffer;
    half __align__(16) reg_input[8];

    if constexpr (stage == Stage::QUK_DEEPSEEK) {
        for (int i = 1; i < cluster.num_blocks(); i++) {
            if (tid == 0) {
                asm volatile (
                    // [DSMEM] mbarrier init (arrival count 1: the local expect_tx arrive)
                    "mbarrier.init.shared::cta.b64 [%0], %1;"
                    :
                    : "r"(barrier), "r"(1)
                );
                asm volatile (
                    // [DSMEM] arm my mbarrier: expect `size` incoming bytes from the peer push
                    "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                    :
                    : "r"(barrier), "r"(size)
                );
            }
            // [DSMEM] barrier.cluster: all threads of all CTAs in the cluster
            cluster.sync();
            if (tid == 0) {
                dst_cta = (cluster_block_id + i) % cluster.num_blocks();
                asm volatile (
                    // [DSMEM] mapa: local smem address -> same offset in the destination CTA (shared::cluster address)
                    "mapa.shared::cluster.u32 %0, %1, %2;\n"
                    : "=r"(neighbor_dst_addr)
                    : "r"(dst_addr), "r"(dst_cta)
                );
                asm volatile (
                    // [DSMEM] mapa: local smem address -> same offset in the destination CTA (shared::cluster address)
                    "mapa.shared::cluster.u32 %0, %1, %2;\n"
                    : "=r"(neighbor_dst_bar)
                    : "r"(barrier), "r"(dst_cta)
                );
                asm volatile (
                    // [DSMEM] bulk async push: local smem -> peer smem, completes `size` tx bytes on the peer mbarrier
                    "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                    :
                    :"r"(neighbor_dst_addr), "r"(src_addr), "r"(size), "r"(neighbor_dst_bar)
                    : "memory"
                );
            }
            asm volatile (
                "{\n"
                ".reg .pred                P1;\n"
                "LAB_WAIT:\n"
                // [DSMEM] spin until the incoming bytes landed (phase parity)
                "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
                "@P1                       bra.uni DONE;\n"
                "bra.uni                   LAB_WAIT;\n"
                "DONE:\n"
                "}\n"
                :: "r"(barrier),
                "r"(0)
            );
            // [DSMEM] barrier.cluster: all threads of all CTAs in the cluster
            cluster.sync();
        }
        return;
    } else {
        if (tid == 0) {
            asm volatile (
                // [DSMEM] mbarrier init (arrival count 1: the local expect_tx arrive)
                "mbarrier.init.shared::cta.b64 [%0], %1;"
                :
                : "r"(barrier), "r"(1)
            );
        }
        // [DSMEM] barrier.cluster: all threads of all CTAs in the cluster
        cluster.sync();
        for (int i = 1; i < cluster.num_blocks() - 1; i++) {
            if (tid == 0) {
                asm volatile (
                    // [DSMEM] arm my mbarrier: expect `size` incoming bytes from the peer push
                    "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                    :
                    : "r"(barrier), "r"(size)
                );
                dst_cta = (cluster_block_id + i) % cluster.num_blocks();
                asm volatile (
                    // [DSMEM] mapa: local smem address -> same offset in the destination CTA (shared::cluster address)
                    "mapa.shared::cluster.u32 %0, %1, %2;\n"
                    : "=r"(neighbor_dst_addr)
                    : "r"(dst_addr), "r"(dst_cta)
                );
                asm volatile (
                    // [DSMEM] mapa: local smem address -> same offset in the destination CTA (shared::cluster address)
                    "mapa.shared::cluster.u32 %0, %1, %2;\n"
                    : "=r"(neighbor_dst_bar)
                    : "r"(barrier), "r"(dst_cta)
                );
                asm volatile (
                    // [DSMEM] bulk async push: local smem -> peer smem, completes `size` tx bytes on the peer mbarrier
                    "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                    :
                    :"r"(neighbor_dst_addr), "r"(src_addr), "r"(size), "r"(neighbor_dst_bar)
                    : "memory"
                );
            }
            asm volatile (
                "{\n"
                ".reg .pred                P1;\n"
                "LAB_WAIT:\n"
                // [DSMEM] spin until the incoming bytes landed (phase parity)
                "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
                "@P1                       bra.uni DONE;\n"
                "bra.uni                   LAB_WAIT;\n"
                "DONE:\n"
                "}\n"
                :: "r"(barrier),
                "r"(i ^ 1)
            );

            // Local reduce-add
            if constexpr (stage == Stage::LINEAR) {
                if (tid < tile_size / 2) {
                    for (int j = 0; j < 3; j++) {
                        buffer = *(half2*)(&dst[j * tile_size + tid * 2]);
                        *(half2*)(&src[j * tile_size + tid * 2]) = __hadd2(*(half2*)(&src[j * tile_size + tid * 2]), buffer);
                    }
                }
            } else if constexpr (stage == Stage::ATTN) {
                if (tid < tile_size / 2) {
                    buffer = *(half2*)(&dst[tid * 2]);
                    *(half2*)(&src[tid * 2]) = __hadd2(*(half2*)(&src[tid * 2]), buffer);
                }
            } else if constexpr (stage == Stage::FFN) {
                if (tid < tile_size / 2) {
                    for (int j = 0; j < 3; j++) {
                        buffer = *(half2*)(&dst[j * tile_size + tid * 2]);
                        if (i == cluster.num_blocks() - 2) // ReLU
                            *(half2*)(&src[j * tile_size + tid * 2]) = __hmax2(__hadd2(*(half2*)(&src[j * tile_size + tid * 2]), buffer), __float22half2_rn({0.0f, 0.0f}));
                        else
                            *(half2*)(&src[j * tile_size + tid * 2]) = __hadd2(*(half2*)(&src[j * tile_size + tid * 2]), buffer);
                    }
                    for (int j = 0; j < 3; j++) {
                        buffer = *(half2*)(&dst[tile_size * 3 + j * tile_size + tid * 2]);
                        *(half2*)(&src[tile_size * 3 + j * tile_size + tid * 2]) = __hadd2(*(half2*)(&src[tile_size * 3 + j * tile_size + tid * 2]), buffer);
                    }
                }
            } else if constexpr (stage == Stage::LINEAR_DEEPSEEK) {
                *(uint4*)(&reg_input[0]) = *(uint4*)(&dst[tid * 8]);
                for (int di = 0; di < 8; di++) 
                    src[tid * 8 + di] = __hadd(src[tid * 8 + di], reg_input[di]);
                src[tile_size + tid] = __hadd(src[tile_size + tid], dst[tile_size + tid]);
            } else if constexpr (stage == Stage::ATTN_DEEPSEEK) {
                if (tid < tile_size / 8) {
                    *(uint4*)(&reg_input[0]) = *(uint4*)(&dst[tid * 8]);
                    for (int di = 0; di < 8; di++) 
                        src[tid * 8 + di] = __hadd(src[tid * 8 + di], reg_input[di]);
                }
            } else
                assert(false && "Unknown stage");
    
            // [DSMEM] barrier.cluster: all threads of all CTAs in the cluster
            cluster.sync();
        }
    }
}

#endif // DSM_CUH
// ======================================================================
// verbatim from third_party/ClusterFusion/include/H100/llama/kernel.cuh
// ======================================================================
#include <cuda/barrier>
#include <cudaTypedefs.h>
#include <math_constants.h> 
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;
namespace cg = cooperative_groups;

// Neox-style RoPE for sglang.
// If commented, we will use GPT-J style RoPE for tests/models/llama.py
// #define NEOX_STYLE_ROPE

__forceinline__ __device__ float ptx_exp2(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) LlamaDecoderLayerKernel(
    half* output, // 1 * hidden_dim
    half* k_output,
    half* v_output,
    half* input,  // 1 * hidden_dim
    half* w_rms_input,// hidden_dim
    float* cos,       // head_dim
    float* sin,       // head_dim
    half* k_cache,
    half* v_cache,
    const __grid_constant__ CUtensorMap tensor_map, // 3 * hidden_dim * hidden_dim
    const __grid_constant__ CUtensorMap tensor_map_k_cache, // seqlen * head_num * head_dim
    const __grid_constant__ CUtensorMap tensor_map_v_cache, // seqlen * head_num * head_dim
    const __grid_constant__ CUtensorMap tensor_map_weight_o, // hidden_dim * hidden_dim
    const uint32_t SEQ_LEN,
    const uint32_t KV_DIM_PER_BLOCK
) {
    cg::grid_group grid             = cg::this_grid();
    cg::cluster_group cluster       = cg::this_cluster();
    cg::thread_block block          = cg::this_thread_block();
    const uint32_t head_id          = grid.cluster_rank();
    const uint32_t cluster_block_id = cluster.block_rank();
    const uint32_t tid              = block.thread_rank();
    const uint32_t lane_id = tid % WARP_SIZE; 
    const uint32_t warp_id = tid / WARP_SIZE;
    const uint32_t tile_row = tid / NUM_THREAD_PER_ROW_2;
    const uint32_t tile_col = tid % NUM_THREAD_PER_ROW_2;

    // Init shared memory
    extern __shared__ uint8_t shmem_base[];
    half* weight = reinterpret_cast<half*>((uintptr_t)(shmem_base) + 127 & ~127);
    half* local_qkv = weight + 2 * TMA_LOAD_ONCE * MAX_SMEM_DIM;
    half* input_shmem = local_qkv + 3 * HEAD_DIM;
    float* reduction = reinterpret_cast<float*>(input_shmem + DIM_PER_BLOCK);

    __shared__ float cluster_local_sum, cluster_local_max;

    // Init registers
    float local_sum = 0.0, eps = 1e-6, rms_rcp = 0.0, tmp = 0.0, local_max = -CUDART_INF_F, pre_max = -CUDART_INF_F, scale = 0.0, softmax_scale = __frsqrt_rn(HEAD_DIM) * 1.44269504088896340736f;
    half __align__(16) reg_input[NUM_PER_THREAD], reg_weight[NUM_PER_THREAD];
    float reg_reduce[NUM_PER_THREAD];
    float* dst_shmem;
    float q_rope, q_rope_1, k_rope, k_rope_1, cos_reg, sin_reg;
    uint32_t size;
    uint32_t src_addr, dst_addr, neighbor_dst_bar = 0;
    float __align__(16) qk[DEC_TILE];

    // Init barrier
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar[4];
    barrier::arrival_token token[4];
    __shared__ uint64_t barrier;
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
    if (tid == 0) {
        init(&bar[0], blockDim.x);
        cde::fence_proxy_async_shared_cta();
        init(&bar[1], blockDim.x);
        cde::fence_proxy_async_shared_cta();
        init(&bar[2], blockDim.x);
        cde::fence_proxy_async_shared_cta();
        init(&bar[3], blockDim.x);
        cde::fence_proxy_async_shared_cta();
    }
    block.sync();

    // Precompute some indices
    uint input_idx = (lane_id % NUM_THREAD_PER_ROW) * NUM_PER_THREAD;
    uint weight_idx = warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW;
    uint input_idx_2 = (lane_id % NUM_THREAD_PER_ROW_2) * NUM_PER_THREAD;
    uint weight_idx_2 = warp_id * NUM_ROW_PER_WARP_2 + (lane_id / NUM_THREAD_PER_ROW_2) * DEC_TILE;
    uint input_idx_3 = (lane_id % NUM_THREAD_PER_ROW_3) * NUM_PER_THREAD;
    uint weight_idx_3 = warp_id * NUM_ROW_PER_WARP_3 + lane_id / NUM_THREAD_PER_ROW_3;
    uint cluster_block_st_id = cluster_block_id * DIM_PER_BLOCK;
    uint cluster_head_idx = head_id * HEAD_DIM;

    // RMSNorm
    for (int d = tid * 8; d < DIM_PER_BLOCK; d+=BLOCK_SIZE * 8) { 
        *(uint4*)(&reg_input[0]) = *(uint4*)(&input[cluster_block_st_id + d]);
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
    // [DSMEM] barrier.cluster: all threads of all CTAs in the cluster
    cluster.sync();
    // ClusterReduce
    for (int i = 1; i < cluster.num_blocks() - 1; i++) {
        if (tid == 0) {
            local_sum = cluster_local_sum;
            int dst_cta = (cluster_block_id + i) % cluster.num_blocks();
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
    }
    rms_rcp = __frsqrt_rn(cluster_local_sum / HIDDEN_DIM + eps);
    for (int d = tid * 8; d < DIM_PER_BLOCK; d+=BLOCK_SIZE * 8) { 
        *(uint4*)(&reg_weight[0]) = *(uint4*)(&w_rms_input[cluster_block_st_id + d]);
        for (int i = 0; i < 8; i++) {
            reg_input[i] = __float2half(__half2float(reg_input[i]) * rms_rcp * __half2float(reg_weight[i]));
        }
        *(uint4*)(&input_shmem[d]) = *(uint4*)(&reg_input[0]);
    }
    block.sync();

    // Compute input @ w_q
    // Preload weight_q
    if (tid == 0) {
        // [DSMEM] TMA: global -> this CTA's smem (not a cluster op; weights/KV streaming)
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map, cluster_head_idx, cluster_block_st_id, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE);
    } else {
        token[0] = bar[0].arrive();
    }

    for (int id = 1; id < DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
        if (tid == 0) {
            // [DSMEM] TMA: global -> this CTA's smem (not a cluster op; weights/KV streaming)
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], &tensor_map, cluster_head_idx, cluster_block_st_id + id * TMA_LOAD_ONCE, bar[id % 2]);
            token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE);
        } else {
            token[id % 2] = bar[id % 2].arrive();
        }
        bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
            *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                tmp += __half2float(reg_input[d]) * __half2float(weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + (input_idx + i + d) * HEAD_DIM + weight_idx]);
            }
        }
    }
    bar[(DIM_PER_BLOCK / TMA_LOAD_ONCE - 1) % 2].wait(std::move(token[(DIM_PER_BLOCK / TMA_LOAD_ONCE - 1) % 2]));
    for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
        *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + ((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) * TMA_LOAD_ONCE + i]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(reg_input[d]) * __half2float(weight[TMA_LOAD_ONCE_NUM + (input_idx + i + d) * HEAD_DIM + weight_idx]);
        }
    }
    #pragma unroll
    for (int mask = (NUM_THREAD_PER_ROW >> 1); mask > 0; mask >>= 1) {
        tmp += __shfl_down_sync(0xffffffff, tmp, mask);
    }
    if (lane_id % NUM_THREAD_PER_ROW == 0) {
        local_qkv[warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW] = __float2half(tmp);
    }
    block.sync();

    // Compute input @ w_k
    // Preload weight_k
    tmp = 0.0;
    if (tid == 0) {
        // [DSMEM] TMA: global -> this CTA's smem (not a cluster op; weights/KV streaming)
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map, cluster_head_idx, HIDDEN_DIM + cluster_block_st_id, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE);
    } else {
        token[0] = bar[0].arrive();
    }

    for (int id = 1; id < DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
        if (tid == 0) {
            // [DSMEM] TMA: global -> this CTA's smem (not a cluster op; weights/KV streaming)
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], &tensor_map, cluster_head_idx, HIDDEN_DIM + cluster_block_st_id + id * TMA_LOAD_ONCE, bar[id % 2]);
            token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE);
        } else {
            token[id % 2] = bar[id % 2].arrive();
        }
        bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
            *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                tmp += __half2float(reg_input[d]) * __half2float(weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + (input_idx + i + d) * HEAD_DIM + weight_idx]);
            }
        }
    }
    bar[(DIM_PER_BLOCK / TMA_LOAD_ONCE - 1) % 2].wait(std::move(token[(DIM_PER_BLOCK / TMA_LOAD_ONCE - 1) % 2]));
    for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
        *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + ((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) * TMA_LOAD_ONCE + i]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(reg_input[d]) * __half2float(weight[TMA_LOAD_ONCE_NUM + (input_idx + i + d) * HEAD_DIM + weight_idx]);
        }
    }
    #pragma unroll
    for (int mask = (NUM_THREAD_PER_ROW >> 1); mask > 0; mask >>= 1) {
        tmp += __shfl_down_sync(0xffffffff, tmp, mask);
    }
    if (lane_id % NUM_THREAD_PER_ROW == 0) {
        local_qkv[HEAD_DIM + warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW] = __float2half(tmp);
    }
    block.sync();

    // Compute input @ w_v
    // Preload weight_v
    tmp = 0.0;
    if (tid == 0) {
        // [DSMEM] TMA: global -> this CTA's smem (not a cluster op; weights/KV streaming)
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map, cluster_head_idx, HIDDEN_DIM * 2 + cluster_block_st_id, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE);
    } else {
        token[0] = bar[0].arrive();
    }

    for (int id = 1; id < DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
        if (tid == 0) {
            // [DSMEM] TMA: global -> this CTA's smem (not a cluster op; weights/KV streaming)
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], &tensor_map, cluster_head_idx, HIDDEN_DIM * 2 + cluster_block_st_id + id * TMA_LOAD_ONCE, bar[id % 2]);
            token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE);
        } else {
            token[id % 2] = bar[id % 2].arrive();
        }
        bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
            *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                tmp += __half2float(reg_input[d]) * __half2float(weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + (input_idx + i + d) * HEAD_DIM + weight_idx]);
            }
        }
    }
    bar[(DIM_PER_BLOCK / TMA_LOAD_ONCE - 1) % 2].wait(std::move(token[(DIM_PER_BLOCK / TMA_LOAD_ONCE - 1) % 2]));
    for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
        *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + ((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) * TMA_LOAD_ONCE + i]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(reg_input[d]) * __half2float(weight[TMA_LOAD_ONCE_NUM + (input_idx + i + d) * HEAD_DIM + weight_idx]);
        }
    }
    #pragma unroll
    for (int mask = (NUM_THREAD_PER_ROW >> 1); mask > 0; mask >>= 1) {
        tmp += __shfl_down_sync(0xffffffff, tmp, mask);
    }
    if (lane_id % NUM_THREAD_PER_ROW == 0) {
        local_qkv[HEAD_DIM * 2 + warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW] = __float2half(tmp);
    }
    block.sync();

    // ClusterReduce
    size = (HEAD_DIM * 3) * sizeof(half);
    src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(local_qkv));
    dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(weight));
    // [DSMEM] ClusterReduce (vector all-reduce, shift-and-accumulate bulk push, N-2 rounds)
    cluster_reduce<CLUSTER_SIZE, Stage::LINEAR>(
        size, tid, HEAD_DIM, cluster_block_id,  
        src_addr, dst_addr, bar_ptr, 
        neighbor_dst_bar, local_qkv, weight);

    q_rope = __half2float(local_qkv[tid]);
    k_rope = __half2float(local_qkv[HEAD_DIM + tid]);
    cos_reg = cos[tid];
    sin_reg = sin[tid];
#ifdef NEOX_STYLE_ROPE
    if (tid < HEAD_DIM / 2) {
        q_rope_1 = __half2float(local_qkv[HEAD_DIM / 2 + tid]);
        k_rope_1 = __half2float(local_qkv[HEAD_DIM + HEAD_DIM / 2 + tid]);
    } else {
        q_rope_1 = __half2float(local_qkv[tid - HEAD_DIM / 2]);
        k_rope_1 = __half2float(local_qkv[HEAD_DIM + tid - HEAD_DIM / 2]);
    }
    block.sync();
    if (tid < HEAD_DIM / 2) {
        local_qkv[tid] = __float2half(q_rope * cos_reg - q_rope_1 * sin_reg);
        local_qkv[HEAD_DIM + tid] = __float2half(k_rope * cos_reg - k_rope_1 * sin_reg);
    } else {
        local_qkv[tid] = __float2half(q_rope * cos_reg + q_rope_1 * sin_reg);
        local_qkv[HEAD_DIM + tid] = __float2half(k_rope * cos_reg + k_rope_1 * sin_reg);
    }
#else
    // NOTE: RoPE from llama/model.py. GPT-J stype RoPE
    if (tid % 2 == 0) {
        q_rope_1 = __half2float(local_qkv[tid + 1]);
        k_rope_1 = __half2float(local_qkv[HEAD_DIM + (tid + 1)]);
    } else {
        q_rope_1 = __half2float(local_qkv[tid - 1]);
        k_rope_1 = __half2float(local_qkv[HEAD_DIM + (tid - 1)]);
    }
    block.sync();
    if (tid % 2 == 0) {
        local_qkv[tid] = __float2half(q_rope * cos_reg - q_rope_1 * sin_reg);
        local_qkv[HEAD_DIM + tid] = __float2half(k_rope * cos_reg - k_rope_1 * sin_reg);
    } else {
        local_qkv[tid] = __float2half(q_rope * cos_reg + q_rope_1 * sin_reg);
        local_qkv[HEAD_DIM + tid] = __float2half(k_rope * cos_reg + k_rope_1 * sin_reg);
    }
#endif

    // Output kv
    // [DSMEM] barrier.cluster: all threads of all CTAs in the cluster
    cluster.sync();
    if (cluster_block_id == 0) {
        k_output[head_id * HEAD_DIM + tid] = local_qkv[HEAD_DIM + tid];
        v_output[head_id * HEAD_DIM + tid] = local_qkv[2 * HEAD_DIM + tid];
    }
    // [DSMEM] barrier.cluster: all threads of all CTAs in the cluster
    cluster.sync();

    // Compute flash-decoding
    local_sum = 0.0f;
    for(int i = 0; i < NUM_PER_THREAD; i++)
        reg_reduce[i] = 0.0f;
    *(uint4*)(&reg_input[0]) = *(uint4*)(&local_qkv[input_idx_2]);
    block.sync();

    // Preload kv_cache
    if (tid == 0) {
        // [DSMEM] TMA: global -> this CTA's smem (not a cluster op; weights/KV streaming)
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map_k_cache, cluster_head_idx, cluster_block_id * KV_DIM_PER_BLOCK, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE_ATTN);
        // [DSMEM] TMA: global -> this CTA's smem (not a cluster op; weights/KV streaming)
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[TMA_LOAD_ONCE_NUM_ATTN], &tensor_map_v_cache, cluster_head_idx, cluster_block_id * KV_DIM_PER_BLOCK, bar[2]);
        token[2] = cuda::device::barrier_arrive_tx(bar[2], 1, TMA_LOAD_ONCE_SIZE_ATTN);
    } else {
        token[0] = bar[0].arrive();
        token[2] = bar[2].arrive();
    }

    for (int id = 1; id < KV_DIM_PER_BLOCK / TMA_LOAD_ONCE_ATTN; id++) {
        if (tid == 0) {
            // [DSMEM] TMA: global -> this CTA's smem (not a cluster op; weights/KV streaming)
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], &tensor_map_k_cache, cluster_head_idx, cluster_block_id * KV_DIM_PER_BLOCK + id * TMA_LOAD_ONCE_ATTN, bar[id % 2]);
            token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE_ATTN);
        } else {
            token[id % 2] = bar[id % 2].arrive();
        }
        bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        pre_max = local_max;
        #pragma unroll
        for (int j = 0; j < DEC_TILE; j++) {
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + (weight_idx_2 + j) * HEAD_DIM + input_idx_2]);
            qk[j] = 0.0f;
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                qk[j] += __half2float(reg_input[d]) * __half2float(reg_weight[d]);
            }
            #pragma unroll
            for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
                qk[j] += __shfl_xor_sync(0xffffffff, qk[j], mask);
            }
            qk[j] = qk[j] * softmax_scale;
            local_max = max(local_max, qk[j]);
        }
        scale = ptx_exp2(pre_max - local_max);
        local_sum *= scale;
        // For filled zeros
        #pragma unroll
        for (int j = 0; j < DEC_TILE; j++) {
            if ((KV_DIM_PER_BLOCK * cluster_block_id + (id - 1) * TMA_LOAD_ONCE_ATTN + weight_idx_2 + j) < SEQ_LEN) {
                qk[j] = ptx_exp2(qk[j] - local_max);
                local_sum += qk[j];
            }
        }
        #pragma unroll
        for (int j = 0; j < NUM_PER_THREAD; j++) {
            reg_reduce[j] = reg_reduce[j] * scale;
        }
        if (tid == 0) {
            // [DSMEM] TMA: global -> this CTA's smem (not a cluster op; weights/KV streaming)
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM + TMA_LOAD_ONCE_NUM_ATTN], &tensor_map_v_cache, cluster_head_idx, cluster_block_id * KV_DIM_PER_BLOCK + id * TMA_LOAD_ONCE_ATTN, bar[2 + id % 2]);
            token[2 + id % 2] = cuda::device::barrier_arrive_tx(bar[2 + id % 2], 1, TMA_LOAD_ONCE_SIZE_ATTN);
        } else {
            token[2 + id % 2] = bar[2 + id % 2].arrive();
        }
        bar[2 + (id - 1) % 2].wait(std::move(token[2 + (id - 1) % 2]));
        for (int j = 0; j < DEC_TILE; j++) {
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + TMA_LOAD_ONCE_NUM_ATTN + (weight_idx_2 + j) * HEAD_DIM + input_idx_2]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                reg_reduce[d] = reg_reduce[d] + qk[j] * __half2float(reg_weight[d]);
            }
        }
    }
    if (KV_DIM_PER_BLOCK > TMA_LOAD_ONCE_ATTN) {
        bar[(KV_DIM_PER_BLOCK / TMA_LOAD_ONCE_ATTN - 1) % 2].wait(std::move(token[(KV_DIM_PER_BLOCK / TMA_LOAD_ONCE_ATTN - 1) % 2]));
    } else {
        bar[0].wait(std::move(token[0]));
    }
    pre_max = local_max;
    #pragma unroll
    for (int j = 0; j < DEC_TILE; j++) {
        *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[((KV_DIM_PER_BLOCK / TMA_LOAD_ONCE_ATTN - 1) % 2) * TMA_LOAD_ONCE_NUM + (weight_idx_2 + j) * HEAD_DIM + input_idx_2]);
        qk[j] = 0.0f;
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            qk[j] += __half2float(reg_input[d]) * __half2float(reg_weight[d]);
        }
        #pragma unroll
        for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
            qk[j] += __shfl_xor_sync(0xffffffff, qk[j], mask);
        }
        qk[j] = qk[j] * softmax_scale;
        local_max = max(local_max, qk[j]);
    }
    scale = ptx_exp2(pre_max - local_max);
    local_sum *= scale;
    #pragma unroll
    for (int j = 0; j < DEC_TILE; j++) {
        if ((KV_DIM_PER_BLOCK * cluster_block_id + (KV_DIM_PER_BLOCK / TMA_LOAD_ONCE_ATTN - 1) * TMA_LOAD_ONCE_ATTN + weight_idx_2 + j) < SEQ_LEN) {
            qk[j] = ptx_exp2(qk[j] - local_max);
            local_sum += qk[j];
        }
    }
    #pragma unroll
    for (int j = 0; j < NUM_PER_THREAD; j++) {
        reg_reduce[j] = reg_reduce[j] * scale;
    }
    if (KV_DIM_PER_BLOCK > TMA_LOAD_ONCE_ATTN) {
        bar[(KV_DIM_PER_BLOCK / TMA_LOAD_ONCE_ATTN - 1) % 2 + 2].wait(std::move(token[(KV_DIM_PER_BLOCK / TMA_LOAD_ONCE_ATTN - 1) % 2 + 2]));
    } else {
        bar[2].wait(std::move(token[2]));
    }
    for (int j = 0; j < DEC_TILE; j++) {
        *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[((KV_DIM_PER_BLOCK / TMA_LOAD_ONCE_ATTN - 1) % 2) * TMA_LOAD_ONCE_NUM + TMA_LOAD_ONCE_NUM_ATTN + (weight_idx_2 + j) * HEAD_DIM + input_idx_2]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            reg_reduce[d] = reg_reduce[d] + qk[j] * __half2float(reg_weight[d]);
        }
    }
    block.sync();

    // Process KV of current token
    if (cluster_block_id == 0 && warp_id == 0) {
        if (lane_id / NUM_THREAD_PER_ROW_2 == 1) {
            pre_max = local_max;
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&local_qkv[HEAD_DIM + input_idx_2]); 
            qk[0] = 0.0f;
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                qk[0] += __half2float(reg_input[d]) * __half2float(reg_weight[d]);
            }
        }
        #pragma unroll
        for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
            qk[0] += __shfl_xor_sync(0xffffffff, qk[0], mask);
        }
        if (lane_id / NUM_THREAD_PER_ROW_2 == 1) {
            qk[0] = qk[0] * softmax_scale;
            local_max = max(local_max, qk[0]); 
            scale = ptx_exp2(pre_max - local_max);
            local_sum *= scale;
            qk[0] = ptx_exp2(qk[0] - local_max);
            local_sum += qk[0];
            #pragma unroll
            for (int j = 0; j < NUM_PER_THREAD; j++) {
                reg_reduce[j] = reg_reduce[j] * scale;
            }
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&local_qkv[2 * HEAD_DIM + input_idx_2]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                reg_reduce[d] = reg_reduce[d] + qk[0] * __half2float(reg_weight[d]);
            }
        }
    }
    block.sync();

    #pragma unroll
    for (int i = 0; i < NUM_PER_THREAD; i++) {
        weight[tile_row * HEAD_DIM + tile_col * NUM_PER_THREAD + i] = __float2half(reg_reduce[i]);
    }
    if (lane_id % NUM_THREAD_PER_ROW_2 == 0) {
        reduction[tile_row * 2] = local_max;
        reduction[tile_row * 2 + 1] = local_sum;
    }
    block.sync();
    for(int i = 0; i < NUM_PER_THREAD; i++)
        reg_reduce[i] = 0.0f;
    local_sum = 0.0, local_max = 0.0;
    #pragma unroll
    for(int j = 0; j < DIM_BLOCK_REDUCE / 2; j++) {
        *(uint4*)(&reg_input[0]) = *(uint4*)(&weight[j * HEAD_DIM + tile_col * NUM_PER_THREAD]);
        float m = reduction[j * 2], s = reduction[j * 2 + 1];
        pre_max = local_max;
        local_max = max(m, local_max);
        scale = ptx_exp2(m - local_max);
        s *= scale;
        local_sum = local_sum * ptx_exp2(pre_max - local_max) + s;
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            reg_reduce[d] = reg_reduce[d] * ptx_exp2(pre_max - local_max) + __half2float(reg_input[d]) * scale;
        }
    }
    block.sync();

    pre_max = local_max;
    if(tid == 0) {
        cluster_local_max = local_max;
    }
    // [DSMEM] barrier.cluster: all threads of all CTAs in the cluster
    cluster.sync();
    // ClusterReduce: local_max
    for (int i = 1; i < cluster.num_blocks() - 1; i++) {
        if (tid == 0) {
            local_max = cluster_local_max;
            int dst_cta = (cluster_block_id + i) % cluster.num_blocks();
            // [DSMEM] generic DSMEM pointer into the peer CTA's shared memory
            dst_shmem = cluster.map_shared_rank(&cluster_local_max, dst_cta);  
        }
        // [DSMEM] barrier.cluster: all threads of all CTAs in the cluster
        cluster.sync();
        if (tid == 0) {
            // [DSMEM] remote read-modify-write store into the peer's smem scalar (max ClusterReduce; one writer per slot per round)
            *dst_shmem = fmaxf(*dst_shmem, local_max);
        }
        // [DSMEM] barrier.cluster: all threads of all CTAs in the cluster
        cluster.sync();
    }
    scale = ptx_exp2(pre_max - cluster_local_max);
    local_sum *= scale;
    #pragma unroll
    for (int j = 0; j < NUM_PER_THREAD; j++) {
        reg_reduce[j] = reg_reduce[j] * scale;
    }
    if(tid == 0) {
        cluster_local_sum = local_sum;
    }
    // [DSMEM] barrier.cluster: all threads of all CTAs in the cluster
    cluster.sync();
    // ClusterReduce: local_sum
    for (int i = 1; i < cluster.num_blocks() - 1; i++) {
        if (tid == 0) {
            local_sum = cluster_local_sum;
            int dst_cta = (cluster_block_id + i) % cluster.num_blocks();
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
    }
    #pragma unroll
    for (int j = 0; j < NUM_PER_THREAD; j++) {
        reg_reduce[j] = reg_reduce[j] * __frcp_rn(cluster_local_sum);
    }
    if(tid < NUM_THREAD_PER_ROW_2) {
        // NOTE: revised vectorized store
        #pragma unroll
        for (int i = 0; i < NUM_PER_THREAD; i++) {
            local_qkv[2 * HEAD_DIM + tid * NUM_PER_THREAD + i] = __float2half(reg_reduce[i]);
        }
    }
    block.sync();

    // ClusterReduce
    size = HEAD_DIM * sizeof(half);
    src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&local_qkv[2 * HEAD_DIM]));
    dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(weight));
    // [DSMEM] ClusterReduce (vector all-reduce, shift-and-accumulate bulk push, N-2 rounds)
    cluster_reduce<CLUSTER_SIZE, Stage::ATTN>(
        size, tid, HEAD_DIM, cluster_block_id,  
        src_addr, dst_addr, bar_ptr, 
        neighbor_dst_bar, &local_qkv[2 * HEAD_DIM], weight);

    // Compute output @ w_o
    // Preload w_o
    if (tid == 0) {
        // [DSMEM] TMA: global -> this CTA's smem (not a cluster op; weights/KV streaming)
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map_weight_o, cluster_block_st_id, cluster_head_idx, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE);
    } else {
        token[0] = bar[0].arrive();
    }

    for (int id = 1; id < DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
        if (tid == 0) {
            // [DSMEM] TMA: global -> this CTA's smem (not a cluster op; weights/KV streaming)
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], &tensor_map_weight_o, cluster_block_st_id + id * TMA_LOAD_ONCE, cluster_head_idx, bar[id % 2]);
            token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE);
        } else {
            token[id % 2] = bar[id % 2].arrive();
        }
        bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        tmp = 0.0;
        for (int j = 0; j < HEAD_DIM; j+=NUM_PER_ROW_3) {
            *(uint4*)(&reg_input[0]) = *(uint4*)(&local_qkv[2 * HEAD_DIM + input_idx_3 + j]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                tmp += __half2float(reg_input[d]) * __half2float(weight[(id - 1) % 2 * TMA_LOAD_ONCE_NUM + (input_idx_3 + j + d) * TMA_LOAD_ONCE + weight_idx_3]);
            }
        }
        #pragma unroll
        for (int mask = (NUM_THREAD_PER_ROW_3 >> 1); mask > 0; mask >>= 1) {
            tmp += __shfl_down_sync(0xffffffff, tmp, mask);
        }
        if (lane_id % NUM_THREAD_PER_ROW_3 == 0) {
            atomicAdd(&output[cluster_block_st_id + weight_idx_3 + (id - 1) * TMA_LOAD_ONCE], __float2half(tmp));
        }
        block.sync();
    }
    bar[(DIM_PER_BLOCK / TMA_LOAD_ONCE - 1) % 2].wait(std::move(token[(DIM_PER_BLOCK / TMA_LOAD_ONCE - 1) % 2]));
    tmp = 0.0;
    for (int j = 0; j < HEAD_DIM; j+=NUM_PER_ROW_3) {
        *(uint4*)(&reg_input[0]) = *(uint4*)(&local_qkv[2 * HEAD_DIM + input_idx_3 + j]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(reg_input[d]) * __half2float(weight[TMA_LOAD_ONCE_NUM + (input_idx_3 + j + d) * TMA_LOAD_ONCE + weight_idx_3]);
        }
    }
    #pragma unroll
    for (int mask = (NUM_THREAD_PER_ROW_3 >> 1); mask > 0; mask >>= 1) {
        tmp += __shfl_down_sync(0xffffffff, tmp, mask);
    }
    if (lane_id % NUM_THREAD_PER_ROW_3 == 0) {
        atomicAdd(&output[cluster_block_st_id + weight_idx_3 + ((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) * TMA_LOAD_ONCE], __float2half(tmp));
    }
}
