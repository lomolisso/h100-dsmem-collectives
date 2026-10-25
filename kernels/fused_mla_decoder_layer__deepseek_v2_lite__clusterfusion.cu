/*
 * Fused MLA (Multi-head Latent Attention) decoder-layer kernel for DeepSeek-V2-Lite decoding (ClusterFusion).
 * Source: third_party/ClusterFusion/include/H100/deepseek/kernel.cuh (+ dsm.cuh, config.h); paper Appendix B.1, Alg. 4.
 *
 * 1. Computation (one token; one head per cluster; D = 2048, kv_lora_rank l = 512, rope r = 64, nope h = 128,
 *    MLA_HEAD_DIM = l + r = 576, S = 4096; weight-absorbed MLA):
 *    - RMSNorm(x)                                                       x in R^{1xD}
 *    - q_nope = x_hat @ W_q[:, head] (h), q_pe = x_hat @ W_q_pe (r), c_kv = x_hat @ W_kv (l), k_pe = x_hat @ W_k_pe (r)
 *    - RoPE on q_pe, k_pe; RMSNorm on c_kv (the new latent KV entry)
 *    - absorption: q_lat = q_nope @ W_uk[head]  (h -> l), so attention runs in latent space:
 *      s = [q_lat | q_pe] . [c_kv_cache | k_pe_cache]^T * H^{-1/2},  online softmax, a_lat = softmax(s) @ c_kv_cache (l)
 *    - a = a_lat @ W_uv[head] (l -> h);   out[d-slice] += a @ W_o[head,:]
 *
 * 2. Split across N = CLUSTER_SIZE (4) CTAs:
 *    - projections: split-K over input dims [b*D/N, ...) -> partial (q_nope|q_pe|c_kv|k_pe) per CTA (2*576 halves);
 *    - q @ W_uk: CTA b computes 128 of the 512 latent outputs (split-N);
 *    - attention: CTA b owns KV tokens [b*S/N, ...) -> local (m_b, l_b, a_lat_b);
 *    - a_lat @ W_uv: CTA b contracts its 128-slice of a_lat (split-K) -> partial head output (h);
 *    - O-proj: CTA b writes output dims [b*D/N, ...); heads summed via global atomicAdd.
 *
 * 3. Collectives (all DSMEM, dsm.cuh):
 *    a. AllReduce(sum) scalar for the input RMS sum (remote atomicAdd, shift schedule).
 *    b. AllReduce(sum) of the 2*576-half projection partials: cluster_reduce<LINEAR_DEEPSEEK> (bulk push).
 *    c. AllGather of the four 128-half q_lat slices: cluster_reduce<QUK_DEEPSEEK> (direct shift push, N-1 rounds)
 *       -> every CTA holds the full 512-dim latent query.
 *    d. AllReduce(max) m_b and AllReduce(sum) l_b: scalar remote store / atomicAdd.
 *    e. AllReduce(sum) of a_lat_b (512 halves = 1 KB): cluster_reduce<ATTN_DEEPSEEK>.
 *    f. AllReduce(sum) of the W_uv partials (128 halves): cluster_reduce<ATTN>.
 */
#include <cuda/barrier>
#include <cudaTypedefs.h>

// ======================================================================
// verbatim from third_party/ClusterFusion/include/H100/deepseek/config.h
// ======================================================================

#define NOPE_HEAD_DIM 128    
#define ROPE_HEAD_DIM 64    
#define HEAD_DIM (ROPE_HEAD_DIM + NOPE_HEAD_DIM) 
#define MLA_HEAD_DIM (KV_LORA_RANK + ROPE_HEAD_DIM)
#define KV_LORA_RANK 512    
#define HEAD_NUM 16     
#define HIDDEN_DIM 2048 
#define SEQ_LEN 4096   

#define NUM_WARPS 4 // 4 8 16 32
#define WARP_SIZE 32
#define BLOCK_SIZE (NUM_WARPS * WARP_SIZE) 
#define CLUSTER_SIZE 4 // 2 4 8 16
#define NUM_PER_THREAD 8
#define NUM_ROW_PER_WARP (NOPE_HEAD_DIM / NUM_WARPS) 
#define NUM_THREAD_PER_ROW (WARP_SIZE / NUM_ROW_PER_WARP) 
#define NUM_PER_ROW (NUM_PER_THREAD * NUM_THREAD_PER_ROW) 
#define DIM_PER_BLOCK (HIDDEN_DIM / CLUSTER_SIZE)
#define KV_DIM_PER_BLOCK (SEQ_LEN / CLUSTER_SIZE) 

#define TMA_LOAD_ONCE 64 // 8 16 32 64 128 256
#define TMA_LOAD_ONCE_NUM (TMA_LOAD_ONCE * NOPE_HEAD_DIM)
#define TMA_LOAD_ONCE_SIZE (TMA_LOAD_ONCE_NUM * sizeof(half))
#define TMA_LOAD_ONCE_ATTN (TMA_LOAD_ONCE / 4) 
#define TMA_LOAD_ONCE_NUM_ATTN (TMA_LOAD_ONCE_ATTN * KV_LORA_RANK) 
#define TMA_LOAD_ONCE_SIZE_ATTN (TMA_LOAD_ONCE_NUM_ATTN * sizeof(half))

#define NUM_ROW_PER_WARP_2 (TMA_LOAD_ONCE / NUM_WARPS) 
#define NUM_THREAD_PER_ROW_2 (WARP_SIZE / NUM_ROW_PER_WARP_2) 
#define NUM_PER_ROW_2 (NUM_PER_THREAD * NUM_THREAD_PER_ROW_2) 
#define NUM_THREAD_PER_ROW_3 (KV_LORA_RANK / NUM_PER_THREAD) 
#define NUM_ROW_PER_BLOCK (BLOCK_SIZE / NUM_THREAD_PER_ROW_3) 
#define DEC_TILE (TMA_LOAD_ONCE_ATTN / NUM_ROW_PER_BLOCK) 
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
// verbatim from third_party/ClusterFusion/include/H100/deepseek/kernel.cuh
// ======================================================================
#include <cuda/barrier>
#include <cudaTypedefs.h>
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;
namespace cg = cooperative_groups;

__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) DeepSeekDecoderLayerKernel(
    half* output, // 1 * hidden_dim
    half* input,  // 1 * hidden_dim
    half* w_rms_input,// hidden_dim
    half* w_rms_ckv,// kv_lora_rank
    float* cos,       // head_dim
    float* sin,       // head_dim
    const __grid_constant__ CUtensorMap tensor_map_weight_q, // hidden_dim * (head_num * nope_head_dim)
    const __grid_constant__ CUtensorMap tensor_map_weight_q_pe, // hidden_dim * (head_num * rope_head_dim)
    const __grid_constant__ CUtensorMap tensor_map_weight_uk, // nope_head_dim * (head_num * kv_lora_rank)
    const __grid_constant__ CUtensorMap tensor_map_weight_kv, // hidden_dim * kv_lora_rank
    const __grid_constant__ CUtensorMap tensor_map_weight_k_pe, // hidden_dim * rope_head_dim
    const __grid_constant__ CUtensorMap tensor_map_kv_cache, // seqlen * mla_head_dim
    const __grid_constant__ CUtensorMap tensor_map_kv_cache_, // seqlen * mla_head_dim
    const __grid_constant__ CUtensorMap tensor_map_weight_uv, // kv_lora_rank * (head_num * nope_head_dim)
    const __grid_constant__ CUtensorMap tensor_map_weight_o // (head_num * nope_head_dim) * hidden_dim
) {
    cg::grid_group grid             = cg::this_grid();
    cg::cluster_group cluster       = cg::this_cluster();
    cg::thread_block block          = cg::this_thread_block();
    const uint32_t head_id          = grid.cluster_rank();
    const uint32_t cluster_block_id = cluster.block_rank();
    const uint32_t tid              = block.thread_rank();
    const uint32_t lane_id = tid % WARP_SIZE; 
    const uint32_t warp_id = tid / WARP_SIZE;
    const uint32_t tile_row = tid / NUM_THREAD_PER_ROW_3;
    const uint32_t tile_col = tid % NUM_THREAD_PER_ROW_3;

    // Init shared memory
    __shared__ __align__(16) half input_shmem[DIM_PER_BLOCK];
    __shared__ float reduction[NUM_WARPS];
    __shared__ float cluster_local_sum, cluster_local_max;
    __shared__ alignas(128) half weight[2 * TMA_LOAD_ONCE * NOPE_HEAD_DIM];
    __shared__ __align__(16) half local_qkv[MLA_HEAD_DIM * 2];

    // Init registers
    float local_sum = 0.0, eps = 1e-6, rms_rcp = 0.0, tmp = 0.0, local_max = 0.0, pre_max = 0.0, scale = 0.0, softmax_scale = __frsqrt_rn(HEAD_DIM);
    half __align__(16) reg_input[NUM_PER_THREAD], reg_weight[NUM_PER_THREAD], reg_reduce[NUM_PER_THREAD];
    float* dst_shmem;
    half2 q_rope, q_rope_1, k_rope, k_rope_1;
    float2 cos_reg, sin_reg;
    uint32_t size;
    uint32_t src_addr, dst_addr, neighbor_dst_bar = 0;
    float __align__(16) qk[DEC_TILE];

    // Init barrier
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar[2];
    barrier::arrival_token token[2];
    __shared__ uint64_t barrier;
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
    if (tid == 0) {
        init(&bar[0], blockDim.x);
        cde::fence_proxy_async_shared_cta();
        init(&bar[1], blockDim.x);
        cde::fence_proxy_async_shared_cta();
    }
    block.sync();

    // Precompute some indices
    uint input_idx = (lane_id % NUM_THREAD_PER_ROW) * NUM_PER_THREAD;
    uint weight_idx = warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW;
    uint input_idx_2 = (lane_id % NUM_THREAD_PER_ROW_2) * NUM_PER_THREAD;
    uint weight_idx_2 = warp_id * NUM_ROW_PER_WARP_2 + lane_id / NUM_THREAD_PER_ROW_2;
    uint input_idx_3 = tile_col * NUM_PER_THREAD;
    uint weight_idx_3 = tile_row * DEC_TILE;
    uint cluster_head_idx = head_id * NOPE_HEAD_DIM;
    uint cluster_head_pe_idx = head_id * ROPE_HEAD_DIM;
    uint cluster_head_uk_idx = head_id * KV_LORA_RANK + cluster_block_id * 128;
    uint cluster_block_st_idx = cluster_block_id * DIM_PER_BLOCK;
    uint cluster_block_st_uv_idx = cluster_block_id * (KV_LORA_RANK / CLUSTER_SIZE);

    // RMSNorm
    for (int d = tid * 4; d < DIM_PER_BLOCK; d+=BLOCK_SIZE * 4) { 
        *(uint64_t*)(&reg_input[0]) = *(uint64_t*)(&input[cluster_block_st_idx + d]);
        for (int di = 0; di < 4; di++)
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
    block.sync();
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
    for (int d = tid * 4; d < DIM_PER_BLOCK; d+=BLOCK_SIZE * 4) { 
        *(uint64_t*)(&reg_weight[0]) = *(uint64_t*)(&w_rms_input[cluster_block_st_idx + d]);
        for (int i = 0; i < 4; i++) {
            reg_input[i] = __float2half(__half2float(reg_input[i]) * rms_rcp * __half2float(reg_weight[i]));
        }
        *(uint64_t*)(&input_shmem[d]) = *(uint64_t*)(&reg_input[0]);
    }
    block.sync();

    // Compute input @ w_q
    // Preload w_q
    if (tid == 0) {
        // [DSMEM] TMA: global -> this CTA's smem (not a cluster op; weights/KV streaming)
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map_weight_q, cluster_head_idx, cluster_block_st_idx, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE);
    } else {
        token[0] = bar[0].arrive();
    }

    for (int id = 1; id < DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
        if (tid == 0) {
            // [DSMEM] TMA: global -> this CTA's smem (not a cluster op; weights/KV streaming)
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], &tensor_map_weight_q, cluster_head_idx, cluster_block_st_idx + id * TMA_LOAD_ONCE, bar[id % 2]);
            token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE);
        } else {
            token[id % 2] = bar[id % 2].arrive();
        }
        bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
            *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                tmp += __half2float(__hmul(reg_input[d], weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + (input_idx + i + d) * NOPE_HEAD_DIM + weight_idx]));
            }
        }
    }
    bar[1].wait(std::move(token[1]));
    for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
        *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + (DIM_PER_BLOCK - TMA_LOAD_ONCE) + i]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(__hmul(reg_input[d], weight[TMA_LOAD_ONCE_NUM + (input_idx + i + d) * NOPE_HEAD_DIM + weight_idx]));
        }
    }
    if (lane_id % NUM_THREAD_PER_ROW == 0) {
        local_qkv[warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW] = __float2half(tmp);
    }
    block.sync();

    // Compute input @ w_q_rope
    if (tid == 0) {
        // [DSMEM] TMA: global -> this CTA's smem (not a cluster op; weights/KV streaming)
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map_weight_q_pe, cluster_head_pe_idx, cluster_block_st_idx, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE);
    } else {
        token[0] = bar[0].arrive();
    }

    tmp = 0.0;
    for (int id = 1; id < DIM_PER_BLOCK / 128; id++) {
        if (tid == 0) {
            // [DSMEM] TMA: global -> this CTA's smem (not a cluster op; weights/KV streaming)
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], &tensor_map_weight_q_pe, cluster_head_pe_idx, cluster_block_st_idx + id * 128, bar[id % 2]);
            token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE);
        } else {
            token[id % 2] = bar[id % 2].arrive();
        }
        bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        for (int i = 0; i < 128; i+=NUM_PER_ROW_2) { 
            *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx_2 + (id - 1) * 128 + i]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                tmp += __half2float(__hmul(reg_input[d], weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + (input_idx_2 + i + d) * ROPE_HEAD_DIM + weight_idx_2]));
            }
        }
    }
    bar[1].wait(std::move(token[1]));
    for (int i = 0; i < 128; i+=NUM_PER_ROW_2) { 
        *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx_2 + (DIM_PER_BLOCK - 128) + i]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(__hmul(reg_input[d], weight[TMA_LOAD_ONCE_NUM + (input_idx_2 + i + d) * ROPE_HEAD_DIM + weight_idx_2]));
        }
    }
    #pragma unroll
    for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
        tmp += __shfl_down_sync(0xffffffff, tmp, mask);
    }
    if (lane_id % NUM_THREAD_PER_ROW_2 == 0) {
        local_qkv[KV_LORA_RANK + warp_id * NUM_ROW_PER_WARP_2 + lane_id / NUM_THREAD_PER_ROW_2] = __float2half(tmp);
    }
    block.sync();

    // Compute input @ w_kv
    for (int s = 0; s < KV_LORA_RANK / 128; s++) {
        tmp = 0.0;
        if (tid == 0) {
            // [DSMEM] TMA: global -> this CTA's smem (not a cluster op; weights/KV streaming)
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map_weight_kv, s * 128, cluster_block_st_idx, bar[0]);
            token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE);
        } else {
            token[0] = bar[0].arrive();
        }

        for (int id = 1; id < DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
            if (tid == 0) {
                // [DSMEM] TMA: global -> this CTA's smem (not a cluster op; weights/KV streaming)
                cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], &tensor_map_weight_kv, s * 128, cluster_block_st_idx + id * TMA_LOAD_ONCE, bar[id % 2]);
                token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE);
            } else {
                token[id % 2] = bar[id % 2].arrive();
            }
            bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
            for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
                *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
                #pragma unroll
                for (int d = 0; d < NUM_PER_THREAD; d++) {
                    tmp += __half2float(__hmul(reg_input[d], weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + (input_idx + i + d) * 128 + weight_idx]));
                }
            }
        }
        bar[1].wait(std::move(token[1]));
        for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
            *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + (DIM_PER_BLOCK - TMA_LOAD_ONCE) + i]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                tmp += __half2float(__hmul(reg_input[d], weight[TMA_LOAD_ONCE_NUM + (input_idx + i + d) * 128 + weight_idx]));
            }
        }
        if (lane_id % NUM_THREAD_PER_ROW == 0) {
            local_qkv[MLA_HEAD_DIM + s * 128 + warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW] = __float2half(tmp);
        }
    }
    block.sync();

    // Compute input @ w_k_pe
    if (tid == 0) {
        // [DSMEM] TMA: global -> this CTA's smem (not a cluster op; weights/KV streaming)
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map_weight_k_pe, 0, cluster_block_st_idx, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE);
    } else {
        token[0] = bar[0].arrive();
    }

    tmp = 0.0;
    for (int id = 1; id < DIM_PER_BLOCK / 128; id++) {
        if (tid == 0) {
            // [DSMEM] TMA: global -> this CTA's smem (not a cluster op; weights/KV streaming)
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], &tensor_map_weight_k_pe, 0, cluster_block_st_idx + id * 128, bar[id % 2]);
            token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE);
        } else {
            token[id % 2] = bar[id % 2].arrive();
        }
        bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        for (int i = 0; i < 128; i+=NUM_PER_ROW_2) { 
            *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx_2 + (id - 1) * 128 + i]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                tmp += __half2float(__hmul(reg_input[d], weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + (input_idx_2 + i + d) * ROPE_HEAD_DIM + weight_idx_2]));
            }
        }
    }
    bar[1].wait(std::move(token[1]));
    for (int i = 0; i < 128; i+=NUM_PER_ROW_2) { 
        *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx_2 + (DIM_PER_BLOCK - 128) + i]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(__hmul(reg_input[d], weight[TMA_LOAD_ONCE_NUM + (input_idx_2 + i + d) * ROPE_HEAD_DIM + weight_idx_2]));
        }
    }
    #pragma unroll
    for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
        tmp += __shfl_down_sync(0xffffffff, tmp, mask);
    }
    if (lane_id % NUM_THREAD_PER_ROW_2 == 0) {
        local_qkv[MLA_HEAD_DIM + KV_LORA_RANK + warp_id * NUM_ROW_PER_WARP_2 + lane_id / NUM_THREAD_PER_ROW_2] = __float2half(tmp);
    }
    block.sync();

    // DSM All-reduce
    size = MLA_HEAD_DIM * 2 * sizeof(half);
    src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(local_qkv));
    dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(weight));
    // [DSMEM] ClusterReduce (vector all-reduce, shift-and-accumulate bulk push, N-2 rounds)
    cluster_reduce<CLUSTER_SIZE, Stage::LINEAR_DEEPSEEK>(
        size, tid, BLOCK_SIZE * 8, cluster_block_id,  
        src_addr, dst_addr, bar_ptr, 
        neighbor_dst_bar, local_qkv, weight);

    // Compute partial RoPE
    if (tid < ROPE_HEAD_DIM / 2) {
        q_rope = *(half2*)(&local_qkv[KV_LORA_RANK + tid * 2]);
        k_rope = *(half2*)(&local_qkv[MLA_HEAD_DIM + KV_LORA_RANK + tid * 2]);
        if (tid * 2 < ROPE_HEAD_DIM / 2) {
            q_rope_1 = *(half2*)(&local_qkv[KV_LORA_RANK + ROPE_HEAD_DIM / 2 + tid * 2]);
            k_rope_1 = *(half2*)(&local_qkv[MLA_HEAD_DIM + ROPE_HEAD_DIM / 2 + tid * 2]);
            cos_reg = {cos[tid * 2], cos[tid * 2 + 1]};
            sin_reg = {-sin[ROPE_HEAD_DIM / 2 + tid * 2], -sin[ROPE_HEAD_DIM / 2 + tid * 2 + 1]};
        } else {
            q_rope_1 = *(half2*)(&local_qkv[KV_LORA_RANK + tid * 2 - ROPE_HEAD_DIM / 2]);
            k_rope_1 = *(half2*)(&local_qkv[MLA_HEAD_DIM + KV_LORA_RANK + tid * 2 - ROPE_HEAD_DIM / 2]);
            cos_reg = {cos[tid * 2], cos[tid * 2 + 1]};
            sin_reg = {sin[tid * 2 - ROPE_HEAD_DIM / 2], sin[tid * 2 + 1 - ROPE_HEAD_DIM / 2]};
        }
        *(half2*)(&local_qkv[KV_LORA_RANK + tid * 2]) = __hadd2(__hmul2(q_rope, __float22half2_rn(cos_reg)), __hmul2(q_rope_1, __float22half2_rn(sin_reg)));
        *(half2*)(&local_qkv[MLA_HEAD_DIM + KV_LORA_RANK + tid * 2]) = __hadd2(__hmul2(k_rope, __float22half2_rn(cos_reg)), __hmul2(k_rope_1, __float22half2_rn(sin_reg)));
    }

    // RMSNorm
    local_sum = 0.0;
    for (int d = tid * 4; d < KV_LORA_RANK; d+=BLOCK_SIZE * 4) { 
        *(uint64_t*)(&reg_input[0]) = *(uint64_t*)(&local_qkv[MLA_HEAD_DIM + d]);
        for (int di = 0; di < 4; di++)
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
    rms_rcp = __frsqrt_rn(cluster_local_sum / KV_LORA_RANK + eps);
    for (int d = tid * 4; d < KV_LORA_RANK; d+=BLOCK_SIZE * 4) { 
        *(uint64_t*)(&reg_weight[0]) = *(uint64_t*)(&w_rms_ckv[d]);
        for (int i = 0; i < 4; i++) {
            reg_input[i] = __float2half(__half2float(reg_input[i]) * rms_rcp * __half2float(reg_weight[i]));
        }
        *(uint64_t*)(&local_qkv[MLA_HEAD_DIM + d]) = *(uint64_t*)(&reg_input[0]);
    }
    block.sync();
    
    // Compute q @ w_uk
    if (tid == 0) {
        // [DSMEM] TMA: global -> this CTA's smem (not a cluster op; weights/KV streaming)
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map_weight_uk, cluster_head_uk_idx, 0, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE);
    } else {
        token[0] = bar[0].arrive();
    }

    tmp = 0.0;
    for (int id = 1; id < NOPE_HEAD_DIM / TMA_LOAD_ONCE; id++) {
        if (tid == 0) {
            // [DSMEM] TMA: global -> this CTA's smem (not a cluster op; weights/KV streaming)
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], &tensor_map_weight_uk, cluster_head_uk_idx, id * TMA_LOAD_ONCE, bar[id % 2]);
            token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE);
        } else {
            token[id % 2] = bar[id % 2].arrive();
        }
        bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
            *(uint4*)(&reg_input[0]) = *(uint4*)(&local_qkv[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                tmp += __half2float(__hmul(reg_input[d], weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + (input_idx + i + d) * 128 + weight_idx]));
            }
        }
    }
    bar[1].wait(std::move(token[1]));
    for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
        *(uint4*)(&reg_input[0]) = *(uint4*)(&local_qkv[input_idx + (NOPE_HEAD_DIM - TMA_LOAD_ONCE) + i]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(__hmul(reg_input[d], weight[TMA_LOAD_ONCE_NUM + (input_idx + i + d) * 128 + weight_idx]));
        }
    }
    if (lane_id % NUM_THREAD_PER_ROW == 0) {
        local_qkv[cluster_block_id * 128 + warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW] = __float2half(tmp);
    }
    block.sync();

    // DSM All-gather
    size = 128 * sizeof(half);
    src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&local_qkv[cluster_block_id * 128]));
    dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&local_qkv[cluster_block_id * 128]));
    // [DSMEM] ClusterGather (all-gather, direct shift push, N-1 rounds)
    cluster_reduce<CLUSTER_SIZE, Stage::QUK_DEEPSEEK>(
        size, tid, 0, cluster_block_id,  
        src_addr, dst_addr, bar_ptr, 
        neighbor_dst_bar, local_qkv, local_qkv);

    // Compute flash-decoding
    local_sum = 0.0f;
    for(int i = 0; i < NUM_PER_THREAD; i++)
        reg_reduce[i] = __float2half(0.0f);
    *(uint4*)(&reg_input[0]) = *(uint4*)(&local_qkv[input_idx_3]);
    block.sync();

    // Preload kv_cache
    if (tid == 0) {
        // [DSMEM] TMA: global -> this CTA's smem (not a cluster op; weights/KV streaming)
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map_kv_cache, 0, cluster_block_id * KV_DIM_PER_BLOCK, bar[0]);
        // [DSMEM] TMA: global -> this CTA's smem (not a cluster op; weights/KV streaming)
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[TMA_LOAD_ONCE_NUM_ATTN / 2], &tensor_map_kv_cache_, KV_LORA_RANK / 2, cluster_block_id * KV_DIM_PER_BLOCK, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE_ATTN);
    } else {
        token[0] = bar[0].arrive();
    }

    for (int id = 1; id < KV_DIM_PER_BLOCK / TMA_LOAD_ONCE_ATTN; id++) {
        if (tid == 0) {
            // [DSMEM] TMA: global -> this CTA's smem (not a cluster op; weights/KV streaming)
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM_ATTN], &tensor_map_kv_cache, 0, cluster_block_id * KV_DIM_PER_BLOCK + id * TMA_LOAD_ONCE_ATTN, bar[id % 2]);
            // [DSMEM] TMA: global -> this CTA's smem (not a cluster op; weights/KV streaming)
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM_ATTN + TMA_LOAD_ONCE_NUM_ATTN / 2], &tensor_map_kv_cache_, KV_LORA_RANK / 2, cluster_block_id * KV_DIM_PER_BLOCK + id * TMA_LOAD_ONCE_ATTN, bar[id % 2]);
            token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE_ATTN);
        } else {
            token[id % 2] = bar[id % 2].arrive();
        }
        bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        pre_max = local_max;
        #pragma unroll
        for (int j = 0; j < DEC_TILE; j++) {
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM_ATTN + (warp_id % 2) * TMA_LOAD_ONCE_NUM_ATTN / 2 + (weight_idx_3 + j) * KV_LORA_RANK / 2 + lane_id * NUM_PER_THREAD]);
            qk[j] = 0.0f;
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                qk[j] += __half2float(__hmul(reg_input[d], reg_weight[d]));
            }
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                qk[j] += __shfl_xor_sync(0xffffffff, qk[j], mask);
            }
            block.sync();
            if (lane_id == 0)
                reduction[warp_id] = qk[j];
            block.sync();
            if (tile_row == 1)
                qk[j] += reduction[2 + (warp_id + 1) % 2];
            else
                qk[j] += reduction[(warp_id + 1) % 2];
            qk[j] = qk[j] * softmax_scale;
            local_max = fmaxf(local_max, qk[j]);
        }
        scale = __expf(pre_max - local_max);
        local_sum *= scale;
        #pragma unroll
        for (int j = 0; j < DEC_TILE; j++) {
            qk[j] = __expf(qk[j] - local_max);
            local_sum += qk[j];
        }
        #pragma unroll
        for (int j = 0; j < NUM_PER_THREAD; j++) {
            reg_reduce[j] = __hmul(reg_reduce[j], __float2half(scale));
        }
        #pragma unroll
        for (int j = 0; j < DEC_TILE; j++) {
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM_ATTN + (warp_id % 2) * TMA_LOAD_ONCE_NUM_ATTN / 2 + (weight_idx_3 + j) * KV_LORA_RANK / 2 + lane_id * NUM_PER_THREAD]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                reg_reduce[d] = __hadd(reg_reduce[d], __float2half(qk[j] * __half2float(reg_weight[d])));
            }
        }
    }
    bar[1].wait(std::move(token[1]));
    pre_max = local_max;
    #pragma unroll
    for (int j = 0; j < DEC_TILE; j++) {
        if (cluster_block_id == CLUSTER_SIZE - 1 && tile_row == 1 && j == DEC_TILE - 1)
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&local_qkv[MLA_HEAD_DIM + input_idx_3]);
        else
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[TMA_LOAD_ONCE_NUM_ATTN + (warp_id % 2) * TMA_LOAD_ONCE_NUM_ATTN / 2 + (weight_idx_3 + j) * KV_LORA_RANK / 2 + lane_id * NUM_PER_THREAD]);
        qk[j] = 0.0f;
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            qk[j] += __half2float(__hmul(reg_input[d], reg_weight[d]));
        }
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            qk[j] += __shfl_xor_sync(0xffffffff, qk[j], mask);
        }
        if (lane_id == 0)
            reduction[warp_id] = qk[j];
        block.sync();
        if (tile_row == 1)
            qk[j] += reduction[2 + (warp_id + 1) % 2];
        else
            qk[j] += reduction[(warp_id + 1) % 2];
        qk[j] = qk[j] * softmax_scale;
        local_max = fmaxf(local_max, qk[j]);
    }
    scale = __expf(pre_max - local_max);
    local_sum *= scale;
    #pragma unroll
    for (int j = 0; j < DEC_TILE; j++) {
        qk[j] = __expf(qk[j] - local_max);
        local_sum += qk[j];
    }
    #pragma unroll
    for (int j = 0; j < NUM_PER_THREAD; j++) {
        reg_reduce[j] = __hmul(reg_reduce[j], __float2half(scale));
    }
    #pragma unroll
    for (int j = 0; j < DEC_TILE; j++) {
        if (cluster_block_id == CLUSTER_SIZE - 1 && tile_row == 1 && j == DEC_TILE - 1)
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&local_qkv[MLA_HEAD_DIM + input_idx_3]);
        else
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[TMA_LOAD_ONCE_NUM_ATTN + (warp_id % 2) * TMA_LOAD_ONCE_NUM_ATTN / 2 + (weight_idx_3 + j) * KV_LORA_RANK / 2 + lane_id * NUM_PER_THREAD]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            reg_reduce[d] = __hadd(reg_reduce[d], __float2half(qk[j] * __half2float(reg_weight[d])));
        }
    }
    block.sync();

    *(uint4*)(&weight[tile_row * MLA_HEAD_DIM + tile_col * NUM_PER_THREAD]) = *(uint4*)(&reg_reduce[0]);
    if (tid % NUM_THREAD_PER_ROW_3 == 0) {
        reduction[tile_row * 2] = local_max;
        reduction[tile_row * 2 + 1] = local_sum;
    }
    block.sync();
    for(int i = 0; i < NUM_PER_THREAD; i++)
        reg_reduce[i] = __float2half(0.0f);
    local_sum = 0.0, local_max = 0.0;
    #pragma unroll
    for(int j = 0; j < NUM_ROW_PER_BLOCK; j++) {
        *(uint4*)(&reg_input[0]) = *(uint4*)(&weight[j * MLA_HEAD_DIM + tile_col * NUM_PER_THREAD]);
        float m = reduction[j * 2], s = reduction[j * 2 + 1];
        pre_max = local_max;
        local_max = fmaxf(m, local_max);
        scale = __expf(m - local_max);
        s *= scale;
        local_sum = local_sum * __expf(pre_max - local_max) + s;
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            reg_reduce[d] = __hadd(__hmul(reg_reduce[d], __float2half(__expf(pre_max - local_max))), __hmul(reg_input[d], __float2half(scale)));
        }
    }
    block.sync();
    pre_max = local_max;
    if(tid == 0) {
        cluster_local_max = local_max;
    }
    // [DSMEM] barrier.cluster: all threads of all CTAs in the cluster
    cluster.sync();
    // ClusterReduce
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
    scale = __expf(pre_max - cluster_local_max);
    local_sum *= scale;
    #pragma unroll
    for (int j = 0; j < NUM_PER_THREAD; j++) {
        reg_reduce[j] = __hmul(reg_reduce[j], __float2half(scale));
    }
    if(tid == 0) {
        cluster_local_sum = local_sum;
    }
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
    #pragma unroll
    for (int j = 0; j < NUM_PER_THREAD; j++) {
        reg_reduce[j] = __hmul(reg_reduce[j], __float2half(__frcp_rn(cluster_local_sum)));
    }
    if(tid < NUM_THREAD_PER_ROW_3) {
        *(uint4*)(&local_qkv[tid * NUM_PER_THREAD]) = *(uint4*)(&reg_reduce[0]);
    }
    block.sync();
    // ClusterReduce
    size = KV_LORA_RANK * sizeof(half);
    src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(local_qkv));
    dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(weight));
    // [DSMEM] ClusterReduce (vector all-reduce, shift-and-accumulate bulk push, N-2 rounds)
    cluster_reduce<CLUSTER_SIZE, Stage::ATTN_DEEPSEEK>(
        size, tid, KV_LORA_RANK, cluster_block_id,  
        src_addr, dst_addr, bar_ptr, 
        neighbor_dst_bar, local_qkv, weight);
    
    // Compute output @ w_uv
    if (tid == 0) {
        // [DSMEM] TMA: global -> this CTA's smem (not a cluster op; weights/KV streaming)
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map_weight_uv, cluster_head_idx, cluster_block_st_uv_idx, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE);
    } else {
        token[0] = bar[0].arrive();
    }

    tmp = 0.0;
    for (int id = 1; id < 128 / TMA_LOAD_ONCE; id++) {
        if (tid == 0) {
            // [DSMEM] TMA: global -> this CTA's smem (not a cluster op; weights/KV streaming)
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], &tensor_map_weight_uv, cluster_head_idx, cluster_block_st_uv_idx + id * TMA_LOAD_ONCE, bar[id % 2]);
            token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE);
        } else {
            token[id % 2] = bar[id % 2].arrive();
        }
        bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
            *(uint4*)(&reg_input[0]) = *(uint4*)(&local_qkv[cluster_block_st_uv_idx + input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                tmp += __half2float(__hmul(reg_input[d], weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + (input_idx + i + d) * 128 + weight_idx]));
            }
        }
    }
    bar[1].wait(std::move(token[1]));
    for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
        *(uint4*)(&reg_input[0]) = *(uint4*)(&local_qkv[cluster_block_st_uv_idx + input_idx + (NOPE_HEAD_DIM - TMA_LOAD_ONCE) + i]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(__hmul(reg_input[d], weight[TMA_LOAD_ONCE_NUM + (input_idx + i + d) * 128 + weight_idx]));
        }
    }
    if (lane_id % NUM_THREAD_PER_ROW == 0) {
        local_qkv[warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW] = __float2half(tmp);
    }
    block.sync();

    size = NOPE_HEAD_DIM * sizeof(half);
    src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(local_qkv));
    dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(weight));
    // [DSMEM] ClusterReduce (vector all-reduce, shift-and-accumulate bulk push, N-2 rounds)
    cluster_reduce<CLUSTER_SIZE, Stage::ATTN>(
        size, tid, NOPE_HEAD_DIM, cluster_block_id,  
        src_addr, dst_addr, bar_ptr, 
        neighbor_dst_bar, local_qkv, weight);

    // Compute output @ w_o
    // Preload w_o
    if (tid == 0) {
        // [DSMEM] TMA: global -> this CTA's smem (not a cluster op; weights/KV streaming)
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map_weight_o, cluster_block_st_idx, cluster_head_idx, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE);
    } else {
        token[0] = bar[0].arrive();
    }

    for (int id = 1; id < DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
        if (tid == 0) {
            // [DSMEM] TMA: global -> this CTA's smem (not a cluster op; weights/KV streaming)
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], &tensor_map_weight_o, cluster_block_st_idx + id * TMA_LOAD_ONCE, cluster_head_idx, bar[id % 2]);
            token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE);
        } else {
            token[id % 2] = bar[id % 2].arrive();
        }
        bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        tmp = 0.0;
        for (int j = 0; j < NOPE_HEAD_DIM; j+=NUM_PER_ROW_2) {
            *(uint4*)(&reg_input[0]) = *(uint4*)(&local_qkv[input_idx_2 + j]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                tmp += __half2float(__hmul(reg_input[d], weight[(id - 1) % 2 * TMA_LOAD_ONCE_NUM + (input_idx_2 + j + d) * TMA_LOAD_ONCE + weight_idx_2]));
            }
        }
        #pragma unroll
        for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
            tmp += __shfl_down_sync(0xffffffff, tmp, mask);
        }
        if (lane_id % NUM_THREAD_PER_ROW_2 == 0) {
            atomicAdd(&output[cluster_block_st_idx + weight_idx_2 + (id - 1) * TMA_LOAD_ONCE], __float2half(tmp));
        }
        block.sync();
    }
    bar[1].wait(std::move(token[1]));
    tmp = 0.0;
    for (int j = 0; j < NOPE_HEAD_DIM; j+=NUM_PER_ROW_2) {
        *(uint4*)(&reg_input[0]) = *(uint4*)(&local_qkv[input_idx_2 + j]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(__hmul(reg_input[d], weight[TMA_LOAD_ONCE_NUM + (input_idx_2 + j + d) * TMA_LOAD_ONCE + weight_idx_2]));
        }
    }
    #pragma unroll
    for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
        tmp += __shfl_down_sync(0xffffffff, tmp, mask);
    }
    if (lane_id % NUM_THREAD_PER_ROW_2 == 0) {
        atomicAdd(&output[cluster_block_st_idx + weight_idx_2 + ((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) * TMA_LOAD_ONCE], __float2half(tmp));
    }
}
