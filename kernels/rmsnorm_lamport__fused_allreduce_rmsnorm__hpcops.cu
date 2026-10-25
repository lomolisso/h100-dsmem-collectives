/*
 * RMSNorm half of the fused multi-GPU AllReduce + RMSNorm, low-latency Lamport mode (Tencent hpc-ops).
 * Source: third_party/hpc-ops/src/allreduce/fuse_allreduce_rmsnorm_low_latency.cu, kernels::rmsNormLamport,
 * kept verbatim. Needs -I third_party/hpc-ops. The multi-GPU parts (LamportFlags, polling the NVLink
 * broadcast buffer for non -0.0 values, PDL) are OUT OF SCOPE for this study and are marked below; only the
 * intra-GPU cluster reduction of the sum of squares is a DSMEM collective.
 *
 * 1. Computation: per token, h = allreduce_input + residual (written back as the pre-norm residual), then
 *    y = h * rsqrt(sum(h^2)/dim + eps) * gamma; bf16 in/out, fp32 statistics.
 * 2. Split: grid (numTokens, clusterSize) with cluster dims (1, clusterSize, 1): one cluster per token, the
 *    hidden dimension split into clusterSize contiguous chunks (clusterSize = 8 halved until smem fits);
 *    each CTA loads its chunk of input/residual/gamma to smem and reduces its partial sum of squares.
 * 3. Collective: AllReduce(sum) of one float by direct replicated push: thread t < numBlocks stores blockSum
 *    into sharedVal[blockRank] of CTA t (cluster.map_shared_rank), one split barrier.cluster arrive/wait,
 *    then each CTA sums its 8 local slots. Same pattern as CREDIT's cluster_reduce_pair.
 */

// ======================================================================
// verbatim from third_party/hpc-ops/src/allreduce/fuse_allreduce_rmsnorm_low_latency.cu (lines 1-15, 143-300)
// ======================================================================
// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>

#include <algorithm>

#include "src/allreduce/fuse_allreduce_rmsnorm_low_latency.h"
#include "src/utils/utils.cuh"
#include "src/utils/utils.h"

namespace hpc {
namespace allreduce {
namespace kernels {
template <typename T_IN, typename T_OUT, int LoadsPerThread = 1>
__global__ __launch_bounds__(1024) void rmsNormLamport(T_IN* outputPreNorm, T_OUT* outputNorm,
                                                       T_IN* bufferInput, T_IN const* gamma,
                                                       float epsilon, T_IN const* residual,
                                                       uint32_t numTokens, uint32_t dim,
                                                       uint32_t worldSize, uint32_t* bufferFlags) {
  static_assert(std::is_same_v<T_IN, T_OUT>, "T_IN and T_OUT must be the same type");
  static int const kELTS_PER_LOAD = sizeof(float4) / sizeof(T_IN);

  uint32_t const token = blockIdx.x;
  uint32_t const blockSize = blockDim.x;
  uint32_t const threadOffset = threadIdx.x;

  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  uint32_t numThreads = cluster.num_threads();
  uint32_t clusterSize = cluster.num_blocks();
  uint32_t blockOffset = cluster.block_rank();
  uint32_t const dimPadded = round_up(dim, kELTS_PER_LOAD * numThreads);
  uint32_t const elemsPerThread = dimPadded / numThreads;
  uint32_t const loadStride = blockSize;

  extern __shared__ uint8_t smem[];
  float rInput[LoadsPerThread * kELTS_PER_LOAD];
  uint32_t offsets[LoadsPerThread * kELTS_PER_LOAD];

  uint32_t const smemBufferSize = blockSize * elemsPerThread * sizeof(T_IN);
  T_IN* smemInput = (T_IN*)&smem[0];
  T_IN* smemResidual = (T_IN*)&smem[smemBufferSize];
  T_IN* smemGamma = (T_IN*)&smem[2 * smemBufferSize];

  // [DSMEM] multi-GPU (out of scope): Lamport buffer bookkeeping for the NVLink two-shot allreduce
  LamportFlags<float4> flag(bufferFlags, MNNVLTwoShotStage::NUM_STAGES);
  T_IN* input = reinterpret_cast<T_IN*>(
      flag.getCurLamportBuf(reinterpret_cast<void*>(bufferInput), MNNVLTwoShotStage::BROADCAST));

  cudaTriggerProgrammaticLaunchCompletion();
  // The offset that current thread should load from. Note that the hidden dimension is split by CGA
  // size and each block loads a contiguous chunk; The size of chunk that each block processes
  uint32_t const blockChunkSize = ceil_div(dim, clusterSize * kELTS_PER_LOAD) * kELTS_PER_LOAD;
  uint32_t const blockLoadOffset = token * dim + blockOffset * blockChunkSize;

#pragma unroll
  for (uint32_t i = 0; i < LoadsPerThread; i++) {
    // Each block load a contiguous chunk of tokens
    uint32_t const threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
    offsets[i] = blockLoadOffset + threadLoadOffset;
  }

#pragma unroll
  for (uint32_t i = 0; i < LoadsPerThread; i++) {
    uint32_t const threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
    if (blockOffset * blockChunkSize + threadLoadOffset < dim) {
      copyF4(&smemResidual[threadLoadOffset], &residual[blockLoadOffset + threadLoadOffset]);
    }
  }
  __pipeline_commit();
#pragma unroll
  for (uint32_t i = 0; i < LoadsPerThread; i++) {
    uint32_t const threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
    if (blockOffset * blockChunkSize + threadLoadOffset < dim) {
      copyF4(&smemGamma[threadLoadOffset], &gamma[blockOffset * blockChunkSize + threadLoadOffset]);
    }
  }
  __pipeline_commit();

  // [DSMEM] multi-GPU (out of scope): per-cluster arrival counter in global memory
  flag.ctaArrive();
  bool valid = false;
  // ACQBLK if not lamport
  // [DSMEM] multi-GPU (out of scope): poll the broadcast buffer until all 16-B packets are non -0.0
  while (!valid) {
    valid = true;
#pragma unroll
    for (uint32_t i = 0; i < LoadsPerThread; i++) {
      uint32_t threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;

      if (blockOffset * blockChunkSize + threadLoadOffset < dim) {
        float4* dst4 = reinterpret_cast<float4*>(&smemInput[threadLoadOffset]);
        float4 const* src4 = reinterpret_cast<float4 const*>(&input[offsets[i]]);

        float4 value = loadPackedVolatile<float4>(src4);
        // Assume that the 16B were written atomically, so we only need to check one value
        valid &= !isNegZero(value.x);
        *dst4 = value;
      }
    }
  }

  __pipeline_wait_prior(1);
  __syncthreads();

  float threadSum = 0.f;
#pragma unroll
  for (int i = 0; i < LoadsPerThread; i++) {
    int threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
    if (blockOffset * blockChunkSize + threadLoadOffset < dim) {
      PackedVec<float4, T_IN> inp{.packed = loadPacked<float4>(&smemInput[threadLoadOffset])};
      PackedVec<float4, T_IN> res{.packed = loadPacked<float4>(&smemResidual[threadLoadOffset])};

      PackedVec<float4, T_IN> inp_plus_res = inp + res;
#pragma unroll
      for (int j = 0; j < kELTS_PER_LOAD; j++) {
        rInput[i * kELTS_PER_LOAD + j] = toFloat<T_IN>(inp_plus_res.elements[j]);
        threadSum += toFloat<T_IN>(inp_plus_res.elements[j] * inp_plus_res.elements[j]);
      }

      *reinterpret_cast<float4*>(&outputPreNorm[blockLoadOffset + threadLoadOffset]) =
          inp_plus_res.packed;
    }
  }

  __pipeline_wait_prior(0);

  float blockSum = blockReduceSum<float, true>(threadSum);

  float fullSum = blockSum;
  __shared__ float sharedVal[8];
  // B200 always supports CGA Reduction
  int const numBlocks = cluster.num_blocks();
  if (numBlocks > 1) {
    fullSum = 0.F;
    // Need to reduce over the entire cluster
    int const blockRank = cluster.block_rank();
    if (threadIdx.x < numBlocks) {
      // [DSMEM] generic DSMEM pointer into a peer CTA's smem
      cluster.map_shared_rank(&sharedVal[0], threadIdx.x)[blockRank] = blockSum;
    }
    // [DSMEM] split barrier.cluster arrive/wait
    cluster.barrier_wait(cluster.barrier_arrive());
    for (int i = 0; i < numBlocks; ++i) {
      fullSum += sharedVal[i];
    }
  }

  float rcpRms = rsqrtf(fullSum / dim + epsilon);

#pragma unroll
  for (int i = 0; i < LoadsPerThread; i++) {
    PackedVec<float4, T_OUT> r_out;
    uint32_t threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
    if (blockOffset * blockChunkSize + threadLoadOffset < dim) {
      PackedVec<float4, T_IN> gamma = {.packed = loadPacked<float4>(&smemGamma[threadLoadOffset])};

#pragma unroll
      for (uint32_t j = 0; j < kELTS_PER_LOAD; j++) {
        r_out.elements[j] = fromFloat<T_OUT>(toFloat<T_IN>(gamma.elements[j]) *
                                             rInput[i * kELTS_PER_LOAD + j] * rcpRms);
      }

      *reinterpret_cast<float4*>(&outputNorm[blockLoadOffset + threadLoadOffset]) = r_out.packed;
    }
  }
  constexpr int kELTS_SIZE = sizeof(T_IN);

  // Assume the previous kernel does not modify the buffer_flags.
  cudaGridDependencySynchronize();
  // Update the buffer pointers
  // [DSMEM] multi-GPU (out of scope): rotate Lamport buffers
  flag.waitAndUpdate({static_cast<uint32_t>(round_up(numTokens, worldSize) * dim * kELTS_SIZE),
                      static_cast<uint32_t>(numTokens * dim * kELTS_SIZE), 0, 0});
}

}  // namespace kernels

}  // namespace allreduce
}  // namespace hpc
