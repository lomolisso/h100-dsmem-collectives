/*
 * Cluster-8 exact top-k over one row of logits (Tencent hpc-ops, LLM sampling / filtered top-k).
 * Source: third_party/hpc-ops/src/topk/topk_filtered.cu, topk_filtered_cluster8_exact_kernel and the device
 * helpers it needs (other top-k paths removed). Needs -I third_party/hpc-ops for src/topk/*.h and src/utils.
 *
 * 1. Computation: for a row of `length` fp32 scores select the indices of the kTopK (2048) largest values.
 *    Radix-select: (i) coarse 11-bit histogram of a monotone key (RADIX = 2048 bins); (ii) prefix sum to find
 *    the threshold bin tau such that count(key > tau) <= k; (iii) scan again: keys above tau are selected
 *    outright, keys in bin tau are candidates; (iv) refine the candidate bin with up to 3 finer 11/11/10-bit
 *    histogram rounds until exactly k - selected ties remain.
 *
 * 2. Split: one cluster of kSplits = 8 CTAs (1024 threads each) per row; CTA b scans the contiguous segment
 *    [b*seg_len, (b+1)*seg_len) and builds a *replicated* full histogram of its segment in its own smem.
 *    Every CTA then needs the row-global histogram (same threshold everywhere), and rank 0 needs all
 *    selected indices / boundary candidates to write the output.
 *
 * 3. Collectives:
 *    a. AllReduce(sum) of the 2048-bin histogram, cluster_allreduce_histogram: rank b owns 256 bins; for each
 *       owned bin 8 consecutive lanes each pull that bin from a distinct peer (map_shared_rank + load),
 *       reduce with 3 xor-shuffles, then push the total back to all 8 copies (store); cluster.sync before/after.
 *       Pull reduce-scatter + push all-gather, 2 barriers, 8 KB in + 8 KB out per CTA.
 *    b. Rooted gather to rank 0 (`primary`): peers reserve disjoint output ranges with remote atomicAdd on
 *       rank 0's counters, publish their candidate interval with remote stores, and (aligned path) rank 0
 *       pulls one (index,key) pair per thread straight from the owning CTA's smem; (unaligned path) peers
 *       push indices/candidates into rank 0's smem. Several cluster.sync() rendezvous keep every CTA's DSMEM
 *       alive until rank 0 is done (a CTA may not exit while peers still address its shared memory).
 */

// ======================================================================
// verbatim from third_party/hpc-ops/src/topk/topk_filtered.cu (constants, smem layout, helpers)
// ======================================================================
// Copyright (C) 2026 Tencent.

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_fp16.h>

#include <algorithm>

#include "src/topk/topk.h"
#include "src/topk/topk_filtered_boundary.cuh"
#include "src/utils/utils.h"

namespace hpc {
namespace topk {
namespace filtered {

constexpr int RADIX = 2048;  // 11-bit fp16 coarse / 11-bit fp32 fine (last round 10-bit)
constexpr int NUM_REFINE_ROUNDS = 3;

// The sampled paths derive a coarse threshold from a fixed systematic view.
constexpr int SAMPLE_STRIDE = 64;
constexpr int SAMPLE_RANK = 48;
constexpr int WIDE_SAMPLE_STRIDE = 128;
constexpr int WIDE_SAMPLE_RANK = 24;

// Bit layouts of the three fp32-ordered fine refinement rounds.
//   round 0 → bits [21..31]  (top 11 bits)
//   round 1 → bits [10..20]  (middle 11 bits)
//   round 2 → bits [0..9]    (low 10 bits, back-fill round)
__device__ constexpr int kFineShifts[NUM_REFINE_ROUNDS] = {21, 10, 0};
__device__ constexpr uint32_t kFineMasks[NUM_REFINE_ROUNDS] = {0x7FFu, 0x7FFu, 0x3FFu};

constexpr int THREADS = 512;
constexpr int MIN_BLOCKS = 3;

// smem candidate capacity per ping-pong slot. The sampled path keeps twice as many
// because its predicted threshold intentionally admits more than top_k.
constexpr int COARSE_CANDIDATE_CAPACITY = 2048;
constexpr int SAMPLED_CANDIDATE_CAPACITY = 4096;

// Persistent grid: a fixed pool of BLOCKS_PER_SM CTAs per SM pulling rows from
// the work queue. Also sizes the gmem spill region ([num_blocks, 2, n]).
constexpr int BLOCKS_PER_SM = 3;

// Persistent grid size (num_sms * BLOCKS_PER_SM). Shared by the launch and the
// workspace sizing so the spill buffer always has exactly gridDim.x rows.
inline int grid_num_blocks() { return get_sm_count() * BLOCKS_PER_SM; }

// The low/high halves hold the descending coarse keys for x/y.
__device__ __forceinline__ uint32_t to_coarse_key2(float x, float y) {
  return static_cast<uint32_t>(to_coarse_key(x)) | (static_cast<uint32_t>(to_coarse_key(y)) << 16);
}

__device__ __forceinline__ uint32_t to_ordered(float x) {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? bits : ((bits ^ 0xFFFFFFFFu) & 0x7FFFFFFFu);
}

// Vectorized float4 load (cache at global level only).
__device__ __forceinline__ float4 ld_cg_float4(const float* p) {
  float4 v;
  asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];\n"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
               : "l"(p));
  return v;
}

// Shared-memory layout for one row. The sampled path keeps more candidates
// because its predicted threshold intentionally admits more than K.
template <int kBlockThreads>
struct PrefixStorage {
  int warp_sum[kBlockThreads / 32];
};

template <int kTopK, bool kSampled, int kBlockThreads = THREADS>
struct RowSmem {
  static constexpr int CANDIDATE_CAPACITY =
      kSampled ? SAMPLED_CANDIDATE_CAPACITY : COARSE_CANDIDATE_CAPACITY;

  int histogram[RADIX + 1];  // coarse / refine histogram (prefix-summed in place)
  int threshold_bin;         // bin holding the top-k boundary
  int num_selected;          // front fill cursor into indices
  int ties_remaining;        // back fill cursor for the last round's ties
  int num_cand[2];           // ping-pong candidate counts (smem portion)
  int num_spilled_cand[2];   // ping-pong candidate counts (gmem spill portion)
  int32_t indices[kTopK];    // accumulated result indices
  alignas(8) int32_t cand_idx[2][CANDIDATE_CAPACITY];  // ping-pong candidate index buffers
  PrefixStorage<kBlockThreads> prefix;
};

struct RowCandidate {
  int32_t index;
  uint32_t key;
};

template <int kTopK, int kBlockThreads>
struct RowLocalSmem {
  static constexpr int CANDIDATE_CAPACITY = kBlockThreads;

  int histogram[RADIX + 1];
  int threshold_bin;
  int threshold_above;
  int threshold_count;
  int num_selected;
  int ties_remaining;
  int num_candidates;
  int32_t indices[kTopK];
  alignas(8) RowCandidate candidates[CANDIDATE_CAPACITY];
  PrefixStorage<kBlockThreads> prefix;
};

// Cluster8 reuses the candidate allocation as aligned (index, FP32 bits) pairs.
__device__ __forceinline__ uint64_t pack_cluster_candidate(int32_t index, uint32_t key) {
  return static_cast<uint64_t>(static_cast<uint32_t>(index)) | (static_cast<uint64_t>(key) << 32);
}

__device__ __forceinline__ int32_t cluster_candidate_index(uint64_t pair) {
  return static_cast<int32_t>(static_cast<uint32_t>(pair));
}

__device__ __forceinline__ uint32_t cluster_candidate_key(uint64_t pair) {
  return static_cast<uint32_t>(pair >> 32);
}

template <int kTopK, int kBlockThreads>
__device__ __forceinline__ uint64_t* cluster_candidate_pairs(
    RowSmem<kTopK, false, kBlockThreads>* s) {
  return reinterpret_cast<uint64_t*>(&s->cand_idx[0][0]);
}

__device__ __forceinline__ int warp_inclusive_sum(int value) {
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const int upper = __shfl_up_sync(0xFFFFFFFFu, value, offset);
    if ((threadIdx.x & 31) >= offset) {
      value += upper;
    }
  }
  return value;
}
template <int kTopK, bool kSampled, int kSecondaryRemaining = -1, int kBlockThreads = THREADS,
          bool kResetState = true>
__device__ __forceinline__ void prefix_sum_find_threshold(
    RowSmem<kTopK, kSampled, kBlockThreads>* s, int tidx, int remaining, int next_idx,
    int* reset_target, int reset_value) {
  static_assert(kBlockThreads >= 32 && kBlockThreads <= 1024);
  static_assert((kBlockThreads & 31) == 0);
  static_assert(RADIX % kBlockThreads == 0);
  constexpr int kItems = RADIX / kBlockThreads;
  constexpr int kNumWarps = kBlockThreads / 32;
  int values[kItems];
  int local_sum = 0;
#pragma unroll
  for (int item = 0; item < kItems; ++item) {
    const int value = s->histogram[tidx * kItems + item];
    values[item] = value;
    local_sum += value;
  }

  if constexpr (kResetState) {
    if (tidx == 0) {
      s->num_cand[next_idx] = 0;
      s->num_spilled_cand[next_idx] = 0;
      *reset_target = reset_value;
    }
  }
  const int lane = tidx & 31;
  const int warp = tidx >> 5;
  const int warp_inclusive = warp_inclusive_sum(local_sum);
  const int warp_exclusive = warp_inclusive - local_sum;
  if (lane == 31) {
    s->prefix.warp_sum[warp] = warp_inclusive;
  }
  __syncthreads();
  const int preceding_warp_value = lane < kNumWarps && lane < warp ? s->prefix.warp_sum[lane] : 0;
  const int preceding_warps = __reduce_add_sync(0xFFFFFFFFu, preceding_warp_value);

  int prefix = preceding_warps + warp_exclusive;
  int found_bin = -1;
  int secondary_bin = -1;
#pragma unroll
  for (int item = 0; item < kItems; ++item) {
    const int ibin = tidx * kItems + item;
    const int exclusive = prefix;
    prefix += values[item];
    s->histogram[ibin] = prefix;
    if (found_bin < 0 && exclusive <= remaining && prefix > remaining) {
      found_bin = ibin;
    }
    if constexpr (kSecondaryRemaining >= 0) {
      if (secondary_bin < 0 && exclusive <= kSecondaryRemaining && prefix > kSecondaryRemaining) {
        secondary_bin = ibin;
      }
    }
  }
  if (found_bin >= 0) {
    s->threshold_bin = found_bin;
  }
  if constexpr (kSecondaryRemaining >= 0) {
    if (secondary_bin >= 0) {
      s->ties_remaining = secondary_bin;
    }
  }
  __syncthreads();
}

template <int kTopK, int kBlockThreads = THREADS>
__device__ __noinline__ void refine_resident_keys(RowSmem<kTopK, false, kBlockThreads>* s,
                                                  int topk_remaining, int tidx);

template <uint32_t VEC, int kTopK, bool kSampled, int kBlockThreads = THREADS>
__device__ __noinline__ int run_exact_coarse_stage(RowSmem<kTopK, kSampled, kBlockThreads>* s,
                                                   const float* score, int32_t* buf0, int length,
                                                   int tidx) {
  using Smem = RowSmem<kTopK, kSampled, kBlockThreads>;
  int remaining = kTopK;
  for (int i = tidx; i < RADIX + 1; i += kBlockThreads) {
    s->histogram[i] = 0;
  }
  if (tidx == 0) {
    s->num_selected = 0;
  }
  __syncthreads();

  auto add_coarse = [&](float raw) { atomicAdd(&s->histogram[to_coarse_key(raw)], 1); };

  if constexpr (VEC == 4) {
    // 8-wide per thread (two float4 loads in flight) to match num_copy_bits=256
    // and hide global-load latency.
    const int n8 = length >> 3;
    const float4* v4 = reinterpret_cast<const float4*>(score);
    for (int g = tidx; g < n8; g += kBlockThreads) {
      float4 a = ld_cg_float4(reinterpret_cast<const float*>(v4 + 2 * g));
      float4 b = ld_cg_float4(reinterpret_cast<const float*>(v4 + 2 * g + 1));
      add_coarse(a.x);
      add_coarse(a.y);
      add_coarse(a.z);
      add_coarse(a.w);
      add_coarse(b.x);
      add_coarse(b.y);
      add_coarse(b.z);
      add_coarse(b.w);
    }
    for (int i = (n8 << 3) + tidx; i < length; i += kBlockThreads) {
      add_coarse(score[i]);
    }
  } else {
    for (int i = tidx; i < length; i += kBlockThreads) {
      add_coarse(score[i]);
    }
  }
  __syncthreads();

  prefix_sum_find_threshold(s, tidx, remaining, /*next_idx=*/0, &s->num_selected,
                            /*reset_value=*/0);

  int threshold_bin = s->threshold_bin;
  if (threshold_bin > 0) {
    remaining -= s->histogram[threshold_bin - 1];
  }
  // All threads must finish reading the prefix value before histogram reset.
  __syncthreads();

  if (remaining == 0) {
    // Boundary bin contributes nothing; just collect bin < threshold_bin.
    const PackedCoarseBoundary select_boundary =
        pack_coarse_boundary(coarse_boundary(threshold_bin - 1));
    auto collect = [&](float raw, int i) {
      if (coarse_accepts(raw, select_boundary)) {
        int pos = atomicAdd(&s->num_selected, 1);
        s->indices[pos] = i;
      }
    };
    if constexpr (VEC == 4) {
      const int n4 = length >> 2;
      const float4* v4 = reinterpret_cast<const float4*>(score);
      for (int g = tidx; g < n4; g += kBlockThreads) {
        float4 v = ld_cg_float4(reinterpret_cast<const float*>(v4 + g));
        int base = g << 2;
        collect(v.x, base);
        collect(v.y, base + 1);
        collect(v.z, base + 2);
        collect(v.w, base + 3);
      }
      for (int i = (n4 << 2) + tidx; i < length; i += kBlockThreads) {
        collect(score[i], i);
      }
    } else {
      for (int i = tidx; i < length; i += kBlockThreads) {
        collect(score[i], i);
      }
    }
    __syncthreads();
    return 0;
  }

  // Reset histogram for the first refine round.
  for (int i = tidx; i < RADIX + 1; i += kBlockThreads) {
    s->histogram[i] = 0;
  }
  __syncthreads();

  // Per-element exact coarse split: bin<thr -> select; bin==thr -> candidate +
  // first refine histogram; bin>thr -> drop.
  const PackedCoarseBoundary select_boundary =
      pack_coarse_boundary(coarse_boundary(threshold_bin - 1));
  const PackedCoarseBoundary candidate_boundary =
      pack_coarse_boundary(coarse_boundary(threshold_bin));
  auto split = [&](float raw, int i) {
    const CoarseClass coarse_class = coarse_classify(raw, select_boundary, candidate_boundary);
    if (coarse_class == CoarseClass::kSelected) {
      int pos = atomicAdd(&s->num_selected, 1);
      s->indices[pos] = i;
    } else if (coarse_class == CoarseClass::kCandidate) {
      int pos = atomicAdd(&s->num_cand[0], 1);
      const uint32_t ordered = to_ordered(raw);
      if (pos < Smem::CANDIDATE_CAPACITY) {
        s->cand_idx[0][pos] = i;
        if constexpr (!kSampled) {
          s->cand_idx[1][pos] = static_cast<int32_t>(ordered);
        }
      } else {
        int gpos = atomicAdd(&s->num_spilled_cand[0], 1);
        buf0[gpos] = i;
      }
      int sub = (ordered >> kFineShifts[0]) & kFineMasks[0];
      atomicAdd(&s->histogram[sub], 1);
    }
  };

  if constexpr (VEC == 4) {
    const int n8 = length >> 3;
    const float4* v4 = reinterpret_cast<const float4*>(score);
    for (int g = tidx; g < n8; g += kBlockThreads) {
      float4 a = ld_cg_float4(reinterpret_cast<const float*>(v4 + 2 * g));
      float4 b = ld_cg_float4(reinterpret_cast<const float*>(v4 + 2 * g + 1));
      int base = g << 3;
      split(a.x, base);
      split(a.y, base + 1);
      split(a.z, base + 2);
      split(a.w, base + 3);
      split(b.x, base + 4);
      split(b.y, base + 5);
      split(b.z, base + 6);
      split(b.w, base + 7);
    }
    for (int i = (n8 << 3) + tidx; i < length; i += kBlockThreads) {
      split(score[i], i);
    }
  } else {
    for (int i = tidx; i < length; i += kBlockThreads) {
      split(score[i], i);
    }
  }
  __syncthreads();
  return remaining;
}

// Exact fp32 refinement rounds over the round-0 candidate set already staged in
// s->cand_idx[0] / buf0 with s->histogram holding the round-0 fine
// histogram. Fills s->indices with exactly kTopK entries.
template <int kTopK, bool kSampled, int kBlockThreads = THREADS>
__device__ __noinline__ void refine_rounds(RowSmem<kTopK, kSampled, kBlockThreads>* s,
                                           const float* score, int32_t* buf0, int32_t* buf1,
                                           int topk_remaining, int tidx) {
  using Smem = RowSmem<kTopK, kSampled, kBlockThreads>;
  bool run = true;
  for (int round = 0; round < NUM_REFINE_ROUNDS; ++round) {
    if (!run) {
      break;
    }
    int r = round & 1;

    // prefix sum + threshold; seeds s->ties_remaining = remaining - prefix[thr-1].
    prefix_sum_find_threshold(s, tidx, topk_remaining, /*next_idx=*/r ^ 1, &s->ties_remaining,
                              topk_remaining);
    // last_remain reset_value must be remaining-previous, recomputed below.
    __syncthreads();

    int num_input = min(s->num_cand[r], Smem::CANDIDATE_CAPACITY);
    int g_num = s->num_spilled_cand[r];
    int threshold = s->threshold_bin;
    if (threshold > 0) {
      topk_remaining -= s->histogram[threshold - 1];
    }
    // Fix last_remain to the post-subtraction remaining (the # of ties to
    // back-fill in the final round). prefix_sum seeded it with the pre-round
    // remaining; recompute now that we know prefix[threshold-1].
    if (tidx == 0) {
      s->ties_remaining = topk_remaining;
    }
    __syncthreads();

    int cur_shift = kFineShifts[round];
    uint32_t cur_mask = kFineMasks[round];
    bool is_last = (round == NUM_REFINE_ROUNDS - 1);
    int next_shift = is_last ? 0 : kFineShifts[round + 1];
    uint32_t next_mask = is_last ? 0 : kFineMasks[round + 1];
    // Select the ping-pong spill slices with ternaries rather than a local
    // pointer array: an array captured by the lambda below would be forced to
    // the stack and spill.
    int32_t* cur_buf = r ? buf1 : buf0;
    int32_t* next_buf = r ? buf0 : buf1;

    if (topk_remaining == 0) {
      for (int i = tidx; i < num_input; i += kBlockThreads) {
        int idx = s->cand_idx[r][i];
        int b = (to_ordered(score[idx]) >> cur_shift) & cur_mask;
        if (b < threshold) {
          int pos = atomicAdd(&s->num_selected, 1);
          s->indices[pos] = idx;
        }
      }
      for (int i = tidx; i < g_num; i += kBlockThreads) {
        int idx = cur_buf[i];
        int b = (to_ordered(score[idx]) >> cur_shift) & cur_mask;
        if (b < threshold) {
          int pos = atomicAdd(&s->num_selected, 1);
          s->indices[pos] = idx;
        }
      }
      __syncthreads();
      run = false;
    } else {
      for (int i = tidx; i < RADIX + 1; i += kBlockThreads) {
        s->histogram[i] = 0;
      }
      __syncthreads();

      // Split one candidate: select, promote to the next round, or drop.
      auto refine_one = [&](int idx) {
        float raw = score[idx];
        uint32_t bits = to_ordered(raw);
        int b = (bits >> cur_shift) & cur_mask;
        if (b < threshold) {
          int pos = atomicAdd(&s->num_selected, 1);
          s->indices[pos] = idx;
        } else if (b == threshold) {
          if (is_last) {
            int cur = atomicAdd(&s->ties_remaining, -1);
            if (cur > 0) {
              s->indices[kTopK - cur] = idx;
            }
          } else {
            int cur = atomicAdd(&s->num_cand[r ^ 1], 1);
            if (cur < Smem::CANDIDATE_CAPACITY) {
              s->cand_idx[r ^ 1][cur] = idx;
            } else {
              int gpos = atomicAdd(&s->num_spilled_cand[r ^ 1], 1);
              next_buf[gpos] = idx;
            }
            int sub = (bits >> next_shift) & next_mask;
            atomicAdd(&s->histogram[sub], 1);
          }
        }
      };

      // smem candidates
      for (int i = tidx; i < num_input; i += kBlockThreads) {
        refine_one(s->cand_idx[r][i]);
      }
      // gmem-spilled candidates
      for (int i = tidx; i < g_num; i += kBlockThreads) {
        refine_one(cur_buf[i]);
      }
      __syncthreads();
    }
  }
}

// Refine a resident exact boundary without rereading scores or ping-ponging
// candidate indices. Sampled and spilled boundaries use refine_rounds.
template <int kTopK, int kBlockThreads>
__device__ __noinline__ void refine_resident_keys(RowSmem<kTopK, false, kBlockThreads>* s,
                                                  int topk_remaining, int tidx) {
  const int num_input = s->num_cand[0];
  uint32_t active_mask = 0;
  uint32_t active_prefix = 0;

#pragma unroll
  for (int round = 0; round < NUM_REFINE_ROUNDS; ++round) {
    prefix_sum_find_threshold<kTopK, false, -1, kBlockThreads,
                              /*kResetState=*/false>(s, tidx, topk_remaining, /*next_idx=*/0,
                                                     &s->ties_remaining, topk_remaining);
    const int threshold = s->threshold_bin;
    if (threshold > 0) {
      topk_remaining -= s->histogram[threshold - 1];
    }
    if (tidx == 0) {
      s->ties_remaining = topk_remaining;
    }
    __syncthreads();

    const int cur_shift = kFineShifts[round];
    const uint32_t cur_mask = kFineMasks[round];
    const bool is_last = round == NUM_REFINE_ROUNDS - 1;
    if (topk_remaining != 0 && !is_last) {
      for (int bin = tidx; bin < RADIX + 1; bin += kBlockThreads) {
        s->histogram[bin] = 0;
      }
      __syncthreads();
    }

    for (int pos = tidx; pos < num_input; pos += kBlockThreads) {
      const uint32_t ordered = static_cast<uint32_t>(s->cand_idx[1][pos]);
      const int idx = s->cand_idx[0][pos];
      if ((ordered & active_mask) != active_prefix) {
        continue;
      }
      const int bin = (ordered >> cur_shift) & cur_mask;
      if (bin < threshold) {
        const int output_pos = atomicAdd(&s->num_selected, 1);
        s->indices[output_pos] = idx;
      } else if (bin == threshold && topk_remaining != 0) {
        if (is_last) {
          const int cur = atomicAdd(&s->ties_remaining, -1);
          if (cur > 0) {
            s->indices[kTopK - cur] = idx;
          }
        } else {
          const int next_shift = kFineShifts[round + 1];
          const uint32_t next_mask = kFineMasks[round + 1];
          const int next_bin = (ordered >> next_shift) & next_mask;
          atomicAdd(&s->histogram[next_bin], 1);
        }
      }
    }
    __syncthreads();
    if (topk_remaining == 0 || is_last) {
      break;
    }
    active_mask |= cur_mask << cur_shift;
    active_prefix |= static_cast<uint32_t>(threshold) << cur_shift;
  }
}

// One primary thread owns one Cluster8 boundary candidate across 11-11-10.
template <int kTopK, int kBlockThreads>
__device__ __noinline__ void refine_cluster_register_candidate(
    RowSmem<kTopK, false, kBlockThreads>* s, uint64_t raw_pair, bool active, int topk_remaining,
    int tidx, int32_t* direct_output) {
  uint32_t ordered = 0;
  int idx = -1;
  int write_pos = kTopK;
  if (active) {
    idx = cluster_candidate_index(raw_pair);
    ordered = to_ordered(__uint_as_float(cluster_candidate_key(raw_pair)));
  }

  for (int bin = tidx; bin < RADIX + 1; bin += kBlockThreads) {
    s->histogram[bin] = 0;
  }
  __syncthreads();
  if (active) {
    const int sub = (ordered >> kFineShifts[0]) & kFineMasks[0];
    atomicAdd(&s->histogram[sub], 1);
  }
  __syncthreads();

#pragma unroll
  for (int round = 0; round < NUM_REFINE_ROUNDS; ++round) {
    prefix_sum_find_threshold<kTopK, false, -1, kBlockThreads,
                              /*kResetState=*/false>(s, tidx, topk_remaining, /*next_idx=*/0,
                                                     &s->ties_remaining, topk_remaining);
    const int threshold = s->threshold_bin;
    if (threshold > 0) {
      topk_remaining -= s->histogram[threshold - 1];
    }
    if (tidx == 0) {
      s->ties_remaining = topk_remaining;
    }
    __syncthreads();

    const int cur_shift = kFineShifts[round];
    const uint32_t cur_mask = kFineMasks[round];
    const bool is_last = round == NUM_REFINE_ROUNDS - 1;
    if (topk_remaining != 0 && !is_last) {
      for (int bin = tidx; bin < RADIX + 1; bin += kBlockThreads) {
        s->histogram[bin] = 0;
      }
      __syncthreads();
    }

    if (active) {
      const int bin = (ordered >> cur_shift) & cur_mask;
      if (bin < threshold) {
        write_pos = atomicAdd(&s->num_selected, 1);
        active = false;
      } else if (bin > threshold) {
        active = false;
      } else if (topk_remaining != 0) {
        if (is_last) {
          const int cur = atomicAdd(&s->ties_remaining, -1);
          if (cur > 0) {
            write_pos = kTopK - cur;
          }
          active = false;
        } else {
          const int next_shift = kFineShifts[round + 1];
          const uint32_t next_mask = kFineMasks[round + 1];
          const int next_bin = (ordered >> next_shift) & next_mask;
          atomicAdd(&s->histogram[next_bin], 1);
        }
      } else {
        active = false;
      }
    }
    __syncthreads();
    if (topk_remaining == 0 || is_last) {
      break;
    }
  }
  if (write_pos < kTopK) {
    direct_output[write_pos] = idx;
  }
}

constexpr int SHORT_THREADS = 1024;
constexpr size_t HEAD_COUNTER_INTS = 2;  // work_counter, exit_counter

inline size_t counters_bytes_for(int m) {
  return (HEAD_COUNTER_INTS + static_cast<size_t>(m > 0 ? m : 0)) * sizeof(int32_t);
}

inline size_t spill_ints(int n) { return static_cast<size_t>(grid_num_blocks()) * 2 * n; }

inline size_t one_cta_workspace_bytes(int n) { return spill_ints(n) * sizeof(int32_t); }

inline size_t split_spill_ints(int m, int n) {
  return static_cast<size_t>(std::max(grid_num_blocks(), m)) * 2 * n;
}

inline size_t split_workspace_bytes(int m, int splits, int n) {
  const size_t split_ints = static_cast<size_t>(m) * splits +          // counts
                            static_cast<size_t>(m) * splits * RADIX +  // hists
                            static_cast<size_t>(m) * n;                // cands
  return (split_spill_ints(m, n) + split_ints) * sizeof(int32_t);
inline bool opt_in_dynamic_smem(const void* kernel, size_t smem_bytes) {
  static const int max_smem = [] {
    int dev = 0;
    int bytes = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    return bytes;
  }();
  if (smem_bytes > static_cast<size_t>(max_smem)) {
    return false;
  }
  return cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                              static_cast<int>(smem_bytes)) == cudaSuccess;
}

// Vector width for the row scans, from the row-stride alignment.
inline uint32_t vector_width(int row_stride) {
  return (row_stride % 4 == 0) ? 4u : (row_stride % 2 == 0) ? 2u : 1u;
}
// ======================================================================
// verbatim from third_party/hpc-ops/src/topk/topk_filtered.cu (cluster collectives + kernel)
// ======================================================================
template <int kTopK, int kSplits, int kBlockThreads>
__device__ __forceinline__ void cluster_allreduce_histogram(
    RowSmem<kTopK, false, kBlockThreads>* s, cooperative_groups::cluster_group cluster, int rank,
    int tidx) {
  static_assert(kSplits == 8);
  static_assert(kBlockThreads == 1024);
  static_assert(RADIX % kSplits == 0);
  static_assert(RADIX % kBlockThreads == 0);
  constexpr int kBinsPerRank = RADIX / kSplits;
  constexpr int kItems = RADIX / kBlockThreads;

  // [DSMEM] barrier.cluster
  cluster.sync();
#pragma unroll
  for (int item = 0; item < kItems; ++item) {
    const int linear = tidx + item * kBlockThreads;
    const int bin = rank * kBinsPerRank + linear / kSplits;
    const int peer = linear & (kSplits - 1);
    // [DSMEM] generic DSMEM pointer: element `bin` of peer `peer`'s histogram (pull then push)
    int* addr = cluster.map_shared_rank(&s->histogram[bin], peer);
    int value = *addr;
#pragma unroll
    for (int offset = 1; offset < kSplits; offset <<= 1) {
      // [DSMEM] reduce the 8 pulled values inside the 8-lane group
      value += __shfl_xor_sync(0xFFFFFFFFu, value, offset);
    }
    *addr = value;
  }
  // [DSMEM] barrier.cluster
  cluster.sync();
}
template <int kTopK, int kBlockThreads>
__device__ __noinline__ void refine_large_boundary_single_cta(
    RowSmem<kTopK, false, kBlockThreads>* s, const float* score, int32_t* dst, int length,
    PackedCoarseBoundary select_boundary, PackedCoarseBoundary candidate_boundary,
    int topk_remaining) {
  const int tidx = threadIdx.x;
  uint32_t active_mask = 0;
  uint32_t active_prefix = 0;

  for (int bin = tidx; bin < RADIX + 1; bin += kBlockThreads) {
    s->histogram[bin] = 0;
  }
  if (tidx == 0) {
    s->num_selected = 0;
  }
  __syncthreads();
  for (int i = tidx; i < length; i += kBlockThreads) {
    const float raw = score[i];
    const CoarseClass coarse_class = coarse_classify(raw, select_boundary, candidate_boundary);
    if (coarse_class == CoarseClass::kSelected) {
      const int pos = atomicAdd(&s->num_selected, 1);
      if (pos < kTopK) {
        dst[pos] = i;
      }
    } else if (coarse_class == CoarseClass::kCandidate) {
      const uint32_t ordered = to_ordered(raw);
      const int sub = (ordered >> kFineShifts[0]) & kFineMasks[0];
      atomicAdd(&s->histogram[sub], 1);
    }
  }
  __syncthreads();

#pragma unroll 1
  for (int round = 0; round < NUM_REFINE_ROUNDS; ++round) {
    prefix_sum_find_threshold<kTopK, false, -1, kBlockThreads,
                              /*kResetState=*/false>(s, tidx, topk_remaining, /*next_idx=*/0,
                                                     &s->ties_remaining, topk_remaining);

    const int threshold = s->threshold_bin;
    if (threshold > 0) {
      topk_remaining -= s->histogram[threshold - 1];
    }
    const int shift = kFineShifts[round];
    const uint32_t mask = kFineMasks[round];
    const bool is_last = round == NUM_REFINE_ROUNDS - 1;

    if (topk_remaining != 0 && !is_last) {
      for (int bin = tidx; bin < RADIX + 1; bin += kBlockThreads) {
        s->histogram[bin] = 0;
      }
    }
    __syncthreads();

    // Only values in the coarse boundary and the active FP32 prefix remain.
    for (int i = tidx; i < length; i += kBlockThreads) {
      const float raw = score[i];
      if (coarse_classify(raw, select_boundary, candidate_boundary) != CoarseClass::kCandidate) {
        continue;
      }
      const uint32_t ordered = to_ordered(raw);
      if ((ordered & active_mask) != active_prefix) {
        continue;
      }
      const int bin = (ordered >> shift) & mask;
      if (bin < threshold) {
        const int pos = atomicAdd(&s->num_selected, 1);
        if (pos < kTopK) {
          dst[pos] = i;
        }
      } else if (bin == threshold && topk_remaining != 0) {
        if (is_last) {
          const int pos = atomicAdd(&s->num_selected, 1);
          if (pos < kTopK) {
            dst[pos] = i;
          }
        } else {
          const int next_shift = kFineShifts[round + 1];
          const uint32_t next_mask = kFineMasks[round + 1];
          const int next_bin = (ordered >> next_shift) & next_mask;
          atomicAdd(&s->histogram[next_bin], 1);
        }
      }
    }
    __syncthreads();

    if (topk_remaining == 0 || is_last) {
      break;
    }
    active_mask |= mask << shift;
    active_prefix |= static_cast<uint32_t>(threshold) << shift;
  }
}
template <int kTopK, int kSplits, int kBlockThreads>
__device__ __noinline__ void run_cluster_unaligned_overflow(RowSmem<kTopK, false, kBlockThreads>* s,
                                                            const float* score, int32_t* dst,
                                                            int32_t* spill0, int32_t* spill1,
                                                            int length, int seg_lo, int seg_len,
                                                            int topk_remaining) {
  namespace cg = cooperative_groups;
  using Smem = RowSmem<kTopK, false, kBlockThreads>;
  cg::cluster_group cluster = cg::this_cluster();
  const int tidx = threadIdx.x;
  const int rank = static_cast<int>(cluster.block_rank());

  for (int bin = tidx; bin < RADIX + 1; bin += kBlockThreads) {
    s->histogram[bin] = 0;
  }
  __syncthreads();
  const int local_resident = min(s->num_cand[0], Smem::CANDIDATE_CAPACITY);
  for (int pos = tidx; pos < local_resident; pos += kBlockThreads) {
    const uint32_t ordered = static_cast<uint32_t>(s->cand_idx[1][pos]);
    const int sub = (ordered >> kFineShifts[0]) & kFineMasks[0];
    atomicAdd(&s->histogram[sub], 1);
  }
  for (int pos = tidx; pos < s->num_spilled_cand[0]; pos += kBlockThreads) {
    const int idx = spill1[seg_lo + pos];
    const uint32_t ordered = to_ordered(score[idx]);
    const int sub = (ordered >> kFineShifts[0]) & kFineMasks[0];
    atomicAdd(&s->histogram[sub], 1);
  }
  __syncthreads();
  // [DSMEM] AllReduce of the 2048-bin histogram: pull reduce-scatter + push all-gather
  cluster_allreduce_histogram<kTopK, kSplits, kBlockThreads>(s, cluster, rank, tidx);

  if (rank == 0) {
    if (tidx == 0) {
      s->num_cand[1] = s->num_spilled_cand[0];
      s->num_selected = s->threshold_bin;
      s->num_cand[0] = min(s->ties_remaining, Smem::CANDIDATE_CAPACITY);
      s->num_spilled_cand[0] = 0;
    }
    __syncthreads();

#pragma unroll
    for (int sp = 1; sp < kSplits; ++sp) {
      // [DSMEM] generic DSMEM pointer into a peer CTA's smem
      Smem* peer = cluster.map_shared_rank(s, sp);
      const int selected = peer->num_selected;
      for (int i = tidx; i < selected; i += kBlockThreads) {
        const int pos = atomicAdd(&s->num_selected, 1);
        s->indices[pos] = peer->indices[i];
      }
      const int candidates = min(peer->num_cand[0], Smem::CANDIDATE_CAPACITY);
      for (int i = tidx; i < candidates; i += kBlockThreads) {
        const int pos = atomicAdd(&s->num_cand[0], 1);
        const int idx = peer->cand_idx[0][i];
        if (pos < Smem::CANDIDATE_CAPACITY) {
          s->cand_idx[0][pos] = idx;
        } else {
          const int gpos = atomicAdd(&s->num_spilled_cand[0], 1);
          spill0[gpos] = idx;
        }
      }
      __syncthreads();
    }

#pragma unroll
    for (int sp = 0; sp < kSplits; ++sp) {
      // [DSMEM] generic DSMEM pointer into a peer CTA's smem
      Smem* peer = cluster.map_shared_rank(s, sp);
      const int overflow = (sp == 0) ? s->num_cand[1] : peer->num_spilled_cand[0];
      const int source_lo = min(sp * seg_len, length);
      for (int i = tidx; i < overflow; i += kBlockThreads) {
        const int pos = atomicAdd(&s->num_cand[0], 1);
        const int idx = spill1[source_lo + i];
        if (pos < Smem::CANDIDATE_CAPACITY) {
          s->cand_idx[0][pos] = idx;
        } else {
          const int gpos = atomicAdd(&s->num_spilled_cand[0], 1);
          spill0[gpos] = idx;
        }
      }
      __syncthreads();
    }
  }
  // [DSMEM] barrier.cluster
  cluster.sync();

  if (rank != 0) {
    return;
  }
  if (topk_remaining != 0) {
    refine_rounds<kTopK, false, kBlockThreads>(s, score, spill0, spill1, topk_remaining, tidx);
  }
  for (int i = tidx; i < kTopK; i += kBlockThreads) {
    dst[i] = s->indices[i];
  }
}
template <uint32_t VEC, int kTopK, int kSplits, int kBlockThreads>
__global__ void __launch_bounds__(kBlockThreads, 2)
    topk_filtered_cluster8_exact_kernel(const float* __restrict__ input,
                                        int32_t* __restrict__ output,
                                        const int32_t* __restrict__ ke,
                                        int32_t* __restrict__ spill_buffer,
                                        const int32_t* __restrict__ num_valid_rows,
                                        uint32_t row_stride, uint32_t out_stride, uint32_t n_cols) {
  namespace cg = cooperative_groups;
  using Smem = RowSmem<kTopK, false, kBlockThreads>;

  cg::cluster_group cluster = cg::this_cluster();
  const int tidx = threadIdx.x;
  const int rank = static_cast<int>(cluster.block_rank());
  const int row_i = static_cast<int>(blockIdx.y);
  if (row_i >= *num_valid_rows) {
    return;
  }

  extern __shared__ uint8_t smem_raw[];
  Smem* s = reinterpret_cast<Smem*>(smem_raw);
  // [DSMEM] generic DSMEM pointer to rank 0's whole RowSmem (root of the gather)
  Smem* primary = cluster.map_shared_rank(s, 0);
  const float* score = input + static_cast<size_t>(row_i) * row_stride;
  int32_t* dst = output + static_cast<size_t>(row_i) * out_stride;
  const int length = ke[row_i];
  int32_t* spill0 = spill_buffer + static_cast<size_t>(row_i) * 2 * n_cols;
  int32_t* spill1 = spill0 + n_cols;

  if (length <= kTopK) {
    if (rank == 0) {
      for (int i = tidx; i < kTopK; i += kBlockThreads) {
        dst[i] = (i < length) ? i : -1;
      }
    }
    return;
  }

  const int seg_len = (length + kSplits - 1) / kSplits;
  const int seg_lo = min(rank * seg_len, length);
  const int seg_hi = min(seg_lo + seg_len, length);

  // Pass 1: exact coarse histogram, distributed over the row segments.
  for (int i = tidx; i < RADIX + 1; i += kBlockThreads) {
    s->histogram[i] = 0;
  }
  __syncthreads();

  auto add_coarse = [&](float raw) { atomicAdd(&s->histogram[to_coarse_key(raw)], 1); };
  if constexpr (VEC == 4) {
    const int v_lo = min((seg_lo + 7) & ~7, seg_hi);
    const int v_hi = max(seg_hi & ~7, v_lo);
    for (int i = seg_lo + tidx; i < v_lo; i += kBlockThreads) {
      add_coarse(score[i]);
    }
    const int n8 = (v_hi - v_lo) >> 3;
    for (int g = tidx; g < n8; g += kBlockThreads) {
      const float* p = score + v_lo + (g << 3);
      const float4 a = ld_cg_float4(p);
      const float4 b = ld_cg_float4(p + 4);
      add_coarse(a.x);
      add_coarse(a.y);
      add_coarse(a.z);
      add_coarse(a.w);
      add_coarse(b.x);
      add_coarse(b.y);
      add_coarse(b.z);
      add_coarse(b.w);
    }
    for (int i = v_hi + tidx; i < seg_hi; i += kBlockThreads) {
      add_coarse(score[i]);
    }
  } else {
    for (int i = seg_lo + tidx; i < seg_hi; i += kBlockThreads) {
      add_coarse(score[i]);
    }
  }
  __syncthreads();

  // [DSMEM] AllReduce of the 2048-bin histogram: pull reduce-scatter + push all-gather
  cluster_allreduce_histogram<kTopK, kSplits, kBlockThreads>(s, cluster, rank, tidx);
  // histogram[RADIX] was cleared with the rest of the allocation and is not
  // touched by the all-reduce. The prefix helper resets num_selected when it
  // publishes the threshold, so no intermediate CTA rendezvous is required.
  prefix_sum_find_threshold<kTopK, false, -1, kBlockThreads>(s, tidx, kTopK, /*next_idx=*/0,
                                                             &s->num_selected,
                                                             /*reset_value=*/0);
  if (tidx == 0) {
    const int local_threshold = s->threshold_bin;
    s->ties_remaining = kTopK - (local_threshold == 0 ? 0 : s->histogram[local_threshold - 1]);
  }
  // Every rank owns the same reduced histogram and computes the same boundary.
  // Only the thread-0 remainder write needs publication here; no peer state is
  // consumed until the post-collect cluster barrier.
  __syncthreads();

  const int threshold = s->threshold_bin;
  const int topk_remaining = s->ties_remaining;
  const int coarse_candidate_count =
      s->histogram[threshold] - (threshold == 0 ? 0 : s->histogram[threshold - 1]);
  const bool large_boundary = topk_remaining != 0 && coarse_candidate_count > kBlockThreads;
  const PackedCoarseBoundary select_boundary = pack_coarse_boundary(coarse_boundary(threshold - 1));
  const PackedCoarseBoundary candidate_boundary = pack_coarse_boundary(coarse_boundary(threshold));

  if constexpr (VEC == 4) {
    if (large_boundary) {
      if (rank == 0) {
        refine_large_boundary_single_cta<kTopK, kBlockThreads>(
            s, score, dst, length, select_boundary, candidate_boundary, topk_remaining);
      }
      // [DSMEM] barrier.cluster
      cluster.sync();
      return;
    }
  }

  // Pass 2: retain only the compact exact upper-tail state from this segment.
  // The cumulative coarse histogram tells us before the scan whether the
  // boundary can use the one-candidate-per-thread tail.  A larger boundary
  // reuses the dead allocation as its first exact histogram instead of
  // materializing a list that must later be rebuilt.
  auto classify = [&](float raw, int i) {
    const CoarseClass coarse_class = coarse_classify(raw, select_boundary, candidate_boundary);
    if (coarse_class == CoarseClass::kSelected) {
      const int pos = atomicAdd(&s->num_selected, 1);
      if constexpr (VEC == 4) {
        if (rank == 0) {
          dst[pos] = i;
        } else {
          s->indices[pos] = i;
        }
      } else {
        s->indices[pos] = i;
      }
    } else if (coarse_class == CoarseClass::kCandidate && topk_remaining != 0) {
      const int pos = atomicAdd(&s->num_cand[0], 1);
      if (pos < Smem::CANDIDATE_CAPACITY) {
        if constexpr (VEC == 4) {
          cluster_candidate_pairs(s)[pos] = pack_cluster_candidate(i, __float_as_uint(raw));
        } else {
          s->cand_idx[0][pos] = i;
          s->cand_idx[1][pos] = static_cast<int32_t>(to_ordered(raw));
        }
      } else {
        if constexpr (VEC != 4) {
          const int gpos = atomicAdd(&s->num_spilled_cand[0], 1);
          spill1[seg_lo + gpos] = i;
        }
      }
    }
  };

  if constexpr (VEC == 4) {
    const int v_lo = min((seg_lo + 3) & ~3, seg_hi);
    const int v_hi = max(seg_hi & ~3, v_lo);
    for (int i = seg_lo + tidx; i < v_lo; i += kBlockThreads) {
      classify(score[i], i);
    }
    const int n4 = (v_hi - v_lo) >> 2;
    int g = tidx;
    float4 next;
    if (g < n4) {
      next = ld_cg_float4(score + v_lo + (g << 2));
    }
    while (g < n4) {
      const float4 current = next;
      const int base = v_lo + (g << 2);
      g += kBlockThreads;
      if (g < n4) {
        next = ld_cg_float4(score + v_lo + (g << 2));
      }
      classify(current.x, base);
      classify(current.y, base + 1);
      classify(current.z, base + 2);
      classify(current.w, base + 3);
    }
    for (int i = v_hi + tidx; i < seg_hi; i += kBlockThreads) {
      classify(score[i], i);
    }
  } else {
    for (int i = seg_lo + tidx; i < seg_hi; i += kBlockThreads) {
      classify(score[i], i);
    }
  }

  // Reuse dead coarse-stage scalars as this CTA's output reservations. The
  // aligned path can merge peer atomics into the primary's live counters while
  // the primary finishes its own collect. The unaligned fallback still stages
  // selected indices through primary DSM and therefore retains its snapshot.
  if constexpr (VEC != 4) {
    __syncthreads();
    if (tidx == 0 && rank == 0) {
      s->threshold_bin = s->num_selected;
      s->ties_remaining = s->num_cand[0];
    }
    // [DSMEM] barrier.cluster
    cluster.sync();
  }

  // Only peers consume their CTA-local counters and staging buffers here.  For
  // aligned rows, the primary's local and peer DSM atomics may safely overlap:
  // every reservation is atomic and therefore receives a disjoint range.  Let
  // the primary proceed directly to the cluster rendezvous instead of making
  // it pay two block barriers whose data it never reads.
  if constexpr (VEC == 4) {
    if (rank != 0) {
      __syncthreads();
    }
  }
  if (rank != 0 && tidx == 0) {
    // [DSMEM] remote atomic add into rank 0's smem counters (reserve output range)
    s->threshold_bin = atomicAdd(&primary->num_selected, s->num_selected);
    // [DSMEM] remote atomic add into rank 0's smem counters (reserve output range)
    s->ties_remaining = atomicAdd(&primary->num_cand[0], s->num_cand[0]);
    if constexpr (VEC == 4) {
      // Publish the peer's reserved candidate interval in dead coarse
      // histogram storage.  The primary can then load one candidate directly
      // from its owner CTA instead of materializing a second DSM copy.
      // [DSMEM] remote store into rank 0's smem (push)
      primary->histogram[rank] = s->ties_remaining;
      // [DSMEM] remote store into rank 0's smem (push)
      primary->histogram[kSplits + rank] = s->num_cand[0];
    }
  }
  if constexpr (VEC == 4) {
    if (rank != 0) {
      __syncthreads();
    }
  } else {
    __syncthreads();
  }

  if constexpr (VEC == 4) {
    // [DSMEM] barrier.cluster
    cluster.sync();
    // [DSMEM] remote load from rank 0's smem (pull)
    const int aggregate_candidates = primary->num_cand[0];
    if (rank == 0) {
      bool active = tidx < aggregate_candidates;
      uint64_t raw_pair = 0;
      if (active) {
        int owner = 0;
        int local_pos = tidx;
#pragma unroll
        for (int sp = 1; sp < kSplits; ++sp) {
          // [DSMEM] remote load from rank 0's smem (pull)
          const int start = primary->histogram[sp];
          // [DSMEM] remote load from rank 0's smem (pull)
          const int count = primary->histogram[kSplits + sp];
          if (tidx >= start && tidx < start + count) {
            owner = sp;
            local_pos = tidx - start;
          }
        }
        // [DSMEM] generic DSMEM pointer to the owning CTA's RowSmem (pull one candidate)
        Smem* source = cluster.map_shared_rank(s, owner);
        raw_pair = cluster_candidate_pairs(source)[local_pos];
      }
      // All metadata reads must finish before refinement recycles the coarse
      // histogram as its first fine histogram.
      __syncthreads();
      if (topk_remaining != 0) {
        refine_cluster_register_candidate<kTopK, kBlockThreads>(s, raw_pair, active, topk_remaining,
                                                                tidx, dst);
      }
    } else {
      const int selected_prefix = s->threshold_bin;
      for (int i = tidx; i < s->num_selected; i += kBlockThreads) {
        dst[selected_prefix + i] = s->indices[i];
      }
    }
    // [DSMEM] barrier.cluster
    cluster.sync();
    return;
  }

  if (rank != 0) {
    const int candidate_prefix = s->ties_remaining;
    const int selected_prefix = s->threshold_bin;
    for (int i = tidx; i < s->num_selected; i += kBlockThreads) {
      // [DSMEM] remote store into rank 0's smem (push)
      primary->indices[selected_prefix + i] = s->indices[i];
    }
    if (candidate_prefix < Smem::CANDIDATE_CAPACITY) {
      const int copy_candidates = min(s->num_cand[0], Smem::CANDIDATE_CAPACITY - candidate_prefix);
      for (int i = tidx; i < copy_candidates; i += kBlockThreads) {
        // [DSMEM] remote store into rank 0's smem (push)
        primary->cand_idx[0][candidate_prefix + i] = s->cand_idx[0][i];
        // [DSMEM] remote store into rank 0's smem (push)
        primary->cand_idx[1][candidate_prefix + i] = s->cand_idx[1][i];
      }
    }
  }
  // [DSMEM] barrier.cluster
  cluster.sync();

  if (tidx == 0) {
    // num_cand counts every boundary element, including local overflow, so the
    // aggregate capacity test also proves that no CTA spilled candidates.
    // [DSMEM] remote load from rank 0's smem (pull)
    s->num_cand[1] = primary->num_cand[0] <= Smem::CANDIDATE_CAPACITY ? 1 : 0;
  }
  __syncthreads();
  const bool resident = s->num_cand[1] != 0;
  if (resident) {
    if (rank == 0) {
      s->num_spilled_cand[0] = 0;
      if (topk_remaining != 0) {
        for (int bin = tidx; bin < RADIX + 1; bin += kBlockThreads) {
          s->histogram[bin] = 0;
        }
        __syncthreads();
        for (int pos = tidx; pos < s->num_cand[0]; pos += kBlockThreads) {
          const uint32_t ordered = static_cast<uint32_t>(s->cand_idx[1][pos]);
          const int sub = (ordered >> kFineShifts[0]) & kFineMasks[0];
          atomicAdd(&s->histogram[sub], 1);
        }
        __syncthreads();
        refine_resident_keys(s, topk_remaining, tidx);
      }
      for (int i = tidx; i < kTopK; i += kBlockThreads) {
        dst[i] = s->indices[i];
      }
    }
    // Peers publish disjoint selected ranges while the primary refines the
    // boundary. Cluster blocks must still rendezvous before any peer exits;
    // otherwise the cluster's DSM lifetime ends while the primary is active.
    // [DSMEM] barrier.cluster
    cluster.sync();
    return;
  }

  // Keep shared counters alive until every peer observes the overflow decision.
  // [DSMEM] barrier.cluster
  cluster.sync();
  run_cluster_unaligned_overflow<kTopK, kSplits, kBlockThreads>(
      s, score, dst, spill0, spill1, length, seg_lo, seg_len, topk_remaining);
}
template <int kTopK, int kSplits, int kBlockThreads>
bool launch_cluster8_exact(int* topk_indices, const float* logits, const int* ke,
                           const int* num_valid_rows, int m, int n, int row_stride, int out_stride,
                           void* workspace, size_t workspace_bytes, cudaStream_t stream) {
  if (get_sm_major_version() < 9 || m <= 0 || workspace_bytes < one_cta_workspace_bytes(n)) {
    return false;
  }

  int32_t* spill = static_cast<int32_t*>(workspace);
  const uint32_t vec = vector_width(row_stride);
  const auto kernel =
      (vec == 4)   ? topk_filtered_cluster8_exact_kernel<4, kTopK, kSplits, kBlockThreads>
      : (vec == 2) ? topk_filtered_cluster8_exact_kernel<2, kTopK, kSplits, kBlockThreads>
                   : topk_filtered_cluster8_exact_kernel<1, kTopK, kSplits, kBlockThreads>;

  const size_t smem_bytes = sizeof(RowSmem<kTopK, false, kBlockThreads>);
  if (!opt_in_dynamic_smem(reinterpret_cast<const void*>(kernel), smem_bytes)) {
    return false;
  }

  cudaLaunchAttribute attribute[1]{};
  attribute[0].id = cudaLaunchAttributeClusterDimension;
  attribute[0].val.clusterDim.x = kSplits;
  attribute[0].val.clusterDim.y = 1;
  attribute[0].val.clusterDim.z = 1;

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(kSplits, m);
  config.blockDim = dim3(kBlockThreads);
  config.dynamicSmemBytes = smem_bytes;
  config.stream = stream;
  config.attrs = attribute;
  config.numAttrs = 1;

  return cudaLaunchKernelEx(&config, kernel, logits, topk_indices, ke, spill, num_valid_rows,
                            static_cast<uint32_t>(row_stride), static_cast<uint32_t>(out_stride),
                            static_cast<uint32_t>(n)) == cudaSuccess;
}

}  // namespace filtered
}  // namespace topk
}  // namespace hpc
