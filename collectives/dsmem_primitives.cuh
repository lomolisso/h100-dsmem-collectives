// dsmem_primitives.cuh - thin, documented wrappers around the Hopper (sm_90a)
// thread-block-cluster primitives used by every file in collectives/.
//
// Terminology
//   cluster      : group of N CTAs co-scheduled on one GPC (launch attribute or
//                  __cluster_dims__). Rank b in [0,N) = %cluster_ctarank.
//   DSMEM        : a CTA may address the shared memory of any CTA in its
//                  cluster. A local 32-bit smem address is turned into a
//                  "shared::cluster" address of rank r with `mapa`.
//   generic ptr  : cg::this_cluster().map_shared_rank(p, r) gives a generic
//                  pointer that ordinary ld/st/atomics may use; the compiler
//                  emits generic-space accesses that resolve to DSMEM.
//   mbarrier     : 64-bit arrival counter in smem with a transaction-byte
//                  count. Async copies "complete_tx" bytes on it; waiters spin
//                  on the phase parity. A remote CTA may arrive on it through a
//                  shared::cluster address.
//   bulk copy    : cp.async.bulk smem(cta) -> smem(cluster) moves `bytes`
//                  (multiple of 16, 16-B aligned) and signals the destination
//                  CTA's mbarrier. One thread issues it.
//   st.async     : 4/8/16-byte remote store that also completes_tx on the
//                  destination mbarrier (fine-grained alternative to bulk copy).
//   barrier.cluster : hardware cluster-wide barrier (arrive + wait). Default
//                  arrive has release semantics and wait has acquire semantics,
//                  so plain remote stores issued before arrive are visible to
//                  every CTA after wait.
//
// All functions are __device__ __forceinline__. Nothing here is a collective
// by itself; the algorithms live in the per-file <op>__<alg>__<src>.cu.
#pragma once
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace dsmem {

namespace cg = cooperative_groups;

// ---------------------------------------------------------------- identity ---
__device__ __forceinline__ uint32_t cluster_rank() {
    uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r;
}
__device__ __forceinline__ uint32_t cluster_size() {
    uint32_t n; asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(n)); return n;
}
__device__ __forceinline__ uint32_t smid() {
    uint32_t s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s)); return s;
}
// Global nanosecond timer, consistent across SMs (unlike clock64()).
__device__ __forceinline__ unsigned long long globaltimer() {
    unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}

// -------------------------------------------------------------- addressing ---
// Generic pointer -> 32-bit shared::cta address (what PTX smem instructions take).
template <typename T>
__device__ __forceinline__ uint32_t smem_u32(const T* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
// mapa: map a local shared::cta address to the same offset inside CTA `rank`'s
// shared memory, producing a shared::cluster address usable by
// ld/st/atom.shared::cluster, mbarrier.*.shared::cluster and cp.async.bulk.
__device__ __forceinline__ uint32_t mapa(uint32_t smem_addr, uint32_t rank) {
    uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(smem_addr), "r"(rank)); return r;
}
// Generic-pointer flavour of mapa (cooperative groups). The result can be
// dereferenced, passed to atomicAdd(), etc.
template <typename T>
__device__ __forceinline__ T* map_ptr(T* local, uint32_t rank) {
    return cg::this_cluster().map_shared_rank(local, rank);
}

// ------------------------------------------------- remote scalar ld / st ----
// Direct DSMEM load/store through a shared::cluster address (from mapa).
__device__ __forceinline__ uint32_t ld_cluster_u32(uint32_t caddr) {
    uint32_t v; asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(v) : "r"(caddr) : "memory"); return v;
}
__device__ __forceinline__ void st_cluster_u32(uint32_t caddr, uint32_t v) {
    asm volatile("st.shared::cluster.u32 [%0], %1;" :: "r"(caddr), "r"(v) : "memory");
}
__device__ __forceinline__ float ld_cluster_f32(uint32_t caddr) {
    float v; asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(caddr) : "memory"); return v;
}
__device__ __forceinline__ void st_cluster_f32(uint32_t caddr, float v) {
    asm volatile("st.shared::cluster.f32 [%0], %1;" :: "r"(caddr), "f"(v) : "memory");
}
// 16-byte vector variants (the unit most push/pull loops use).
__device__ __forceinline__ uint4 ld_cluster_v4(uint32_t caddr) {
    uint4 v;
    asm volatile("ld.shared::cluster.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(caddr) : "memory");
    return v;
}
__device__ __forceinline__ void st_cluster_v4(uint32_t caddr, uint4 v) {
    asm volatile("st.shared::cluster.v4.u32 [%0], {%1,%2,%3,%4};"
                 :: "r"(caddr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
// Remote reduction without return value (fire-and-forget atomic add).
__device__ __forceinline__ void red_cluster_add_f32(uint32_t caddr, float v) {
    asm volatile("red.shared::cluster.add.f32 [%0], %1;" :: "r"(caddr), "f"(v) : "memory");
}
__device__ __forceinline__ void red_cluster_add_u32(uint32_t caddr, uint32_t v) {
    asm volatile("red.shared::cluster.add.u32 [%0], %1;" :: "r"(caddr), "r"(v) : "memory");
}
// fence.proxy.async: make generic-proxy smem writes visible to the async proxy
// (TMA / cp.async.bulk / st.async) and vice versa. Required between writing a
// buffer with ordinary stores and handing it to a bulk copy.
__device__ __forceinline__ void fence_proxy_async_smem() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// ---------------------------------------------------------------- mbarrier ---
// `bar` is always the 32-bit shared::cta address of an 8-byte aligned uint64_t.
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t arrival_count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(arrival_count));
}
// Make mbarrier.init visible cluster-wide (needed before any remote CTA touches it).
__device__ __forceinline__ void fence_mbar_init() {
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
// Local arrive that also registers `bytes` of expected transactions.
__device__ __forceinline__ void mbar_arrive_expect_tx(uint32_t bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory");
}
// Same, but on the mbarrier at the same offset inside CTA `rank` (remote arrive).
__device__ __forceinline__ void mbar_arrive_expect_tx_remote(uint32_t bar, uint32_t rank, uint32_t bytes) {
    uint32_t rbar = mapa(bar, rank);
    asm volatile("mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%0], %1;" :: "r"(rbar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(bar) : "memory");
}
// Remote arrive on CTA `rank`'s mbarrier (no transaction bytes).
__device__ __forceinline__ void mbar_arrive_remote(uint32_t bar, uint32_t rank) {
    uint32_t rbar = mapa(bar, rank);
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" :: "r"(rbar) : "memory");
}
// Spin until the phase with the given parity completes (parity flips every
// time arrival count reaches zero and tx-count is zero).
__device__ __forceinline__ void mbar_wait_parity(uint32_t bar, uint32_t parity) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
        "@P bra DONE;\n"
        "bra WAIT;\n"
        "DONE:\n"
        "}\n" :: "r"(bar), "r"(parity) : "memory");
}

// ------------------------------------------------- bulk / async DSMEM push ---
// One thread issues: copy `bytes` from local smem `src` into CTA `dst_rank`'s
// smem at the same offset as local address `dst`, then complete_tx `bytes` on
// that CTA's mbarrier at the same offset as local `dst_bar`.
// Requirements: src/dst 16-B aligned, bytes % 16 == 0, the destination CTA has
// already done mbar_arrive_expect_tx(dst_bar, bytes) (or will, before waiting).
__device__ __forceinline__ void bulk_push(uint32_t dst_rank, uint32_t dst, uint32_t src,
                                          uint32_t bytes, uint32_t dst_bar) {
    uint32_t rdst = mapa(dst, dst_rank), rbar = mapa(dst_bar, dst_rank);
    asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(rdst), "r"(src), "r"(bytes), "r"(rbar) : "memory");
}
// 16-byte asynchronous remote store that completes 16 tx bytes on the remote mbarrier.
__device__ __forceinline__ void st_async_v4(uint32_t dst_rank, uint32_t dst, uint4 v, uint32_t dst_bar) {
    uint32_t rdst = mapa(dst, dst_rank), rbar = mapa(dst_bar, dst_rank);
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1,%2,%3,%4}, [%5];"
                 :: "r"(rdst), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "r"(rbar) : "memory");
}

// --------------------------------------------------------- cluster barrier ---
// Split-phase cluster barrier. arrive(release) ... wait(acquire).
__device__ __forceinline__ void cluster_arrive() {
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
}
__device__ __forceinline__ void cluster_arrive_relaxed() {   // no memory ordering
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}
__device__ __forceinline__ void cluster_wait() {
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}
// Full barrier: every thread of every CTA in the cluster. Equivalent to
// cg::this_cluster().sync().
__device__ __forceinline__ void cluster_sync() { cluster_arrive(); cluster_wait(); }

// ------------------------------------------------------------- reductions ---
// Reduction functors with identities (identity is needed by lanes that take
// part in a warp shuffle but own no element).
template <typename T> __device__ __forceinline__ T lowest_value();
template <> __device__ __forceinline__ float  lowest_value<float>()  { return -__int_as_float(0x7f800000); }
template <> __device__ __forceinline__ double lowest_value<double>() { return -__longlong_as_double(0x7ff0000000000000ll); }
template <> __device__ __forceinline__ int    lowest_value<int>()    { return (int)0x80000000; }
template <> __device__ __forceinline__ unsigned lowest_value<unsigned>() { return 0u; }
template <> __device__ __forceinline__ __half lowest_value<__half>() { return __float2half(-__int_as_float(0x7f800000)); }

struct Sum {
    template <typename T> __device__ __forceinline__ T operator()(T a, T b) const { return a + b; }
    template <typename T> static __device__ __forceinline__ T identity() { return T(0); }
};
struct Max {
    template <typename T> __device__ __forceinline__ T operator()(T a, T b) const { return a > b ? a : b; }
    template <typename T> static __device__ __forceinline__ T identity() { return lowest_value<T>(); }
};

// --------------------------------------------------------- harness contract ---
// Every <op>__<alg>__<src>.cu exposes
//   __global__ void __cluster_dims__(CLUSTER_N,1,1)
//   collective_kernel(const ELEM_T* in, ELEM_T* out, unsigned long long* ts, unsigned* smid);
// with one cluster per launch (gridDim.x == CLUSTER_N) and per-rank layouts
//   allreduce     : in [b*M .. b*M+M)            out[b*M .. )          (all ranks equal)
//   allgather     : in [b*M .. )                 out[b*N*M + r*M] = rank r's M
//   reducescatter : in [b*N*M + j*M] chunk j     out[b*M .. ) = sum_r chunk b of rank r
//   alltoall      : in [b*N*M + j*M] -> rank j   out[b*N*M + r*M] = chunk b of rank r
// Timestamps: ts[b*kStamps + 0] entry (after the entry cluster_sync),
//             ts[b*kStamps + 1] exit (after the last barrier of the collective),
//             ts[b*kStamps + 2 + r] end of round r (optional). smid[b] = SM id.
constexpr int kStamps = 32;

struct Stamps {
    unsigned long long* base;   // &ts[rank * kStamps]
    __device__ __forceinline__ void mark(int i) const { if (threadIdx.x == 0) base[i] = globaltimer(); }
    __device__ __forceinline__ void round(int r) const { mark(2 + r); }
};

// Copy M elements global -> smem (all threads), block-synchronous.
template <typename T, int M>
__device__ __forceinline__ void load_block(T* smem, const T* g) {
    for (int i = threadIdx.x; i < M; i += blockDim.x) smem[i] = g[i];
    __syncthreads();
}
template <typename T, int M>
__device__ __forceinline__ void store_block(T* g, const T* smem) {
    __syncthreads();
    for (int i = threadIdx.x; i < M; i += blockDim.x) g[i] = smem[i];
}

}  // namespace dsmem

// Default compile-time knobs (override with -D).
#ifndef CLUSTER_N
#define CLUSTER_N 4
#endif
#ifndef MSG_ELEMS
#define MSG_ELEMS 1024
#endif
#ifndef ELEM_T
#define ELEM_T float
#endif
