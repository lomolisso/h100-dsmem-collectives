/*
 * harness.cuh — device-side helpers shared by every experiment (sm_90a).
 *
 * Every helper is a single inline-PTX statement (or a fixed short sequence) in `asm volatile`, with a
 * "memory" clobber where ordering matters, so that the instruction in SASS is exactly the one written here.
 * PTX forms follow the project's validated templates (ptxas 12.9, sm_90a):
 *   timers / placement   read_clock64, read_clock32, read_globaltimer, read_timers_pair,
 *                        read_clock64_after, read_timers_after, TIMED_REGION_BEGIN/END,
 *                        read_smid, read_nsmid, read_laneid, read_cluster_regs, claim_target_sm
 *   cluster / DSMEM      smem_u32, mapa_u32, cluster_arrive{,_release,_relaxed}, cluster_wait{,_acquire},
 *                        cluster_sync, ld_dsmem_u32, st_dsmem_u32
 *   mbarrier             mbar_init, mbar_inval, fence_mbarrier_init, mbar_arrive, mbar_arrive_expect_tx,
 *                        mbar_remote_arrive, mbar_try_wait_parity{,_acq_cluster}, mbar_wait_parity{,_acq_cluster},
 *                        mbar_try_wait_parity_hint (suspendTimeHint), mbar_test_wait_parity (non-blocking)
 *   1-D bulk (TMA)       bulk_g2s, bulk_g2s_hint (.L2::cache_hint), bulk_g2s_multicast, bulk_s2g, bulk_s2s_dsmem,
 *                        bulk_prefetch_l2, bulk_commit, bulk_wait<N>, bulk_wait_read<N>, createpolicy_evict_last
 *   tensor TMA           tma_load_1d/2d/3d (.tile, G -> own smem, mbarrier completion), prefetch_tensormap
 *   cp.async (Ampere)    cp_async_cg16, cp_async_commit, cp_async_wait_all
 *   fences / misc        fence_proxy_async_shared_cta (+ other fences), bar_sync0, elect_one, nanosleep_ns,
 *                        relaxed/acquire/release global loads and stores at .gpu scope (cross-SM flags)
 *   synchronisation (07) bar_sync_n, bar_arrive_n, bar_warp_sync, non-aligned cluster_*_na, fence_acq_rel_cta,
 *                        fence_sc_cta, fence_sc_cluster, mbar_arrive_state, mbar_arrive_cnt_state,
 *                        mbar_arrive_expect_tx_state, mbar_arrive_drop_state, mbar_expect_tx, mbar_complete_tx,
 *                        mbar_remote_arrive_relaxed, st_async_u32, red_async_add_u32 (mbarrier complete_tx in the
 *                        destination CTA), DSMEM flags {ld,st}_{volatile,relaxed,acquire/release}_cluster_u32,
 *                        red_release_cluster_add_u32, atom_exch_acq_rel_cluster_u32
 *
 * Timed-region rule: the end timestamp must not be taken before the last measured load has returned.
 * read_*_after(dep, zero) first consumes `dep` (the last loaded value) in an AND with a runtime zero
 * (pass a kernel argument that is 0; a literal 0 would be folded away by ptxas), then reads the timer(s)
 * and adds the zero-valued result into them. The AND waits on the load's scoreboard, and the timer read
 * issues after it in program order. Check this ordering in the SASS of every timed loop.
 */
#pragma once
#include <cstdint>

// ---------------------------------------------------------------- timers and placement
struct TimerPair { uint64_t cycles, ns; };

__device__ __forceinline__ uint64_t read_clock64() {
    uint64_t t; asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory"); return t; }
__device__ __forceinline__ uint32_t read_clock32() {
    uint32_t t; asm volatile("mov.u32 %0, %%clock;" : "=r"(t) :: "memory"); return t; }
__device__ __forceinline__ uint64_t read_globaltimer() {
    uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory"); return t; }

// Both timers back to back in one asm block (cycles first).
__device__ __forceinline__ TimerPair read_timers_pair() {
    TimerPair p;
    asm volatile("mov.u64 %0, %%clock64;\n\tmov.u64 %1, %%globaltimer;" : "=l"(p.cycles), "=l"(p.ns) :: "memory");
    return p;
}

// %clock64 read that cannot issue before `dep` is available (zero must be a runtime 0).
__device__ __forceinline__ uint64_t read_clock64_after(uint64_t dep, uint64_t zero) {
    uint64_t t;
    asm volatile("{\n\t.reg .u64 d;\n\tand.b64 d, %1, %2;\n\tmov.u64 %0, %%clock64;\n\tadd.u64 %0, %0, d;\n\t}"
                 : "=l"(t) : "l"(dep), "l"(zero) : "memory");
    return t;
}
__device__ __forceinline__ TimerPair read_timers_after(uint64_t dep, uint64_t zero) {
    TimerPair p;
    asm volatile("{\n\t.reg .u64 d;\n\tand.b64 d, %2, %3;\n\tmov.u64 %0, %%clock64;\n\tmov.u64 %1, %%globaltimer;\n\t"
                 "add.u64 %0, %0, d;\n\tadd.u64 %1, %1, d;\n\t}"
                 : "=l"(p.cycles), "=l"(p.ns) : "l"(dep), "l"(zero) : "memory");
    return p;
}

// TIMED_REGION_BEGIN(t);  ...dependent chain producing `last`...;  TIMED_REGION_END(t, last, zero);
// defines t_begin / t_end (TimerPair). Store results only after TIMED_REGION_END.
#define TIMED_REGION_BEGIN(name) const TimerPair name##_begin = read_timers_pair()
#define TIMED_REGION_END(name, dep, zero) \
    const TimerPair name##_end = read_timers_after((uint64_t)(dep), (uint64_t)(zero))

__device__ __forceinline__ uint32_t read_smid()   { uint32_t r; asm volatile("mov.u32 %0, %%smid;"   : "=r"(r)); return r; }
__device__ __forceinline__ uint32_t read_nsmid()  { uint32_t r; asm volatile("mov.u32 %0, %%nsmid;"  : "=r"(r)); return r; }
__device__ __forceinline__ uint32_t read_laneid() { uint32_t r; asm volatile("mov.u32 %0, %%laneid;" : "=r"(r)); return r; }

// Pin a single-warp block to SM `target` without reserving that SM's shared memory (the L1 carveout stays free):
// launch many one-warp blocks (hb::SmClaim::blocks()); the first block that runs on `target` wins the claim and gets
// true in all its lanes, every other block should return at once. claim[0] = 0 before the launch (SmClaim::reset);
// afterwards claim[0] = winner blockIdx.x + 1 and claim[1] = its %smid.
__device__ __forceinline__ bool claim_target_sm(uint32_t target, uint32_t* claim) {
    uint32_t won = 0;
    if (threadIdx.x == 0 && read_smid() == target && atomicCAS(claim, 0u, blockIdx.x + 1) == 0u) {
        claim[1] = target;
        won = 1;
    }
    return __shfl_sync(0xffffffffu, won, 0) != 0;
}

struct ClusterRegs { uint32_t clusterid_x, clusterid_y, clusterid_z, ctarank, nctarank; };
__device__ __forceinline__ ClusterRegs read_cluster_regs() {
    ClusterRegs c;
    asm volatile("mov.u32 %0, %%clusterid.x;" : "=r"(c.clusterid_x));
    asm volatile("mov.u32 %0, %%clusterid.y;" : "=r"(c.clusterid_y));
    asm volatile("mov.u32 %0, %%clusterid.z;" : "=r"(c.clusterid_z));
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(c.ctarank));
    asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(c.nctarank));
    return c;
}

// ---------------------------------------------------------------- cluster / DSMEM
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
// .shared::cta address -> .shared::cluster address of the same offset in CTA `rank`.
__device__ __forceinline__ uint32_t mapa_u32(uint32_t smem_addr, uint32_t rank) {
    uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(smem_addr), "r"(rank)); return r; }
__device__ __forceinline__ void cluster_arrive()         { asm volatile("barrier.cluster.arrive.aligned;" ::: "memory"); }
__device__ __forceinline__ void cluster_arrive_release() { asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory"); }
__device__ __forceinline__ void cluster_arrive_relaxed() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }
__device__ __forceinline__ void cluster_wait()           { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }
__device__ __forceinline__ void cluster_wait_acquire()   { asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory"); }
__device__ __forceinline__ void cluster_sync()           { cluster_arrive_release(); cluster_wait_acquire(); }
__device__ __forceinline__ uint32_t ld_dsmem_u32(uint32_t a) {
    uint32_t v; asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(v) : "r"(a) : "memory"); return v; }
__device__ __forceinline__ void st_dsmem_u32(uint32_t a, uint32_t v) {
    asm volatile("st.shared::cluster.u32 [%0], %1;" :: "r"(a), "r"(v) : "memory"); }

// ---------------------------------------------------------------- mbarrier (64-bit object in .shared::cta)
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory"); }
__device__ __forceinline__ void mbar_inval(uint32_t bar) {
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(bar) : "memory"); }
// Publish initialised barriers to the async proxy and to peer CTAs (follow with a CTA or cluster barrier).
__device__ __forceinline__ void fence_mbarrier_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(bar) : "memory"); }
// arrive(1) + expect_tx(bytes) on a local barrier (.release.cta).
__device__ __forceinline__ void mbar_arrive_expect_tx(uint32_t bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory"); }
// Arrive on a peer CTA's barrier (address from mapa_u32).
__device__ __forceinline__ void mbar_remote_arrive(uint32_t rbar) {
    asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" :: "r"(rbar) : "memory"); }
__device__ __forceinline__ bool mbar_try_wait_parity(uint32_t bar, uint32_t parity) {
    uint32_t ok;
    asm volatile("{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\tselp.u32 %0, 1, 0, p;\n\t}"
                 : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
    return ok != 0;
}
// Use when the phase is completed by other CTAs, or by async copies issued from other CTAs.
__device__ __forceinline__ bool mbar_try_wait_parity_acq_cluster(uint32_t bar, uint32_t parity) {
    uint32_t ok;
    asm volatile("{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n\tselp.u32 %0, 1, 0, p;\n\t}"
                 : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
    return ok != 0;
}
__device__ __forceinline__ void mbar_wait_parity(uint32_t bar, uint32_t parity) {
    while (!mbar_try_wait_parity(bar, parity)) { } }
__device__ __forceinline__ void mbar_wait_parity_acq_cluster(uint32_t bar, uint32_t parity) {
    while (!mbar_try_wait_parity_acq_cluster(bar, parity)) { } }

// ---------------------------------------------------------------- 1-D bulk copies (async proxy, TMA unit)
// size: multiple of 16 B; src/dst 16-B aligned. Completion: mbarrier complete_tx (-> smem) or bulk_group (-> gmem).
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory"); }
// Same CTA-relative dst/bar offsets in every CTA of `mask`; each destination CTA must expect_tx(bytes).
__device__ __forceinline__ void bulk_g2s_multicast(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar, uint16_t mask) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1], %2, [%3], %4;"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "h"(mask) : "memory"); }
__device__ __forceinline__ void bulk_s2g(void* dst, uint32_t src, uint32_t bytes) {
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                 :: "l"(dst), "r"(src), "r"(bytes) : "memory"); }
// Put into a peer CTA's smem; dst and rbar are .shared::cluster addresses (mapa_u32) of the same peer.
__device__ __forceinline__ void bulk_s2s_dsmem(uint32_t dst, uint32_t src, uint32_t bytes, uint32_t rbar) {
    asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(dst), "r"(src), "r"(bytes), "r"(rbar) : "memory"); }
__device__ __forceinline__ void bulk_prefetch_l2(const void* src, uint32_t bytes) {
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(src), "r"(bytes) : "memory"); }
__device__ __forceinline__ void bulk_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void bulk_wait()      { asm volatile("cp.async.bulk.wait_group %0;" :: "n"(N) : "memory"); }
template <int N> __device__ __forceinline__ void bulk_wait_read() { asm volatile("cp.async.bulk.wait_group.read %0;" :: "n"(N) : "memory"); }

// suspendTimeHint variant: the thread may sleep up to hint_ns while the phase is incomplete.
__device__ __forceinline__ bool mbar_try_wait_parity_hint(uint32_t bar, uint32_t parity, uint32_t hint_ns) {
    uint32_t ok;
    asm volatile("{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n\tselp.u32 %0, 1, 0, p;\n\t}"
                 : "=r"(ok) : "r"(bar), "r"(parity), "r"(hint_ns) : "memory");
    return ok != 0;
}
// Non-blocking phase test.
__device__ __forceinline__ bool mbar_test_wait_parity(uint32_t bar, uint32_t parity) {
    uint32_t ok;
    asm volatile("{\n\t.reg .pred p;\n\tmbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n\tselp.u32 %0, 1, 0, p;\n\t}"
                 : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
    return ok != 0;
}

// ---------------------------------------------------------------- async-proxy extras: cache hints, tensor TMA, cp.async
// L2 policy operand for .L2::cache_hint (evict_last for the whole access).
__device__ __forceinline__ uint64_t createpolicy_evict_last() {
    uint64_t p; asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p)); return p; }
__device__ __forceinline__ void bulk_g2s_hint(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar, uint64_t policy) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(policy) : "memory"); }
// Tensor tile loads; tmap = address of a __grid_constant__ CUtensorMap parameter (or a 64-B aligned global copy),
// coordinates in elements, innermost first. The barrier receives complete_tx(box bytes).
__device__ __forceinline__ void tma_load_1d(uint32_t dst, const void* tmap, int32_t c0, uint32_t bar) {
    asm volatile("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2}], [%3];"
                 :: "r"(dst), "l"(tmap), "r"(c0), "r"(bar) : "memory"); }
__device__ __forceinline__ void tma_load_2d(uint32_t dst, const void* tmap, int32_t c0, int32_t c1, uint32_t bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 :: "r"(dst), "l"(tmap), "r"(c0), "r"(c1), "r"(bar) : "memory"); }
__device__ __forceinline__ void tma_load_3d(uint32_t dst, const void* tmap, int32_t c0, int32_t c1, int32_t c2, uint32_t bar) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
                 :: "r"(dst), "l"(tmap), "r"(c0), "r"(c1), "r"(c2), "r"(bar) : "memory"); }
__device__ __forceinline__ void prefetch_tensormap(const void* tmap) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(tmap) : "memory"); }
// Ampere-style asynchronous 16-B copy G -> S (L1 bypass), group completion.
__device__ __forceinline__ void cp_async_cg16(uint32_t dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory"); }
__device__ __forceinline__ void cp_async_commit()   { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;" ::: "memory"); }

// ---------------------------------------------------------------- fences, barriers, misc
__device__ __forceinline__ void fence_proxy_async_shared_cta() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void fence_proxy_async_dsmem()      { asm volatile("fence.proxy.async.shared::cluster;" ::: "memory"); }
__device__ __forceinline__ void fence_proxy_async_all()        { asm volatile("fence.proxy.async;" ::: "memory"); }
__device__ __forceinline__ void fence_acq_rel_cluster()        { asm volatile("fence.acq_rel.cluster;" ::: "memory"); }
__device__ __forceinline__ void fence_acq_rel_gpu()            { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }
__device__ __forceinline__ void fence_sc_gpu()                 { asm volatile("fence.sc.gpu;" ::: "memory"); }
__device__ __forceinline__ void bar_sync0()                    { asm volatile("bar.sync 0;" ::: "memory"); }
__device__ __forceinline__ void nanosleep_ns(uint32_t ns)      { asm volatile("nanosleep.u32 %0;" :: "r"(ns)); }
__device__ __forceinline__ bool elect_one() {
    uint32_t p;
    asm volatile("{\n\t.reg .pred q;\n\telect.sync _|q, 0xffffffff;\n\tselp.u32 %0, 1, 0, q;\n\t}" : "=r"(p));
    return p != 0;
}

// Cross-SM flags in global memory (.gpu scope).
__device__ __forceinline__ uint32_t ld_relaxed_gpu_u32(const uint32_t* a) {
    uint32_t v; asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory"); return v; }
__device__ __forceinline__ uint32_t ld_acquire_gpu_u32(const uint32_t* a) {
    uint32_t v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory"); return v; }
__device__ __forceinline__ void st_relaxed_gpu_u32(uint32_t* a, uint32_t v) {
    asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" :: "l"(a), "r"(v) : "memory"); }
__device__ __forceinline__ void st_release_gpu_u32(uint32_t* a, uint32_t v) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(a), "r"(v) : "memory"); }

// ---------------------------------------------------------------- synchronisation extras (experiment 07)
// Named CTA barriers: id 0..15, n = participating threads (multiple of 32).
__device__ __forceinline__ void bar_sync_n(uint32_t id, uint32_t n)   { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void bar_arrive_n(uint32_t id, uint32_t n) { asm volatile("bar.arrive %0, %1;" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void bar_warp_sync()                        { asm volatile("bar.warp.sync -1;" ::: "memory"); }
// Cluster barrier without .aligned: may be executed by a subset of a warp's threads (e.g. one active thread per CTA).
__device__ __forceinline__ void cluster_arrive_release_na() { asm volatile("barrier.cluster.arrive.release;" ::: "memory"); }
__device__ __forceinline__ void cluster_arrive_relaxed_na() { asm volatile("barrier.cluster.arrive.relaxed;" ::: "memory"); }
__device__ __forceinline__ void cluster_wait_acquire_na()   { asm volatile("barrier.cluster.wait.acquire;" ::: "memory"); }
__device__ __forceinline__ void fence_acq_rel_cta()         { asm volatile("fence.acq_rel.cta;" ::: "memory"); }
__device__ __forceinline__ void fence_sc_cta()              { asm volatile("fence.sc.cta;" ::: "memory"); }
__device__ __forceinline__ void fence_sc_cluster()          { asm volatile("fence.sc.cluster;" ::: "memory"); }
// mbarrier ops that return the arrival state (so that a timer can depend on their completion).
__device__ __forceinline__ uint64_t mbar_arrive_state(uint32_t bar) {
    uint64_t s; asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];" : "=l"(s) : "r"(bar) : "memory"); return s; }
__device__ __forceinline__ uint64_t mbar_arrive_cnt_state(uint32_t bar, uint32_t count) {
    uint64_t s; asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1], %2;" : "=l"(s) : "r"(bar), "r"(count) : "memory"); return s; }
__device__ __forceinline__ uint64_t mbar_arrive_expect_tx_state(uint32_t bar, uint32_t bytes) {
    uint64_t s; asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 %0, [%1], %2;" : "=l"(s) : "r"(bar), "r"(bytes) : "memory"); return s; }
__device__ __forceinline__ uint64_t mbar_arrive_drop_state(uint32_t bar) {
    uint64_t s; asm volatile("mbarrier.arrive_drop.shared::cta.b64 %0, [%1];" : "=l"(s) : "r"(bar) : "memory"); return s; }
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
    asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(bytes) : "memory"); }
__device__ __forceinline__ void mbar_complete_tx(uint32_t bar, uint32_t bytes) {
    asm volatile("mbarrier.complete_tx.relaxed.cta.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(bytes) : "memory"); }
__device__ __forceinline__ void mbar_remote_arrive_relaxed(uint32_t rbar) {
    asm volatile("mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [%0];" :: "r"(rbar) : "memory"); }
// Register → peer smem with complete_tx(4 B) on the peer's mbarrier (a, rbar: .shared::cluster addresses of one peer).
__device__ __forceinline__ void st_async_u32(uint32_t a, uint32_t v, uint32_t rbar) {
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.u32 [%0], %1, [%2];" :: "r"(a), "r"(v), "r"(rbar) : "memory"); }
__device__ __forceinline__ void red_async_add_u32(uint32_t a, uint32_t v, uint32_t rbar) {
    asm volatile("red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.u32 [%0], %1, [%2];"
                 :: "r"(a), "r"(v), "r"(rbar) : "memory"); }
// DSMEM flags (.shared::cluster addresses, own rank allowed).
__device__ __forceinline__ uint32_t ld_volatile_cluster_u32(uint32_t a) {
    uint32_t v; asm volatile("ld.volatile.shared::cluster.u32 %0, [%1];" : "=r"(v) : "r"(a) : "memory"); return v; }
__device__ __forceinline__ uint32_t ld_relaxed_cluster_u32(uint32_t a) {
    uint32_t v; asm volatile("ld.relaxed.cluster.shared::cluster.u32 %0, [%1];" : "=r"(v) : "r"(a) : "memory"); return v; }
__device__ __forceinline__ uint32_t ld_acquire_cluster_u32(uint32_t a) {
    uint32_t v; asm volatile("ld.acquire.cluster.shared::cluster.u32 %0, [%1];" : "=r"(v) : "r"(a) : "memory"); return v; }
__device__ __forceinline__ void st_volatile_cluster_u32(uint32_t a, uint32_t v) {
    asm volatile("st.volatile.shared::cluster.u32 [%0], %1;" :: "r"(a), "r"(v) : "memory"); }
__device__ __forceinline__ void st_relaxed_cluster_u32(uint32_t a, uint32_t v) {
    asm volatile("st.relaxed.cluster.shared::cluster.u32 [%0], %1;" :: "r"(a), "r"(v) : "memory"); }
__device__ __forceinline__ void st_release_cluster_u32(uint32_t a, uint32_t v) {
    asm volatile("st.release.cluster.shared::cluster.u32 [%0], %1;" :: "r"(a), "r"(v) : "memory"); }
__device__ __forceinline__ void red_release_cluster_add_u32(uint32_t a, uint32_t v) {
    asm volatile("red.release.cluster.shared::cluster.add.u32 [%0], %1;" :: "r"(a), "r"(v) : "memory"); }
__device__ __forceinline__ uint32_t atom_exch_acq_rel_cluster_u32(uint32_t a, uint32_t v) {
    uint32_t o; asm volatile("atom.acq_rel.cluster.shared::cluster.exch.b32 %0, [%1], %2;" : "=r"(o) : "r"(a), "r"(v) : "memory"); return o; }
