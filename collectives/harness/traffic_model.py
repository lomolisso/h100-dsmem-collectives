#!/usr/bin/env python3
"""traffic_model.py — per-rank DSMEM traffic model of the 29 collective files (ROADMAP C5 step 4, EXPERIMENTS §5).

For every (file, N, M, rank) the model replays the communication schedule of the kernel (the loops of the
<op>__<alg>__<src>.cu file, with exact integer arithmetic) as a list of transfers
    (initiator, src, dst, bytes)
and sums them per rank. Only DSMEM payload counts: bytes moved between the shared memories of two DIFFERENT CTAs
by cp.async.bulk (bulk_push), st.async, plain remote stores through map_shared_rank/mapa pointers, remote loads
(ld.shared::cluster, generic loads through map_shared_rank) and remote atomics/red (the element payload of one
atomic). Local smem copies, mbarrier arrives/tx counts and global memory are not DSMEM payload.

Columns per rank r:
  bytes_sent       payload leaving r's SM: r's pushes/atomics + what peers pull out of r's smem
  bytes_recv       payload arriving at r's SM: what peers push/atomically add into r + what r pulls from peers
  bytes_initiated  payload moved by r's own instructions (pushes + pulls + atomics)
  bytes_self       payload of DSMEM-addressed operations whose target is r's OWN smem (map_shared_rank/mapa to
                   the own rank); not inter-CTA traffic, not included in the three columns above
  rounds           number of st.round(...) stamps r writes (all ranks execute every st.round in every file)
  global_read_bytes   r's reads of its `in` slice
  global_write_bytes  r's writes of its `out` slice + (2 + rounds) * 8 B of stamps (st.mark(0), st.mark(1),
                      st.round(i)) + 4 B smid
Invariant: sum_r bytes_sent = sum_r bytes_recv = sum_r bytes_initiated = kernel-wide DSMEM payload.

Usage (host only, stdlib, no GPU):
  python3 collectives/harness/traffic_model.py [--set perf|default]   append rows for every configuration of the
        driver's run set (driver.runnable(latest_build(), ..., which)) to results/model.csv (one block per run)
  python3 collectives/harness/traffic_model.py --check                 totals vs README, invariants, ncu ground truth;
        writes nothing; exit 1 on an invariant, self-consistency or ground-truth failure
"""
import argparse
import os
import sys

HARNESS = os.path.dirname(os.path.abspath(__file__))


class Illegal(ValueError):
    """(file, N, M) violates one of the file's static_asserts (or the cluster-size range)."""


def _need(cond, msg):
    if not cond:
        raise Illegal(msg)


def _pow2(n):
    return n >= 1 and n & (n - 1) == 0


def _log2(n):  # exact for powers of two
    return n.bit_length() - 1


def _clog2(n):  # ceil(log2 n)
    return (n - 1).bit_length()


class Sched:
    """Transfers (initiator, src, dst, bytes) and the number of st.round stamps per rank."""

    def __init__(self, N):
        self.N, self.moves, self.rounds = N, [], [0] * N

    def push(self, src, dst, nbytes):  # bulk copy / st.async / remote store / remote atomic issued by src
        self.moves.append((src, src, dst, nbytes))

    def pull(self, dst, src, nbytes):  # remote load issued by dst out of src's smem
        self.moves.append((dst, src, dst, nbytes))

    def all_rounds(self, k):
        self.rounds = [k] * self.N


def _v16(nbytes, what="bulk copy / 16-byte vector access"):
    _need(nbytes % 16 == 0, f"{what} needs a multiple of 16 bytes (got {nbytes})")


# ------------------------------------------------------------------------------------------------ AllReduce --

def ar_2d_mesh(N, M, B):
    # allreduce_2d_mesh: Grid2D R x C; phase 1 bulk_push j=1..C-1 to (row,(col+j)%C), phase 2 j=1..R-1 to
    # ((row+j)%R,col) (only if R>1), phase 3 j=1..C-1 again; chunk K=M/C; st.round(0..2) always.
    R = 4 if N >= 16 else (2 if N >= 4 else 1)
    C = N // R
    _need(R * C == N, "grid must cover the cluster")
    _need(M % C == 0, "M % C")
    K = M // C
    _v16(K * B)
    s = Sched(N)
    for rank in range(N):
        row, col = divmod(rank, C)
        for j in range(1, C):
            s.push(rank, row * C + (col + j) % C, K * B)
        for j in range(1, R):
            s.push(rank, ((row + j) % R) * C + col, K * B)
        for j in range(1, C):
            s.push(rank, row * C + (col + j) % C, K * B)
    s.all_rounds(3)
    return s


def ar_atomic_all_push(N, M, B):
    # allreduce_atomic_all_push: idx < (N-1)*M, peer = (rank+1+idx/M)%N (never self), one red/atomicAdd of B bytes
    # per (peer, element); st.round(0).
    s = Sched(N)
    for rank in range(N):
        for k in range(N - 1):
            s.push(rank, (rank + 1 + k) % N, M * B)
    s.all_rounds(1)
    return s


def ar_direct_replicated_push(N, M, B):
    # allreduce_direct_replicated_push: idx < N*M, p = idx/M over ALL ranks incl. self, *map_ptr(&slots[rank][e], p)
    # = plain remote store; the p == rank stores are self-addressed (bytes_self); st.round(0).
    s = Sched(N)
    for rank in range(N):
        for p in range(N):
            s.push(rank, p, M * B)
    s.all_rounds(1)
    return s


def ar_one_shot_pull(N, M, B):
    # allreduce_one_shot_pull: for r in 0..N-1 (incl. self) ld_cluster_v4(mapa(buf_a + v*16, r)) for v < M/V;
    # the r == rank loads are self-addressed; st.round(0).
    _v16(M * B)
    s = Sched(N)
    for rank in range(N):
        for r in range(N):
            s.pull(rank, r, M * B)
    s.all_rounds(1)
    return s


def ar_pull_rs_push_ag(N, M, B):
    # allreduce_pull_rs_push_ag: linear < C*N, e = rank*C + linear/N, peer = linear%N (incl. self):
    # v = *map_ptr(&buf[e], peer) (pull) then *addr = v (push), C = M/N elements per peer; st.round(0).
    _need(32 % N == 0, "N | 32")
    _need(M % N == 0, "N | M")
    C = M // N
    s = Sched(N)
    for rank in range(N):
        for p in range(N):
            s.pull(rank, p, C * B)
            s.push(rank, p, C * B)
    s.all_rounds(1)
    return s


def ar_rabenseifner(N, M, B):
    # allreduce_rabenseifner: phase 1 d=N/2..1 bulk_push(rank^d, ..., (send_hi-send_lo)*B) with the live range
    # halving; phase 2 d=1..N/2 bulk_push(rank^d, ..., len*B) with len doubling; st.round per round (2 log2 N).
    _need(_pow2(N), "power-of-two N")
    _need(M % N == 0, "M % N")
    _v16((M // N) * B)
    s = Sched(N)
    for rank in range(N):
        lo, hi = 0, M
        d = N // 2
        while d >= 1:
            mid = (lo + hi) // 2
            upper = (rank & d) != 0
            send_lo, send_hi = (lo, mid) if upper else (mid, hi)
            keep_lo = mid if upper else lo
            s.push(rank, rank ^ d, (send_hi - send_lo) * B)
            length = hi - mid
            lo, hi = keep_lo, keep_lo + length
            d //= 2
        d = 1
        while d < N:
            length = hi - lo
            s.push(rank, rank ^ d, length * B)
            if rank & d:
                lo -= length
            else:
                hi += length
            d *= 2
    s.all_rounds(2 * _log2(N))
    return s


def ar_recursive_doubling(N, M, B):
    # allreduce_recursive_doubling: s = 1,2,..<N: bulk_push((rank+s)%N, scratch, cur, M*B); st.round per s.
    _need(_pow2(N), "power-of-two N")
    _v16(M * B)
    s = Sched(N)
    for rank in range(N):
        st = 1
        while st < N:
            s.push(rank, (rank + st) % N, M * B)
            st *= 2
    s.all_rounds(_log2(N))
    return s


def ar_reduce_bcast_binomial(N, M, B):
    # allreduce_reduce_bcast_binomial: reduce d=1..N/2, active rank with bit d set pushes M*B to rank-d and drops
    # out; bcast d=N/2..1, rank%(2d)==0 pushes M*B to rank+d; st.round every round on every rank (2 log2 N).
    _need(_pow2(N), "power-of-two N")
    _v16(M * B)
    s = Sched(N)
    for rank in range(N):
        active, d = True, 1
        while d < N:
            if active and (rank & d):
                s.push(rank, rank - d, M * B)
                active = False
            d *= 2
        d = N // 2
        while d >= 1:
            if rank % (2 * d) == 0:
                s.push(rank, rank + d, M * B)
            d //= 2
    s.all_rounds(2 * _log2(N))
    return s


def ar_ring(N, M, B):
    # allreduce_ring: N-1 reduce-scatter + N-1 all-gather steps, each bulk_push(next, ..., C*B), C = M/N;
    # st.round per step (2(N-1)).
    _need(M % N == 0, "M % N")
    C = M // N
    _v16(C * B)
    s = Sched(N)
    for rank in range(N):
        for _ in range(2 * (N - 1)):
            s.push(rank, (rank + 1) % N, C * B)
    s.all_rounds(2 * (N - 1))
    return s


def ar_shift_accumulate_atomic(N, M, B):
    # allreduce_shift_accumulate_atomic: rounds = 1 (N=2) or N-2; round i: atomicAdd(&map_ptr(val,(rank+i)%N)[e])
    # for all M elements (B bytes each); st.round(i-1).
    _need(N in (2, 4), "N in {2, 4}")
    rounds = 1 if N == 2 else N - 2
    s = Sched(N)
    for rank in range(N):
        for i in range(1, rounds + 1):
            s.push(rank, (rank + i) % N, M * B)
    s.all_rounds(rounds)
    return s


def ar_shift_accumulate_bulk(N, M, B):
    # allreduce_shift_accumulate: i = 1..N-2, bulk_push((rank+i)%N, scratch, cur, M*B); st.round(i-1).
    _need(N == 4, "N == 4")
    _v16(M * B)
    s = Sched(N)
    for rank in range(N):
        for i in range(1, N - 1):
            s.push(rank, (rank + i) % N, M * B)
    s.all_rounds(N - 2)
    return s


def ar_two_shot_push(N, M, B):
    # allreduce_two_shot_push: shot 1 j=1..N-1 bulk_push((rank+j)%N, slot, chunk, C*B); shot 2 same peers, C*B;
    # st.round(0), st.round(1).
    _need(M % N == 0, "M % N")
    C = M // N
    _v16(C * B)
    s = Sched(N)
    for rank in range(N):
        for _shot in range(2):
            for j in range(1, N):
                s.push(rank, (rank + j) % N, C * B)
    s.all_rounds(2)
    return s


# ------------------------------------------------------------------------------------------------ AllGather --

def ag_bruck(N, M, B):
    # allgather_bruck: d = 1,2,4..<N, cnt = min(d, N-d), bulk_push((rank+N-d)%N, D+d*blk, D, cnt*blk);
    # st.round per d (ceil log2 N).
    _v16(M * B)
    s = Sched(N)
    for rank in range(N):
        d = 1
        while d < N:
            s.push(rank, (rank + N - d) % N, min(d, N - d) * M * B)
            d *= 2
    s.all_rounds(_clog2(N))
    return s


def ag_direct_pull(N, M, B):
    # allgather_direct_pull: r != rank (own slot copied locally), ld_cluster_v4(mapa(mine_a + u*16, r)), M*B per
    # peer; st.round(0).
    _v16(M * B)
    s = Sched(N)
    for rank in range(N):
        for r in range(N):
            if r != rank:
                s.pull(rank, r, M * B)
    s.all_rounds(1)
    return s


def ag_direct_replicated_push(N, M, B):
    # allgather_direct_replicated_push: idx < N*M, p = idx/M over ALL ranks incl. self, *map_ptr(&slots[rank][e], p)
    # = mine[e]; p == rank is self-addressed; st.round(0).
    s = Sched(N)
    for rank in range(N):
        for p in range(N):
            s.push(rank, p, M * B)
    s.all_rounds(1)
    return s


def ag_direct_shift(N, M, B):
    # allgather_direct_shift: i = 1..N-1, bulk_push((rank+i)%N, seg, seg, M*B); st.round(i-1).
    _v16(M * B)
    s = Sched(N)
    for rank in range(N):
        for i in range(1, N):
            s.push(rank, (rank + i) % N, M * B)
    s.all_rounds(N - 1)
    return s


def ag_gather_bcast_binomial(N, M, B):
    # allgather_gather_bcast_binomial: gather d=1..N/2, active rank with bit d pushes d*blk to rank-d (its whole
    # subtree); bcast d=N/2..1, rank%(2d)==0 pushes N*blk to rank+d; st.round every round (2 log2 N).
    _need(_pow2(N), "power-of-two N")
    _v16(M * B)
    s = Sched(N)
    for rank in range(N):
        active, d = True, 1
        while d < N:
            if active and (rank & d):
                s.push(rank, rank - d, d * M * B)
                active = False
            d *= 2
        d = N // 2
        while d >= 1:
            if rank % (2 * d) == 0:
                s.push(rank, rank + d, N * M * B)
            d //= 2
    s.all_rounds(2 * _log2(N))
    return s


def ag_recursive_doubling(N, M, B):
    # allgather_recursive_doubling: s = 1,2,..<N, bulk_push((rank+s)%N, D+s*M, D, s*M*B); st.round per s.
    _need(_pow2(N), "power-of-two N")
    _v16(M * B)
    s = Sched(N)
    for rank in range(N):
        st = 1
        while st < N:
            s.push(rank, (rank + st) % N, st * M * B)
            st *= 2
    s.all_rounds(_log2(N))
    return s


def ag_ring(N, M, B):
    # allgather_ring: s = 0..N-2, bulk_push(next, slot, slot, M*B); st.round(s).
    _v16(M * B)
    s = Sched(N)
    for rank in range(N):
        for _ in range(N - 1):
            s.push(rank, (rank + 1) % N, M * B)
    s.all_rounds(N - 1)
    return s


# ------------------------------------------------------------------------------------------- ReduceScatter --

def rs_direct_pull(N, M, B):
    # reducescatter_direct_pull: for r in 0..N-1 (incl. self) ld_cluster_v4(mapa(chunk_a + v*16, r)), M*B per r;
    # the r == rank loads are self-addressed; st.round(0).
    _v16(M * B)
    s = Sched(N)
    for rank in range(N):
        for r in range(N):
            s.pull(rank, r, M * B)
    s.all_rounds(1)
    return s


def rs_direct_push(N, M, B):
    # reducescatter_direct_push: j = 1..N-1, bulk_push((rank+j)%N, slot[rank], chunk dst, M*B); st.round(0).
    _v16(M * B)
    s = Sched(N)
    for rank in range(N):
        for j in range(1, N):
            s.push(rank, (rank + j) % N, M * B)
    s.all_rounds(1)
    return s


def rs_recursive_halving(N, M, B):
    # reducescatter_recursive_halving: live range [0, N*M) halves, d = N/2..1, bulk_push(rank^d, scratch, ...,
    # len*B) with len = hi - mid; st.round per d (log2 N).
    _need(_pow2(N), "power-of-two N")
    _v16(M * B)
    s = Sched(N)
    for rank in range(N):
        lo, hi = 0, N * M
        d = N // 2
        while d >= 1:
            mid = (lo + hi) // 2
            length = hi - mid
            keep_lo = mid if (rank & d) else lo
            s.push(rank, rank ^ d, length * B)
            lo, hi = keep_lo, keep_lo + length
            d //= 2
    s.all_rounds(_log2(N))
    return s


def rs_ring(N, M, B):
    # reducescatter_ring: s = 0..N-2, bulk_push(next, scratch, chunk send_c, M*B); st.round(s).
    _v16(M * B)
    s = Sched(N)
    for rank in range(N):
        for _ in range(N - 1):
            s.push(rank, (rank + 1) % N, M * B)
    s.all_rounds(N - 1)
    return s


# ------------------------------------------------------------------------------------------------ AlltoAll --

def a2a_bruck(N, M, B):
    # alltoall_bruck: d = 1,2,..<N, one bulk_push((rank+d)%N, tmp+s*blk, D+pos*blk, blk) per pos with (pos & d)
    # (N/2 blocks); st.round per d (log2 N).
    _need(_pow2(N), "power-of-two N")
    _v16(M * B)
    s = Sched(N)
    for rank in range(N):
        d = 1
        while d < N:
            for pos in range(N):
                if pos & d:
                    s.push(rank, (rank + d) % N, M * B)
            d *= 2
    s.all_rounds(_log2(N))
    return s


def a2a_direct_pull(N, M, B):
    # alltoall_direct_pull: r != rank -> ld_cluster_v4(mapa(my_chunk_a + u*16, r)); r == rank is a LOCAL uint4
    # load (not DSMEM); M*B per peer; st.round(0).
    _v16(M * B)
    s = Sched(N)
    for rank in range(N):
        for r in range(N):
            if r != rank:
                s.pull(rank, r, M * B)
    s.all_rounds(1)
    return s


def a2a_direct_push(N, M, B):
    # alltoall_direct_push: j = 1..N-1, bulk_push((rank+j)%N, recv[rank], chunk dst, M*B); own chunk copied
    # locally; st.round(0).
    _v16(M * B)
    s = Sched(N)
    for rank in range(N):
        for j in range(1, N):
            s.push(rank, (rank + j) % N, M * B)
    s.all_rounds(1)
    return s


def a2a_pairwise_exchange(N, M, B):
    # alltoall_pairwise_exchange: N == 2, bulk_push(rank^1, recv, chunks + (rank^1)*M, M*B) (the remote
    # mbarrier.arrive.expect_tx carries no payload); st.round(0).
    _need(N == 2, "N == 2")
    _v16(M * B)
    s = Sched(N)
    for rank in range(N):
        s.push(rank, rank ^ 1, M * B)
    s.all_rounds(1)
    return s


def a2a_pairwise_xor(N, M, B):
    # alltoall_pairwise_xor: i = 1..N-1, bulk_push(rank^i, recv[rank], chunk p, M*B); st.round(i-1).
    _need(_pow2(N), "power-of-two N")
    _v16(M * B)
    s = Sched(N)
    for rank in range(N):
        for i in range(1, N):
            s.push(rank, rank ^ i, M * B)
    s.all_rounds(N - 1)
    return s


def a2a_shift(N, M, B):
    # alltoall_shift: i = 1..N-1, bulk_push((rank+i)%N, recv[rank], chunk dst, M*B); st.round(i-1).
    _v16(M * B)
    s = Sched(N)
    for rank in range(N):
        for i in range(1, N):
            s.push(rank, (rank + i) % N, M * B)
    s.all_rounds(N - 1)
    return s


FILES = {
    "allreduce__2d_mesh__generic": ar_2d_mesh,
    "allreduce__atomic_all_push__generic": ar_atomic_all_push,
    "allreduce__direct_replicated_push__credit": ar_direct_replicated_push,
    "allreduce__one_shot_pull__trtllm": ar_one_shot_pull,
    "allreduce__pull_reduce_scatter_push_allgather__hpcops": ar_pull_rs_push_ag,
    "allreduce__rabenseifner__mpich": ar_rabenseifner,
    "allreduce__recursive_doubling__clusterfusion": ar_recursive_doubling,
    "allreduce__reduce_bcast_binomial__mpich": ar_reduce_bcast_binomial,
    "allreduce__ring__nccl": ar_ring,
    "allreduce__shift_accumulate_atomic__clusterfusion": ar_shift_accumulate_atomic,
    "allreduce__shift_accumulate_bulk__clusterfusion": ar_shift_accumulate_bulk,
    "allreduce__two_shot_push__trtllm": ar_two_shot_push,
    "allgather__bruck__mpich": ag_bruck,
    "allgather__direct_pull__generic": ag_direct_pull,
    "allgather__direct_replicated_push__credit": ag_direct_replicated_push,
    "allgather__direct_shift__clusterfusion": ag_direct_shift,
    "allgather__gather_bcast_binomial__mpich": ag_gather_bcast_binomial,
    "allgather__recursive_doubling__clusterfusion": ag_recursive_doubling,
    "allgather__ring__nccl": ag_ring,
    "reducescatter__direct_pull__generic": rs_direct_pull,
    "reducescatter__direct_push__generic": rs_direct_push,
    "reducescatter__recursive_halving__mpich": rs_recursive_halving,
    "reducescatter__ring__nccl": rs_ring,
    "alltoall__bruck__mpich": a2a_bruck,
    "alltoall__direct_pull__generic": a2a_direct_pull,
    "alltoall__direct_push__generic": a2a_direct_push,
    "alltoall__pairwise_exchange__fa4": a2a_pairwise_exchange,
    "alltoall__pairwise_xor__mpich": a2a_pairwise_xor,
    "alltoall__shift__generic": a2a_shift,
}

# Elements of `in` read and of `out` written per rank (load_block / store_block / explicit loops of each kernel):
# allreduce in M, out M; allgather in M, out N*M; reducescatter in N*M, out M; alltoall in N*M, out N*M
# (fa4: two store_block<M> = N*M at N = 2; alltoall bruck: rotation loop reads N*M, un-rotation writes N*M).
GLOBAL_ELEMS = {
    "allreduce": lambda N, M: (M, M),
    "allgather": lambda N, M: (M, N * M),
    "reducescatter": lambda N, M: (N * M, M),
    "alltoall": lambda N, M: (N * M, N * M),
}
STAMP_BYTES, SMID_BYTES = 8, 4

KEYS = ["bytes_sent", "bytes_recv", "bytes_initiated", "bytes_self", "rounds", "global_read_bytes",
        "global_write_bytes"]


def model(file: str, N: int, M: int, elem_bytes: int = 4) -> list:
    """N dicts (rank order) with the KEYS columns. Raises Illegal if (N, M) violates the file's static_asserts."""
    if file not in FILES:
        raise KeyError(f"unknown file {file}")
    _need(2 <= N <= 16, "cluster size 2..16")
    _need(M >= 1, "M >= 1")
    s = FILES[file](N, M, elem_bytes)
    rows = [dict.fromkeys(KEYS, 0) for _ in range(N)]
    for init, src, dst, nbytes in s.moves:
        if src == dst:
            rows[init]["bytes_self"] += nbytes
            continue
        rows[src]["bytes_sent"] += nbytes
        rows[dst]["bytes_recv"] += nbytes
        rows[init]["bytes_initiated"] += nbytes
    rd, wr = GLOBAL_ELEMS[file.split("__")[0]](N, M)
    for r, row in enumerate(rows):
        row["rounds"] = s.rounds[r]
        row["global_read_bytes"] = rd * elem_bytes
        row["global_write_bytes"] = wr * elem_bytes + (2 + s.rounds[r]) * STAMP_BYTES + SMID_BYTES
    return rows


# ------------------------------------------------------------------------------- closed forms (for --check) --
# Kernel-wide inter-CTA payload, kernel-wide self-addressed payload and per-rank rounds, written independently of
# the schedule replay above, in units of MB = M*B (C = M/N where noted). --check requires replay == closed form.

def _grid(N):
    R = 4 if N >= 16 else (2 if N >= 4 else 1)
    return R, N // R


FORMULA = {  # file: (total(N, M, B), self(N, M, B), rounds(N))
    "allreduce__2d_mesh__generic":
        (lambda N, M, B: N * (2 * (_grid(N)[1] - 1) + (_grid(N)[0] - 1)) * (M // _grid(N)[1]) * B,
         lambda N, M, B: 0, lambda N: 3),
    "allreduce__atomic_all_push__generic": (lambda N, M, B: N * (N - 1) * M * B, lambda N, M, B: 0, lambda N: 1),
    "allreduce__direct_replicated_push__credit":
        (lambda N, M, B: N * (N - 1) * M * B, lambda N, M, B: N * M * B, lambda N: 1),
    "allreduce__one_shot_pull__trtllm":
        (lambda N, M, B: N * (N - 1) * M * B, lambda N, M, B: N * M * B, lambda N: 1),
    "allreduce__pull_reduce_scatter_push_allgather__hpcops":
        (lambda N, M, B: 2 * (N - 1) * M * B, lambda N, M, B: 2 * M * B, lambda N: 1),
    "allreduce__rabenseifner__mpich":
        (lambda N, M, B: 2 * (N - 1) * M * B, lambda N, M, B: 0, lambda N: 2 * _log2(N)),
    "allreduce__recursive_doubling__clusterfusion":
        (lambda N, M, B: N * _log2(N) * M * B, lambda N, M, B: 0, lambda N: _log2(N)),
    "allreduce__reduce_bcast_binomial__mpich":
        (lambda N, M, B: 2 * (N - 1) * M * B, lambda N, M, B: 0, lambda N: 2 * _log2(N)),
    "allreduce__ring__nccl": (lambda N, M, B: 2 * (N - 1) * M * B, lambda N, M, B: 0, lambda N: 2 * (N - 1)),
    "allreduce__shift_accumulate_atomic__clusterfusion":
        (lambda N, M, B: N * (1 if N == 2 else N - 2) * M * B, lambda N, M, B: 0, lambda N: 1 if N == 2 else N - 2),
    "allreduce__shift_accumulate_bulk__clusterfusion":
        (lambda N, M, B: N * (N - 2) * M * B, lambda N, M, B: 0, lambda N: N - 2),
    "allreduce__two_shot_push__trtllm": (lambda N, M, B: 2 * (N - 1) * M * B, lambda N, M, B: 0, lambda N: 2),
    "allgather__bruck__mpich": (lambda N, M, B: N * (N - 1) * M * B, lambda N, M, B: 0, lambda N: _clog2(N)),
    "allgather__direct_pull__generic": (lambda N, M, B: N * (N - 1) * M * B, lambda N, M, B: 0, lambda N: 1),
    "allgather__direct_replicated_push__credit":
        (lambda N, M, B: N * (N - 1) * M * B, lambda N, M, B: N * M * B, lambda N: 1),
    "allgather__direct_shift__clusterfusion":
        (lambda N, M, B: N * (N - 1) * M * B, lambda N, M, B: 0, lambda N: N - 1),
    "allgather__gather_bcast_binomial__mpich":
        (lambda N, M, B: ((N // 2) * _log2(N) + N * (N - 1)) * M * B, lambda N, M, B: 0, lambda N: 2 * _log2(N)),
    "allgather__recursive_doubling__clusterfusion":
        (lambda N, M, B: N * (N - 1) * M * B, lambda N, M, B: 0, lambda N: _log2(N)),
    "allgather__ring__nccl": (lambda N, M, B: N * (N - 1) * M * B, lambda N, M, B: 0, lambda N: N - 1),
    "reducescatter__direct_pull__generic":
        (lambda N, M, B: N * (N - 1) * M * B, lambda N, M, B: N * M * B, lambda N: 1),
    "reducescatter__direct_push__generic": (lambda N, M, B: N * (N - 1) * M * B, lambda N, M, B: 0, lambda N: 1),
    "reducescatter__recursive_halving__mpich":
        (lambda N, M, B: N * (N - 1) * M * B, lambda N, M, B: 0, lambda N: _log2(N)),
    "reducescatter__ring__nccl": (lambda N, M, B: N * (N - 1) * M * B, lambda N, M, B: 0, lambda N: N - 1),
    "alltoall__bruck__mpich":
        (lambda N, M, B: N * (N // 2) * _log2(N) * M * B, lambda N, M, B: 0, lambda N: _log2(N)),
    "alltoall__direct_pull__generic": (lambda N, M, B: N * (N - 1) * M * B, lambda N, M, B: 0, lambda N: 1),
    "alltoall__direct_push__generic": (lambda N, M, B: N * (N - 1) * M * B, lambda N, M, B: 0, lambda N: 1),
    "alltoall__pairwise_exchange__fa4": (lambda N, M, B: 2 * M * B, lambda N, M, B: 0, lambda N: 1),
    "alltoall__pairwise_xor__mpich": (lambda N, M, B: N * (N - 1) * M * B, lambda N, M, B: 0, lambda N: N - 1),
    "alltoall__shift__generic": (lambda N, M, B: N * (N - 1) * M * B, lambda N, M, B: 0, lambda N: N - 1),
}

# README.md table as written ("Total DSMEM bytes per CTA" x N, or the cluster-wide figure of the rooted trees;
# "Rounds"). hpcops' "2 phases, 1 round each" is taken as 2.
L2 = _log2
README = {
    "allreduce__recursive_doubling__clusterfusion": (lambda N, M, B: N * M * B * L2(N), lambda N: L2(N)),
    "allreduce__shift_accumulate_bulk__clusterfusion": (lambda N, M, B: N * (N - 2) * M * B, lambda N: N - 2),
    "allreduce__shift_accumulate_atomic__clusterfusion":
        (lambda N, M, B: N * (M * B if N == 2 else (N - 2) * M * B), lambda N: 1 if N == 2 else N - 2),
    "allreduce__direct_replicated_push__credit": (lambda N, M, B: N * (N - 1) * M * B, lambda N: 1),
    "allreduce__pull_reduce_scatter_push_allgather__hpcops":
        (lambda N, M, B: N * 2 * (N - 1) * M * B // N, lambda N: 2),
    "allreduce__ring__nccl": (lambda N, M, B: N * 2 * (N - 1) * M * B // N, lambda N: 2 * (N - 1)),
    "allreduce__rabenseifner__mpich": (lambda N, M, B: N * 2 * (N - 1) * M * B // N, lambda N: 2 * L2(N)),
    "allreduce__one_shot_pull__trtllm": (lambda N, M, B: N * (N - 1) * M * B, lambda N: 1),
    "allreduce__two_shot_push__trtllm": (lambda N, M, B: N * 2 * (N - 1) * M * B // N, lambda N: 2),
    "allreduce__reduce_bcast_binomial__mpich": (lambda N, M, B: 2 * (N - 1) * M * B, lambda N: 2 * L2(N)),
    "allreduce__atomic_all_push__generic": (lambda N, M, B: N * (N - 1) * M * B, lambda N: 1),
    "allreduce__2d_mesh__generic":
        (lambda N, M, B: N * (2 * (_grid(N)[1] - 1) + (_grid(N)[0] - 1)) * M * B // _grid(N)[1], lambda N: 3),
    "allgather__recursive_doubling__clusterfusion": (lambda N, M, B: N * (N - 1) * M * B, lambda N: L2(N)),
    "allgather__direct_shift__clusterfusion": (lambda N, M, B: N * (N - 1) * M * B, lambda N: N - 1),
    "allgather__direct_replicated_push__credit": (lambda N, M, B: N * (N - 1) * M * B, lambda N: 1),
    "allgather__ring__nccl": (lambda N, M, B: N * (N - 1) * M * B, lambda N: N - 1),
    "allgather__bruck__mpich": (lambda N, M, B: N * (N - 1) * M * B, lambda N: _clog2(N)),
    "allgather__direct_pull__generic": (lambda N, M, B: N * (N - 1) * M * B, lambda N: 1),
    "allgather__gather_bcast_binomial__mpich":
        (lambda N, M, B: (N - 1) * M * B + (N - 1) * N * M * B, lambda N: 2 * L2(N)),
    "reducescatter__ring__nccl": (lambda N, M, B: N * (N - 1) * M * B, lambda N: N - 1),
    "reducescatter__recursive_halving__mpich": (lambda N, M, B: N * (N - 1) * M * B, lambda N: L2(N)),
    "reducescatter__direct_push__generic": (lambda N, M, B: N * (N - 1) * M * B, lambda N: 1),
    "reducescatter__direct_pull__generic": (lambda N, M, B: N * (N - 1) * M * B, lambda N: 1),
    "alltoall__pairwise_exchange__fa4": (lambda N, M, B: N * M * B, lambda N: 1),
    "alltoall__direct_push__generic": (lambda N, M, B: N * (N - 1) * M * B, lambda N: 1),
    "alltoall__pairwise_xor__mpich": (lambda N, M, B: N * (N - 1) * M * B, lambda N: N - 1),
    "alltoall__shift__generic": (lambda N, M, B: N * (N - 1) * M * B, lambda N: N - 1),
    "alltoall__bruck__mpich": (lambda N, M, B: N * (N // 2) * L2(N) * M * B, lambda N: L2(N)),
    "alltoall__direct_pull__generic": (lambda N, M, B: N * (N - 1) * M * B, lambda N: 1),
}

# Nsight Compute, kernel-wide DSMEM payload (l1tex__m_l1tex2xbar_write_bytes_mem_dshared.sum), N = 4, M = 256, float
# (C1 ncu discovery, results/ncu_discovery__*__N4_M256_float.csv).
GROUND_TRUTH = [("alltoall__direct_push__generic", 4, 256, 12288),
                ("allgather__direct_replicated_push__credit", 4, 256, 12288),
                ("allgather__direct_pull__generic", 4, 256, 12288)]


def totals(rows):
    return tuple(sum(r[k] for r in rows) for k in ("bytes_sent", "bytes_recv", "bytes_initiated", "bytes_self"))


def check():
    fails = []
    # 1. exhaustive invariant + closed-form self-consistency over every legal N in 2..16 and a range of M
    n_cfg = n_illegal = 0
    for f in FILES:
        ftot, fself, frounds = FORMULA[f]
        for N in range(2, 17):
            for M in (1, 4, 16, 64, 96, 256, 1000, 1024, 4096, 16384, 54656):
                try:
                    rows = model(f, N, M)
                except Illegal:
                    n_illegal += 1
                    continue
                n_cfg += 1
                s, r, i, slf = totals(rows)
                if not (s == r == i):
                    fails.append(f"INVARIANT {f} N={N} M={M}: sent {s} recv {r} initiated {i}")
                if s != ftot(N, M, 4) or slf != fself(N, M, 4):
                    fails.append(f"CLOSED-FORM {f} N={N} M={M}: replay {s}/{slf} formula {ftot(N, M, 4)}/{fself(N, M, 4)}")
                if any(x["rounds"] != frounds(N) for x in rows):
                    fails.append(f"ROUNDS {f} N={N} M={M}: replay {[x['rounds'] for x in rows]} formula {frounds(N)}")
    print(f"invariant + closed-form check: {n_cfg} legal (file, N in 2..16, M) configurations, "
          f"{n_illegal} illegal skipped, {len(fails)} failures")

    # 2. legality vs the build matrix: every OK build must be legal in the model; SKIP:static_assert builds the
    #    model accepts are listed (stale rows of a since-removed assert, or the smem/static_assert precedence)
    sys.path.insert(0, HARNESS)
    import driver
    cwd = os.getcwd()
    os.chdir(driver.COLL)
    lb = driver.latest_build()
    os.chdir(cwd)
    n_ok = 0
    for (f, n, m, t), r in sorted(lb.items()):
        try:
            model(f, n, m, T_BYTES[t])
            legal = True
        except Illegal:
            legal = False
        if r["status"] == "OK":
            n_ok += 1
            if not legal:
                fails.append(f"LEGALITY {f} N={n} M={m} {t}: OK build, model says illegal")
        elif legal != (r["status"] == "SKIP:smem"):
            print(f"  note: build {r['status']} but model {'legal' if legal else 'illegal'}: {f} N={n} M={m} {t}"
                  f" (build run {r['run_id']})")
    print(f"legality: {n_ok} OK builds of build_matrix_v2 checked against the model's static_assert rules")

    # 3. table vs README at N in {2,4,8,16}, M = 1024, float
    print(f"\n{'file':54s} {'N':>2s} {'M':>5s} {'model B':>9s} {'README B':>9s} {'self B':>7s} "
          f"{'rnd':>3s} {'README':>6s}  status")
    readme_diff = set()
    for f in FILES:
        rtot, rrounds = README[f]
        for N in (2, 4, 8, 16):
            for M in (1024,):
                try:
                    rows = model(f, N, M)
                except Illegal:
                    continue
                s, _, _, slf = totals(rows)
                rnd = rows[0]["rounds"]
                notes = []
                if s != rtot(N, M, 4):
                    notes.append("README bytes differ")
                if rnd != rrounds(N):
                    notes.append("README rounds differ")
                if notes:
                    readme_diff.add(f)
                print(f"{f:54s} {N:2d} {M:5d} {s:9d} {rtot(N, M, 4):9d} {slf:7d} {rnd:3d} {rrounds(N):6d}  "
                      + ("; ".join(notes) or "ok"))
    print(f"\nREADME disagreements (reported, not failures): {sorted(readme_diff) or 'none'}")

    # 4. ncu ground truth
    print("\nground truth (ncu, kernel-wide DSMEM payload, float):")
    for f, N, M, want in GROUND_TRUTH:
        got = totals(model(f, N, M))[0]
        ok = got == want
        print(f"  {f} N={N} M={M}: model {got} B, measured {want} B  {'PASS' if ok else 'FAIL'}")
        if not ok:
            fails.append(f"GROUND TRUTH {f} N={N} M={M}: model {got} measured {want}")

    for x in fails:
        print("FAIL:", x)
    print(f"\n--check {'FAILED' if fails else 'passed'} ({len(fails)} failures)")
    return 1 if fails else 0


T_BYTES = {"float": 4, "int": 4, "double": 8, "__half": 2, "half": 2}


def write(which):
    sys.path.insert(0, HARNESS)
    import driver
    os.chdir(driver.COLL)
    lb = driver.latest_build()
    ns = argparse.Namespace(files=None, ops=None, N=None, M=None, T=None)
    cfgs = driver.runnable(lb, ns, which)
    header = driver.MANDATORY + ["file", "op", "alg", "N", "M", "elem_bytes", "rank"] + KEYS
    run_id = driver.iso_now()
    rows, seen = [], set()
    for f, n, m, t in cfgs:
        b = T_BYTES[t]
        if (f, n, m, b) in seen:  # float and int builds have the same traffic: one block per (file, N, M, B)
            continue
        seen.add((f, n, m, b))
        op, alg = driver.split(f)
        try:
            ranks = model(f, n, m, b)
        except Illegal as e:  # an OK build the model calls illegal is a model bug: write nothing
            raise SystemExit(f"model rejects built configuration {f} N={n} M={m} {t}: {e}")
        for r, d in enumerate(ranks):
            rows.append(["", "", "", "", "", "", run_id, -1, f, op, alg, n, m, b, r] + [d[k] for k in KEYS])
    if not rows:
        raise SystemExit(f"{which} set is empty (no OK builds in {driver.MATRIX}?): nothing written")
    path = os.path.join(driver.RES, "model.csv")
    driver.append_rows(path, header, rows)
    print(f"{which} set: {len(cfgs)} configurations -> {len(seen)} (file, N, M, elem_bytes) -> {len(rows)} rows "
          f"appended to {os.path.join(driver.COLL, path)} (run_id {run_id})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", choices=["perf", "default"], default="perf", help="driver.runnable set (default perf)")
    ap.add_argument("--check", action="store_true", help="totals vs README, invariants, ground truth; no CSV")
    a = ap.parse_args()
    if a.check:
        sys.exit(check())
    write(a.set)


if __name__ == "__main__":
    main()
