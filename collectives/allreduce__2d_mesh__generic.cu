/*
 * AllReduce - 2D mesh / hierarchical (row reduce-scatter, column all-reduce, row all-gather)
 *
 * 1. Op: every CTA ends with the element-wise sum of N M-element vectors.
 *
 * 2. Algorithm: view the N CTAs as an R x C grid (16 -> 4x4, 8 -> 2x4, 4 -> 2x2), rank = row*C + col.
 *    Phase 1 (row reduce-scatter, direct push): CTA (r,c) pushes chunk j (size M/C) to row peer (r,j)
 *    and reduces the C partials of chunk c -> row-partial of chunk c.
 *    Phase 2 (column all-reduce, direct replicated push): CTA (r,c) pushes its row-partial chunk to
 *    the R-1 column peers (*,c), every column CTA reduces the R copies -> fully reduced chunk c.
 *    Phase 3 (row all-gather, direct push): CTA (r,c) pushes chunk c into position c of every row peer.
 *    Every message stays inside one row or one column of the logical grid, which is what a 2D
 *    mesh/torus SM-to-SM network (one of the NoC topologies under study) can route without crossings.
 *
 * 3. Pattern: 2D mesh, 3 rounds. Messages per CTA: (C-1)+(R-1)+(C-1), each M/C*sizeof(T) bytes.
 *    Total per CTA: (2(C-1)+(R-1))/C * M*sizeof(T) (for 4x4: 9/4 M, vs 2(N-1)/N = 15/8 M optimal, but
 *    N-1 -> 3 rounds and all traffic is "local" in the grid).
 *    Primitive: cp.async.bulk pushes into per-phase slot arrays, one aggregate-armed mbarrier per phase
 *    (expect_tx = total incoming bytes), parity waits, fence.proxy.async before re-pushing reduced data;
 *    exit barrier.cluster for DSMEM lifetime.
 *
 * Source: generic 2D (torus) algorithm as used on TPU pods / Trainium 2D meshes and in hierarchical NCCL.
 */
#include "dsmem_primitives.cuh"
using namespace dsmem;

template <int N> struct Grid2D {
    static constexpr int R = (N >= 16) ? 4 : (N >= 4 ? 2 : 1);
    static constexpr int C = N / R;
};

template <int N, typename T, int M, class Op>
// On return: `chunk` holds the fully reduced chunk `col`; buf holds the other chunks (chunk col in buf is stale).
__device__ void allreduce_2d_mesh(T* buf, T (*row_slots)[M / Grid2D<N>::C], T (*col_slots)[M / Grid2D<N>::C], T* chunk,
                                  uint32_t bar_row, uint32_t bar_col, uint32_t bar_ag, Op op, Stamps st) {
    constexpr int R = Grid2D<N>::R, C = Grid2D<N>::C;
    static_assert(R * C == N, "grid must cover the cluster");
    static_assert(M % C == 0 && ((M / C) * sizeof(T)) % 16 == 0, "chunk must be a multiple of 16 bytes");
    constexpr int K = M / C;                                   // chunk elements
    constexpr uint32_t bytes = K * sizeof(T);
    const uint32_t rank = cluster_rank();
    const uint32_t row = rank / C, col = rank % C;
    const uint32_t buf_a = smem_u32(buf), rs_a = smem_u32(&row_slots[0][0]), cs_a = smem_u32(&col_slots[0][0]);
    const uint32_t chunk_a = smem_u32(chunk);

    // phase 1: row reduce-scatter (push chunk j to (row, j))
    if (threadIdx.x == 0) {
        mbar_arrive_expect_tx(bar_row, (C - 1) * bytes);
        for (uint32_t j = 1; j < C; ++j) {
            const uint32_t dc = (col + j) % C, dst = row * C + dc;
            bulk_push(dst, rs_a + col * bytes, buf_a + dc * bytes, bytes, bar_row);   // chunk dc -> row_slots[col] of (row, dc)
        }
    }
    mbar_wait_parity(bar_row, 0);
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        T acc = buf[col * K + i];
        #pragma unroll
        for (int c = 0; c < C; ++c) if (c != (int)col) acc = op(acc, row_slots[c][i]);
        buf[col * K + i] = acc;                                // row-partial of chunk `col`
    }
    fence_proxy_async_smem(); __syncthreads();
    st.round(0);

    // phase 2: column all-reduce of chunk `col` (replicated push among (*, col))
    if constexpr (R > 1) {
        if (threadIdx.x == 0) {
            mbar_arrive_expect_tx(bar_col, (R - 1) * bytes);
            for (uint32_t j = 1; j < R; ++j) {
                const uint32_t dr = (row + j) % R, dst = dr * C + col;
                bulk_push(dst, cs_a + row * bytes, buf_a + col * bytes, bytes, bar_col);   // my row-partial -> col_slots[row] of (dr, col)
            }
        }
        mbar_wait_parity(bar_col, 0);
        for (int i = threadIdx.x; i < K; i += blockDim.x) {    // my column push may still read buf[col]: write elsewhere
            T acc = buf[col * K + i];
            #pragma unroll
            for (int r = 0; r < R; ++r) if (r != (int)row) acc = op(acc, col_slots[r][i]);
            chunk[i] = acc;                                    // fully reduced chunk `col`
        }
    } else {
        for (int i = threadIdx.x; i < K; i += blockDim.x) chunk[i] = buf[col * K + i];
    }
    fence_proxy_async_smem(); __syncthreads();
    st.round(1);

    // phase 3: row all-gather (push chunk `col` into position col of every row peer)
    if (threadIdx.x == 0) {
        mbar_arrive_expect_tx(bar_ag, (C - 1) * bytes);
        for (uint32_t j = 1; j < C; ++j) {
            const uint32_t dst = row * C + (col + j) % C;
            bulk_push(dst, buf_a + col * bytes, chunk_a, bytes, bar_ag);   // reduced chunk -> position col in row peer
        }
    }
    mbar_wait_parity(bar_ag, 0);
    st.round(2);
}

__global__ void __cluster_dims__(CLUSTER_N, 1, 1)
collective_kernel(const ELEM_T* in, ELEM_T* out, unsigned long long* ts, unsigned* smid_out) {
    constexpr int N = CLUSTER_N, M = MSG_ELEMS, C = Grid2D<N>::C, R = Grid2D<N>::R;
    __shared__ __align__(16) ELEM_T buf[M];
    __shared__ __align__(16) ELEM_T row_slots[C][M / C];
    __shared__ __align__(16) ELEM_T col_slots[R][M / C];
    __shared__ __align__(16) ELEM_T chunk[M / C];
    __shared__ __align__(8) uint64_t bars[3];
    const uint32_t rank = cluster_rank();
    Stamps st{ts + rank * kStamps};

    load_block<ELEM_T, M>(buf, in + rank * M);
    fence_proxy_async_smem();
    if (threadIdx.x == 0) { for (int i = 0; i < 3; ++i) mbar_init(smem_u32(&bars[i]), 1); fence_mbar_init(); smid_out[rank] = smid(); }
    cluster_sync();
    st.mark(0);
    allreduce_2d_mesh<N, ELEM_T, M>(buf, row_slots, col_slots, chunk, smem_u32(&bars[0]), smem_u32(&bars[1]), smem_u32(&bars[2]), Sum{}, st);
    st.mark(1);
    __syncthreads();
    for (int i = threadIdx.x; i < M; i += blockDim.x) {       // chunk col comes from `chunk`, the rest from buf
        const int c = i / (M / C);
        out[rank * M + i] = (c == (int)(rank % C)) ? chunk[i % (M / C)] : buf[i];
    }
    cluster_sync();
}
