/*
 * AllGather - recursive doubling (ClusterFusion paper, Algorithm 2 "ClusterGather")
 *
 * 1. Op: every CTA ends with the concatenation of the N M-element vectors, in rank order.
 *
 * 2. Algorithm: buffer D_b of N*M elements; own data sits in segment 0. log2(N) rounds with
 *    stride s = 1, 2, 4, ...: CTA b sends D_b[0 : s*M] to CTA (b + s) mod N, which stores it at
 *    D[s*M : 2*s*M]; b receives the same range from (b - s) mod N. The transferred block
 *    doubles every round. After the loop D_b holds ranks in *rotated* order
 *    [b, b-1, b-2, ...], so the final store un-rotates: out[r] = D_b[(b - r) mod N].
 *
 * 3. Pattern: ring with doubling shift. Rounds: log2(N). Message in round k: 2^k * M*sizeof(T)
 *    bytes. Total per CTA: (N-1)*M*sizeof(T) (= Traffic_Gather/N in the paper).
 *    Primitive: cp.async.bulk push to the peer's receive range + complete_tx on the peer's
 *    mbarrier; receiver waits on parity; barrier.cluster per round (origin "wait for arrival").
 *
 * Source: ClusterFusion (NeurIPS'25) Alg. 2.
 */
#include "dsmem_primitives.cuh"
using namespace dsmem;

template <int N, typename T, int M>
__device__ void allgather_recursive_doubling(T* D, uint32_t bar, Stamps st) {
    static_assert((N & (N - 1)) == 0, "recursive doubling needs a power-of-two cluster size");
    static_assert((M * sizeof(T)) % 16 == 0, "bulk copy needs a multiple of 16 bytes");
    const uint32_t rank = cluster_rank();
    const uint32_t D_a = smem_u32(D);
    uint32_t phase = 0;
    int round = 0;
    for (int s = 1; s < N; s <<= 1, ++round) {
        const uint32_t bytes = s * M * sizeof(T);              // block size doubles with the stride
        if (threadIdx.x == 0) {
            mbar_arrive_expect_tx(bar, bytes);
            // send D[0 : s*M] -> peer's D[s*M : 2*s*M]; peer = (b + stride) mod N
            bulk_push((rank + s) % N, D_a + bytes, D_a, bytes, bar);
        }
        mbar_wait_parity(bar, phase);                          // data from (b - stride) landed in my D[s*M : 2*s*M]
        phase ^= 1;
        cluster_sync();                                        // all CTAs finished the round (keeps send ranges stable)
        st.round(round);
    }
}

__global__ void __cluster_dims__(CLUSTER_N, 1, 1)
collective_kernel(const ELEM_T* in, ELEM_T* out, unsigned long long* ts, unsigned* smid_out) {
    constexpr int N = CLUSTER_N, M = MSG_ELEMS;
    __shared__ __align__(16) ELEM_T D[N * M];
    __shared__ __align__(8) uint64_t bar;
    const uint32_t rank = cluster_rank();
    Stamps st{ts + rank * kStamps};

    load_block<ELEM_T, M>(D, in + rank * M);                   // own data in segment 0
    fence_proxy_async_smem();
    if (threadIdx.x == 0) { mbar_init(smem_u32(&bar), 1); fence_mbar_init(); smid_out[rank] = smid(); }
    cluster_sync();
    st.mark(0);
    allgather_recursive_doubling<N, ELEM_T, M>(D, smem_u32(&bar), st);
    st.mark(1);
    // un-rotate: segment k of D holds rank (b - k) mod N
    ELEM_T* o = out + rank * N * M;
    for (int i = threadIdx.x; i < N * M; i += blockDim.x) {
        const int k = i / M, e = i % M;
        o[((rank + N - k) % N) * M + e] = D[k * M + e];
    }
    cluster_sync();
}
