/*
 * trap_cs16.cu — settles how kernels with a compile-time __cluster_dims__ may be launched (C1 step 4).
 *
 * The collective files declare `__global__ void __cluster_dims__(CLUSTER_N,1,1) collective_kernel(...)`.
 * This probe uses the same declaration form for CS 8 (portable control) and CS 16 (non-portable) and checks,
 * for each CS, on GPU 0 of the container (GPU 3 under h100 run):
 *   T0  (CS 16 only) plain <<<CS,256>>> launch *before* NonPortableClusterSizeAllowed is set (expected to fail)
 *   T1  cudaFuncSetAttribute(NonPortableClusterSizeAllowed, 1) (CS 16 only), then plain <<<CS,256>>>
 *   T2  cudaLaunchKernelEx with no launch attributes
 *   T3  cudaLaunchKernelEx with cudaLaunchAttributeClusterDimension equal to the compile-time CS
 *       (+ scheduling-policy attribute, i.e. exactly what hb::ClusterLaunch with explicit_cluster sends)
 *   T4  cudaOccupancyMaxActiveClusters with the T2 config and with the T3 config
 * A launch "passes" if it returns cudaSuccess, synchronises without error, and every CTA saw
 * %cluster_nctarank == CS, %clusterid.x == 0 and a distinct %smid. Output: plain text on stdout only.
 */
#include <host.hpp>

#include <cooperative_groups.h>

template <int CS>
__global__ void __cluster_dims__(CS, 1, 1) probe(unsigned* rec) {
    if (threadIdx.x == 0) {
        unsigned nct, cid, rank, sm;
        asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(nct));
        asm volatile("mov.u32 %0, %%clusterid.x;" : "=r"(cid));
        asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
        asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
        rec[blockIdx.x * 4 + 0] = nct;
        rec[blockIdx.x * 4 + 1] = cid;
        rec[blockIdx.x * 4 + 2] = rank;
        rec[blockIdx.x * 4 + 3] = sm;
    }
    cooperative_groups::this_cluster().sync();
}

static std::string verdict(cudaError_t launch, unsigned* d, int cs) {
    if (launch != cudaSuccess) {
        cudaGetLastError();
        return std::string("FAIL launch: ") + cudaGetErrorString(launch);
    }
    cudaError_t s = cudaDeviceSynchronize();
    if (s != cudaSuccess) return std::string("FAIL sync: ") + cudaGetErrorString(s);
    std::vector<unsigned> h(cs * 4);
    CUDA_CHECK(cudaMemcpy(h.data(), d, h.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));
    std::vector<uint32_t> sm;
    std::string smids;
    for (int b = 0; b < cs; ++b) {
        if (h[b * 4 + 0] != (unsigned)cs || h[b * 4 + 1] != 0 || h[b * 4 + 2] != (unsigned)b)
            return "FAIL cluster regs of block " + std::to_string(b) + ": nctarank " + std::to_string(h[b * 4]) +
                   " clusterid " + std::to_string(h[b * 4 + 1]) + " rank " + std::to_string(h[b * 4 + 2]);
        sm.push_back(h[b * 4 + 3]);
        smids += (b ? "|" : "") + std::to_string(h[b * 4 + 3]);
    }
    if (!hb::all_distinct(sm)) return "FAIL smids not distinct: " + smids;
    return "PASS smids " + smids;
}

template <int CS>
static void run_cs() {
    auto k = probe<CS>;
    unsigned* d;
    CUDA_CHECK(cudaMalloc(&d, CS * 4 * sizeof(unsigned)));
    auto clear = [&] { CUDA_CHECK(cudaMemset(d, 0xff, CS * 4 * sizeof(unsigned))); };
    std::printf("== CS %d (compile-time __cluster_dims__(%d,1,1)), grid %d, block 256\n", CS, CS, CS);

    if (CS > 8) {
        clear();
        k<<<CS, 256>>>(d);
        std::printf("T0 <<<>>> before NonPortableClusterSizeAllowed : %s\n", verdict(cudaGetLastError(), d, CS).c_str());
        cudaError_t a = cudaFuncSetAttribute(k, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
        std::printf("   cudaFuncSetAttribute(NonPortableClusterSizeAllowed, 1): %s\n", cudaGetErrorString(a));
    }
    clear();
    k<<<CS, 256>>>(d);
    std::printf("T1 <<<>>>                                     : %s\n", verdict(cudaGetLastError(), d, CS).c_str());

    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(CS);
    cfg.blockDim = dim3(256);
    cfg.numAttrs = 0;
    clear();
    std::printf("T2 cudaLaunchKernelEx, no attributes          : %s\n", verdict(cudaLaunchKernelEx(&cfg, k, d), d, CS).c_str());

    cudaLaunchAttribute at[2];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = CS;
    at[0].val.clusterDim.y = 1;
    at[0].val.clusterDim.z = 1;
    at[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
    at[1].val.clusterSchedulingPolicyPreference = cudaClusterSchedulingPolicyDefault;
    cudaLaunchConfig_t cfg3 = cfg;
    cfg3.attrs = at;
    cfg3.numAttrs = 2;
    clear();
    std::printf("T3 cudaLaunchKernelEx, cluster attr = %2d      : %s\n", CS, verdict(cudaLaunchKernelEx(&cfg3, k, d), d, CS).c_str());

    int n2 = -1, n3 = -1;
    cudaError_t e2 = cudaOccupancyMaxActiveClusters(&n2, k, &cfg);
    if (e2 != cudaSuccess) cudaGetLastError();
    cudaError_t e3 = cudaOccupancyMaxActiveClusters(&n3, k, &cfg3);
    if (e3 != cudaSuccess) cudaGetLastError();
    std::printf("T4 cudaOccupancyMaxActiveClusters             : no attr -> %d (%s); cluster attr -> %d (%s)\n", n2,
                cudaGetErrorString(e2), n3, cudaGetErrorString(e3));
    CUDA_CHECK(cudaFree(d));
}

int main() {
    hb::DeviceInfo dev = hb::DeviceInfo::open(true);
    std::printf("device %s uuid %s driver %s cuda %s\n", dev.name.c_str(), dev.uuid.c_str(), dev.driver_version.c_str(),
                dev.cuda_version.c_str());
    run_cs<8>();
    run_cs<16>();
    return 0;
}
