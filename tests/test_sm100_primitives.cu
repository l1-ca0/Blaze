/**
 * test_tcgen05.cu — SM100 tcgen05 primitive isolation tests.
 *
 * Tests TMEM, tcgen05.mma, and TMA in increasing complexity:
 *   Test 1:  TMEM alloc/dealloc (no MMA, no TMA)
 *   Test 1b: tcgen05.ld from all 4 warps (no MMA)
 *   Test 2:  Manual SMEM fill + tcgen05.mma (no TMA)
 *   Test 3:  TMA load + tcgen05.mma (B64/B128 swizzle)
 *   Test 3b: TMA load + tcgen05.mma (no swizzle)
 *
 * All tests use FP8 E4M3, 128x128x64 tile, single CTA.
 */

#include "gemm/tmem_utils.cuh"
#include "gemm/tma_utils.cuh"
#include "gemm/fp8_gemm_sm100.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

namespace blaze {

static constexpr int TILE_M = 128;
static constexpr int TILE_N = 128;
static constexpr int TILE_K = 64;

struct __align__(128) TestSmem {
    __nv_fp8_e4m3 A[TILE_M * TILE_K];
    __nv_fp8_e4m3 B[TILE_K * TILE_N];
    float C[TILE_M * TILE_N];
    alignas(8) uint64_t mbar[1];
    alignas(8) uint64_t mbar_mma[1];
    uint32_t tmem_addr;
};

// ===== Test 1: TMEM alloc/dealloc only =====
__cluster_dims__(1, 1, 1)
__global__ void __launch_bounds__(128)
test_tmem_only_kernel(int* result) {
    extern __shared__ char smem_raw[];
    TestSmem* smem = reinterpret_cast<TestSmem*>(smem_raw);

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;

    if (warp_id == 0) {
        tmem_alloc(&smem->tmem_addr, TILE_N);
    }
    __syncthreads();
    tmem_addr_t tmem_addr = smem->tmem_addr;

    if (tid == 0) {
        result[0] = (int)tmem_addr;
    }

    __syncthreads();
    if (warp_id == 0) {
        tmem_dealloc(tmem_addr, TILE_N);
    }
}

// ===== Test 1b: TMEM alloc + tcgen05.ld from all 4 warps (no MMA) =====
__cluster_dims__(1, 1, 1)
__global__ void __launch_bounds__(128)
test_tmem_ld_all_warps_kernel(int* result) {
    extern __shared__ char smem_raw[];
    TestSmem* smem = reinterpret_cast<TestSmem*>(smem_raw);

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;

    if (warp_id == 0) {
        tmem_alloc(&smem->tmem_addr, TILE_N);
    }
    __syncthreads();
    tmem_addr_t tmem_addr = smem->tmem_addr;
    __syncthreads();

    // Each warp reads its own 32-row slice (row selection is implicit in hardware)
    float vals[8];
    tmem_load_8xf32(vals, tmem_addr, 0);
    tmem_wait_ld();

    __syncthreads();
    result[0] = 1;

    __syncthreads();
    if (warp_id == 0) {
        tmem_dealloc(tmem_addr, TILE_N);
    }
}

// ===== Test 2: Manual SMEM fill + MMA (no TMA) =====
__cluster_dims__(1, 1, 1)
__global__ void __launch_bounds__(128)
test_mma_no_tma_kernel(float* C_out) {
    extern __shared__ char smem_raw[];
    TestSmem* smem = reinterpret_cast<TestSmem*>(smem_raw);

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;

    // Fill A and B with FP8 1.0 (0x38 = E4M3 1.0)
    for (int i = tid; i < TILE_M * TILE_K; i += blockDim.x) {
        smem->A[i].__x = 0x38;
    }
    for (int i = tid; i < TILE_K * TILE_N; i += blockDim.x) {
        smem->B[i].__x = 0x38;
    }
    __syncthreads();

    if (warp_id == 0) {
        tmem_alloc(&smem->tmem_addr, TILE_N);
    }
    __syncthreads();
    tmem_addr_t tmem_addr = smem->tmem_addr;
    __syncthreads();

    // MMA K-loop: FP8 E4M3 K_per_mma=32, TILE_K=64 → 2 iterations
    {
        constexpr int K_PER_MMA = 32;
        constexpr int NUM_K_ITERS = TILE_K / K_PER_MMA;

        uint32_t smem_a_base = static_cast<uint32_t>(
            __cvta_generic_to_shared(smem->A));
        uint32_t smem_b_base = static_cast<uint32_t>(
            __cvta_generic_to_shared(smem->B));

        uint32_t idesc = make_idesc_f8f6f4(0, 0, TILE_M, TILE_N);

        for (int ki = 0; ki < NUM_K_ITERS; ki++) {
            asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

            uint32_t smem_a_addr = smem_a_base + ki * K_PER_MMA;
            uint32_t smem_b_addr = smem_b_base + ki * K_PER_MMA * TILE_N;

            uint64_t desc_a = make_smem_desc(smem_a_addr, TILE_K * 1, 4);
            uint64_t desc_b = make_smem_desc(smem_b_addr, TILE_N * 1, 2);

            if (elect_one_sync()) {
                if (ki == 0) {
                    asm volatile(
                        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, 0;\n"
                        : : "r"(tmem_addr), "l"(desc_a), "l"(desc_b), "r"(idesc) : "memory");
                } else {
                    asm volatile(
                        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, 1;\n"
                        : : "r"(tmem_addr), "l"(desc_a), "l"(desc_b), "r"(idesc) : "memory");
                }
            }

            asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
        }
    }

    // Cooperative TMEM → SMEM (all warps, each reads its own 32-row slice)
    __syncthreads();
    tmem_store_to_smem_warp(smem->C, tmem_addr, TILE_N, warp_id);
    __syncthreads();

    for (int i = tid; i < TILE_M * TILE_N; i += blockDim.x) {
        C_out[i] = smem->C[i];
    }

    __syncthreads();
    if (warp_id == 0) {
        tmem_dealloc(tmem_addr, TILE_N);
    }
}

// ===== Test 3: TMA load + MMA (with swizzle) =====
__cluster_dims__(1, 1, 1)
__global__ void __launch_bounds__(128)
test_mma_with_tma_kernel(
    const TmaDescriptor* __restrict__ desc_A,
    const TmaDescriptor* __restrict__ desc_B,
    float* __restrict__ C_out
) {
    extern __shared__ char smem_raw[];
    TestSmem* smem = reinterpret_cast<TestSmem*>(smem_raw);

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;

    if (tid == 0) {
        mbarrier_init(&smem->mbar[0], 1);
    }
    __syncthreads();

    if (warp_id == 0) {
        tmem_alloc(&smem->tmem_addr, TILE_N);
    }
    __syncthreads();
    tmem_addr_t tmem_addr = smem->tmem_addr;

    // TMA load
    if (tid == 0) {
        uint32_t tx_bytes = TILE_M * TILE_K + TILE_K * TILE_N;
        mbarrier_expect_tx(&smem->mbar[0], tx_bytes);
        tma_load_2d(smem->A, desc_A, 0, 0, &smem->mbar[0]);
        tma_load_2d(smem->B, desc_B, 0, 0, &smem->mbar[0]);
    }

    mbarrier_wait(&smem->mbar[0], 0);
    __syncthreads();

    // MMA K-loop: FP8 E4M3 K_per_mma=32, TILE_K=64 → 2 iterations
    {
        constexpr int K_PER_MMA = 32;
        constexpr int NUM_K_ITERS = TILE_K / K_PER_MMA;

        uint32_t smem_a_base = static_cast<uint32_t>(
            __cvta_generic_to_shared(smem->A));
        uint32_t smem_b_base = static_cast<uint32_t>(
            __cvta_generic_to_shared(smem->B));
        uint32_t idesc = make_idesc_f8f6f4(0, 0, TILE_M, TILE_N);

        for (int ki = 0; ki < NUM_K_ITERS; ki++) {
            asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

            uint32_t smem_a_addr = smem_a_base + ki * K_PER_MMA;
            uint32_t smem_b_addr = smem_b_base + ki * K_PER_MMA * TILE_N;

            uint64_t desc_a = make_smem_desc(smem_a_addr, TILE_K * 1, 4);
            uint64_t desc_b = make_smem_desc(smem_b_addr, TILE_N * 1, 2);

            if (elect_one_sync()) {
                if (ki == 0) {
                    asm volatile(
                        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, 0;\n"
                        : : "r"(tmem_addr), "l"(desc_a), "l"(desc_b), "r"(idesc) : "memory");
                } else {
                    asm volatile(
                        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, 1;\n"
                        : : "r"(tmem_addr), "l"(desc_a), "l"(desc_b), "r"(idesc) : "memory");
                }
            }

            asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
        }
    }

    __syncthreads();
    tmem_store_to_smem_warp(smem->C, tmem_addr, TILE_N, warp_id);
    __syncthreads();

    for (int i = tid; i < TILE_M * TILE_N; i += blockDim.x) {
        C_out[i] = smem->C[i];
    }

    __syncthreads();
    if (warp_id == 0) {
        tmem_dealloc(tmem_addr, TILE_N);
    }
}

// ===== Test 3b: TMA load (no swizzle) + MMA =====
__cluster_dims__(1, 1, 1)
__global__ void __launch_bounds__(128)
test_mma_tma_no_swizzle_kernel(
    const TmaDescriptor* __restrict__ desc_A,
    const TmaDescriptor* __restrict__ desc_B,
    float* __restrict__ C_out
) {
    extern __shared__ char smem_raw[];
    TestSmem* smem = reinterpret_cast<TestSmem*>(smem_raw);

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;

    if (tid == 0) {
        mbarrier_init(&smem->mbar[0], 1);
    }
    __syncthreads();

    if (warp_id == 0) {
        tmem_alloc(&smem->tmem_addr, TILE_N);
    }
    __syncthreads();
    tmem_addr_t tmem_addr = smem->tmem_addr;

    // TMA load
    if (tid == 0) {
        uint32_t tx_bytes = TILE_M * TILE_K + TILE_K * TILE_N;
        mbarrier_expect_tx(&smem->mbar[0], tx_bytes);
        tma_load_2d(smem->A, desc_A, 0, 0, &smem->mbar[0]);
        tma_load_2d(smem->B, desc_B, 0, 0, &smem->mbar[0]);
    }

    mbarrier_wait(&smem->mbar[0], 0);
    __syncthreads();

    // MMA K-loop with no-swizzle descriptors (layout_type=0)
    {
        constexpr int K_PER_MMA = 32;
        constexpr int NUM_K_ITERS = TILE_K / K_PER_MMA;

        uint32_t smem_a_base = static_cast<uint32_t>(
            __cvta_generic_to_shared(smem->A));
        uint32_t smem_b_base = static_cast<uint32_t>(
            __cvta_generic_to_shared(smem->B));
        uint32_t idesc = make_idesc_f8f6f4(0, 0, TILE_M, TILE_N);

        for (int ki = 0; ki < NUM_K_ITERS; ki++) {
            asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

            uint32_t smem_a_addr = smem_a_base + ki * K_PER_MMA;
            uint32_t smem_b_addr = smem_b_base + ki * K_PER_MMA * TILE_N;

            uint64_t desc_a = make_smem_desc(smem_a_addr, TILE_K * 1, 0);
            uint64_t desc_b = make_smem_desc(smem_b_addr, TILE_N * 1, 0);

            if (elect_one_sync()) {
                if (ki == 0) {
                    asm volatile(
                        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, 0;\n"
                        : : "r"(tmem_addr), "l"(desc_a), "l"(desc_b), "r"(idesc) : "memory");
                } else {
                    asm volatile(
                        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, 1;\n"
                        : : "r"(tmem_addr), "l"(desc_a), "l"(desc_b), "r"(idesc) : "memory");
                }
            }

            asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
        }
    }

    __syncthreads();
    tmem_store_to_smem_warp(smem->C, tmem_addr, TILE_N, warp_id);
    __syncthreads();

    for (int i = tid; i < TILE_M * TILE_N; i += blockDim.x) {
        C_out[i] = smem->C[i];
    }

    __syncthreads();
    if (warp_id == 0) {
        tmem_dealloc(tmem_addr, TILE_N);
    }
}

}  // namespace blaze

using namespace blaze;

bool run_test_tmem_only() {
    printf("\n--- Test 1: TMEM alloc/dealloc only ---\n");
    int* d_result;
    cudaMalloc(&d_result, sizeof(int));

    size_t smem_size = sizeof(TestSmem);
    cudaFuncSetAttribute(test_tmem_only_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(1);
    cfg.blockDim = dim3(128);
    cfg.dynamicSmemBytes = smem_size;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim = {1, 1, 1};
    cfg.attrs = attrs;
    cfg.numAttrs = 1;

    cudaLaunchKernelEx(&cfg, test_tmem_only_kernel, d_result);
    cudaError_t err = cudaDeviceSynchronize();
    cudaFree(d_result);

    if (err != cudaSuccess) {
        printf("FAILED: %s\n", cudaGetErrorString(err));
        return false;
    }
    printf("PASSED\n");
    return true;
}

bool run_test_tmem_ld_all_warps() {
    printf("\n--- Test 1b: tcgen05.ld from all 4 warps (no MMA) ---\n");
    int* d_result;
    cudaMalloc(&d_result, sizeof(int));
    cudaMemset(d_result, 0, sizeof(int));

    size_t smem_size = sizeof(TestSmem);
    cudaFuncSetAttribute(test_tmem_ld_all_warps_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(1);
    cfg.blockDim = dim3(128);
    cfg.dynamicSmemBytes = smem_size;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim = {1, 1, 1};
    cfg.attrs = attrs;
    cfg.numAttrs = 1;

    cudaLaunchKernelEx(&cfg, test_tmem_ld_all_warps_kernel, d_result);
    cudaError_t err = cudaDeviceSynchronize();
    cudaFree(d_result);

    if (err != cudaSuccess) {
        printf("FAILED: %s\n", cudaGetErrorString(err));
        return false;
    }
    printf("PASSED\n");
    return true;
}

bool run_test_mma_no_tma() {
    printf("\n--- Test 2: Manual SMEM fill + MMA (no TMA) ---\n");
    float* d_C;
    cudaMalloc(&d_C, TILE_M * TILE_N * sizeof(float));
    cudaMemset(d_C, 0, TILE_M * TILE_N * sizeof(float));

    size_t smem_size = sizeof(TestSmem);
    cudaFuncSetAttribute(test_mma_no_tma_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(1);
    cfg.blockDim = dim3(128);
    cfg.dynamicSmemBytes = smem_size;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim = {1, 1, 1};
    cfg.attrs = attrs;
    cfg.numAttrs = 1;

    cudaLaunchKernelEx(&cfg, test_mma_no_tma_kernel, d_C);
    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        printf("FAILED: %s\n", cudaGetErrorString(err));
        cudaFree(d_C);
        return false;
    }

    // Verify: A=all 1.0, B=all 1.0, C[i][j] should = K = 64
    float* h_C = (float*)malloc(TILE_M * TILE_N * sizeof(float));
    cudaMemcpy(h_C, d_C, TILE_M * TILE_N * sizeof(float), cudaMemcpyDeviceToHost);

    float expected = (float)TILE_K;
    int errors = 0;
    for (int i = 0; i < TILE_M * TILE_N && errors < 10; i++) {
        if (fabsf(h_C[i] - expected) > 1.0f) {
            printf("  C[%d] = %f, expected %f\n", i, h_C[i], expected);
            errors++;
        }
    }

    if (errors == 0) printf("PASSED: All elements = %.1f\n", expected);
    else printf("FAILED: %d+ mismatches\n", errors);

    free(h_C);
    cudaFree(d_C);
    return errors == 0;
}

bool run_test_mma_with_tma() {
    printf("\n--- Test 3: TMA load + MMA (swizzle) ---\n");

    __nv_fp8_e4m3 *d_A, *d_B;
    float* d_C;
    cudaMalloc(&d_A, TILE_M * TILE_K);
    cudaMalloc(&d_B, TILE_K * TILE_N);
    cudaMalloc(&d_C, TILE_M * TILE_N * sizeof(float));

    uint8_t* h_buf = (uint8_t*)malloc(TILE_M * TILE_K);
    memset(h_buf, 0x38, TILE_M * TILE_K);
    cudaMemcpy(d_A, h_buf, TILE_M * TILE_K, cudaMemcpyHostToDevice);
    memset(h_buf, 0x38, TILE_K * TILE_N);
    cudaMemcpy(d_B, h_buf, TILE_K * TILE_N, cudaMemcpyHostToDevice);
    free(h_buf);

    TmaDescriptor h_desc_A, h_desc_B;
    create_tma_desc_2d(&h_desc_A, d_A, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                       TILE_M, TILE_K, TILE_M, TILE_K, TmaSwizzle::B64);
    create_tma_desc_2d(&h_desc_B, d_B, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                       TILE_K, TILE_N, TILE_K, TILE_N, TmaSwizzle::B128);

    TmaDescriptor *dd_A, *dd_B;
    cudaMalloc(&dd_A, sizeof(TmaDescriptor));
    cudaMalloc(&dd_B, sizeof(TmaDescriptor));
    cudaMemcpy(dd_A, &h_desc_A, sizeof(TmaDescriptor), cudaMemcpyHostToDevice);
    cudaMemcpy(dd_B, &h_desc_B, sizeof(TmaDescriptor), cudaMemcpyHostToDevice);

    size_t smem_size = sizeof(TestSmem);
    cudaFuncSetAttribute(test_mma_with_tma_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(1);
    cfg.blockDim = dim3(128);
    cfg.dynamicSmemBytes = smem_size;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim = {1, 1, 1};
    cfg.attrs = attrs;
    cfg.numAttrs = 1;

    cudaLaunchKernelEx(&cfg, test_mma_with_tma_kernel, dd_A, dd_B, d_C);
    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        printf("FAILED: %s\n", cudaGetErrorString(err));
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        cudaFree(dd_A); cudaFree(dd_B);
        return false;
    }

    float* h_C = (float*)malloc(TILE_M * TILE_N * sizeof(float));
    cudaMemcpy(h_C, d_C, TILE_M * TILE_N * sizeof(float), cudaMemcpyDeviceToHost);

    float expected = (float)TILE_K;
    int errors = 0;
    for (int i = 0; i < TILE_M * TILE_N && errors < 10; i++) {
        if (fabsf(h_C[i] - expected) > 1.0f) {
            printf("  C[%d] = %f, expected %f\n", i, h_C[i], expected);
            errors++;
        }
    }

    if (errors == 0) printf("PASSED: All elements = %.1f\n", expected);
    else printf("FAILED: %d+ mismatches\n", errors);

    free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(dd_A); cudaFree(dd_B);
    return errors == 0;
}

bool run_test_mma_tma_no_swizzle() {
    printf("\n--- Test 3b: TMA load (no swizzle) + MMA ---\n");

    __nv_fp8_e4m3 *d_A, *d_B;
    float* d_C;
    cudaMalloc(&d_A, TILE_M * TILE_K);
    cudaMalloc(&d_B, TILE_K * TILE_N);
    cudaMalloc(&d_C, TILE_M * TILE_N * sizeof(float));

    uint8_t* h_buf = (uint8_t*)malloc(TILE_M * TILE_K);
    memset(h_buf, 0x38, TILE_M * TILE_K);
    cudaMemcpy(d_A, h_buf, TILE_M * TILE_K, cudaMemcpyHostToDevice);
    memset(h_buf, 0x38, TILE_K * TILE_N);
    cudaMemcpy(d_B, h_buf, TILE_K * TILE_N, cudaMemcpyHostToDevice);
    free(h_buf);

    TmaDescriptor h_desc_A, h_desc_B;
    create_tma_desc_2d(&h_desc_A, d_A, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                       TILE_M, TILE_K, TILE_M, TILE_K, TmaSwizzle::NONE);
    create_tma_desc_2d(&h_desc_B, d_B, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                       TILE_K, TILE_N, TILE_K, TILE_N, TmaSwizzle::NONE);

    TmaDescriptor *dd_A, *dd_B;
    cudaMalloc(&dd_A, sizeof(TmaDescriptor));
    cudaMalloc(&dd_B, sizeof(TmaDescriptor));
    cudaMemcpy(dd_A, &h_desc_A, sizeof(TmaDescriptor), cudaMemcpyHostToDevice);
    cudaMemcpy(dd_B, &h_desc_B, sizeof(TmaDescriptor), cudaMemcpyHostToDevice);

    size_t smem_size = sizeof(TestSmem);
    cudaFuncSetAttribute(test_mma_tma_no_swizzle_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(1);
    cfg.blockDim = dim3(128);
    cfg.dynamicSmemBytes = smem_size;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim = {1, 1, 1};
    cfg.attrs = attrs;
    cfg.numAttrs = 1;

    cudaLaunchKernelEx(&cfg, test_mma_tma_no_swizzle_kernel, dd_A, dd_B, d_C);
    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        printf("FAILED: %s\n", cudaGetErrorString(err));
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        cudaFree(dd_A); cudaFree(dd_B);
        return false;
    }

    float* h_C = (float*)malloc(TILE_M * TILE_N * sizeof(float));
    cudaMemcpy(h_C, d_C, TILE_M * TILE_N * sizeof(float), cudaMemcpyDeviceToHost);

    float expected = (float)TILE_K;
    int errors = 0;
    for (int i = 0; i < TILE_M * TILE_N && errors < 10; i++) {
        if (fabsf(h_C[i] - expected) > 1.0f) {
            printf("  C[%d] = %f, expected %f\n", i, h_C[i], expected);
            errors++;
        }
    }

    if (errors == 0) printf("PASSED: All elements = %.1f\n", expected);
    else printf("FAILED: %d+ mismatches\n", errors);

    free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(dd_A); cudaFree(dd_B);
    return errors == 0;
}

int main() {
    printf("=== SM100 tcgen05 Primitive Tests ===\n");
    printf("TestSmem size: %zu bytes\n", sizeof(TestSmem));

    bool ok = true;

    ok &= run_test_tmem_only();
    ok &= run_test_tmem_ld_all_warps();
    if (ok) ok &= run_test_mma_no_tma();
    if (ok) ok &= run_test_mma_with_tma();
    if (ok || true) {  // Always run to compare with Test 3
        ok &= run_test_mma_tma_no_swizzle();
    }

    printf("\n=== %s ===\n", ok ? "ALL PASSED" : "SOME FAILED");
    return ok ? 0 : 1;
}
