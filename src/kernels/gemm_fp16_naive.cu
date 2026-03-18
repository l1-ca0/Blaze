/**
 * gemm_fp16_naive.cu — Phase 0 naive FP16 GEMM with TMEM lifecycle validation.
 *
 * Purpose: Validate two things simultaneously:
 *   1. CUDA-core GEMM correctness vs cuBLAS (proves our matrix indexing is right)
 *   2. tcgen05 TMEM alloc/dealloc lifecycle under real workload (1024 CTAs)
 *
 * Computes C = A × B where:
 *   A: (M, K) FP16, row-major
 *   B: (K, N) FP16, row-major
 *   C: (M, N) FP32 accumulate → store as FP16, row-major
 *
 * Fixed size: M=N=K=4096 for bringup.
 *
 * Architecture:
 *   - 128 threads (4 warps) per CTA, one CTA per 128×128 output tile
 *   - Grid: 32×32 = 1024 CTAs (each allocates/deallocates TMEM)
 *   - Tiled GEMM: 128×128×16 tiles, cooperative SMEM loads, FP32 accumulators
 *   - TMEM is allocated and deallocated but NOT used for compute — that's Phase 1
 *
 * SM100 requirements exercised:
 *   - __cluster_dims__(1,1,1) + cudaLaunchKernelEx (cluster launch for tcgen05)
 *   - tcgen05.alloc with num_columns=32 (minimum, must be power of 2)
 *   - tcgen05.dealloc before kernel exit (mandatory, hardware hangs otherwise)
 *   - Warp-collective: all 32 threads in warp 0 execute alloc/dealloc together
 *
 * Correctness target: mean relative error < 1% vs cuBLAS.
 *   Note: max relative error can be large (~30%) due to FP16 input precision
 *   and different accumulation orders. This is expected — FP16 arithmetic is
 *   not associative, so two correct implementations can produce different results
 *   for near-zero output elements.
 *
 * Performance: ~3 TFLOPS on B200 CUDA cores (not optimized). B200 tensor
 *   cores deliver ~2250 TFLOPS dense FP16 — roughly 700x faster, which is
 *   the point of Phase 1's tcgen05.mma kernels.
 */

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CHECK_CUBLAS(call)                                                     \
    do {                                                                        \
        cublasStatus_t status = (call);                                        \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__,\
                    status);                                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Problem and tile dimensions (fixed for Phase 0 bringup)
// ---------------------------------------------------------------------------
constexpr int M = 4096;
constexpr int N = 4096;
constexpr int K = 4096;

constexpr int TILE_M = 128;    // Output tile rows per CTA
constexpr int TILE_N = 128;    // Output tile columns per CTA
constexpr int TILE_K = 16;     // K-dimension tile (SMEM footprint trade-off)

constexpr int TILES_M = M / TILE_M;  // 32
constexpr int TILES_N = N / TILE_N;  // 32

constexpr int BLOCK_SIZE = 128; // 4 warps

constexpr int SMEM_A_SIZE = TILE_M * TILE_K;  // 2048 halfs = 4 KB
constexpr int SMEM_B_SIZE = TILE_K * TILE_N;  // 2048 halfs = 4 KB

using tmem_addr_t = uint32_t;

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

/**
 * Naive tiled GEMM on CUDA cores + TMEM lifecycle validation.
 *
 * Each CTA:
 *   1. Allocates 32 TMEM columns (warp 0, warp-collective)
 *   2. Computes a 128×128 output tile via tiled GEMM on CUDA cores
 *   3. Deallocates TMEM (warp 0, warp-collective)
 *
 * Thread mapping: thread tid owns column (tid % TILE_N) across all rows.
 *   Each thread computes TILE_M * TILE_N / BLOCK_SIZE = 128 output elements.
 *
 * SMEM layout: [128B TMEM header | A tile (4KB) | B tile (4KB)]
 *   Total dynamic SMEM: 8320 bytes per CTA.
 */
__cluster_dims__(1, 1, 1)
__global__ void gemm_fp16_naive_kernel(
    const half* __restrict__ A,   // (M, K) row-major
    const half* __restrict__ B,   // (K, N) row-major
    half* __restrict__ C,         // (M, N) row-major
    int m, int n, int k
) {
    extern __shared__ char smem_raw[];
    uint32_t* smem_tmem_addr = reinterpret_cast<uint32_t*>(smem_raw);
    half* smem_A = reinterpret_cast<half*>(smem_raw + 128);  // 128B padding for alignment
    half* smem_B = smem_A + SMEM_A_SIZE;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;

    const int bm = blockIdx.y * TILE_M;  // Output tile row start
    const int bn = blockIdx.x * TILE_N;  // Output tile col start

    // --- TMEM lifecycle: allocate (warp-collective, all 32 threads in warp 0) ---
    // Minimum allocation is 32 columns (must be power of 2, range 32-512).
    // TMEM is not used for compute here — just validating the lifecycle works
    // under real workload conditions (1024 concurrent CTAs).
    constexpr uint32_t TMEM_COLS = 32;
    if (warp_id == 0) {
        uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_tmem_addr));
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
            :
            : "r"(smem_addr), "r"(TMEM_COLS)
        );
    }
    __syncthreads();
    tmem_addr_t tmem_addr = *smem_tmem_addr;

    // --- Tiled GEMM on CUDA cores ---
    // Each thread accumulates 128 output elements in FP32 registers.
    // Elements are striped: thread tid handles elements tid, tid+128, tid+256, ...
    constexpr int ELEMS_PER_THREAD = (TILE_M * TILE_N) / BLOCK_SIZE;  // 128

    float acc[ELEMS_PER_THREAD];
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        acc[i] = 0.0f;
    }

    // Iterate over K-dimension in TILE_K chunks
    for (int tk = 0; tk < k; tk += TILE_K) {
        // Cooperative load: all 128 threads load A tile (128×16) into SMEM
        for (int i = tid; i < SMEM_A_SIZE; i += BLOCK_SIZE) {
            int row = i / TILE_K;
            int col = i % TILE_K;
            int gr = bm + row;
            int gc = tk + col;
            smem_A[i] = (gr < m && gc < k) ? A[gr * k + gc] : __float2half(0.0f);
        }

        // Cooperative load: all 128 threads load B tile (16×128) into SMEM
        for (int i = tid; i < SMEM_B_SIZE; i += BLOCK_SIZE) {
            int row = i / TILE_N;
            int col = i % TILE_N;
            int gr = tk + row;
            int gc = bn + col;
            smem_B[i] = (gr < k && gc < n) ? B[gr * n + gc] : __float2half(0.0f);
        }

        __syncthreads();

        // Compute: C_tile[row][col] += sum_kk( A_tile[row][kk] * B_tile[kk][col] )
        for (int idx = 0; idx < ELEMS_PER_THREAD; idx++) {
            int elem = tid + idx * BLOCK_SIZE;
            int row = elem / TILE_N;
            int col = elem % TILE_N;

            float sum = 0.0f;
            for (int kk = 0; kk < TILE_K; kk++) {
                sum += __half2float(smem_A[row * TILE_K + kk]) *
                       __half2float(smem_B[kk * TILE_N + col]);
            }
            acc[idx] += sum;
        }

        __syncthreads();
    }

    // Store FP32 accumulators to global memory as FP16
    for (int idx = 0; idx < ELEMS_PER_THREAD; idx++) {
        int elem = tid + idx * BLOCK_SIZE;
        int row = elem / TILE_N;
        int col = elem % TILE_N;
        int gr = bm + row;
        int gc = bn + col;
        if (gr < m && gc < n) {
            C[gr * n + gc] = __float2half(acc[idx]);
        }
    }

    // --- TMEM lifecycle: deallocate (mandatory before kernel exit) ---
    __syncthreads();
    if (warp_id == 0) {
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
            :
            : "r"(tmem_addr), "r"(TMEM_COLS)
        );
    }
}

// ---------------------------------------------------------------------------
// Host utilities
// ---------------------------------------------------------------------------

void init_random_fp16(half* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = __float2half((static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f);
    }
}

/**
 * Error statistics for comparing two FP16 arrays.
 *
 * Uses max(|ref|, |test|, 1.0) as the denominator for relative error,
 * which avoids the pathological divide-by-near-zero that inflates max
 * relative error when both values are close to zero (common in FP16 GEMM
 * where different accumulation orders yield different rounding near zero).
 */
struct ErrorStats {
    float max_abs_err;
    float mean_abs_err;
    float max_rel_err;
    float mean_rel_err;
    int num_mismatched;
};

ErrorStats compute_error(const half* ref, const half* test, int size,
                         float atol, float rtol) {
    ErrorStats s = {};
    double sum_abs = 0, sum_rel = 0;
    for (int i = 0; i < size; i++) {
        float r = __half2float(ref[i]);
        float t = __half2float(test[i]);
        float abs_err = fabsf(r - t);
        float denom = fmaxf(fmaxf(fabsf(r), fabsf(t)), 1.0f);
        float rel_err = abs_err / denom;

        s.max_abs_err = fmaxf(s.max_abs_err, abs_err);
        s.max_rel_err = fmaxf(s.max_rel_err, rel_err);
        sum_abs += abs_err;
        sum_rel += rel_err;
        if (abs_err > atol + rtol * fabsf(r)) s.num_mismatched++;
    }
    s.mean_abs_err = (float)(sum_abs / size);
    s.mean_rel_err = (float)(sum_rel / size);
    return s;
}

// ---------------------------------------------------------------------------
// Main: cuBLAS reference → our kernel → compare
// ---------------------------------------------------------------------------

int main() {
    printf("=== Blaze: Phase 0 — Naive FP16 GEMM (CUDA cores + TMEM lifecycle) ===\n\n");
    printf("Problem: C[%d×%d] = A[%d×%d] × B[%d×%d] (FP16)\n\n", M, N, M, K, K, N);

    // --- Allocate and initialize ---
    half* h_A = new half[M * K];
    half* h_B = new half[K * N];
    half* h_C = new half[M * N];
    half* h_C_ref = new half[M * N];

    srand(42);
    init_random_fp16(h_A, M * K);
    init_random_fp16(h_B, K * N);

    half *d_A, *d_B, *d_C, *d_C_ref;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_C_ref, M * N * sizeof(half)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice));

    // --- cuBLAS reference (FP16 GEMM with tensor ops) ---
    // cuBLAS is column-major. To compute row-major C = A × B:
    //   C^T = B^T × A^T
    // Row-major data reinterpreted as column-major is transposed, so:
    //   cublasHgemm(N, M, K, d_B, N, d_A, K, d_C_ref, N)
    printf("Computing cuBLAS reference...\n");
    cublasHandle_t cublas;
    CHECK_CUBLAS(cublasCreate(&cublas));
    CHECK_CUBLAS(cublasSetMathMode(cublas, CUBLAS_TENSOR_OP_MATH));

    half alpha_h = __float2half(1.0f);
    half beta_h = __float2half(0.0f);

    CHECK_CUBLAS(cublasHgemm(cublas,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha_h,
                             d_B, N,
                             d_A, K,
                             &beta_h,
                             d_C_ref, N));

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_C_ref, d_C_ref, M * N * sizeof(half), cudaMemcpyDeviceToHost));
    printf("cuBLAS reference computed.\n\n");

    // --- Our kernel ---
    printf("Launching naive GEMM kernel...\n");

    size_t smem_size = 128 + (SMEM_A_SIZE + SMEM_B_SIZE) * sizeof(half);
    printf("  Shared memory per block: %zu bytes (%.1f KB)\n", smem_size, smem_size / 1024.0);

    CHECK_CUDA(cudaFuncSetAttribute(
        gemm_fp16_naive_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size));

    dim3 grid(TILES_N, TILES_M);
    dim3 block(BLOCK_SIZE);

    printf("  Grid: (%d, %d), Block: %d threads\n", grid.x, grid.y, block.x);
    printf("  Tile: %d × %d × %d\n\n", TILE_M, TILE_N, TILE_K);

    // Cluster launch (required for tcgen05 instructions on SM100).
    // Even though cta_group::1 is a single-CTA operation, the hardware
    // expects cluster scheduling context to be initialized.
    cudaLaunchConfig_t launch_config = {};
    launch_config.gridDim = grid;
    launch_config.blockDim = block;
    launch_config.dynamicSmemBytes = smem_size;
    launch_config.stream = 0;

    cudaLaunchAttribute launch_attrs[1];
    launch_attrs[0].id = cudaLaunchAttributeClusterDimension;
    launch_attrs[0].val.clusterDim.x = 1;
    launch_attrs[0].val.clusterDim.y = 1;
    launch_attrs[0].val.clusterDim.z = 1;
    launch_config.attrs = launch_attrs;
    launch_config.numAttrs = 1;

    // Warmup
    cudaLaunchKernelEx(&launch_config, gemm_fp16_naive_kernel, d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());

    // Benchmark: 3 warmup + 10 timed iterations
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int bench_iters = 10;
    for (int i = 0; i < 3; i++) {
        cudaLaunchKernelEx(&launch_config, gemm_fp16_naive_kernel, d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < bench_iters; i++) {
        cudaLaunchKernelEx(&launch_config, gemm_fp16_naive_kernel, d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float time_ms;
    CHECK_CUDA(cudaEventElapsedTime(&time_ms, start, stop));
    float avg_ms = time_ms / bench_iters;

    double flops = 2.0 * M * N * K;
    double tflops = flops / (avg_ms * 1e-3) / 1e12;

    printf("Performance:\n");
    printf("  Average time: %.3f ms\n", avg_ms);
    printf("  Throughput:   %.2f TFLOPS (FP16, CUDA cores — not optimized)\n\n", tflops);

    // --- Correctness check ---
    // FP16 tolerance must account for:
    //   - FP16 input precision (~3 decimal digits)
    //   - Non-associative accumulation: our 16-wide K-tiles vs cuBLAS tiling
    //   - cuBLAS tensor-op math may use different intermediate precision
    // atol=0.5, rtol=0.05 is appropriate for K=4096 FP16 GEMM.
    CHECK_CUDA(cudaMemcpy(h_C, d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost));

    ErrorStats err = compute_error(h_C_ref, h_C, M * N, 0.5f, 0.05f);
    printf("Correctness vs cuBLAS:\n");
    printf("  Max absolute error:  %.6f\n", err.max_abs_err);
    printf("  Mean absolute error: %.6f\n", err.mean_abs_err);
    printf("  Max relative error:  %.6f (denom = max(|ref|, |test|, 1.0))\n", err.max_rel_err);
    printf("  Mean relative error: %.6f\n", err.mean_rel_err);
    printf("  Mismatched elements: %d / %d (atol=0.5, rtol=0.05)\n", err.num_mismatched, M * N);
    bool pass = err.mean_rel_err < 1e-2f && err.num_mismatched < (M * N / 100);
    printf("  Status: %s\n", pass ? "PASS" : "FAIL");

    // --- Cleanup ---
    CHECK_CUBLAS(cublasDestroy(cublas));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_C_ref));
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;

    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
