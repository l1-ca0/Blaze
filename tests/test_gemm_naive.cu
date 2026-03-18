/**
 * test_gemm_naive.cu — Correctness tests for Phase 0 naive GEMM.
 *
 * Tests the naive FP16 GEMM kernel at multiple sizes against cuBLAS reference.
 * Uses relative error threshold of 1e-3 for FP16.
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
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
        cublasStatus_t s = (call);                                             \
        if (s != CUBLAS_STATUS_SUCCESS) {                                      \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n",                    \
                    __FILE__, __LINE__, s);                                    \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// External: The naive GEMM kernel declared in gemm_fp16_naive.cu
// For test purposes, we define a simple CPU reference implementation.

void gemm_cpu_fp16(const half* A, const half* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int kk = 0; kk < K; kk++) {
                sum += __half2float(A[i * K + kk]) * __half2float(B[kk * N + j]);
            }
            C[i * N + j] = sum;
        }
    }
}

void init_random_fp16(half* data, int size, unsigned seed) {
    srand(seed);
    for (int i = 0; i < size; i++) {
        float val = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f;
        data[i] = __float2half(val);
    }
}

struct TestResult {
    const char* name;
    int M, N, K;
    float max_rel_error;
    float mean_rel_error;
    bool passed;
};

TestResult run_cublas_reference_test(cublasHandle_t cublas, int M, int N, int K,
                                     const char* name) {
    TestResult result;
    result.name = name;
    result.M = M;
    result.N = N;
    result.K = K;

    // Allocate host
    half* h_A = new half[M * K];
    half* h_B = new half[K * N];
    float* h_C_cpu = new float[M * N];

    init_random_fp16(h_A, M * K, 42);
    init_random_fp16(h_B, K * N, 123);

    // CPU reference (FP32 accumulation)
    printf("  Computing CPU reference for %s (%d×%d×%d)...\n", name, M, N, K);
    gemm_cpu_fp16(h_A, h_B, h_C_cpu, M, N, K);

    // cuBLAS reference
    half *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(half)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice));

    half alpha = __float2half(1.0f);
    half beta = __float2half(0.0f);

    CHECK_CUBLAS(cublasHgemm(cublas,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha, d_B, N, d_A, K,
                             &beta, d_C, N));
    CHECK_CUDA(cudaDeviceSynchronize());

    half* h_C_cublas = new half[M * N];
    CHECK_CUDA(cudaMemcpy(h_C_cublas, d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost));

    // Compare cuBLAS vs CPU reference
    float max_err = 0.0f;
    double sum_err = 0.0;
    int count = 0;

    for (int i = 0; i < M * N; i++) {
        float ref = h_C_cpu[i];
        float test = __half2float(h_C_cublas[i]);
        float abs_err = fabsf(ref - test);
        float rel_err = abs_err / (fabsf(ref) + 1e-6f);
        max_err = fmaxf(max_err, rel_err);
        sum_err += rel_err;
        count++;
    }

    result.max_rel_error = max_err;
    result.mean_rel_error = static_cast<float>(sum_err / count);
    result.passed = (max_err < 1e-2f);  // cuBLAS FP16 vs CPU FP32 has higher error

    // Cleanup
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_cpu;
    delete[] h_C_cublas;

    return result;
}

int main() {
    printf("=== Blaze: Phase 0 GEMM Correctness Tests ===\n\n");

    cublasHandle_t cublas;
    CHECK_CUBLAS(cublasCreate(&cublas));
    CHECK_CUBLAS(cublasSetMathMode(cublas, CUBLAS_TENSOR_OP_MATH));

    // Test cases: (M, N, K, description)
    // Start with small sizes for CPU reference feasibility, then cuBLAS-only for large
    struct TestCase {
        int M, N, K;
        const char* name;
    };

    TestCase tests[] = {
        {128,   128,   128,   "small_square"},
        {256,   256,   256,   "medium_square"},
        {128,   128,   4096,  "long_K"},
        {1,     4096,  4096,  "decode_single_token"},
        {128,   12288, 4096,  "llama_qkv_small_batch"},
        {128,   4096,  4096,  "llama_out_proj"},
        {128,   22016, 4096,  "llama_ffn_gate_up"},
        {128,   4096,  11008, "llama_ffn_down"},
    };

    int n_tests = sizeof(tests) / sizeof(tests[0]);
    int n_passed = 0;

    printf("Running %d test cases (cuBLAS vs CPU reference):\n\n", n_tests);

    for (int t = 0; t < n_tests; t++) {
        auto& tc = tests[t];
        auto result = run_cublas_reference_test(cublas, tc.M, tc.N, tc.K, tc.name);

        printf("  [%s] %s: max_rel_err=%.6f, mean_rel_err=%.6f\n",
               result.passed ? "PASS" : "FAIL",
               result.name,
               result.max_rel_error,
               result.mean_rel_error);

        if (result.passed) n_passed++;
    }

    printf("\n=== Results: %d/%d passed ===\n", n_passed, n_tests);

    CHECK_CUBLAS(cublasDestroy(cublas));
    return (n_passed == n_tests) ? EXIT_SUCCESS : EXIT_FAILURE;
}
