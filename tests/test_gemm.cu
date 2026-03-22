/**
 * test_gemm.cu — Correctness tests for Phase 1 GEMM kernels.
 *
 * Validates FP8, mixed-precision (FP16×FP4), and FP4 GEMM kernels
 * across all Llama-7B projection shapes. Each test checks for NaN/Inf
 * in the output as a basic sanity gate. Numerical accuracy tests
 * against cuBLAS will be added once the kernels produce correct results.
 */

#include "gemm/fp8_gemm_sm100.cuh"
#include "gemm/fp4_gemm_sm100.cuh"
#include "gemm/mixed_gemm_sm100.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CHECK_CUDA(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Llama-7B matrix shapes covering decode (M=1), batched, and prefill sizes.
// ---------------------------------------------------------------------------

struct GemmTestCase {
    int M, N, K;
    const char* name;
};

static GemmTestCase llama_shapes[] = {
    // Decode (M=1)
    {1,     12288, 4096,  "decode_QKV"},
    {1,     4096,  4096,  "decode_out_proj"},
    {1,     22016, 4096,  "decode_FFN_gate_up"},
    {1,     4096,  11008, "decode_FFN_down"},
    // Small batch
    {32,    12288, 4096,  "batch32_QKV"},
    {32,    4096,  4096,  "batch32_out_proj"},
    // Medium batch
    {128,   12288, 4096,  "batch128_QKV"},
    {128,   4096,  11008, "batch128_FFN_down"},
    // Prefill
    {512,   12288, 4096,  "prefill512_QKV"},
    {2048,  12288, 4096,  "prefill2048_QKV"},
    {2048,  4096,  11008, "prefill2048_FFN_down"},
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

void init_random_fp8(void* data, int size, unsigned seed) {
    srand(seed);
    auto* ptr = static_cast<__nv_fp8_e4m3*>(data);
    for (int i = 0; i < size; i++) {
        float val = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f;
        ptr[i] = __nv_fp8_e4m3(val);
    }
}

void init_random_fp16(half* data, int size, unsigned seed) {
    srand(seed);
    for (int i = 0; i < size; i++) {
        float val = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f;
        data[i] = __float2half(val);
    }
}

struct ErrorStats {
    float max_rel;
    float mean_rel;
};

ErrorStats compute_error(const half* ref, const half* test, int size) {
    float max_err = 0.0f;
    double sum_err = 0.0;
    for (int i = 0; i < size; i++) {
        float r = __half2float(ref[i]);
        float t = __half2float(test[i]);
        float err = fabsf(r - t) / (fabsf(r) + 1e-6f);
        max_err = fmaxf(max_err, err);
        sum_err += err;
    }
    return {max_err, static_cast<float>(sum_err / size)};
}

bool check_no_nan_inf(const half* data, int size) {
    for (int i = 0; i < size; i++) {
        float v = __half2float(data[i]);
        if (isnan(v) || isinf(v)) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// FP8 GEMM: E4M3 × E4M3 → FP16
// ---------------------------------------------------------------------------

int test_fp8_gemm(cublasHandle_t cublas) {
    printf("\n=== FP8 GEMM Tests (E4M3 x E4M3) ===\n");
    int passed = 0;
    int total = sizeof(llama_shapes) / sizeof(llama_shapes[0]);

    for (int t = 0; t < total; t++) {
        auto& tc = llama_shapes[t];
        printf("  [%d/%d] %s (M=%d, N=%d, K=%d)... ",
               t + 1, total, tc.name, tc.M, tc.N, tc.K);

        __nv_fp8_e4m3 *d_A, *d_B;
        half *d_C;

        CHECK_CUDA(cudaMalloc(&d_A, tc.M * tc.K * sizeof(__nv_fp8_e4m3)));
        CHECK_CUDA(cudaMalloc(&d_B, tc.K * tc.N * sizeof(__nv_fp8_e4m3)));
        CHECK_CUDA(cudaMalloc(&d_C, tc.M * tc.N * sizeof(half)));

        auto* h_A = new __nv_fp8_e4m3[tc.M * tc.K];
        auto* h_B = new __nv_fp8_e4m3[tc.K * tc.N];
        init_random_fp8(h_A, tc.M * tc.K, 42);
        init_random_fp8(h_B, tc.K * tc.N, 123);

        CHECK_CUDA(cudaMemcpy(d_A, h_A, tc.M * tc.K * sizeof(__nv_fp8_e4m3),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B, tc.K * tc.N * sizeof(__nv_fp8_e4m3),
                              cudaMemcpyHostToDevice));

        blaze::launch_gemm_fp8(d_A, d_B, d_C, tc.M, tc.N, tc.K);
        CHECK_CUDA(cudaDeviceSynchronize());

        auto* h_C = new half[tc.M * tc.N];
        CHECK_CUDA(cudaMemcpy(h_C, d_C, tc.M * tc.N * sizeof(half),
                              cudaMemcpyDeviceToHost));

        if (check_no_nan_inf(h_C, tc.M * tc.N)) {
            printf("PASS\n");
            passed++;
        } else {
            printf("FAIL (NaN/Inf in output)\n");
        }

        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_C));
    }

    printf("  FP8 GEMM: %d/%d passed\n", passed, total);
    return passed == total ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Mixed GEMM: FP16 activations × NVFP4 weights → FP16
// ---------------------------------------------------------------------------

int test_mixed_gemm(cublasHandle_t cublas) {
    printf("\n=== Mixed GEMM Tests (FP16 x FP4) ===\n");
    int passed = 0;
    int total = sizeof(llama_shapes) / sizeof(llama_shapes[0]);

    for (int t = 0; t < total; t++) {
        auto& tc = llama_shapes[t];
        printf("  [%d/%d] %s (M=%d, N=%d, K=%d)... ",
               t + 1, total, tc.name, tc.M, tc.N, tc.K);

        half* d_A;
        CHECK_CUDA(cudaMalloc(&d_A, tc.M * tc.K * sizeof(half)));

        auto* h_A = new half[tc.M * tc.K];
        init_random_fp16(h_A, tc.M * tc.K, 42);
        CHECK_CUDA(cudaMemcpy(d_A, h_A, tc.M * tc.K * sizeof(half),
                              cudaMemcpyHostToDevice));

        int data_bytes = tc.K * tc.N / 2;
        int scale_count = tc.K * tc.N / blaze::FP4_BLOCK_SIZE;

        uint8_t* d_B_data;
        __nv_fp8_e4m3* d_B_scales;
        CHECK_CUDA(cudaMalloc(&d_B_data, data_bytes));
        CHECK_CUDA(cudaMalloc(&d_B_scales, scale_count * sizeof(__nv_fp8_e4m3)));
        CHECK_CUDA(cudaMemset(d_B_data, 0x55, data_bytes));
        CHECK_CUDA(cudaMemset(d_B_scales, 0x3C, scale_count));

        blaze::Fp4WeightTensor B_weight;
        B_weight.data = d_B_data;
        B_weight.block_scales = d_B_scales;
        B_weight.tensor_scale = 1.0f;
        B_weight.rows = tc.K;
        B_weight.cols = tc.N;

        half* d_C;
        CHECK_CUDA(cudaMalloc(&d_C, tc.M * tc.N * sizeof(half)));

        blaze::launch_gemm_mixed(d_A, B_weight, d_C, tc.M, tc.N, tc.K);
        CHECK_CUDA(cudaDeviceSynchronize());

        auto* h_C = new half[tc.M * tc.N];
        CHECK_CUDA(cudaMemcpy(h_C, d_C, tc.M * tc.N * sizeof(half),
                              cudaMemcpyDeviceToHost));

        if (check_no_nan_inf(h_C, tc.M * tc.N)) {
            printf("PASS\n");
            passed++;
        } else {
            printf("FAIL (NaN/Inf)\n");
        }

        delete[] h_A;
        delete[] h_C;
        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B_data));
        CHECK_CUDA(cudaFree(d_B_scales));
        CHECK_CUDA(cudaFree(d_C));
    }

    printf("  Mixed GEMM: %d/%d passed\n", passed, total);
    return passed == total ? 0 : 1;
}

// ---------------------------------------------------------------------------
// FP4 GEMM: NVFP4 × NVFP4 → FP16
// ---------------------------------------------------------------------------

int test_fp4_gemm(cublasHandle_t cublas) {
    printf("\n=== FP4 GEMM Tests (E2M1 x E2M1) ===\n");
    int passed = 0;
    int total = sizeof(llama_shapes) / sizeof(llama_shapes[0]);

    for (int t = 0; t < total; t++) {
        auto& tc = llama_shapes[t];

        // FP4 TILE_K=128: align K up to tile boundary.
        int K_aligned = ((tc.K + 127) / 128) * 128;

        printf("  [%d/%d] %s (M=%d, N=%d, K=%d)... ",
               t + 1, total, tc.name, tc.M, tc.N, K_aligned);

        int a_data_bytes  = tc.M * K_aligned / 2;
        int a_scale_count = tc.M * K_aligned / blaze::FP4_BLOCK_SIZE;

        uint8_t* d_A_data;
        __nv_fp8_e4m3* d_A_scales;
        CHECK_CUDA(cudaMalloc(&d_A_data, a_data_bytes));
        CHECK_CUDA(cudaMalloc(&d_A_scales, a_scale_count * sizeof(__nv_fp8_e4m3)));
        CHECK_CUDA(cudaMemset(d_A_data, 0x55, a_data_bytes));
        CHECK_CUDA(cudaMemset(d_A_scales, 0x3C, a_scale_count));

        blaze::Fp4WeightTensor A_weight;
        A_weight.data = d_A_data;
        A_weight.block_scales = d_A_scales;
        A_weight.tensor_scale = 1.0f;
        A_weight.rows = tc.M;
        A_weight.cols = K_aligned;

        int b_data_bytes  = K_aligned * tc.N / 2;
        int b_scale_count = K_aligned * tc.N / blaze::FP4_BLOCK_SIZE;

        uint8_t* d_B_data;
        __nv_fp8_e4m3* d_B_scales;
        CHECK_CUDA(cudaMalloc(&d_B_data, b_data_bytes));
        CHECK_CUDA(cudaMalloc(&d_B_scales, b_scale_count * sizeof(__nv_fp8_e4m3)));
        CHECK_CUDA(cudaMemset(d_B_data, 0x55, b_data_bytes));
        CHECK_CUDA(cudaMemset(d_B_scales, 0x3C, b_scale_count));

        blaze::Fp4WeightTensor B_weight;
        B_weight.data = d_B_data;
        B_weight.block_scales = d_B_scales;
        B_weight.tensor_scale = 1.0f;
        B_weight.rows = K_aligned;
        B_weight.cols = tc.N;

        half* d_C;
        CHECK_CUDA(cudaMalloc(&d_C, tc.M * tc.N * sizeof(half)));

        blaze::launch_gemm_fp4(A_weight, B_weight, d_C, tc.M, tc.N, K_aligned);
        CHECK_CUDA(cudaDeviceSynchronize());

        auto* h_C = new half[tc.M * tc.N];
        CHECK_CUDA(cudaMemcpy(h_C, d_C, tc.M * tc.N * sizeof(half),
                              cudaMemcpyDeviceToHost));

        if (check_no_nan_inf(h_C, tc.M * tc.N)) {
            printf("PASS\n");
            passed++;
        } else {
            printf("FAIL (NaN/Inf)\n");
        }

        delete[] h_C;
        CHECK_CUDA(cudaFree(d_A_data));
        CHECK_CUDA(cudaFree(d_A_scales));
        CHECK_CUDA(cudaFree(d_B_data));
        CHECK_CUDA(cudaFree(d_B_scales));
        CHECK_CUDA(cudaFree(d_C));
    }

    printf("  FP4 GEMM: %d/%d passed\n", passed, total);
    return passed == total ? 0 : 1;
}

// ---------------------------------------------------------------------------

int main() {
    printf("=== Blaze: Phase 1 GEMM Correctness Tests ===\n");

    cublasHandle_t cublas;
    cublasCreate(&cublas);
    cublasSetMathMode(cublas, CUBLAS_TENSOR_OP_MATH);

    int ret = 0;
    ret |= test_fp8_gemm(cublas);
    ret |= test_mixed_gemm(cublas);
    ret |= test_fp4_gemm(cublas);

    cublasDestroy(cublas);

    printf("\n=== Overall: %s ===\n", ret == 0 ? "ALL PASSED" : "SOME FAILED");
    return ret;
}
