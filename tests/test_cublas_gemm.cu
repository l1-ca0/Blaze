/**
 * test_cublas_gemm.cu — Correctness tests for cuBLAS GEMM backends.
 *
 * Tests the cuBLASLt wrappers in cublas_gemm.cuh by comparing their
 * output against a CPU reference computation.
 *
 * Three test suites:
 *   1. FP16 × FP16 — cuBLAS vs CPU reference
 *   2. FP8  × FP8  — cuBLAS vs CPU reference (dequant FP8 → FP32, matmul on CPU)
 *   3. Mixed FP16 × FP4 — cuBLAS (with dequant) vs CPU reference
 *
 * Uses small shapes for fast CPU reference, plus a few Llama-7B shapes
 * for end-to-end sanity.
 *
 * Compile: link against blaze_cublas_gemm
 */

#include "gemm/cublas_gemm.cuh"
#include "gemm/fp4_gemm_sm100.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CHECK_CUDA(call) do {                                          \
    cudaError_t e = (call);                                            \
    if (e != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e));                                 \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
} while (0)

// ============================================================================
// Test shapes
// ============================================================================

struct TestShape {
    int M, N, K;
    const char* name;
};

static TestShape small_shapes[] = {
    {1,    128,   128,  "tiny"},
    {1,    256,   256,  "small_decode"},
    {32,   256,   128,  "small_batch"},
    {128,  128,   256,  "square_ish"},
    {128,  512,   256,  "medium"},
    {256,  1024,  512,  "large"},
};

static TestShape llama_shapes[] = {
    {1,    12288, 4096, "decode_QKV"},
    {1,    4096,  4096, "decode_out_proj"},
    {128,  12288, 4096, "batch128_QKV"},
    {128,  4096,  11008,"batch128_FFN_down"},
};

// ============================================================================
// CPU reference: FP32 matmul (gold standard)
// ============================================================================

static void cpu_matmul_f32(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            double sum = 0.0;
            for (int k = 0; k < K; k++) {
                sum += (double)A[m * K + k] * (double)B[k * N + n];
            }
            C[m * N + n] = (float)sum;
        }
    }
}

// ============================================================================
// Helpers
// ============================================================================

static void init_random_fp16(half* h, int n, unsigned seed) {
    srand(seed);
    for (int i = 0; i < n; i++) {
        // Small values to avoid FP16 overflow: [-0.5, 0.5]
        float v = (float)rand() / RAND_MAX - 0.5f;
        h[i] = __float2half(v);
    }
}

static void init_random_fp8(void* data, int n, unsigned seed) {
    srand(seed);
    auto* p = (__nv_fp8_e4m3*)data;
    for (int i = 0; i < n; i++) {
        float v = (float)rand() / RAND_MAX - 0.5f;
        p[i] = __nv_fp8_e4m3(v);
    }
}

static float fp8_to_float(__nv_fp8_e4m3 v) {
    return (float)v;
}

static float decode_e2m1_host(uint8_t nibble) {
    // NVFP4 E2M1 is signed: bit 3 = sign, bits 2-0 = magnitude
    static const float lut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float mag = lut[nibble & 0x7];
    return (nibble & 0x8) ? -mag : mag;
}

struct ErrorStats {
    float max_abs;
    float max_rel;
    float mean_rel;
    int nan_count;
    int inf_count;
};

static ErrorStats compute_error(const float* ref, const half* test, int n) {
    ErrorStats s = {0, 0, 0, 0, 0};
    double sum_rel = 0.0;
    for (int i = 0; i < n; i++) {
        float t = __half2float(test[i]);
        float r = ref[i];
        if (isnan(t)) { s.nan_count++; continue; }
        if (isinf(t)) { s.inf_count++; continue; }
        float abs_err = fabsf(r - t);
        float rel_err = abs_err / (fabsf(r) + 1e-6f);
        s.max_abs = fmaxf(s.max_abs, abs_err);
        s.max_rel = fmaxf(s.max_rel, rel_err);
        sum_rel += rel_err;
    }
    s.mean_rel = (float)(sum_rel / n);
    return s;
}

// ============================================================================
// Test 1: FP16 × FP16
// ============================================================================

static int test_cublas_fp16(TestShape* shapes, int num_shapes) {
    printf("\n=== cuBLAS FP16 GEMM Tests ===\n");
    int passed = 0;

    for (int t = 0; t < num_shapes; t++) {
        auto& s = shapes[t];
        printf("  [%d/%d] %s (M=%d, N=%d, K=%d)... ",
               t + 1, num_shapes, s.name, s.M, s.N, s.K);

        int sA = s.M * s.K, sB = s.K * s.N, sC = s.M * s.N;

        // Host data
        auto* h_A = new half[sA];
        auto* h_B = new half[sB];
        init_random_fp16(h_A, sA, 42 + t);
        init_random_fp16(h_B, sB, 123 + t);

        // CPU reference (FP32)
        auto* ref_A = new float[sA];
        auto* ref_B = new float[sB];
        auto* ref_C = new float[sC];
        for (int i = 0; i < sA; i++) ref_A[i] = __half2float(h_A[i]);
        for (int i = 0; i < sB; i++) ref_B[i] = __half2float(h_B[i]);
        cpu_matmul_f32(ref_A, ref_B, ref_C, s.M, s.N, s.K);

        // GPU: cuBLAS
        half *d_A, *d_B, *d_C;
        CHECK_CUDA(cudaMalloc(&d_A, sA * sizeof(half)));
        CHECK_CUDA(cudaMalloc(&d_B, sB * sizeof(half)));
        CHECK_CUDA(cudaMalloc(&d_C, sC * sizeof(half)));
        CHECK_CUDA(cudaMemcpy(d_A, h_A, sA * sizeof(half), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B, sB * sizeof(half), cudaMemcpyHostToDevice));

        blaze::cublas_gemm_fp16(d_A, d_B, d_C, s.M, s.N, s.K);
        CHECK_CUDA(cudaDeviceSynchronize());

        auto* h_C = new half[sC];
        CHECK_CUDA(cudaMemcpy(h_C, d_C, sC * sizeof(half), cudaMemcpyDeviceToHost));

        auto err = compute_error(ref_C, h_C, sC);

        // FP16 matmul with K up to 11008: expect rel error < 0.05 (5%)
        // FP16 accumulation loses precision for large K.
        float tol = 0.05f;
        bool ok = (err.nan_count == 0 && err.inf_count == 0 && err.max_rel < tol);
        if (ok) {
            printf("PASS (max_rel=%.4f, mean_rel=%.6f)\n", err.max_rel, err.mean_rel);
            passed++;
        } else {
            printf("FAIL (max_rel=%.4f, mean_rel=%.6f, nan=%d, inf=%d)\n",
                   err.max_rel, err.mean_rel, err.nan_count, err.inf_count);
        }

        delete[] h_A; delete[] h_B; delete[] h_C;
        delete[] ref_A; delete[] ref_B; delete[] ref_C;
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    printf("  FP16 cuBLAS: %d/%d passed\n", passed, num_shapes);
    return passed;
}

// ============================================================================
// Test 2: FP8 × FP8
// ============================================================================

static int test_cublas_fp8(TestShape* shapes, int num_shapes) {
    printf("\n=== cuBLAS FP8 GEMM Tests ===\n");
    int passed = 0;

    for (int t = 0; t < num_shapes; t++) {
        auto& s = shapes[t];
        printf("  [%d/%d] %s (M=%d, N=%d, K=%d)... ",
               t + 1, num_shapes, s.name, s.M, s.N, s.K);

        int sA = s.M * s.K, sB = s.K * s.N, sC = s.M * s.N;

        // Host FP8 data
        auto* h_A = new __nv_fp8_e4m3[sA];
        auto* h_B = new __nv_fp8_e4m3[sB];
        init_random_fp8(h_A, sA, 42 + t);
        init_random_fp8(h_B, sB, 123 + t);

        // CPU reference: dequant FP8 → FP32, then matmul
        auto* ref_A = new float[sA];
        auto* ref_B = new float[sB];
        auto* ref_C = new float[sC];
        for (int i = 0; i < sA; i++) ref_A[i] = fp8_to_float(h_A[i]);
        for (int i = 0; i < sB; i++) ref_B[i] = fp8_to_float(h_B[i]);
        cpu_matmul_f32(ref_A, ref_B, ref_C, s.M, s.N, s.K);

        // GPU: cuBLAS FP8
        __nv_fp8_e4m3 *d_A, *d_B;
        half *d_C;
        CHECK_CUDA(cudaMalloc(&d_A, sA));
        CHECK_CUDA(cudaMalloc(&d_B, sB));
        CHECK_CUDA(cudaMalloc(&d_C, sC * sizeof(half)));
        CHECK_CUDA(cudaMemcpy(d_A, h_A, sA, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B, sB, cudaMemcpyHostToDevice));

        blaze::cublas_gemm_fp8(d_A, d_B, d_C, s.M, s.N, s.K);
        CHECK_CUDA(cudaDeviceSynchronize());

        auto* h_C = new half[sC];
        CHECK_CUDA(cudaMemcpy(h_C, d_C, sC * sizeof(half), cudaMemcpyDeviceToHost));

        auto err = compute_error(ref_C, h_C, sC);

        // FP8 inputs lose precision in quantization. Expect higher error than FP16.
        float tol = 0.10f;
        bool ok = (err.nan_count == 0 && err.inf_count == 0 && err.max_rel < tol);
        if (ok) {
            printf("PASS (max_rel=%.4f, mean_rel=%.6f)\n", err.max_rel, err.mean_rel);
            passed++;
        } else {
            printf("FAIL (max_rel=%.4f, mean_rel=%.6f, nan=%d, inf=%d)\n",
                   err.max_rel, err.mean_rel, err.nan_count, err.inf_count);
        }

        delete[] h_A; delete[] h_B; delete[] h_C;
        delete[] ref_A; delete[] ref_B; delete[] ref_C;
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    printf("  FP8 cuBLAS: %d/%d passed\n", passed, num_shapes);
    return passed;
}

// ============================================================================
// Test 3: Mixed (FP16 × FP4)
// ============================================================================

static int test_cublas_mixed(TestShape* shapes, int num_shapes) {
    printf("\n=== cuBLAS Mixed GEMM Tests (FP16 x FP4) ===\n");
    int passed = 0;

    for (int t = 0; t < num_shapes; t++) {
        auto& s = shapes[t];
        // K and N must be multiples of FP4_BLOCK_SIZE (16) for block scales
        int K = s.K, N = s.N, M = s.M;
        printf("  [%d/%d] %s (M=%d, N=%d, K=%d)... ",
               t + 1, num_shapes, s.name, M, N, K);

        int sA = M * K, sC = M * N;
        int data_bytes = K * N / 2;
        int scale_count = K * (N / blaze::FP4_BLOCK_SIZE);

        // --- Host A (FP16) ---
        auto* h_A = new half[sA];
        init_random_fp16(h_A, sA, 42 + t);

        // --- Host B (FP4: packed data + block scales + tensor scale) ---
        auto* h_B_data = new uint8_t[data_bytes];
        auto* h_B_scales = new __nv_fp8_e4m3[scale_count];
        float tensor_scale = 0.25f;

        // Fill FP4 data with known pattern: all nibbles = 0x2 → E2M1 = 1.0
        memset(h_B_data, 0x22, data_bytes);  // Both nibbles = 0010 = 1.0

        // Block scales: all 1.0 (E4M3 0x38)
        for (int i = 0; i < scale_count; i++) {
            h_B_scales[i] = __nv_fp8_e4m3(1.0f);
        }

        // --- CPU reference ---
        // Dequantize B: each element = decode_e2m1(nibble) * block_scale * tensor_scale
        auto* ref_A = new float[sA];
        auto* ref_B = new float[K * N];
        auto* ref_C = new float[sC];

        for (int i = 0; i < sA; i++) ref_A[i] = __half2float(h_A[i]);

        for (int row = 0; row < K; row++) {
            for (int col = 0; col < N; col++) {
                int byte_idx = row * (N / 2) + col / 2;
                uint8_t packed = h_B_data[byte_idx];
                uint8_t nibble = (col & 1) ? (packed >> 4) : (packed & 0x0F);
                float val = decode_e2m1_host(nibble);

                int scale_idx = row * (N / blaze::FP4_BLOCK_SIZE) + col / blaze::FP4_BLOCK_SIZE;
                float bs = (float)h_B_scales[scale_idx];
                ref_B[row * N + col] = val * bs * tensor_scale;
            }
        }

        cpu_matmul_f32(ref_A, ref_B, ref_C, M, N, K);

        // --- GPU ---
        half *d_A, *d_C;
        uint8_t* d_B_data;
        __nv_fp8_e4m3* d_B_scales;
        CHECK_CUDA(cudaMalloc(&d_A, sA * sizeof(half)));
        CHECK_CUDA(cudaMalloc(&d_C, sC * sizeof(half)));
        CHECK_CUDA(cudaMalloc(&d_B_data, data_bytes));
        CHECK_CUDA(cudaMalloc(&d_B_scales, scale_count * sizeof(__nv_fp8_e4m3)));

        CHECK_CUDA(cudaMemcpy(d_A, h_A, sA * sizeof(half), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B_data, h_B_data, data_bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B_scales, h_B_scales,
                              scale_count * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));

        blaze::Fp4WeightTensor B_weight;
        B_weight.data = d_B_data;
        B_weight.block_scales = d_B_scales;
        B_weight.tensor_scale = tensor_scale;
        B_weight.rows = K;
        B_weight.cols = N;

        blaze::cublas_gemm_mixed(d_A, B_weight, d_C, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());

        auto* h_C = new half[sC];
        CHECK_CUDA(cudaMemcpy(h_C, d_C, sC * sizeof(half), cudaMemcpyDeviceToHost));

        auto err = compute_error(ref_C, h_C, sC);

        // Mixed path: dequant FP4→FP16 loses precision, then FP16 matmul.
        // With constant data=1.0, scale=1.0, tensor_scale=0.25:
        //   each B element = 0.25, C[m,n] = sum(A[m,k] * 0.25, k=0..K-1)
        // The FP16 accumulation for large K can lose precision.
        float tol = 0.05f;
        bool ok = (err.nan_count == 0 && err.inf_count == 0 && err.max_rel < tol);
        if (ok) {
            printf("PASS (max_rel=%.4f, mean_rel=%.6f)\n", err.max_rel, err.mean_rel);
            passed++;
        } else {
            printf("FAIL (max_rel=%.4f, mean_rel=%.6f, nan=%d, inf=%d)\n",
                   err.max_rel, err.mean_rel, err.nan_count, err.inf_count);
            // Print first few mismatches
            int printed = 0;
            for (int i = 0; i < sC && printed < 5; i++) {
                float r = ref_C[i];
                float g = __half2float(h_C[i]);
                float rel = fabsf(r - g) / (fabsf(r) + 1e-6f);
                if (rel > tol) {
                    printf("    [%d] ref=%.6f got=%.6f rel=%.4f\n", i, r, g, rel);
                    printed++;
                }
            }
        }

        delete[] h_A; delete[] h_B_data; delete[] h_B_scales; delete[] h_C;
        delete[] ref_A; delete[] ref_B; delete[] ref_C;
        cudaFree(d_A); cudaFree(d_C); cudaFree(d_B_data); cudaFree(d_B_scales);
    }

    printf("  Mixed cuBLAS: %d/%d passed\n", passed, num_shapes);
    return passed;
}

// ============================================================================
// Test 4: Plan-based API (create → execute × 2 → destroy)
// ============================================================================

static int test_plan_api() {
    printf("\n=== cuBLAS Plan API Tests ===\n");
    int passed = 0;

    int M = 128, N = 256, K = 128;
    int sA = M * K, sB = K * N, sC = M * N;

    // FP16 plan
    {
        printf("  FP16 plan (create → execute × 2 → destroy)... ");
        auto* h_A = new half[sA];
        auto* h_B = new half[sB];
        init_random_fp16(h_A, sA, 77);
        init_random_fp16(h_B, sB, 88);

        half *d_A, *d_B, *d_C1, *d_C2;
        CHECK_CUDA(cudaMalloc(&d_A, sA * sizeof(half)));
        CHECK_CUDA(cudaMalloc(&d_B, sB * sizeof(half)));
        CHECK_CUDA(cudaMalloc(&d_C1, sC * sizeof(half)));
        CHECK_CUDA(cudaMalloc(&d_C2, sC * sizeof(half)));
        CHECK_CUDA(cudaMemcpy(d_A, h_A, sA * sizeof(half), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B, sB * sizeof(half), cudaMemcpyHostToDevice));

        auto* plan = blaze::create_cublas_fp16_plan(d_A, d_B, M, N, K);
        blaze::execute_cublas_gemm(plan, d_C1);
        blaze::execute_cublas_gemm(plan, d_C2);
        CHECK_CUDA(cudaDeviceSynchronize());
        blaze::destroy_cublas_gemm_plan(plan);

        // Both outputs should be identical (same plan, same inputs)
        auto* h_C1 = new half[sC];
        auto* h_C2 = new half[sC];
        CHECK_CUDA(cudaMemcpy(h_C1, d_C1, sC * sizeof(half), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_C2, d_C2, sC * sizeof(half), cudaMemcpyDeviceToHost));

        bool match = true;
        for (int i = 0; i < sC; i++) {
            if (__half2float(h_C1[i]) != __half2float(h_C2[i])) {
                match = false; break;
            }
        }

        // Also check not all zeros
        bool nonzero = false;
        for (int i = 0; i < sC; i++) {
            if (__half2float(h_C1[i]) != 0.0f) { nonzero = true; break; }
        }

        if (match && nonzero) {
            printf("PASS (two executions match, nonzero)\n");
            passed++;
        } else {
            printf("FAIL (match=%d, nonzero=%d)\n", match, nonzero);
        }

        delete[] h_A; delete[] h_B; delete[] h_C1; delete[] h_C2;
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C1); cudaFree(d_C2);
    }

    printf("  Plan API: %d/1 passed\n", passed);
    return passed;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("=== cuBLAS GEMM Correctness Tests ===\n");

    blaze::cublas_init();

    int total_pass = 0, total_tests = 0;

    int n_small = sizeof(small_shapes) / sizeof(small_shapes[0]);
    int n_llama = sizeof(llama_shapes) / sizeof(llama_shapes[0]);

    // FP16: small + llama shapes
    printf("\n--- FP16: Small shapes ---");
    int p1a = test_cublas_fp16(small_shapes, n_small);
    printf("\n--- FP16: Llama shapes ---");
    int p1b = test_cublas_fp16(llama_shapes, n_llama);
    total_pass += p1a + p1b;
    total_tests += n_small + n_llama;

    // FP8: small shapes only (CPU ref is slow for large K)
    printf("\n--- FP8: Small shapes ---");
    int p2 = test_cublas_fp8(small_shapes, n_small);
    total_pass += p2;
    total_tests += n_small;

    // Mixed: small + llama shapes
    printf("\n--- Mixed: Small shapes ---");
    int p3a = test_cublas_mixed(small_shapes, n_small);
    printf("\n--- Mixed: Llama shapes ---");
    int p3b = test_cublas_mixed(llama_shapes, n_llama);
    total_pass += p3a + p3b;
    total_tests += n_small + n_llama;

    // Plan API
    int p4 = test_plan_api();
    total_pass += p4;
    total_tests += 1;

    printf("\n========================================\n");
    printf("Overall: %d/%d passed\n", total_pass, total_tests);
    printf("========================================\n");

    blaze::cublas_destroy();

    return (total_pass == total_tests) ? 0 : 1;
}
