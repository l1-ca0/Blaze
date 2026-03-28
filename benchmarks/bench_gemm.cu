/**
 * bench_gemm.cu — GEMM kernel-only benchmarks for Llama-7B shapes.
 *
 * Uses the prepare/execute API to pre-allocate all workspace, TMA
 * descriptors, and data conversions before timing. The hot loop measures
 * only kernel launch and execution — no allocations or setup overhead.
 *
 * Benchmarks mixed-precision (FP16×FP4) and FP8 (E4M3×E4M3) kernels
 * against cuBLAS FP16 as baseline.
 */

#include "gemm/fp8_gemm_sm100.cuh"
#include "gemm/fp4_gemm_sm100.cuh"
#include "gemm/fp4_blkscaled_gemm_sm100.cuh"
#include "gemm/mixed_gemm_sm100.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cublas_v2.h>
#include <string>
#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call) do { cudaError_t e=(call); if(e) { fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1); } } while(0)

// B200 theoretical peaks (TFLOPS)
constexpr double B200_PEAK_FP16_TFLOPS = 2250.0;
constexpr double B200_PEAK_FP8_TFLOPS  = 4500.0;
constexpr double B200_PEAK_FP4_TFLOPS  = 9000.0;

struct BenchShape {
    int M, N, K;
    const char* name;
};

static BenchShape shapes[] = {
    // Decode (M=1)
    {1,    12288, 4096,  "decode QKV"},
    {1,    4096,  4096,  "decode out_proj"},
    {1,    22016, 4096,  "decode FFN gate+up"},
    {1,    4096,  11008, "decode FFN down"},
    // Small batch (M=32)
    {32,   12288, 4096,  "batch32 QKV"},
    {32,   4096,  11008, "batch32 FFN down"},
    // Medium batch (M=128)
    {128,  12288, 4096,  "batch128 QKV"},
    {128,  22016, 4096,  "batch128 FFN gate+up"},
    {128,  4096,  11008, "batch128 FFN down"},
    // Prefill (M=512)
    {512,  12288, 4096,  "prefill512 QKV"},
    {512,  4096,  11008, "prefill512 FFN down"},
    // Prefill (M=2048)
    {2048, 12288, 4096,  "prefill2048 QKV"},
    {2048, 4096,  11008, "prefill2048 FFN down"},
};

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

float time_cublas_fp16(cublasHandle_t cublas,
                       half* d_A, half* d_B, half* d_C,
                       int M, int N, int K, int warmup, int iters) {
    half alpha = __float2half(1.0f);
    half beta = __float2half(0.0f);

    for (int i = 0; i < warmup; i++)
        cublasHgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; i++)
        cublasHgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / iters;
}

float time_mixed_gemm(blaze::MixedGemmPlan* plan, half* d_C,
                       int warmup, int iters) {
    for (int i = 0; i < warmup; i++)
        blaze::execute_mixed_gemm(plan, d_C);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; i++)
        blaze::execute_mixed_gemm(plan, d_C);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / iters;
}

float time_fp8_gemm(blaze::Fp8GemmPlan* plan, half* d_C,
                     int warmup, int iters) {
    for (int i = 0; i < warmup; i++)
        blaze::execute_fp8_gemm(plan, d_C);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; i++)
        blaze::execute_fp8_gemm(plan, d_C);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / iters;
}

// ---------------------------------------------------------------------------

void print_header(const char* section) {
    printf("\n%-30s %6s %6s %6s | %10s %10s | %8s %8s | %6s %6s\n",
           section, "M", "N", "K",
           "Ours(ms)", "cuBLAS(ms)",
           "Ours TF", "cuBLAS TF",
           "%cuBLAS", "%Peak");
    printf("%s\n", std::string(110, '-').c_str());
}

void print_row(const char* name, int M, int N, int K,
               float our_ms, float cublas_ms,
               double our_tflops, double cublas_tflops,
               double pct_cublas, double pct_peak) {
    printf("%-30s %6d %6d %6d | %10.4f %10.4f | %8.1f %8.1f | %5.1f%% %5.1f%%\n",
           name, M, N, K, our_ms, cublas_ms,
           our_tflops, cublas_tflops, pct_cublas, pct_peak);
}

int main() {
    printf("=== Blaze: GEMM Kernel-Only Benchmarks (Llama-7B shapes on B200) ===\n");

    cublasHandle_t cublas;
    cublasCreate(&cublas);
    cublasSetMathMode(cublas, CUBLAS_TENSOR_OP_MATH);

    int n_shapes = sizeof(shapes) / sizeof(shapes[0]);
    int warmup = 10;
    int iters = 100;

    // Pre-allocate the largest buffers once and reuse.
    int max_M = 0, max_N = 0, max_K = 0;
    for (int i = 0; i < n_shapes; i++) {
        if (shapes[i].M > max_M) max_M = shapes[i].M;
        if (shapes[i].N > max_N) max_N = shapes[i].N;
        if (shapes[i].K > max_K) max_K = shapes[i].K;
    }

    // Shared output and cuBLAS buffers (allocated once)
    half *d_C, *d_C_ref, *d_A_fp16, *d_B_fp16;
    CHECK_CUDA(cudaMalloc(&d_C,       (size_t)max_M * max_N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_C_ref,   (size_t)max_M * max_N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_A_fp16,  (size_t)max_M * max_K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B_fp16,  (size_t)max_K * max_N * sizeof(half)));

    // FP8 buffers
    __nv_fp8_e4m3 *d_A_fp8, *d_B_fp8;
    CHECK_CUDA(cudaMalloc(&d_A_fp8,  (size_t)max_M * max_K));
    CHECK_CUDA(cudaMalloc(&d_B_fp8,  (size_t)max_K * max_N));

    // FP4 weight buffers
    int max_data_bytes = max_K * max_N / 2;
    int max_scale_count = max_K * max_N / blaze::FP4_BLOCK_SIZE;
    uint8_t* d_B_fp4_data;
    __nv_fp8_e4m3* d_B_fp4_scales;
    CHECK_CUDA(cudaMalloc(&d_B_fp4_data, max_data_bytes));
    CHECK_CUDA(cudaMalloc(&d_B_fp4_scales, max_scale_count));

    // -----------------------------------------------------------------------
    // Mixed-precision GEMM (FP16 × FP4)
    // -----------------------------------------------------------------------
    print_header("Mixed GEMM (FP16 x FP4)");

    for (int s = 0; s < n_shapes; s++) {
        auto& sh = shapes[s];

        blaze::Fp4WeightTensor B_fp4 = {
            d_B_fp4_data, d_B_fp4_scales, 1.0f, sh.K, sh.N
        };

        // Create plan (all setup done here, outside timing)
        auto* plan = blaze::create_mixed_gemm_plan(
            d_A_fp16, B_fp4, sh.M, sh.N, sh.K);

        // Time kernel only
        float our_ms = time_mixed_gemm(plan, d_C, warmup, iters);
        float cublas_ms = time_cublas_fp16(cublas, d_A_fp16, d_B_fp16, d_C_ref,
                                            sh.M, sh.N, sh.K, warmup, iters);

        double flops = 2.0 * sh.M * sh.N * sh.K;
        double our_tflops = flops / (our_ms * 1e-3) / 1e12;
        double cublas_tflops = flops / (cublas_ms * 1e-3) / 1e12;
        double pct_cublas = (cublas_tflops > 0) ? (our_tflops / cublas_tflops * 100.0) : 0;
        double pct_peak = our_tflops / B200_PEAK_FP4_TFLOPS * 100.0;

        print_row(sh.name, sh.M, sh.N, sh.K,
                  our_ms, cublas_ms, our_tflops, cublas_tflops,
                  pct_cublas, pct_peak);

        blaze::destroy_mixed_gemm_plan(plan);
    }

    // -----------------------------------------------------------------------
    // FP8 GEMM (E4M3 × E4M3)
    // -----------------------------------------------------------------------
    print_header("FP8 GEMM (E4M3 x E4M3)");

    for (int s = 0; s < n_shapes; s++) {
        auto& sh = shapes[s];

        auto* plan = blaze::create_fp8_gemm_plan(
            d_A_fp8, d_B_fp8, sh.M, sh.N, sh.K);

        float our_ms = time_fp8_gemm(plan, d_C, warmup, iters);
        float cublas_ms = time_cublas_fp16(cublas, d_A_fp16, d_B_fp16, d_C_ref,
                                            sh.M, sh.N, sh.K, warmup, iters);

        double flops = 2.0 * sh.M * sh.N * sh.K;
        double our_tflops = flops / (our_ms * 1e-3) / 1e12;
        double cublas_tflops = flops / (cublas_ms * 1e-3) / 1e12;
        double pct_cublas = (cublas_tflops > 0) ? (our_tflops / cublas_tflops * 100.0) : 0;
        double pct_peak = our_tflops / B200_PEAK_FP8_TFLOPS * 100.0;

        print_row(sh.name, sh.M, sh.N, sh.K,
                  our_ms, cublas_ms, our_tflops, cublas_tflops,
                  pct_cublas, pct_peak);

        blaze::destroy_fp8_gemm_plan(plan);
    }

    // -----------------------------------------------------------------------
    // FP4 GEMM (E2M1 × E2M1)
    // -----------------------------------------------------------------------
    print_header("FP4 GEMM (E2M1 x E2M1)");

    // FP4 A weight buffers (max size with K aligned to 128)
    int max_K_aligned = ((max_K + 127) / 128) * 128;
    int max_a_data_bytes = max_M * max_K_aligned / 2;
    int max_a_scale_count = max_M * max_K_aligned / blaze::FP4_BLOCK_SIZE;
    int max_b_fp4_data_bytes = max_K_aligned * max_N / 2;
    int max_b_fp4_scale_count = max_K_aligned * max_N / blaze::FP4_BLOCK_SIZE;

    uint8_t* d_A_fp4_data;
    __nv_fp8_e4m3* d_A_fp4_scales;
    uint8_t* d_B_fp4_data_aligned;
    __nv_fp8_e4m3* d_B_fp4_scales_aligned;
    CHECK_CUDA(cudaMalloc(&d_A_fp4_data, max_a_data_bytes));
    CHECK_CUDA(cudaMalloc(&d_A_fp4_scales, max_a_scale_count));
    CHECK_CUDA(cudaMalloc(&d_B_fp4_data_aligned, max_b_fp4_data_bytes));
    CHECK_CUDA(cudaMalloc(&d_B_fp4_scales_aligned, max_b_fp4_scale_count));

    for (int s = 0; s < n_shapes; s++) {
        auto& sh = shapes[s];
        int K_aligned = ((sh.K + 127) / 128) * 128;

        blaze::Fp4WeightTensor A_fp4 = {
            d_A_fp4_data, d_A_fp4_scales, 1.0f, sh.M, K_aligned
        };
        blaze::Fp4WeightTensor B_fp4_a = {
            d_B_fp4_data_aligned, d_B_fp4_scales_aligned, 1.0f, K_aligned, sh.N
        };

        auto* plan = blaze::create_fp4_gemm_plan(
            A_fp4, B_fp4_a, sh.M, sh.N, K_aligned);

        // Time kernel only
        for (int i = 0; i < warmup; i++)
            blaze::execute_fp4_gemm(plan, d_C);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < iters; i++)
            blaze::execute_fp4_gemm(plan, d_C);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float our_ms;
        CHECK_CUDA(cudaEventElapsedTime(&our_ms, start, stop));
        our_ms /= iters;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        float cublas_ms = time_cublas_fp16(cublas, d_A_fp16, d_B_fp16, d_C_ref,
                                            sh.M, sh.N, sh.K, warmup, iters);

        double flops = 2.0 * sh.M * sh.N * K_aligned;
        double our_tflops = flops / (our_ms * 1e-3) / 1e12;
        double cublas_tflops = flops / (cublas_ms * 1e-3) / 1e12;
        double pct_cublas = (cublas_tflops > 0) ? (our_tflops / cublas_tflops * 100.0) : 0;
        double pct_peak = our_tflops / B200_PEAK_FP4_TFLOPS * 100.0;

        print_row(sh.name, sh.M, sh.N, K_aligned,
                  our_ms, cublas_ms, our_tflops, cublas_tflops,
                  pct_cublas, pct_peak);

        blaze::destroy_fp4_gemm_plan(plan);
    }

    // -----------------------------------------------------------------------
    // FP4 Block-Scaled GEMM (E2M1 × E2M1, hardware block scale application)
    // -----------------------------------------------------------------------
    print_header("FP4 BlkScaled GEMM");

    for (int s = 0; s < n_shapes; s++) {
        auto& sh = shapes[s];
        int K_aligned = ((sh.K + 127) / 128) * 128;

        blaze::Fp4BlkScaledWeightTensor A_fp4_bs = {
            d_A_fp4_data, d_A_fp4_scales,
            1.0f, sh.M, K_aligned
        };
        blaze::Fp4BlkScaledWeightTensor B_fp4_bs = {
            d_B_fp4_data_aligned, d_B_fp4_scales_aligned,
            1.0f, K_aligned, sh.N
        };

        auto* plan = blaze::create_fp4_blkscaled_gemm_plan(
            A_fp4_bs, B_fp4_bs, sh.M, sh.N, K_aligned);

        // Time kernel only
        for (int i = 0; i < warmup; i++)
            blaze::execute_fp4_blkscaled_gemm(plan, d_C);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < iters; i++)
            blaze::execute_fp4_blkscaled_gemm(plan, d_C);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float our_ms;
        CHECK_CUDA(cudaEventElapsedTime(&our_ms, start, stop));
        our_ms /= iters;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        float cublas_ms = time_cublas_fp16(cublas, d_A_fp16, d_B_fp16, d_C_ref,
                                            sh.M, sh.N, sh.K, warmup, iters);

        double flops = 2.0 * sh.M * sh.N * K_aligned;
        double our_tflops = flops / (our_ms * 1e-3) / 1e12;
        double cublas_tflops = flops / (cublas_ms * 1e-3) / 1e12;
        double pct_cublas = (cublas_tflops > 0) ? (our_tflops / cublas_tflops * 100.0) : 0;
        double pct_peak = our_tflops / B200_PEAK_FP4_TFLOPS * 100.0;

        print_row(sh.name, sh.M, sh.N, K_aligned,
                  our_ms, cublas_ms, our_tflops, cublas_tflops,
                  pct_cublas, pct_peak);

        blaze::destroy_fp4_blkscaled_gemm_plan(plan);
    }

    // -----------------------------------------------------------------------
    // FP4 Block-Scaled Persistent GEMM
    // -----------------------------------------------------------------------
    print_header("FP4 BlkScaled Persistent GEMM");

    for (int s = 0; s < n_shapes; s++) {
        const auto& sh = shapes[s];
        int K_aligned = ((sh.K + 127) / 128) * 128;

        blaze::Fp4BlkScaledWeightTensor A_fp4_bs_p = {
            d_A_fp4_data, d_A_fp4_scales,
            1.0f, sh.M, K_aligned
        };
        blaze::Fp4BlkScaledWeightTensor B_fp4_bs_p = {
            d_B_fp4_data_aligned, d_B_fp4_scales_aligned,
            1.0f, K_aligned, sh.N
        };

        auto* plan_p = blaze::create_fp4_blkscaled_persistent_gemm_plan(
            A_fp4_bs_p, B_fp4_bs_p, sh.M, sh.N, K_aligned);

        // Time kernel only
        for (int i = 0; i < warmup; i++)
            blaze::execute_fp4_blkscaled_persistent_gemm(plan_p, d_C);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < iters; i++)
            blaze::execute_fp4_blkscaled_persistent_gemm(plan_p, d_C);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        float avg_ms = ms / iters;

        float cublas_ms = time_cublas_fp16(cublas, d_A_fp16, d_B_fp16, d_C_ref,
                                            sh.M, sh.N, sh.K, warmup, iters);

        double flops = 2.0 * sh.M * sh.N * K_aligned;
        double our_tflops = flops / (avg_ms * 1e-3) / 1e12;
        double cublas_tflops = flops / (cublas_ms * 1e-3) / 1e12;
        double pct_cublas = (cublas_tflops > 0) ? (our_tflops / cublas_tflops * 100.0) : 0;
        double pct_peak = our_tflops / B200_PEAK_FP4_TFLOPS * 100.0;

        print_row(sh.name, sh.M, sh.N, K_aligned,
                  avg_ms, cublas_ms, our_tflops, cublas_tflops,
                  pct_cublas, pct_peak);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        blaze::destroy_fp4_blkscaled_persistent_gemm_plan(plan_p);
    }

    CHECK_CUDA(cudaFree(d_A_fp4_data));
    CHECK_CUDA(cudaFree(d_A_fp4_scales));
    CHECK_CUDA(cudaFree(d_B_fp4_data_aligned));
    CHECK_CUDA(cudaFree(d_B_fp4_scales_aligned));

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_C_ref));
    CHECK_CUDA(cudaFree(d_A_fp16));
    CHECK_CUDA(cudaFree(d_B_fp16));
    CHECK_CUDA(cudaFree(d_A_fp8));
    CHECK_CUDA(cudaFree(d_B_fp8));
    CHECK_CUDA(cudaFree(d_B_fp4_data));
    CHECK_CUDA(cudaFree(d_B_fp4_scales));

    cublasDestroy(cublas);

    return 0;
}
