/**
 * bench_gemm.cu — GEMM benchmarks for all Llama-7B shapes.
 *
 * Compares our FP8, FP4, and mixed-precision kernels against cuBLAS.
 * Reports TFLOPS, % of cuBLAS, and % of theoretical peak.
 */

#include "gemm/fp8_gemm_sm100.cuh"
#include "gemm/fp4_gemm_sm100.cuh"
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
constexpr double B200_PEAK_FP16_TFLOPS = 2250.0;   // 2.25 PFLOPS
constexpr double B200_PEAK_FP8_TFLOPS  = 4500.0;   // 4.5 PFLOPS
constexpr double B200_PEAK_FP4_TFLOPS  = 9000.0;   // 9 PFLOPS

struct BenchShape {
    int M, N, K;
    const char* precision;
    const char* name;
};

static BenchShape shapes[] = {
    // Decode (M=1)
    {1,    12288, 4096,  "fp4",  "decode QKV"},
    {1,    4096,  4096,  "fp4",  "decode out_proj"},
    {1,    22016, 4096,  "fp4",  "decode FFN gate+up"},
    {1,    4096,  11008, "fp4",  "decode FFN down"},
    // Small batch (M=32)
    {32,   12288, 4096,  "fp4",  "batch32 QKV"},
    {32,   4096,  11008, "fp4",  "batch32 FFN down"},
    // Medium batch (M=128)
    {128,  12288, 4096,  "fp4",  "batch128 QKV"},
    {128,  22016, 4096,  "fp4",  "batch128 FFN gate+up"},
    {128,  4096,  11008, "fp4",  "batch128 FFN down"},
    // Prefill (M=512)
    {512,  12288, 4096,  "fp4",  "prefill512 QKV"},
    {512,  4096,  11008, "fp4",  "prefill512 FFN down"},
    // Prefill (M=2048)
    {2048, 12288, 4096,  "fp4",  "prefill2048 QKV"},
    {2048, 4096,  11008, "fp4",  "prefill2048 FFN down"},
};

struct BenchResult {
    const char* name;
    int M, N, K;
    float our_ms;
    float cublas_ms;
    double our_tflops;
    double cublas_tflops;
    double pct_of_cublas;
    double pct_of_peak;
};

float benchmark_cublas_fp16(cublasHandle_t cublas,
                            half* d_A, half* d_B, half* d_C,
                            int M, int N, int K, int iters) {
    half alpha = __float2half(1.0f);
    half beta = __float2half(0.0f);

    // Warmup
    for (int i = 0; i < 3; i++) {
        cublasHgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        cublasHgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / iters;
}

float benchmark_mixed_gemm(const half* d_A, const blaze::Fp4WeightTensor& B,
                            half* d_C, int M, int N, int K, int iters) {
    // Warmup
    for (int i = 0; i < 3; i++) {
        blaze::launch_gemm_mixed(d_A, B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        blaze::launch_gemm_mixed(d_A, B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / iters;
}

int main() {
    printf("=== Blaze: GEMM Benchmarks (Llama-7B shapes on B200) ===\n\n");

    cublasHandle_t cublas;
    cublasCreate(&cublas);
    cublasSetMathMode(cublas, CUBLAS_TENSOR_OP_MATH);

    int n_shapes = sizeof(shapes) / sizeof(shapes[0]);
    BenchResult results[64];
    int bench_iters = 50;

    printf("%-30s %6s %6s %6s | %10s %10s | %8s %8s | %6s %6s\n",
           "Shape", "M", "N", "K",
           "Ours(ms)", "cuBLAS(ms)",
           "Ours TF", "cuBLAS TF",
           "%cuBLAS", "%Peak");
    printf("%s\n", std::string(110, '-').c_str());

    for (int s = 0; s < n_shapes; s++) {
        auto& sh = shapes[s];

        // Allocate
        half *d_A, *d_B_fp16, *d_C, *d_C_ref;
        CHECK_CUDA(cudaMalloc(&d_A, sh.M * sh.K * sizeof(half)));
        CHECK_CUDA(cudaMalloc(&d_B_fp16, sh.K * sh.N * sizeof(half)));
        CHECK_CUDA(cudaMalloc(&d_C, sh.M * sh.N * sizeof(half)));
        CHECK_CUDA(cudaMalloc(&d_C_ref, sh.M * sh.N * sizeof(half)));

        // FP4 weight
        int data_bytes = sh.K * sh.N / 2;
        int scale_count = sh.K * sh.N / blaze::FP4_BLOCK_SIZE;
        uint8_t* d_B_data;
        __nv_fp8_e4m3* d_B_scales;
        CHECK_CUDA(cudaMalloc(&d_B_data, data_bytes));
        CHECK_CUDA(cudaMalloc(&d_B_scales, scale_count));

        blaze::Fp4WeightTensor B_fp4 = {d_B_data, d_B_scales, 1.0f, sh.K, sh.N};

        // Benchmark cuBLAS (FP16)
        float cublas_ms = benchmark_cublas_fp16(cublas, d_A, d_B_fp16, d_C_ref,
                                                sh.M, sh.N, sh.K, bench_iters);

        // Benchmark our mixed GEMM
        float our_ms = benchmark_mixed_gemm(d_A, B_fp4, d_C,
                                            sh.M, sh.N, sh.K, bench_iters);

        double flops = 2.0 * sh.M * sh.N * sh.K;
        double our_tflops = flops / (our_ms * 1e-3) / 1e12;
        double cublas_tflops = flops / (cublas_ms * 1e-3) / 1e12;
        double pct_cublas = (cublas_tflops > 0) ? (our_tflops / cublas_tflops * 100.0) : 0;
        double pct_peak = our_tflops / B200_PEAK_FP4_TFLOPS * 100.0;

        results[s] = {sh.name, sh.M, sh.N, sh.K,
                     our_ms, cublas_ms,
                     our_tflops, cublas_tflops,
                     pct_cublas, pct_peak};

        printf("%-30s %6d %6d %6d | %10.4f %10.4f | %8.1f %8.1f | %5.1f%% %5.1f%%\n",
               sh.name, sh.M, sh.N, sh.K,
               our_ms, cublas_ms,
               our_tflops, cublas_tflops,
               pct_cublas, pct_peak);

        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B_fp16));
        CHECK_CUDA(cudaFree(d_C));
        CHECK_CUDA(cudaFree(d_C_ref));
        CHECK_CUDA(cudaFree(d_B_data));
        CHECK_CUDA(cudaFree(d_B_scales));
    }

    printf("\nTarget: >= 92%% of cuBLAS across all shapes, >= 95%% for large M\n");

    cublasDestroy(cublas);
    return 0;
}
