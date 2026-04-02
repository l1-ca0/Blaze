/**
 * cublas_gemm.cu — cuBLASLt-backed GEMM implementations.
 *
 * Provides competitive GEMM performance for all Blaze precision modes
 * using cuBLASLt, enabling Phase 2 development while custom kernels
 * are still being optimized.
 */

#include "gemm/cublas_gemm.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cublasLt.h>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace blaze {

// ============================================================================
// Error checking
// ============================================================================

#define CHECK_CUDA_(call) do {                                          \
    cudaError_t e = (call);                                             \
    if (e != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                cudaGetErrorString(e));                                  \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
} while (0)

#define CHECK_CUBLAS(call) do {                                          \
    cublasStatus_t s = (call);                                           \
    if (s != CUBLAS_STATUS_SUCCESS) {                                    \
        fprintf(stderr, "cuBLAS error %s:%d: status=%d\n",              \
                __FILE__, __LINE__, (int)s);                             \
        exit(EXIT_FAILURE);                                              \
    }                                                                    \
} while (0)

// ============================================================================
// Global handle
// ============================================================================

static cublasLtHandle_t g_handle = nullptr;
static std::mutex g_handle_mutex;

void cublas_init() {
    std::lock_guard<std::mutex> lock(g_handle_mutex);
    if (!g_handle) {
        CHECK_CUBLAS(cublasLtCreate(&g_handle));
    }
}

void cublas_destroy() {
    std::lock_guard<std::mutex> lock(g_handle_mutex);
    if (g_handle) {
        cublasLtDestroy(g_handle);
        g_handle = nullptr;
    }
}

cublasLtHandle_t cublas_get_handle() {
    return g_handle;
}

// ============================================================================
// Epilogue kernels
// ============================================================================

__global__ void epilogue_bias_kernel(
    half* __restrict__ C, const half* __restrict__ bias, int M, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        int col = idx % N;
        float val = __half2float(C[idx]) + __half2float(bias[col]);
        C[idx] = __float2half(val);
    }
}

__global__ void epilogue_silu_kernel(
    half* __restrict__ C, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __half2float(C[idx]);
        C[idx] = __float2half(x / (1.0f + expf(-x)));
    }
}

__global__ void epilogue_bias_silu_kernel(
    half* __restrict__ C, const half* __restrict__ bias, int M, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        int col = idx % N;
        float x = __half2float(C[idx]) + __half2float(bias[col]);
        C[idx] = __float2half(x / (1.0f + expf(-x)));
    }
}

__global__ void epilogue_residual_add_kernel(
    half* __restrict__ C, const half* __restrict__ residual, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __half2float(C[idx]) + __half2float(residual[idx]);
        C[idx] = __float2half(val);
    }
}

__global__ void silu_mul_kernel(
    const half* __restrict__ gate, const half* __restrict__ up,
    half* __restrict__ output, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = __half2float(gate[idx]);
        float u = __half2float(up[idx]);
        float silu_g = g / (1.0f + expf(-g));
        output[idx] = __float2half(silu_g * u);
    }
}

static void apply_epilogue_gemm(half* C, int M, int N,
                                const half* bias, GemmEpilogue epilogue,
                                cudaStream_t stream) {
    if (epilogue == GemmEpilogue::NONE) return;
    int total = M * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    switch (epilogue) {
        case GemmEpilogue::BIAS:
            epilogue_bias_kernel<<<blocks, threads, 0, stream>>>(C, bias, M, N);
            break;
        case GemmEpilogue::SILU:
            epilogue_silu_kernel<<<blocks, threads, 0, stream>>>(C, total);
            break;
        case GemmEpilogue::BIAS_SILU:
            epilogue_bias_silu_kernel<<<blocks, threads, 0, stream>>>(C, bias, M, N);
            break;
        default: break;
    }
}

static void apply_epilogue_mixed(half* C, int M, int N,
                                 const half* bias, const half* residual,
                                 MixedEpilogue epilogue, cudaStream_t stream) {
    if (epilogue == MixedEpilogue::NONE) return;
    int total = M * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    switch (epilogue) {
        case MixedEpilogue::BIAS:
            epilogue_bias_kernel<<<blocks, threads, 0, stream>>>(C, bias, M, N);
            break;
        case MixedEpilogue::SILU:
            epilogue_silu_kernel<<<blocks, threads, 0, stream>>>(C, total);
            break;
        case MixedEpilogue::BIAS_SILU:
            epilogue_bias_silu_kernel<<<blocks, threads, 0, stream>>>(C, bias, M, N);
            break;
        case MixedEpilogue::RESIDUAL_ADD:
            epilogue_residual_add_kernel<<<blocks, threads, 0, stream>>>(C, residual, total);
            break;
        default: break;
    }
}

// ============================================================================
// FP4 → FP16 dequantization kernel
// ============================================================================

/**
 * Dequantize NVFP4 (E2M1 + block scales + tensor scale) → FP16.
 *
 * Each thread handles one output element:
 *   output[row, col] = unpack_e2m1(data[row, col]) * block_scales[row, col/16] * tensor_scale
 *
 * E2M1 nibble decoding:
 *   bit3 = sign (0=pos for unsigned E2M1, all values positive)
 *   bits 2-0: 000=0.0, 001=0.5, 010=1.0, 011=1.5, 100=2.0, 101=3.0, 110=4.0, 111=6.0
 */
__device__ __forceinline__ float decode_e2m1(uint8_t nibble) {
    // NVFP4 E2M1 is SIGNED: bit 3 = sign, bits 2-0 = magnitude.
    // Magnitude LUT: 000=0.0, 001=0.5, 010=1.0, 011=1.5, 100=2.0, 101=3.0, 110=4.0, 111=6.0
    static constexpr float lut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float mag = lut[nibble & 0x7];
    return (nibble & 0x8) ? -mag : mag;
}

__global__ void dequant_fp4_to_fp16_kernel(
    const uint8_t* __restrict__ data,               // Packed E2M1 [rows, cols/2]
    const __nv_fp8_e4m3* __restrict__ block_scales,  // [rows, cols/BLOCK_SIZE]
    float tensor_scale,
    half* __restrict__ output,                       // [rows, cols]
    int rows, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) return;

    int row = idx / cols;
    int col = idx % cols;

    // Unpack E2M1 nibble
    int byte_idx = row * (cols / 2) + col / 2;
    uint8_t packed = data[byte_idx];
    uint8_t nibble = (col & 1) ? (packed >> 4) : (packed & 0x0F);
    float val = decode_e2m1(nibble);

    // Apply block scale
    int scale_col = col / FP4_BLOCK_SIZE;
    int scale_idx = row * (cols / FP4_BLOCK_SIZE) + scale_col;
    float bs = static_cast<float>(block_scales[scale_idx]);

    // Apply tensor scale and store as FP16
    output[idx] = __float2half(val * bs * tensor_scale);
}

static void dequant_fp4_to_fp16(
    const Fp4WeightTensor& tensor, half* output, cudaStream_t stream
) {
    int total = tensor.rows * tensor.cols;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    dequant_fp4_to_fp16_kernel<<<blocks, threads, 0, stream>>>(
        tensor.data, tensor.block_scales, tensor.tensor_scale,
        output, tensor.rows, tensor.cols
    );
}

// ============================================================================
// Internal plan struct
// ============================================================================

enum class CublasPrecision { FP16, FP8, MIXED };

struct CublasGemmPlan {
    CublasPrecision precision;

    // cuBLASLt descriptors
    cublasLtMatmulDesc_t matmul_desc;
    cublasLtMatrixLayout_t layout_A;
    cublasLtMatrixLayout_t layout_B;
    cublasLtMatrixLayout_t layout_C;
    cublasLtMatmulPreference_t preference;
    cublasLtMatmulHeuristicResult_t heuristic;

    // Workspace
    void* workspace;
    size_t workspace_size;

    // Input pointers (may point to converted/dequantized buffers)
    const void* A_ptr;
    const void* B_ptr;

    // FP8 per-tensor scale (device pointer, cublasLt requires device scalars)
    float* d_alpha;  // Points to device float = 1.0 (used for A_scale and B_scale)

    // Dequantized weight buffer (for mixed precision)
    half* B_dequant;

    // Original FP4 weight (for on-the-fly dequant in quick-launch path)
    Fp4WeightTensor fp4_weight;
    bool has_fp4_weight;

    // Epilogue state
    GemmEpilogue gemm_epilogue;
    MixedEpilogue mixed_epilogue;
    const half* bias;
    const half* residual;

    // Dimensions
    int M, N, K;
};

// ============================================================================
// Helper: create cuBLASLt descriptors
// ============================================================================

/**
 * Create a matmul descriptor + layouts for a row-major GEMM.
 *
 * cuBLASLt uses column-major by default. For row-major A[M,K] × B[K,N] = C[M,N],
 * we compute it as: C^T = B^T × A^T in column-major, which is equivalent to
 * swapping A and B and transposing the result.
 *
 * Actually, the standard trick:
 *   Row-major C = A × B  ↔  Col-major C = B^T × A^T  (no actual transpose needed)
 *
 * So we pass B as the first matrix and A as the second, both with column-major
 * layout where leading dimension = number of columns in row-major.
 */
static CublasGemmPlan* create_plan_internal(
    const void* A_ptr, cudaDataType_t A_type,
    const void* B_ptr, cudaDataType_t B_type,
    cudaDataType_t C_type,
    cublasComputeType_t compute_type,
    int M, int N, int K,
    CublasPrecision precision
) {
    auto* plan = new CublasGemmPlan{};
    plan->precision = precision;
    plan->M = M;
    plan->N = N;
    plan->K = K;
    plan->A_ptr = A_ptr;
    plan->B_ptr = B_ptr;
    plan->B_dequant = nullptr;
    plan->has_fp4_weight = false;
    plan->gemm_epilogue = GemmEpilogue::NONE;
    plan->mixed_epilogue = MixedEpilogue::NONE;
    plan->bias = nullptr;
    plan->residual = nullptr;
    plan->d_alpha = nullptr;

    cublasLtHandle_t handle = cublas_get_handle();

    // Matmul descriptor
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&plan->matmul_desc, compute_type, CUDA_R_32F));

    // Row-major trick: swap A/B, set transA=N, transB=N
    // Row-major: C[M,N] = A[M,K] × B[K,N]
    // Equivalent col-major: C_col[N,M] = B_col[N,K] × A_col[K,M]
    // where X_col is X interpreted as col-major with lda = num_cols_in_row_major
    cublasOperation_t op_N = CUBLAS_OP_N;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        plan->matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA,
        &op_N, sizeof(op_N)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        plan->matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB,
        &op_N, sizeof(op_N)));

    // For FP8: cuBLASLt requires per-tensor scale factors on device.
    // A_scale and B_scale dequantize the FP8 inputs: A_fp32 = A_fp8 * A_scale.
    // D_scale scales the output: D_fp8 = D_fp32 / D_scale (only if D is FP8).
    // Since our output is FP16, we don't need D_scale.
    // We set A_scale = B_scale = 1.0 (our FP8 data is already at correct scale).
    if (precision == CublasPrecision::FP8) {
        // FP8 cuBLASLt requires per-tensor scale factors as device pointers.
        // A_scale = B_scale = 1.0 (our FP8 data is pre-scaled).
        CHECK_CUDA_(cudaMalloc(&plan->d_alpha, sizeof(float)));
        float h_one = 1.0f;
        CHECK_CUDA_(cudaMemcpy(plan->d_alpha, &h_one, sizeof(float), cudaMemcpyHostToDevice));

        // Note: "A" in matmul desc = our B matrix (row-major swap), but both scales are 1.0.
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
            plan->matmul_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            &plan->d_alpha, sizeof(plan->d_alpha)));
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
            plan->matmul_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            &plan->d_alpha, sizeof(plan->d_alpha)));
    }

    // Layout descriptors (column-major interpretation of row-major data)
    // "First matrix" = B (N rows, K cols in col-major), ld = N
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&plan->layout_A, B_type, N, K, N));
    // "Second matrix" = A (K rows, M cols in col-major), ld = K
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&plan->layout_B, A_type, K, M, K));
    // Output = C (N rows, M cols in col-major), ld = N
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&plan->layout_C, C_type, N, M, N));

    // Algorithm selection via heuristic
    plan->workspace_size = 4 * 1024 * 1024;  // 4 MB workspace
    CHECK_CUDA_(cudaMalloc(&plan->workspace, plan->workspace_size));

    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&plan->preference));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
        plan->preference, CUBLASLT_MATMUL_PREFERENCE_MAX_WORKSPACE_BYTES,
        &plan->workspace_size, sizeof(plan->workspace_size)));

    int returned_results = 0;
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
        handle, plan->matmul_desc,
        plan->layout_A, plan->layout_B, plan->layout_C, plan->layout_C,
        plan->preference, 1, &plan->heuristic, &returned_results));

    if (returned_results == 0) {
        fprintf(stderr, "cuBLASLt: no algorithm found for M=%d N=%d K=%d\n", M, N, K);
        exit(EXIT_FAILURE);
    }

    return plan;
}

// ============================================================================
// Execute helper
// ============================================================================

static void execute_cublas_matmul(CublasGemmPlan* plan, half* C, cudaStream_t stream) {
    cublasLtHandle_t handle = cublas_get_handle();

    // alpha/beta are always host-side float for the matmul itself.
    // (FP8 per-tensor scales are set on the descriptor, separate from alpha/beta.)
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasLtMatmul(
        handle, plan->matmul_desc,
        &alpha,
        plan->B_ptr, plan->layout_A,      // "A" = B (row-major trick)
        plan->A_ptr, plan->layout_B,      // "B" = A
        &beta,
        C, plan->layout_C,
        C, plan->layout_C,
        &plan->heuristic.algo,
        plan->workspace, plan->workspace_size,
        stream));
}

// ============================================================================
// Quick-launch APIs
// ============================================================================

void cublas_gemm_fp16(
    const half* A, const half* B, half* C,
    int M, int N, int K,
    const half* bias, GemmEpilogue epilogue, cudaStream_t stream
) {
    auto* plan = create_cublas_fp16_plan(A, B, M, N, K, bias, epilogue);
    execute_cublas_gemm(plan, C, stream);
    destroy_cublas_gemm_plan(plan);
}

void cublas_gemm_fp8(
    const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* B, half* C,
    int M, int N, int K,
    const half* bias, GemmEpilogue epilogue, cudaStream_t stream
) {
    auto* plan = create_cublas_fp8_plan(A, B, M, N, K, bias, epilogue);
    execute_cublas_gemm(plan, C, stream);
    destroy_cublas_gemm_plan(plan);
}

void cublas_gemm_mixed(
    const half* A, const Fp4WeightTensor& B, half* C,
    int M, int N, int K,
    const half* bias, const half* residual,
    MixedEpilogue epilogue, cudaStream_t stream
) {
    auto* plan = create_cublas_mixed_plan(A, B, M, N, K, bias, residual, epilogue);
    execute_cublas_gemm(plan, C, stream);
    destroy_cublas_gemm_plan(plan);
}

void cublas_fused_gate_up(
    const half* x,
    const Fp4WeightTensor& W_gate,
    const Fp4WeightTensor& W_up,
    half* output,
    int M, int N, int K,
    cudaStream_t stream
) {
    // Allocate temporaries for gate and up results
    half *d_gate, *d_up;
    CHECK_CUDA_(cudaMalloc(&d_gate, (size_t)M * N * sizeof(half)));
    CHECK_CUDA_(cudaMalloc(&d_up, (size_t)M * N * sizeof(half)));

    // Two GEMMs
    cublas_gemm_mixed(x, W_gate, d_gate, M, N, K,
                      nullptr, nullptr, MixedEpilogue::NONE, stream);
    cublas_gemm_mixed(x, W_up, d_up, M, N, K,
                      nullptr, nullptr, MixedEpilogue::NONE, stream);

    // Fused SiLU(gate) * up
    int total = M * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    silu_mul_kernel<<<blocks, threads, 0, stream>>>(d_gate, d_up, output, total);

    // Must sync before freeing — silu_mul_kernel may still be in flight on stream
    CHECK_CUDA_(cudaStreamSynchronize(stream));
    cudaFree(d_gate);
    cudaFree(d_up);
}

// ============================================================================
// Plan-based API: create
// ============================================================================

CublasGemmPlan* create_cublas_fp16_plan(
    const half* A, const half* B,
    int M, int N, int K,
    const half* bias, GemmEpilogue epilogue
) {
    // Use CUBLAS_COMPUTE_32F (not 16F) for two reasons:
    // 1. FP32 accumulation gives better precision for large K
    // 2. Allows float alpha/beta (COMPUTE_16F requires half alpha/beta)
    auto* plan = create_plan_internal(
        A, CUDA_R_16F,
        B, CUDA_R_16F,
        CUDA_R_16F,
        CUBLAS_COMPUTE_32F,
        M, N, K,
        CublasPrecision::FP16
    );
    plan->bias = bias;
    plan->gemm_epilogue = epilogue;
    return plan;
}

CublasGemmPlan* create_cublas_fp8_plan(
    const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* B,
    int M, int N, int K,
    const half* bias, GemmEpilogue epilogue
) {
    auto* plan = create_plan_internal(
        A, CUDA_R_8F_E4M3,
        B, CUDA_R_8F_E4M3,
        CUDA_R_16F,
        CUBLAS_COMPUTE_32F,
        M, N, K,
        CublasPrecision::FP8
    );
    plan->bias = bias;
    plan->gemm_epilogue = epilogue;
    return plan;
}

CublasGemmPlan* create_cublas_mixed_plan(
    const half* A, const Fp4WeightTensor& B,
    int M, int N, int K,
    const half* bias, const half* residual, MixedEpilogue epilogue
) {
    // Dequantize FP4 weights to FP16 (one-time cost, amortized across executions)
    half* B_dequant;
    CHECK_CUDA_(cudaMalloc(&B_dequant, (size_t)B.rows * B.cols * sizeof(half)));
    dequant_fp4_to_fp16(B, B_dequant, 0);
    CHECK_CUDA_(cudaDeviceSynchronize());

    auto* plan = create_plan_internal(
        A, CUDA_R_16F,
        B_dequant, CUDA_R_16F,
        CUDA_R_16F,
        CUBLAS_COMPUTE_32F,
        M, N, K,
        CublasPrecision::MIXED
    );
    plan->B_dequant = B_dequant;
    plan->fp4_weight = B;
    plan->has_fp4_weight = true;
    plan->bias = bias;
    plan->residual = residual;
    plan->mixed_epilogue = epilogue;
    return plan;
}

// ============================================================================
// Plan-based API: execute
// ============================================================================

void execute_cublas_gemm(CublasGemmPlan* plan, half* C, cudaStream_t stream) {
    execute_cublas_matmul(plan, C, stream);

    // Apply epilogue
    if (plan->precision == CublasPrecision::MIXED) {
        apply_epilogue_mixed(C, plan->M, plan->N,
                             plan->bias, plan->residual,
                             plan->mixed_epilogue, stream);
    } else {
        apply_epilogue_gemm(C, plan->M, plan->N,
                            plan->bias, plan->gemm_epilogue, stream);
    }
}

// ============================================================================
// Plan-based API: destroy
// ============================================================================

void destroy_cublas_gemm_plan(CublasGemmPlan* plan) {
    if (!plan) return;

    cublasLtMatmulPreferenceDestroy(plan->preference);
    cublasLtMatrixLayoutDestroy(plan->layout_C);
    cublasLtMatrixLayoutDestroy(plan->layout_B);
    cublasLtMatrixLayoutDestroy(plan->layout_A);
    cublasLtMatmulDescDestroy(plan->matmul_desc);

    if (plan->workspace) cudaFree(plan->workspace);
    if (plan->B_dequant) cudaFree(plan->B_dequant);
    if (plan->d_alpha) cudaFree(plan->d_alpha);

    delete plan;
}

}  // namespace blaze
