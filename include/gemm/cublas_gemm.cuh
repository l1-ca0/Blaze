#pragma once
/**
 * cublas_gemm.cuh — cuBLAS-backed GEMM for all Blaze precision modes.
 *
 * Provides drop-in replacements for the custom tcgen05-based GEMM kernels
 * using cuBLASLt, so Phase 2 (attention, runtime) can use competitive GEMM
 * performance while custom kernels continue to be optimized.
 *
 * Supported operations:
 *   1. FP16 × FP16 → FP16           (cublasLtMatmul, FP16 compute)
 *   2. FP8  × FP8  → FP16           (cublasLtMatmul, E4M3 inputs, FP32 compute)
 *   3. Mixed: FP16 × FP4 → FP16     (dequant FP4→FP16 + cublasLtMatmul)
 *
 * All functions match the existing Blaze API signatures so model_runner.cu
 * can switch backends by changing a single include + function name.
 *
 * Epilogue operations (BIAS, SILU, BIAS_SILU, RESIDUAL_ADD) are applied
 * via lightweight post-GEMM kernels — cuBLASLt's built-in epilogues only
 * cover bias addition, not SiLU.
 */

#include "gemm/fp4_gemm_sm100.cuh"      // Fp4WeightTensor, Fp4Epilogue
#include "gemm/fp8_gemm_sm100.cuh"      // GemmEpilogue
#include "gemm/mixed_gemm_sm100.cuh"    // MixedEpilogue

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cublasLt.h>
#include <cstdint>

namespace blaze {

// ============================================================================
// Global cuBLAS handle management
// ============================================================================

/**
 * Initialize the global cuBLASLt handle. Call once at startup.
 * Thread-safe: uses internal mutex.
 */
void cublas_init();

/**
 * Destroy the global cuBLASLt handle. Call at shutdown.
 */
void cublas_destroy();

/**
 * Get the global cuBLASLt handle.
 * cublas_init() must have been called first.
 */
cublasLtHandle_t cublas_get_handle();

// ============================================================================
// FP16 GEMM: half × half → half
// ============================================================================

/**
 * C[M,N] = A[M,K] × B[K,N]  (all FP16, row-major)
 *
 * Uses cublasLtMatmul with FP16 compute + tensor cores.
 * This is the fastest cuBLAS path and the baseline for %cuBLAS comparisons.
 */
void cublas_gemm_fp16(
    const half* A,
    const half* B,
    half* C,
    int M, int N, int K,
    const half* bias = nullptr,
    GemmEpilogue epilogue = GemmEpilogue::NONE,
    cudaStream_t stream = 0
);

// ============================================================================
// FP8 GEMM: E4M3 × E4M3 → FP16
// ============================================================================

/**
 * C[M,N] = A[M,K] × B[K,N]  (A,B in E4M3, output FP16)
 *
 * Uses cublasLtMatmul with FP8 E4M3 inputs and FP32 compute.
 * cuBLASLt natively supports FP8 on SM89+ (Hopper, Blackwell).
 *
 * Drop-in replacement for launch_gemm_fp8().
 */
void cublas_gemm_fp8(
    const __nv_fp8_e4m3* A,
    const __nv_fp8_e4m3* B,
    half* C,
    int M, int N, int K,
    const half* bias = nullptr,
    GemmEpilogue epilogue = GemmEpilogue::NONE,
    cudaStream_t stream = 0
);

// ============================================================================
// Mixed GEMM: FP16 activations × FP4 weights → FP16
// ============================================================================

/**
 * C[M,N] = A_fp16[M,K] × B_fp4[K,N]
 *
 * cuBLAS does not natively support FP4 inputs. This function:
 *   1. Dequantizes B to FP16 (data * block_scale * tensor_scale)
 *   2. Runs cublasLtMatmul in FP16
 *   3. Applies epilogue (bias, SiLU, residual) if requested
 *
 * The dequantized B is cached in the plan for reuse (weights don't change).
 *
 * Drop-in replacement for launch_gemm_mixed().
 */
void cublas_gemm_mixed(
    const half* A,
    const Fp4WeightTensor& B,
    half* C,
    int M, int N, int K,
    const half* bias = nullptr,
    const half* residual = nullptr,
    MixedEpilogue epilogue = MixedEpilogue::NONE,
    cudaStream_t stream = 0
);

/**
 * Fused gate+up projection: output = SiLU(x @ W_gate) * (x @ W_up)
 *
 * Runs two cuBLAS GEMMs + a SiLU-mul fusion kernel.
 * Drop-in replacement for launch_fused_gate_up().
 */
void cublas_fused_gate_up(
    const half* x,
    const Fp4WeightTensor& W_gate,
    const Fp4WeightTensor& W_up,
    half* output,
    int M, int N, int K,
    cudaStream_t stream = 0
);

// ============================================================================
// Prepare/Execute API — mirrors the custom kernel plan-based API
// ============================================================================

/**
 * Opaque plan struct for cuBLAS GEMM.
 *
 * Pre-computes:
 *   - cublasLtMatmulDesc, layout descriptors
 *   - Dequantized weight buffer (for FP4 inputs)
 *   - Algorithm selection (heuristic search at plan creation)
 *   - Workspace buffer
 *
 * The execute path only calls cublasLtMatmul — zero allocations.
 */
struct CublasGemmPlan;

// --- FP16 plans ---

CublasGemmPlan* create_cublas_fp16_plan(
    const half* A,
    const half* B,
    int M, int N, int K,
    const half* bias = nullptr,
    GemmEpilogue epilogue = GemmEpilogue::NONE
);

// --- FP8 plans ---

CublasGemmPlan* create_cublas_fp8_plan(
    const __nv_fp8_e4m3* A,
    const __nv_fp8_e4m3* B,
    int M, int N, int K,
    const half* bias = nullptr,
    GemmEpilogue epilogue = GemmEpilogue::NONE
);

// --- Mixed plans (FP16 × FP4) ---

CublasGemmPlan* create_cublas_mixed_plan(
    const half* A,
    const Fp4WeightTensor& B,
    int M, int N, int K,
    const half* bias = nullptr,
    const half* residual = nullptr,
    MixedEpilogue epilogue = MixedEpilogue::NONE
);

// --- Common execute/destroy ---

/**
 * Execute a prepared cuBLAS GEMM plan. Only calls cublasLtMatmul +
 * optional epilogue kernel — no allocations, descriptor setup, or
 * algorithm search on this path.
 */
void execute_cublas_gemm(CublasGemmPlan* plan, half* C, cudaStream_t stream = 0);

/** Free all resources held by a cuBLAS plan. */
void destroy_cublas_gemm_plan(CublasGemmPlan* plan);

}  // namespace blaze
