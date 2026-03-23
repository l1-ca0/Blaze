#pragma once
/**
 * mixed_gemm_sm100.cuh — Mixed-precision GEMM for SM100 (Blackwell).
 *
 * Primary inference kernel: FP16 activations × NVFP4 weights.
 * The host launch converts FP16 activations to E4M3, then the kernel
 * computes C[M,N] = A_e4m3[M,K] × B_fp4[K,N] via tcgen05.mma.kind::f8f6f4.
 *
 *   A (activations): FP16 at API boundary, E4M3 inside kernel
 *   B (weights):     NVFP4 (E2M1 data + E4M3 block scales + FP32 tensor scale)
 *   Accumulator:     FP32 in TMEM
 *   Output:          FP16
 *
 * Warp-specialized, 2-stage double-buffered pipeline:
 *   Warp 0: MMA consumer  (tcgen05.mma, E4M3 × E2M1)
 *   Warp 1: TMA producer  (packed data via TMA, scales via global loads)
 *   Warp 2: Idle
 *   Warp 3: Epilogue      (TMEM → SMEM → global store with tensor scale)
 *
 * Tile: 128×128×64 (M×N×K). K_PER_MMA=32, 2 inner iterations per tile.
 */

#include "gemm/fp4_gemm_sm100.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace blaze {

struct MixedGemmConfig {
    static constexpr int TILE_M = 128;
    static constexpr int TILE_N = 128;
    static constexpr int TILE_K = 64;
    static constexpr int BLOCK_SIZE = 128;         // 4 warps
    static constexpr int PIPELINE_STAGES = 2;

    // Per-stage SMEM for A (E4M3 activations, 1 byte each)
    static constexpr int SMEM_A_BYTES = TILE_M * TILE_K;                       // 8 KB

    // Per-stage SMEM for B packed data + block scales
    static constexpr int SMEM_B_DATA_BYTES  = TILE_K * TILE_N / 2;            // 4 KB
    static constexpr int SMEM_B_SCALE_BYTES = TILE_K * (TILE_N / FP4_BLOCK_SIZE);  // 512

    static constexpr int SMEM_C_BYTES = TILE_M * TILE_N * sizeof(float);       // 64 KB

    static constexpr int TOTAL_SMEM_BYTES =
        PIPELINE_STAGES * (SMEM_A_BYTES + SMEM_B_DATA_BYTES + SMEM_B_SCALE_BYTES) +
        SMEM_C_BYTES;

    static constexpr int TMEM_COLUMNS = TILE_N;
};

enum class MixedEpilogue {
    NONE,
    BIAS,
    SILU,
    BIAS_SILU,
    RESIDUAL_ADD,  // output += residual (for skip connections)
};

/**
 * Launch mixed-precision GEMM: C = A_fp16 × B_fp4
 *
 * @param A              Activation matrix (M, K) in FP16, row-major
 * @param B              Weight tensor in NVFP4 format
 * @param C              Output matrix (M, N) in FP16, row-major
 * @param M, N, K        Matrix dimensions
 * @param bias           Optional bias (N,) in FP16
 * @param residual       Optional residual tensor (M, N) in FP16 for skip connection
 * @param epilogue       Epilogue operation
 * @param stream         CUDA stream
 */
void launch_gemm_mixed(
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
 * Fused gate+up projection for SwiGLU FFN.
 *
 * Computes: output = SiLU(x @ W_gate) * (x @ W_up)
 *
 * Fusing both projections avoids reading the activation tensor twice
 * and eliminates the intermediate storage.
 *
 * @param x          Input activations (M, K) in FP16
 * @param W_gate     Gate weight in NVFP4 (K, N)
 * @param W_up       Up weight in NVFP4 (K, N)
 * @param output     Output (M, N) in FP16
 * @param M, N, K    Dimensions
 * @param stream     CUDA stream
 */
void launch_fused_gate_up(
    const half* x,
    const Fp4WeightTensor& W_gate,
    const Fp4WeightTensor& W_up,
    half* output,
    int M, int N, int K,
    cudaStream_t stream = 0
);

// ---------------------------------------------------------------------------
// Prepare/execute API — pre-allocates workspace and TMA descriptors once
// so the hot path (execute) only launches the kernel with zero allocations.
// ---------------------------------------------------------------------------

struct MixedGemmPlan;

/**
 * Create a reusable plan for mixed GEMM with fixed A, B, and dimensions.
 * Performs FP16→E4M3 conversion, M-padding, and TMA descriptor creation.
 * The returned plan can be executed many times with different C pointers.
 */
MixedGemmPlan* create_mixed_gemm_plan(
    const half* A,
    const Fp4WeightTensor& B,
    int M, int N, int K,
    const half* bias = nullptr,
    const half* residual = nullptr,
    MixedEpilogue epilogue = MixedEpilogue::NONE
);

/**
 * Execute a prepared mixed GEMM plan. Only launches the kernel — no
 * allocations, TMA setup, or data conversions on this path.
 */
void execute_mixed_gemm(MixedGemmPlan* plan, half* C, cudaStream_t stream = 0);

/** Free all resources held by a mixed GEMM plan. */
void destroy_mixed_gemm_plan(MixedGemmPlan* plan);

}  // namespace blaze
