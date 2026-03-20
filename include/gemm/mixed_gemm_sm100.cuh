#pragma once
/**
 * mixed_gemm_sm100.cuh — Mixed-precision GEMM for SM100 (Blackwell).
 *
 * The actual inference kernel: FP16/BF16 activations × FP4 weights.
 *   A (activations): FP16/BF16, dynamic per-token values
 *   B (weights):     NVFP4 (E2M1 + block scales + tensor scale)
 *   Accumulator:     FP32 in TMEM
 *   Output:          FP16/BF16
 *
 * This is the most important kernel — it's called for every linear layer
 * during both prefill and decode.
 */

#include "gemm/fp4_gemm_sm100.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace blaze {

struct MixedGemmConfig {
    static constexpr int TILE_M = 128;
    static constexpr int TILE_N = 128;
    static constexpr int TILE_K = 64;   // K in FP16 elements (matched to weight FP4 K/2 bytes)
    static constexpr int BLOCK_SIZE = 128;
    static constexpr int PIPELINE_STAGES = 2;

    // A (activations): FP16, (TILE_M × TILE_K) elements
    static constexpr int SMEM_A_BYTES = TILE_M * TILE_K * sizeof(half);  // 16 KB

    // B (weights): FP4 packed + block scales
    static constexpr int SMEM_B_DATA_BYTES = TILE_K * TILE_N / 2;       // 4 KB (FP4 packed)
    static constexpr int SMEM_B_SCALE_BYTES =
        (TILE_K * TILE_N / FP4_BLOCK_SIZE) * sizeof(__nv_fp8_e4m3);     // scales

    static constexpr int SMEM_C_BYTES = TILE_M * TILE_N * sizeof(float); // 64 KB

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

}  // namespace blaze
