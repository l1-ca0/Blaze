#pragma once
/**
 * fp8_gemm_sm100.cuh — FP8 GEMM kernel interface for SM100 (Blackwell).
 *
 * Warp-specialized, double-buffered TMA pipeline:
 *   Warp 0:    TMA producer (async loads A, B tiles)
 *   Warps 1-2: MMA consumer (tcgen05.mma with TMEM accumulation)
 *   Warp 3:    Epilogue (TMEM → SMEM → global, optional bias/activation fusion)
 *
 * Tile: 128×128×64 (M×N×K), 2-stage double-buffered pipeline.
 * Inputs: E4M3 (FP8), Accumulator: FP32 in TMEM, Output: FP16/BF16.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace blaze {

// Tile dimensions for FP8 GEMM
struct Fp8GemmConfig {
    static constexpr int TILE_M = 128;
    static constexpr int TILE_N = 128;
    static constexpr int TILE_K = 64;   // FP8: 64 K per MMA
    static constexpr int BLOCK_SIZE = 128;  // 4 warps
    static constexpr int PIPELINE_STAGES = 2;  // Double buffer

    static constexpr int SMEM_A_BYTES = TILE_M * TILE_K * sizeof(__nv_fp8_e4m3);  // 8 KB
    static constexpr int SMEM_B_BYTES = TILE_K * TILE_N * sizeof(__nv_fp8_e4m3);  // 8 KB
    static constexpr int SMEM_C_BYTES = TILE_M * TILE_N * sizeof(float);          // 64 KB

    // Double-buffered: 2 × (A + B) + C staging
    static constexpr int TOTAL_SMEM_BYTES =
        PIPELINE_STAGES * (SMEM_A_BYTES + SMEM_B_BYTES) + SMEM_C_BYTES;

    // TMEM columns needed: TILE_N columns for the accumulator
    static constexpr int TMEM_COLUMNS = TILE_N;
};

/**
 * Epilogue operation applied after GEMM accumulation.
 */
enum class GemmEpilogue {
    NONE,       // Just store FP32 → FP16
    BIAS,       // Add bias vector
    SILU,       // SiLU activation (for FFN gate)
    BIAS_SILU,  // Bias + SiLU
};

/**
 * Launch FP8 GEMM: C = A × B
 *
 * @param A        Input matrix (M, K) in E4M3 format, row-major
 * @param B        Weight matrix (K, N) in E4M3 format, row-major
 * @param C        Output matrix (M, N) in FP16, row-major
 * @param M        Number of rows in A and C
 * @param N        Number of columns in B and C
 * @param K        Shared dimension
 * @param bias     Optional bias vector (N,) in FP16. Pass nullptr if unused.
 * @param epilogue Epilogue operation
 * @param stream   CUDA stream
 */
void launch_gemm_fp8(
    const __nv_fp8_e4m3* A,
    const __nv_fp8_e4m3* B,
    half* C,
    int M, int N, int K,
    const half* bias = nullptr,
    GemmEpilogue epilogue = GemmEpilogue::NONE,
    cudaStream_t stream = 0
);

}  // namespace blaze
