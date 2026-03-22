#pragma once
/**
 * fp8_gemm_sm100.cuh — FP8 GEMM kernel interface for SM100 (Blackwell).
 *
 * Warp-specialized, 2-stage double-buffered TMA pipeline:
 *   Warp 0: MMA consumer  (tcgen05.mma, E4M3 × E4M3, FP32 accumulator in TMEM)
 *   Warp 1: TMA producer  (async bulk loads of A and B tiles via TMA)
 *   Warp 2: Idle
 *   Warp 3: Epilogue      (TMEM → SMEM → global, optional bias/activation fusion)
 *
 * Tile: 128×128×64 (M×N×K), K_PER_MMA=32 (2 inner iterations per tile).
 * Inputs: E4M3 (FP8), Accumulator: FP32 in TMEM, Output: FP16.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace blaze {

struct Fp8GemmConfig {
    static constexpr int TILE_M = 128;
    static constexpr int TILE_N = 128;
    static constexpr int TILE_K = 64;
    static constexpr int BLOCK_SIZE = 128;         // 4 warps
    static constexpr int PIPELINE_STAGES = 2;

    // Per-stage SMEM: A (8 KB) + B (8 KB), double-buffered. C staging (64 KB).
    static constexpr int SMEM_A_BYTES = TILE_M * TILE_K * sizeof(__nv_fp8_e4m3);
    static constexpr int SMEM_B_BYTES = TILE_K * TILE_N * sizeof(__nv_fp8_e4m3);
    static constexpr int SMEM_C_BYTES = TILE_M * TILE_N * sizeof(float);

    static constexpr int TOTAL_SMEM_BYTES =
        PIPELINE_STAGES * (SMEM_A_BYTES + SMEM_B_BYTES) + SMEM_C_BYTES;

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
