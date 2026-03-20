#pragma once
/**
 * fp4_gemm_sm100.cuh — NVFP4 GEMM kernel interface for SM100 (Blackwell).
 *
 * NVFP4 (E2M1) format with two-level scaling:
 *   - Tensor-level scale: FP32, one per tensor
 *   - Block-level scale: E4M3 (FP8), one per 16 elements along K
 *   - Data: E2M1, packed 2 values per byte
 *
 * Tile: 128×128×128 (M×N×K) — 2× K of FP8 due to packing.
 * Same warp-specialized architecture as FP8.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace blaze {

// FP4 E2M1 representable values (magnitude)
// 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
constexpr float FP4_E2M1_MAX = 6.0f;
constexpr int FP4_BLOCK_SIZE = 16;  // Block scale group size

struct Fp4GemmConfig {
    static constexpr int TILE_M = 128;
    static constexpr int TILE_N = 128;
    static constexpr int TILE_K = 128;  // 2× FP8 due to FP4 packing (same bytes)
    static constexpr int BLOCK_SIZE = 128;  // 4 warps
    static constexpr int PIPELINE_STAGES = 2;

    // FP4 packed: 2 values per byte → TILE_M * TILE_K / 2 bytes for data
    static constexpr int SMEM_A_DATA_BYTES = TILE_M * TILE_K / 2;           // 8 KB
    static constexpr int SMEM_B_DATA_BYTES = TILE_K * TILE_N / 2;           // 8 KB

    // Block scales: 1 FP8 per 16 values along K
    static constexpr int SMEM_A_SCALE_BYTES = (TILE_M * TILE_K / FP4_BLOCK_SIZE) * sizeof(__nv_fp8_e4m3);
    static constexpr int SMEM_B_SCALE_BYTES = (TILE_K * TILE_N / FP4_BLOCK_SIZE) * sizeof(__nv_fp8_e4m3);

    static constexpr int SMEM_C_BYTES = TILE_M * TILE_N * sizeof(float);    // 64 KB

    static constexpr int TOTAL_SMEM_BYTES =
        PIPELINE_STAGES * (SMEM_A_DATA_BYTES + SMEM_B_DATA_BYTES +
                           SMEM_A_SCALE_BYTES + SMEM_B_SCALE_BYTES) +
        SMEM_C_BYTES;

    static constexpr int TMEM_COLUMNS = TILE_N;
};

/**
 * NVFP4 weight tensor descriptor.
 *
 * A quantized weight matrix consists of:
 *   - data:         Packed FP4 values (2 per byte, uint8)
 *   - block_scales: FP8 E4M3 scales (1 per BLOCK_SIZE values along K)
 *   - tensor_scale: FP32 scalar (1 per tensor)
 */
struct Fp4WeightTensor {
    const uint8_t* data;                    // Packed E2M1 values
    const __nv_fp8_e4m3* block_scales;      // Per-block scales
    float tensor_scale;                      // Per-tensor scale
    int rows;                                // Outer dimension
    int cols;                                // Inner dimension (in elements, not bytes)
};

// Epilogue operations (shared with FP8)
enum class Fp4Epilogue {
    NONE,
    BIAS,
    SILU,
    BIAS_SILU,
};

/**
 * Launch FP4 GEMM: C = A × B
 *
 * Both A and B are in NVFP4 format.
 *
 * @param A            FP4 weight tensor A (M, K)
 * @param B            FP4 weight tensor B (K, N)
 * @param C            Output matrix (M, N) in FP16
 * @param M, N, K      Matrix dimensions (in elements)
 * @param bias         Optional bias (N,) in FP16
 * @param epilogue     Epilogue operation
 * @param stream       CUDA stream
 */
void launch_gemm_fp4(
    const Fp4WeightTensor& A,
    const Fp4WeightTensor& B,
    half* C,
    int M, int N, int K,
    const half* bias = nullptr,
    Fp4Epilogue epilogue = Fp4Epilogue::NONE,
    cudaStream_t stream = 0
);

}  // namespace blaze
