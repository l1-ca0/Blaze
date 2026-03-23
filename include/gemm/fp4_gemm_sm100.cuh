#pragma once
/**
 * fp4_gemm_sm100.cuh — NVFP4 GEMM kernel interface for SM100 (Blackwell).
 *
 * NVFP4 (E2M1) format with two-level scaling:
 *   - Tensor-level scale: FP32 scalar, one per tensor
 *   - Block-level scale:  E4M3 (FP8), one per 16 consecutive elements
 *   - Data:               E2M1, packed 2 values per byte
 *
 * Warp-specialized, 2-stage double-buffered pipeline:
 *   Warp 0: MMA consumer  (tcgen05.mma.kind::f8f6f4, E2M1 × E2M1)
 *   Warp 1: TMA producer  (packed data via TMA, scales via global loads)
 *   Warp 2: Idle
 *   Warp 3: Epilogue      (TMEM → SMEM → global store with tensor scale)
 *
 * Tile: 128×128×128 (M×N×K). K_PER_MMA=64, 2 inner iterations per tile.
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
    static constexpr int TILE_K = 128;             // 2× FP8 tile (same byte count due to packing)
    static constexpr int BLOCK_SIZE = 128;         // 4 warps
    static constexpr int PIPELINE_STAGES = 2;

    // Per-stage SMEM for packed data (2 values per byte)
    static constexpr int SMEM_A_DATA_BYTES = TILE_M * TILE_K / 2;   // 8 KB
    static constexpr int SMEM_B_DATA_BYTES = TILE_K * TILE_N / 2;   // 8 KB

    // Per-stage SMEM for block scales (1 E4M3 per 16 values along K)
    static constexpr int A_SCALE_COLS = TILE_K / FP4_BLOCK_SIZE;    // 8
    static constexpr int B_SCALE_COLS = TILE_N / FP4_BLOCK_SIZE;    // 8
    static constexpr int SMEM_A_SCALE_BYTES = TILE_M * A_SCALE_COLS;  // 1024
    static constexpr int SMEM_B_SCALE_BYTES = TILE_K * B_SCALE_COLS;  // 1024

    static constexpr int SMEM_C_BYTES = TILE_M * TILE_N * sizeof(float);  // 64 KB

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

// ---------------------------------------------------------------------------
// Prepare/execute API — pre-allocates workspace and TMA descriptors once
// so the hot path (execute) only launches the kernel with zero allocations.
// ---------------------------------------------------------------------------

struct Fp4GemmPlan;

/**
 * Create a reusable plan for FP4 GEMM with fixed A, B, and dimensions.
 * Performs M-padding and TMA descriptor creation.
 */
Fp4GemmPlan* create_fp4_gemm_plan(
    const Fp4WeightTensor& A,
    const Fp4WeightTensor& B,
    int M, int N, int K,
    const half* bias = nullptr,
    Fp4Epilogue epilogue = Fp4Epilogue::NONE
);

/**
 * Execute a prepared FP4 GEMM plan. Only launches the kernel — no
 * allocations or TMA setup on this path.
 */
void execute_fp4_gemm(Fp4GemmPlan* plan, half* C, cudaStream_t stream = 0);

/** Free all resources held by an FP4 GEMM plan. */
void destroy_fp4_gemm_plan(Fp4GemmPlan* plan);

}  // namespace blaze
