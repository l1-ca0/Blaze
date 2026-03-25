#pragma once
/**
 * fp4_blkscaled_gemm_sm100.cuh — Block-scaled NVFP4 GEMM interface for SM100 (Blackwell).
 *
 * Uses tcgen05.mma.kind::mxf4nvf4.block_scale.block16 for
 * hardware-native block scale application.
 *
 * Key differences from fp4_gemm_sm100.cuh (software scale path):
 *   - Block scales loaded via TMA → SMEM → tcgen05.cp → TMEM (not global loads)
 *   - MMA instruction natively applies block scales (7-operand form)
 *   - 3 active warps during K-loop (consumer + 2 producers), warp 3 idle
 *   - mbar_load init=2 (2 TMA producer warps)
 *   - TMEM alloc 256 columns (128 accum + 8 SFA + 8 SFB)
 *   - Interleaved scale format required for TMEM-compatible layout (SfAtom)
 *
 * Warp-specialized, 2-stage double-buffered pipeline:
 *   Warp 0: MMA consumer   (tcgen05.cp for scales + block-scaled MMA)
 *   Warp 1: TMA producer A (A_data + SFA loads, 9216 bytes/stage)
 *   Warp 2: TMA producer B (B_data + SFB loads, 9216 bytes/stage)
 *   Warp 3: Idle during K-loop
 *   All warps: Epilogue   (TMEM → registers → global store)
 *
 * Scale factor layout (SfAtom from CUTLASS):
 *   128 M-rows are packed into 32 SMEM rows of 16 bytes each.
 *   Each 16-byte row: [ws0_k0..k3, ws1_k0..k3, ws2_k0..k3, ws3_k0..k3]
 *   where ws=warp_subpartition (4×32=128 rows), k=scale index within MMA atom.
 *   tcgen05.cp.32x128b.warpx4 broadcasts 32 rows to all 4 subpartitions.
 *   One cp call fills 4 TMEM columns (one MMA atom's scales).
 *   NUM_K_ITERS cp calls needed per SF per K-tile (each to different TMEM cols).
 *
 * Tile: 128×128×128 (M×N×K). K_PER_MMA=64, 2 inner iterations per tile.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace blaze {

// FP4 E2M1 constants
constexpr int FP4_BLKSCALED_BLOCK_SIZE = 16;  // Block scale group size (1 scale per 16 E2M1 values)

struct Fp4BlkScaledConfig {
    // Tile dimensions
    static constexpr int TILE_M = 128;
    static constexpr int TILE_N = 128;
    static constexpr int TILE_K = 128;
    static constexpr int K_PER_MMA = 64;                              // MMA atom K for E2M1
    static constexpr int NUM_K_ITERS = TILE_K / K_PER_MMA;           // 2
    static constexpr int BLOCK_SIZE = 128;                            // 4 warps
    static constexpr int PIPELINE_STAGES = 2;

    // Scale factor parameters
    static constexpr int SF_VEC_SIZE = 16;                            // block16: 1 scale per 16 values
    static constexpr int SF_K = TILE_K / SF_VEC_SIZE;                 // 8 scales per row per K-tile

    // SfAtom layout: 128 M-rows packed into 32 SMEM rows × 16 bytes.
    // Each 16-byte row: [ws0_k0..k3, ws1_k0..k3, ws2_k0..k3, ws3_k0..k3]
    // where ws=warp_subpartition, k=K-scale-index within one MMA atom (4 values).
    // One SfAtom = one MMA atom's scales = 32 × 16 = 512 bytes.
    // For NUM_K_ITERS=2 atoms, SMEM holds 2 SfAtoms per K-tile = 1024 bytes.
    //
    // SMEM row stride = NUM_K_ITERS × 16 = 32 bytes (atoms are column-adjacent).
    // tcgen05.cp for atom ki reads from SMEM base + ki*16 with stride 32.
    static constexpr int SF_CP_BYTES_PER_ROW = 16;                    // 128 bits per cp call
    static constexpr int SF_SMEM_ROW_BYTES = SF_CP_BYTES_PER_ROW * NUM_K_ITERS;  // 32

    // Packed data SMEM
    static constexpr int SMEM_A_DATA_BYTES = TILE_M * TILE_K / 2;    // 8 KB
    static constexpr int SMEM_B_DATA_BYTES = TILE_K * TILE_N / 2;    // 8 KB

    // Scale factor SMEM (TMA-loaded, interleaved SfAtom format)
    // SFA: [32, SF_SMEM_ROW_BYTES] = [32, 32] = 1024 bytes per stage
    // SFB: [32, SF_SMEM_ROW_BYTES] = [32, 32] = 1024 bytes per stage
    static constexpr int SF_SMEM_ROWS = 32;                              // lanes (not M-rows)
    static constexpr int SMEM_SFA_BYTES = SF_SMEM_ROWS * SF_SMEM_ROW_BYTES;  // 1024
    static constexpr int SMEM_SFB_BYTES = SF_SMEM_ROWS * SF_SMEM_ROW_BYTES;  // 1024

    // Total SMEM per stage
    static constexpr int SMEM_PER_STAGE = SMEM_A_DATA_BYTES + SMEM_B_DATA_BYTES
                                        + SMEM_SFA_BYTES + SMEM_SFB_BYTES;  // 18 KB

    // TMEM column layout:
    //   [0, 128)    accumulator (TILE_N columns)
    //   [128, 136)  SFA (4 columns × NUM_K_ITERS atoms = 8 columns)
    //   [136, 144)  SFB (4 columns × NUM_K_ITERS atoms = 8 columns)
    // Each tcgen05.cp.32x128b.warpx4 fills 4 TMEM columns per call.
    // MMA atom ki reads SFA from tmem_sfa + ki*4, SFB from tmem_sfb + ki*4.
    static constexpr int TMEM_ACCUM_COLS = TILE_N;                    // 128
    static constexpr int TMEM_SF_COLS_PER_ATOM = 4;                   // filled by one 32x128b cp
    static constexpr int TMEM_SFA_COLS = TMEM_SF_COLS_PER_ATOM * NUM_K_ITERS;  // 8
    static constexpr int TMEM_SFB_COLS = TMEM_SF_COLS_PER_ATOM * NUM_K_ITERS;  // 8
    static constexpr int TMEM_TOTAL_NEEDED = TMEM_ACCUM_COLS + TMEM_SFA_COLS + TMEM_SFB_COLS;  // 144
    static constexpr int TMEM_ALLOC_COLS = 256;                       // next power-of-2 >= 144

    // TMEM column offsets
    static constexpr int TMEM_ACCUM_START = 0;
    static constexpr int TMEM_SFA_START = TMEM_ACCUM_COLS;            // 128
    static constexpr int TMEM_SFB_START = TMEM_SFA_START + TMEM_SFA_COLS;  // 136
};

/**
 * NVFP4 weight tensor for block-scaled GEMM.
 * Same structure as Fp4WeightTensor but used with interleaved scales.
 */
struct Fp4BlkScaledWeightTensor {
    const uint8_t* data;                    // Packed E2M1 values (2 per byte)
    const __nv_fp8_e4m3* block_scales;      // Per-block scales (row-major, will be interleaved at launch)
    float tensor_scale;                      // Per-tensor scale
    int rows;
    int cols;                                // In elements, not bytes
};

// Epilogue operations
enum class Fp4BlkScaledEpilogue {
    NONE,
    BIAS,
    SILU,
    BIAS_SILU,
};

/**
 * Launch block-scaled FP4 GEMM: C = A × B
 *
 * Both A and B are in NVFP4 format. Block scales are applied natively by
 * the MMA hardware. Tensor scales are applied in the epilogue.
 *
 * @param A            FP4 weight tensor A (M, K)
 * @param B            FP4 weight tensor B (K, N)
 * @param C            Output matrix (M, N) in FP16
 * @param M, N, K      Matrix dimensions (in elements)
 * @param bias         Optional bias (N,) in FP16
 * @param epilogue     Epilogue operation
 * @param stream       CUDA stream
 */
void launch_gemm_fp4_blkscaled(
    const Fp4BlkScaledWeightTensor& A,
    const Fp4BlkScaledWeightTensor& B,
    half* C,
    int M, int N, int K,
    const half* bias = nullptr,
    Fp4BlkScaledEpilogue epilogue = Fp4BlkScaledEpilogue::NONE,
    cudaStream_t stream = 0
);

// ---------------------------------------------------------------------------
// Prepare/execute API — pre-allocates workspace and TMA descriptors once
// so the hot path (execute) only launches the kernel with zero allocations.
// ---------------------------------------------------------------------------

struct Fp4BlkScaledGemmPlan;

Fp4BlkScaledGemmPlan* create_fp4_blkscaled_gemm_plan(
    const Fp4BlkScaledWeightTensor& A,
    const Fp4BlkScaledWeightTensor& B,
    int M, int N, int K,
    const half* bias = nullptr,
    Fp4BlkScaledEpilogue epilogue = Fp4BlkScaledEpilogue::NONE
);

void execute_fp4_blkscaled_gemm(Fp4BlkScaledGemmPlan* plan, half* C, cudaStream_t stream = 0);

void destroy_fp4_blkscaled_gemm_plan(Fp4BlkScaledGemmPlan* plan);

}  // namespace blaze
