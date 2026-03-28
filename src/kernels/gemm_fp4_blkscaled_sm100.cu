/**
 * gemm_fp4_blkscaled_sm100.cu — Block-scaled NVFP4 GEMM for SM100 (Blackwell).
 *
 * Uses tcgen05.mma.kind::mxf4nvf4.block_scale.block16 for
 * hardware-native block scale application.
 *
 * Scale factor layout (SfAtom from CUTLASS):
 *   128 M-rows packed into 32 SMEM rows × 16 bytes.
 *   tcgen05.cp.32x128b.warpx4 broadcasts to all 4 subpartitions → 128 TMEM rows.
 *   One cp call fills 4 TMEM columns (one MMA atom's scales).
 *   NUM_K_ITERS cp calls per SF per K-tile (targeting different TMEM columns).
 *
 * GMEM interleaved: [M/4, sf_k*4] for SFA, [N/4, sf_k*4] for SFB (sf_k = K/SF_VEC_SIZE).
 * TMA loads [32, NUM_K_ITERS*16] tiles into SMEM.
 *
 * TMEM: 256 columns = 128 accum + 8 SFA + 8 SFB
 */

#include "gemm/fp4_blkscaled_gemm_sm100.cuh"
#include "gemm/tmem_utils.cuh"
#include "gemm/tma_utils.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdio>

namespace blaze {

using Config = Fp4BlkScaledConfig;


// ---------------------------------------------------------------------------
// SMEM layout
// ---------------------------------------------------------------------------

struct __align__(128) Fp4BlkScaledSmemLayout {
    alignas(8) uint64_t mbar_load[Config::PIPELINE_STAGES];
    alignas(8) uint64_t mbar_mma[Config::PIPELINE_STAGES];
    uint32_t tmem_addr;
    char _pad[128 - (2 * 16 + 4)];

    uint8_t A_data[Config::PIPELINE_STAGES][Config::SMEM_A_DATA_BYTES];    // 2 × 8 KB
    uint8_t B_data[Config::PIPELINE_STAGES][Config::SMEM_B_DATA_BYTES];    // 2 × 8 KB
    uint8_t SFA[Config::PIPELINE_STAGES][Config::SMEM_SFA_BYTES];         // 2 × 1024
    uint8_t SFB[Config::PIPELINE_STAGES][Config::SMEM_SFB_BYTES];         // 2 × 1024
};

// ---------------------------------------------------------------------------
// Interleave scale factors — SfAtom layout
// ---------------------------------------------------------------------------

/**
 * Interleave block scales from [M, sf_k] row-major to SfAtom [M/4, sf_k*4].
 *
 * SfAtom maps: out[rest_m*32+lane, rest_k*16+warp_sub*4+k_sub] = in[m, k_sf]
 *   where lane=m%32, warp_sub=(m%128)/32, rest_m=m/128, k_sub=k_sf%4, rest_k=k_sf/4.
 */
__global__ void interleave_scales_kernel(
    const __nv_fp8_e4m3* __restrict__ src,   // [M, sf_k] row-major
    uint8_t* __restrict__ dst,                // [M/4, sf_k*4] row-major
    int M, int sf_k
) {
    int m = blockIdx.x * 128 + threadIdx.x;
    if (m >= M) return;

    int lane = m % 32;
    int warp_sub = (m % 128) / 32;
    int rest_m = m / 128;
    int out_cols = sf_k * 4;
    int out_row = rest_m * 32 + lane;

    for (int k_sf = 0; k_sf < sf_k; k_sf++) {
        int out_col = (k_sf / 4) * 16 + warp_sub * 4 + (k_sf % 4);
        dst[out_row * out_cols + out_col] =
            *reinterpret_cast<const uint8_t*>(&src[m * sf_k + k_sf]);
    }
}

/**
 * Same interleave but source is transposed: [sf_k, M] row-major.
 * Used for SFB where B.block_scales is [K/16, N] and we interleave over N.
 */
__global__ void interleave_scales_transposed_kernel(
    const __nv_fp8_e4m3* __restrict__ src,   // [sf_k, src_ld] row-major
    uint8_t* __restrict__ dst,                // [M/4, sf_k*4] row-major
    int M, int sf_k, int src_ld
) {
    int m = blockIdx.x * 128 + threadIdx.x;
    if (m >= M) return;

    int lane = m % 32;
    int warp_sub = (m % 128) / 32;
    int rest_m = m / 128;
    int out_cols = sf_k * 4;
    int out_row = rest_m * 32 + lane;

    for (int k_sf = 0; k_sf < sf_k; k_sf++) {
        int out_col = (k_sf / 4) * 16 + warp_sub * 4 + (k_sf % 4);
        // Transposed: element (m, k_sf) is at src[k_sf * src_ld + m]
        dst[out_row * out_cols + out_col] =
            *reinterpret_cast<const uint8_t*>(&src[k_sf * src_ld + m]);
    }
}

// ---------------------------------------------------------------------------
// TMA producers (warps 1 and 2, balanced split by operand)
// ---------------------------------------------------------------------------

// Producer A (warp 1): loads A_data + SFA — one operand's data and scales.
__device__ __forceinline__
void fp4_blkscaled_tma_producer_a(
    Fp4BlkScaledSmemLayout* smem,
    const TmaDescriptor* desc_A_data,
    const TmaDescriptor* desc_SFA,
    int bm, int K, int lane_id
) {
    static constexpr uint32_t TX_BYTES = Config::SMEM_A_DATA_BYTES
                                       + Config::SMEM_SFA_BYTES;  // 9216
    const int num_k_tiles = K / Config::TILE_K;

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int stage = k_tile % Config::PIPELINE_STAGES;
        int k_offset = k_tile * Config::TILE_K;

        // Wait for consumer to finish with this stage before reusing it
        if (k_tile >= Config::PIPELINE_STAGES) {
            mbarrier_wait(&smem->mbar_mma[stage],
                         ((k_tile / Config::PIPELINE_STAGES) + 1) & 1);
        }

        if (lane_id == 0) {
            mbarrier_expect_tx(&smem->mbar_load[stage], TX_BYTES);

            // A_data: [M, K/2] global, tile [TILE_M, TILE_K/2]
            tma_load_2d(smem->A_data[stage], desc_A_data,
                        k_offset / 2, bm, &smem->mbar_load[stage]);
            // SFA: [M/4, K/4] global, tile [32, TILE_K/4=32]
            tma_load_2d(smem->SFA[stage], desc_SFA,
                        k_offset / 4, bm / 4, &smem->mbar_load[stage]);
        }
    }
}

// Producer B (warp 2): loads B_data + SFB — the other operand's data and scales.
__device__ __forceinline__
void fp4_blkscaled_tma_producer_b(
    Fp4BlkScaledSmemLayout* smem,
    const TmaDescriptor* desc_B_data,
    const TmaDescriptor* desc_SFB,
    int bn, int K, int lane_id
) {
    static constexpr uint32_t TX_BYTES = Config::SMEM_B_DATA_BYTES
                                       + Config::SMEM_SFB_BYTES;  // 9216
    const int num_k_tiles = K / Config::TILE_K;

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int stage = k_tile % Config::PIPELINE_STAGES;
        int k_offset = k_tile * Config::TILE_K;

        // Wait for consumer to finish with this stage before reusing it
        if (k_tile >= Config::PIPELINE_STAGES) {
            mbarrier_wait(&smem->mbar_mma[stage],
                         ((k_tile / Config::PIPELINE_STAGES) + 1) & 1);
        }

        if (lane_id == 0) {
            mbarrier_expect_tx(&smem->mbar_load[stage], TX_BYTES);

            // B_data: [K, N/2] global, tile [TILE_K, TILE_N/2]
            tma_load_2d(smem->B_data[stage], desc_B_data,
                        bn / 2, k_offset, &smem->mbar_load[stage]);
            // SFB: [N/4, K/4] global, tile [32, TILE_K/4=32]
            tma_load_2d(smem->SFB[stage], desc_SFB,
                        k_offset / 4, bn / 4, &smem->mbar_load[stage]);
        }
    }
}

// ---------------------------------------------------------------------------
// MMA consumer (warp 0)
// ---------------------------------------------------------------------------

__device__ __forceinline__
void fp4_blkscaled_mma_consumer(
    Fp4BlkScaledSmemLayout* smem,
    tmem_addr_t tmem_base,
    tmem_addr_t tmem_sfa,
    tmem_addr_t tmem_sfb,
    int K
) {
    const int num_k_tiles = K / Config::TILE_K;
    bool first_mma = true;

    uint32_t idesc = make_idesc_blkscaled(
        5, 5, Config::TILE_M, Config::TILE_N, 0, 0);

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int stage = k_tile % Config::PIPELINE_STAGES;

        // Wait for TMA loads
        mbarrier_wait(&smem->mbar_load[stage],
                     (k_tile / Config::PIPELINE_STAGES) & 1);

        // Ensure TMA async writes to SMEM are visible before tcgen05 reads them.
        // mbarrier_wait confirms bytes arrived, but proxy fence is needed for
        // SMEM coherency with subsequent tcgen05.cp and tcgen05.mma operations.
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

        // tcgen05.cp: copy all K-atoms' scales from SMEM → TMEM.
        // All threads in the warp must issue cp (warp-collective).
        {
            uint32_t sfa_smem = static_cast<uint32_t>(
                __cvta_generic_to_shared(smem->SFA[stage]));
            uint32_t sfb_smem = static_cast<uint32_t>(
                __cvta_generic_to_shared(smem->SFB[stage]));

            for (int ki = 0; ki < Config::NUM_K_ITERS; ki++) {
                uint64_t sfa_desc = make_sf_smem_desc(
                    sfa_smem + ki * Config::SF_CP_BYTES_PER_ROW,
                    Config::SF_SMEM_ROW_BYTES);
                tcgen05_cp_32x128b_warpx4(tmem_sfa + ki * 4, sfa_desc);

                uint64_t sfb_desc = make_sf_smem_desc(
                    sfb_smem + ki * Config::SF_CP_BYTES_PER_ROW,
                    Config::SF_SMEM_ROW_BYTES);
                tcgen05_cp_32x128b_warpx4(tmem_sfb + ki * 4, sfb_desc);
            }
        }

        // MMA inner loop
        uint32_t smem_a_base = static_cast<uint32_t>(
            __cvta_generic_to_shared(smem->A_data[stage]));
        uint32_t smem_b_base = static_cast<uint32_t>(
            __cvta_generic_to_shared(smem->B_data[stage]));

        for (int ki = 0; ki < Config::NUM_K_ITERS; ki++) {
            uint32_t smem_a = smem_a_base + ki * (Config::K_PER_MMA / 2);
            uint32_t smem_b = smem_b_base + ki * (Config::K_PER_MMA / 2) * Config::TILE_N;

            uint64_t desc_a = make_smem_desc(smem_a, Config::TILE_K / 2, 4);
            uint64_t desc_b = make_smem_desc(smem_b, Config::TILE_N / 2, 4);

            // Block-scaled MMA
            if (elect_one_sync()) {
                blkscaled_mma_mxf4nvf4(
                    tmem_base, desc_a, desc_b, idesc,
                    tmem_sfa + ki * 4,
                    tmem_sfb + ki * 4,
                    !first_mma
                );
            }
            first_mma = false;
        }

        asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");

        __syncwarp();
        if (elect_one_sync()) {
            mbarrier_arrive(&smem->mbar_mma[stage]);
        }
    }
}

// ---------------------------------------------------------------------------
// Epilogue
// ---------------------------------------------------------------------------

__device__ __forceinline__
void fp4_blkscaled_epilogue(
    tmem_addr_t tmem_addr,
    half* C_global, int bm, int bn, int M, int N,
    float tensor_scale_A, float tensor_scale_B,
    const half* bias, Fp4BlkScaledEpilogue epilogue_op,
    int warp_id, int lane_id
) {
    asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

    float tensor_scale = tensor_scale_A * tensor_scale_B;
    int row = warp_id * 32 + lane_id;
    int global_row = bm + row;

    for (int cg = 0; cg < Config::TILE_N / 8; cg++) {
        float vals[8];
        tmem_load_8xf32(vals, tmem_addr, cg * 8);
        tmem_wait_ld();

        if (global_row < M) {
            int gcb = bn + cg * 8;
            for (int c = 0; c < 8; c++) vals[c] *= tensor_scale;

            switch (epilogue_op) {
                case Fp4BlkScaledEpilogue::BIAS:
                    for (int c = 0; c < 8; c++)
                        vals[c] += __half2float(bias[gcb + c]);
                    break;
                case Fp4BlkScaledEpilogue::SILU:
                    for (int c = 0; c < 8; c++)
                        vals[c] = vals[c] / (1.0f + expf(-vals[c]));
                    break;
                case Fp4BlkScaledEpilogue::BIAS_SILU:
                    for (int c = 0; c < 8; c++) {
                        vals[c] += __half2float(bias[gcb + c]);
                        vals[c] = vals[c] / (1.0f + expf(-vals[c]));
                    }
                    break;
                default: break;
            }

            half h[8];
            for (int c = 0; c < 8; c++) h[c] = __float2half(vals[c]);

            if (gcb + 7 < N) {
                *reinterpret_cast<uint4*>(&C_global[global_row * N + gcb]) =
                    *reinterpret_cast<uint4*>(h);
            } else {
                for (int c = 0; c < 8; c++) {
                    if (gcb + c < N)
                        C_global[global_row * N + gcb + c] = h[c];
                }
            }
        }
    }
}

// ===========================================================================
// Persistent kernel: atomic-counter tile scheduling, all-warp epilogue.
//
// Each CTA atomically claims tiles from a global counter. TMEM persists
// across tiles. All 4 warps cooperate on epilogue between tiles.
// ===========================================================================

// --- Persistent SMEM layout ---

struct __align__(128) Fp4BlkScaledSmemPersistent {
    // Barriers for SMEM data pipeline (TMA ↔ MMA)
    alignas(8) uint64_t mbar_load[Config::PIPELINE_STAGES];    // 16 bytes, offset 0
    alignas(8) uint64_t mbar_mma[Config::PIPELINE_STAGES];     // 16 bytes, offset 16

    // Shared state
    uint32_t tmem_addr;                                          // 4 bytes, offset 32
    int tile_bm;                                                 // 4 bytes, offset 36
    int tile_bn;                                                 // 4 bytes, offset 40
    int tile_linear_id;                                          // 4 bytes, offset 44

    // Pad to 128-byte boundary: 128 - 48 = 80 bytes
    char _pad[128 - 48];

    // Double-buffered data tiles (same as non-persistent)
    uint8_t A_data[Config::PIPELINE_STAGES][Config::SMEM_A_DATA_BYTES];
    uint8_t B_data[Config::PIPELINE_STAGES][Config::SMEM_B_DATA_BYTES];
    uint8_t SFA[Config::PIPELINE_STAGES][Config::SMEM_SFA_BYTES];
    uint8_t SFB[Config::PIPELINE_STAGES][Config::SMEM_SFB_BYTES];
};

// --- Persistent TMA producers ---
// Same K-loop as non-persistent, but barriers are reinitialized each tile
// so phase tracking stays simple (always starts at phase 0).

__device__ __forceinline__
void fp4_persistent_tma_producer_a(
    Fp4BlkScaledSmemPersistent* smem,
    const TmaDescriptor* desc_A_data,
    const TmaDescriptor* desc_SFA,
    int bm, int K, int lane_id
) {
    static constexpr uint32_t TX_BYTES = Config::SMEM_A_DATA_BYTES
                                       + Config::SMEM_SFA_BYTES;
    const int num_k_tiles = K / Config::TILE_K;

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int stage = k_tile % Config::PIPELINE_STAGES;
        int k_offset = k_tile * Config::TILE_K;

        // Wait for consumer to release this stage (skip first PIPELINE_STAGES
        // iterations since barriers were just reinitialized with no prior arrivals)
        if (k_tile >= Config::PIPELINE_STAGES) {
            mbarrier_wait(&smem->mbar_mma[stage],
                         ((k_tile / Config::PIPELINE_STAGES) + 1) & 1);
        }

        if (lane_id == 0) {
            mbarrier_expect_tx(&smem->mbar_load[stage], TX_BYTES);
            tma_load_2d(smem->A_data[stage], desc_A_data,
                        k_offset / 2, bm, &smem->mbar_load[stage]);
            tma_load_2d(smem->SFA[stage], desc_SFA,
                        k_offset / 4, bm / 4, &smem->mbar_load[stage]);
        }
    }
}

__device__ __forceinline__
void fp4_persistent_tma_producer_b(
    Fp4BlkScaledSmemPersistent* smem,
    const TmaDescriptor* desc_B_data,
    const TmaDescriptor* desc_SFB,
    int bn, int K, int lane_id
) {
    static constexpr uint32_t TX_BYTES = Config::SMEM_B_DATA_BYTES
                                       + Config::SMEM_SFB_BYTES;
    const int num_k_tiles = K / Config::TILE_K;

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int stage = k_tile % Config::PIPELINE_STAGES;
        int k_offset = k_tile * Config::TILE_K;

        if (k_tile >= Config::PIPELINE_STAGES) {
            mbarrier_wait(&smem->mbar_mma[stage],
                         ((k_tile / Config::PIPELINE_STAGES) + 1) & 1);
        }

        if (lane_id == 0) {
            mbarrier_expect_tx(&smem->mbar_load[stage], TX_BYTES);
            tma_load_2d(smem->B_data[stage], desc_B_data,
                        bn / 2, k_offset, &smem->mbar_load[stage]);
            tma_load_2d(smem->SFB[stage], desc_SFB,
                        k_offset / 4, bn / 4, &smem->mbar_load[stage]);
        }
    }
}

// --- Persistent MMA consumer ---
// Same as non-persistent MMA consumer but with reinitialized barriers per tile.
// No tcgen05.commit needed: __syncthreads() after mainloop ensures MMA completion
// (tcgen05.fence::before_thread_sync is issued after the last K-tile's MMA).

__device__ __forceinline__
void fp4_persistent_mma_consumer(
    Fp4BlkScaledSmemPersistent* smem,
    tmem_addr_t tmem_base,
    tmem_addr_t tmem_sfa,
    tmem_addr_t tmem_sfb,
    int K
) {
    const int num_k_tiles = K / Config::TILE_K;

    uint32_t idesc = make_idesc_blkscaled(5, 5, Config::TILE_M, Config::TILE_N, 0, 0);

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int stage = k_tile % Config::PIPELINE_STAGES;

        // Wait for TMA loads
        mbarrier_wait(&smem->mbar_load[stage],
                     (k_tile / Config::PIPELINE_STAGES) & 1);

        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

        // tcgen05.cp: copy scales from SMEM → TMEM
        {
            uint32_t sfa_smem = static_cast<uint32_t>(
                __cvta_generic_to_shared(smem->SFA[stage]));
            uint32_t sfb_smem = static_cast<uint32_t>(
                __cvta_generic_to_shared(smem->SFB[stage]));

            for (int ki = 0; ki < Config::NUM_K_ITERS; ki++) {
                uint64_t sfa_desc = make_sf_smem_desc(
                    sfa_smem + ki * Config::SF_CP_BYTES_PER_ROW,
                    Config::SF_SMEM_ROW_BYTES);
                tcgen05_cp_32x128b_warpx4(tmem_sfa + ki * 4, sfa_desc);

                uint64_t sfb_desc = make_sf_smem_desc(
                    sfb_smem + ki * Config::SF_CP_BYTES_PER_ROW,
                    Config::SF_SMEM_ROW_BYTES);
                tcgen05_cp_32x128b_warpx4(tmem_sfb + ki * 4, sfb_desc);
            }
        }

        // MMA inner loop
        uint32_t smem_a_base = static_cast<uint32_t>(
            __cvta_generic_to_shared(smem->A_data[stage]));
        uint32_t smem_b_base = static_cast<uint32_t>(
            __cvta_generic_to_shared(smem->B_data[stage]));

        for (int ki = 0; ki < Config::NUM_K_ITERS; ki++) {
            uint32_t smem_a = smem_a_base + ki * (Config::K_PER_MMA / 2);
            uint32_t smem_b = smem_b_base + ki * (Config::K_PER_MMA / 2) * Config::TILE_N;

            uint64_t desc_a = make_smem_desc(smem_a, Config::TILE_K / 2, 4);
            uint64_t desc_b = make_smem_desc(smem_b, Config::TILE_N / 2, 4);

            bool accumulate = (k_tile > 0 || ki > 0);

            if (elect_one_sync()) {
                blkscaled_mma_mxf4nvf4(
                    tmem_base, desc_a, desc_b, idesc,
                    tmem_sfa + ki * 4,
                    tmem_sfb + ki * 4,
                    accumulate
                );
            }
        }

        asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");

        // Release SMEM stage for TMA reuse
        __syncwarp();
        if (elect_one_sync()) {
            mbarrier_arrive(&smem->mbar_mma[stage]);
        }
    }
}

// --- Persistent kernel entry point ---

__cluster_dims__(1, 1, 1)
__global__ void __launch_bounds__(Config::BLOCK_SIZE)
gemm_fp4_blkscaled_persistent_kernel(
    const TmaDescriptor* __restrict__ desc_A_data,
    const TmaDescriptor* __restrict__ desc_B_data,
    const TmaDescriptor* __restrict__ desc_SFA,
    const TmaDescriptor* __restrict__ desc_SFB,
    half* __restrict__ C,
    int M, int N, int K,
    float tensor_scale_A, float tensor_scale_B,
    const half* __restrict__ bias,
    Fp4BlkScaledEpilogue epilogue_op,
    int* tile_counter,         // Global atomic counter (initialized to 0 by host)
    int total_tiles,           // grid_m * grid_n
    int grid_n                 // Tile grid columns for coord derivation
) {
    extern __shared__ char smem_raw[];
    auto* smem = reinterpret_cast<Fp4BlkScaledSmemPersistent*>(smem_raw);

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // === One-time initialization ===
    if (tid == 0) {
        for (int s = 0; s < Config::PIPELINE_STAGES; s++) {
            mbarrier_inval(&smem->mbar_load[s]);
            mbarrier_inval(&smem->mbar_mma[s]);
        }
        for (int s = 0; s < Config::PIPELINE_STAGES; s++) {
            mbarrier_init(&smem->mbar_load[s], 2);   // 2 TMA producer warps
            mbarrier_init(&smem->mbar_mma[s], 1);    // 1 MMA consumer warp
        }
    }
    __syncthreads();

    // Allocate TMEM once for all tiles (256 cols: 128 accum + 8 SFA + 8 SFB)
    tmem_addr_t tmem_base;
    if (warp_id == 0) {
        tmem_alloc(&smem->tmem_addr, Config::TMEM_ALLOC_COLS);
    }
    __syncthreads();
    tmem_base = smem->tmem_addr;

    tmem_addr_t tmem_sfa = tmem_base + Config::TMEM_SFA_START;
    tmem_addr_t tmem_sfb = tmem_base + Config::TMEM_SFB_START;

    // === Persistent tile loop ===
    for (;;) {
        // --- Phase 1: Thread 0 claims next tile via atomic counter ---
        if (tid == 0) {
            smem->tile_linear_id = atomicAdd(tile_counter, 1);
        }
        __syncthreads();

        int linear_id = smem->tile_linear_id;
        if (linear_id >= total_tiles) break;  // All tiles claimed

        int bm = (linear_id / grid_n) * Config::TILE_M;
        int bn = (linear_id % grid_n) * Config::TILE_N;

        // --- Phase 2: Reinitialize mainloop barriers for this tile ---
        // __syncthreads() above guaranteed all warps finished previous tile.
        // mbarrier_inval clears all internal state (including TX byte tracking
        // from expect_tx arrivals), then mbarrier_init resets to phase 0.
        // Without inval, stale TX tracking from the previous tile's 86+ TMA
        // completions can prevent the barrier from completing on the next tile.
        if (tid == 0) {
            for (int s = 0; s < Config::PIPELINE_STAGES; s++) {
                mbarrier_inval(&smem->mbar_load[s]);
                mbarrier_inval(&smem->mbar_mma[s]);
            }
            for (int s = 0; s < Config::PIPELINE_STAGES; s++) {
                mbarrier_init(&smem->mbar_load[s], 2);  // 2 TMA producer warps
                mbarrier_init(&smem->mbar_mma[s], 1);   // 1 MMA consumer warp
            }
        }
        __syncthreads();

        // --- Phase 3: Mainloop (warp 0 = MMA, warps 1-2 = TMA, warp 3 = idle) ---
        switch (warp_id) {
            case 0:
                fp4_persistent_mma_consumer(smem, tmem_base, tmem_sfa, tmem_sfb, K);
                break;
            case 1:
                fp4_persistent_tma_producer_a(smem, desc_A_data, desc_SFA,
                                              bm, K, lane_id);
                break;
            case 2:
                fp4_persistent_tma_producer_b(smem, desc_B_data, desc_SFB,
                                              bn, K, lane_id);
                break;
            default: break;  // warp 3 idle during K-loop
        }

        // --- Phase 4: All-warp epilogue ---
        // tcgen05.fence::before_thread_sync in MMA consumer + __syncthreads()
        // guarantees accumulator is complete and visible to all warps.
        __syncthreads();

        // Reuse the same epilogue as non-persistent kernel (all 4 warps, 32 rows each)
        fp4_blkscaled_epilogue(tmem_base, C, bm, bn, M, N,
                               tensor_scale_A, tensor_scale_B,
                               bias, epilogue_op, warp_id, lane_id);

        // Fence: ensure epilogue's tcgen05.ld (TMEM reads) are ordered before
        // the next __syncthreads(), so the next tile's tcgen05.mma writes
        // don't overlap with this tile's reads. Without this fence, the
        // tcgen05 pipeline may stall/hang on the second tile.
        asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
    }

    // === Cleanup: deallocate TMEM ===
    __syncthreads();
    if (warp_id == 0) {
        tmem_dealloc(tmem_base, Config::TMEM_ALLOC_COLS);
    }
}

// ===========================================================================
// Non-persistent kernel (original)
// ===========================================================================

// ---------------------------------------------------------------------------
// Kernel entry point
// ---------------------------------------------------------------------------

__cluster_dims__(1, 1, 1)
__global__ void __launch_bounds__(Config::BLOCK_SIZE)
gemm_fp4_blkscaled_kernel(
    const TmaDescriptor* __restrict__ desc_A_data,
    const TmaDescriptor* __restrict__ desc_B_data,
    const TmaDescriptor* __restrict__ desc_SFA,
    const TmaDescriptor* __restrict__ desc_SFB,
    half* __restrict__ C,
    int M, int N, int K,
    float tensor_scale_A, float tensor_scale_B,
    const half* __restrict__ bias,
    Fp4BlkScaledEpilogue epilogue_op
) {
    extern __shared__ char smem_raw[];
    auto* smem = reinterpret_cast<Fp4BlkScaledSmemLayout*>(smem_raw);

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int bm = blockIdx.y * Config::TILE_M;
    const int bn = blockIdx.x * Config::TILE_N;

    if (tid == 0) {
        for (int s = 0; s < Config::PIPELINE_STAGES; s++) {
            mbarrier_inval(&smem->mbar_load[s]);
            mbarrier_inval(&smem->mbar_mma[s]);
        }
        for (int s = 0; s < Config::PIPELINE_STAGES; s++) {
            mbarrier_init(&smem->mbar_load[s], 2);  // 2 producer warps
            mbarrier_init(&smem->mbar_mma[s], 1);
        }
    }
    __syncthreads();

    tmem_addr_t tmem_base;
    if (warp_id == 0) {
        tmem_alloc(&smem->tmem_addr, Config::TMEM_ALLOC_COLS);
    }
    __syncthreads();
    tmem_base = smem->tmem_addr;

    tmem_addr_t tmem_sfa = tmem_base + Config::TMEM_SFA_START;
    tmem_addr_t tmem_sfb = tmem_base + Config::TMEM_SFB_START;

    switch (warp_id) {
        case 0:
            fp4_blkscaled_mma_consumer(smem, tmem_base, tmem_sfa, tmem_sfb, K);
            break;
        case 1:
            fp4_blkscaled_tma_producer_a(smem, desc_A_data, desc_SFA,
                                          bm, K, lane_id);
            break;
        case 2:
            fp4_blkscaled_tma_producer_b(smem, desc_B_data, desc_SFB,
                                          bn, K, lane_id);
            break;
        default: break;  // warp 3 idle during K-loop
    }

    __syncthreads();
    fp4_blkscaled_epilogue(tmem_base, C, bm, bn, M, N,
                           tensor_scale_A, tensor_scale_B,
                           bias, epilogue_op, warp_id, lane_id);

    __syncthreads();
    if (warp_id == 0) {
        tmem_dealloc(tmem_base, Config::TMEM_ALLOC_COLS);
    }
}

// ---------------------------------------------------------------------------
// Host launch
// ---------------------------------------------------------------------------

void launch_gemm_fp4_blkscaled(
    const Fp4BlkScaledWeightTensor& A,
    const Fp4BlkScaledWeightTensor& B,
    half* C,
    int M, int N, int K,
    const half* bias,
    Fp4BlkScaledEpilogue epilogue,
    cudaStream_t stream
) {
    int sf_k = K / Config::SF_VEC_SIZE;  // scales along K for both A and B

    // Interleave SFA: [M, sf_k] → [M/4, sf_k*4] = [M/4, K/4]
    int sf_cols = sf_k * 4;    // = K/4
    int sfb_rows = N / 4;

    // Handle M < TILE_M by padding to TILE_M
    int M_pad = (M < Config::TILE_M) ? Config::TILE_M : M;
    int sfa_rows_pad = M_pad / 4;

    uint8_t* sfa_interleaved = nullptr;
    uint8_t* sfb_interleaved = nullptr;
    cudaMalloc(&sfa_interleaved, (size_t)sfa_rows_pad * sf_cols);
    cudaMalloc(&sfb_interleaved, (size_t)sfb_rows * sf_cols);
    cudaMemsetAsync(sfa_interleaved, 0, (size_t)sfa_rows_pad * sf_cols, stream);
    cudaMemsetAsync(sfb_interleaved, 0, (size_t)sfb_rows * sf_cols, stream);

    // SFA interleave: src [M, sf_k] row-major
    {
        int blocks = (M + 127) / 128;
        interleave_scales_kernel<<<blocks, 128, 0, stream>>>(
            A.block_scales, sfa_interleaved, M, sf_k);
    }

    // SFB interleave: src is [sf_k, N] row-major (K/16 rows × N cols),
    // but we interleave over N, so use transposed kernel with src_ld=N
    {
        int blocks = (N + 127) / 128;
        interleave_scales_transposed_kernel<<<blocks, 128, 0, stream>>>(
            B.block_scales, sfb_interleaved, N, sf_k, N);
    }

    // Pad A data if M < TILE_M
    const uint8_t* A_data_tma = A.data;
    uint8_t* A_data_padded = nullptr;
    if (M < Config::TILE_M) {
        size_t data_bytes = (size_t)M_pad * K / 2;
        cudaMalloc(&A_data_padded, data_bytes);
        cudaMemsetAsync(A_data_padded, 0, data_bytes, stream);
        cudaMemcpyAsync(A_data_padded, A.data, (size_t)M * K / 2,
                        cudaMemcpyDeviceToDevice, stream);
        A_data_tma = A_data_padded;
    }

    // TMA descriptors
    TmaDescriptor h_desc_A, h_desc_B, h_desc_SFA, h_desc_SFB;

    // A_data: [M_pad, K/2] uint8, tile [TILE_M, TILE_K/2], B64 swizzle
    create_tma_desc_2d(&h_desc_A, A_data_tma, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                       M_pad, K / 2, Config::TILE_M, Config::TILE_K / 2, TmaSwizzle::B64);
    // B_data: [K, N/2] uint8, tile [TILE_K, TILE_N/2], B64 swizzle
    create_tma_desc_2d(&h_desc_B, B.data, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                       K, N / 2, Config::TILE_K, Config::TILE_N / 2, TmaSwizzle::B64);
    // SFA: [M_pad/4, K/4] uint8, tile [32, TILE_K/4=32], no swizzle
    create_tma_desc_2d(&h_desc_SFA, sfa_interleaved, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                       sfa_rows_pad, sf_cols, 32, Config::TILE_K / 4, TmaSwizzle::NONE);
    // SFB: [N/4, K/4] uint8, tile [32, TILE_K/4=32], no swizzle
    create_tma_desc_2d(&h_desc_SFB, sfb_interleaved, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                       sfb_rows, sf_cols, 32, Config::TILE_K / 4, TmaSwizzle::NONE);

    // Upload descriptors
    TmaDescriptor* d_descs;
    cudaMalloc(&d_descs, 4 * sizeof(TmaDescriptor));
    cudaMemcpyAsync(d_descs,     &h_desc_A,   sizeof(TmaDescriptor), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_descs + 1, &h_desc_B,   sizeof(TmaDescriptor), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_descs + 2, &h_desc_SFA, sizeof(TmaDescriptor), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_descs + 3, &h_desc_SFB, sizeof(TmaDescriptor), cudaMemcpyHostToDevice, stream);

    // Launch kernel
    dim3 grid((N + Config::TILE_N - 1) / Config::TILE_N,
              (M + Config::TILE_M - 1) / Config::TILE_M);
    dim3 block(Config::BLOCK_SIZE);
    size_t smem_size = sizeof(Fp4BlkScaledSmemLayout);

    cudaFuncSetAttribute(gemm_fp4_blkscaled_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    cudaLaunchConfig_t launch_config = {};
    launch_config.gridDim = grid;
    launch_config.blockDim = block;
    launch_config.dynamicSmemBytes = smem_size;
    launch_config.stream = stream;

    cudaLaunchAttribute launch_attrs[1];
    launch_attrs[0].id = cudaLaunchAttributeClusterDimension;
    launch_attrs[0].val.clusterDim.x = 1;
    launch_attrs[0].val.clusterDim.y = 1;
    launch_attrs[0].val.clusterDim.z = 1;
    launch_config.attrs = launch_attrs;
    launch_config.numAttrs = 1;

    cudaLaunchKernelEx(&launch_config, gemm_fp4_blkscaled_kernel,
        d_descs, d_descs + 1, d_descs + 2, d_descs + 3,
        C, M, N, K,
        A.tensor_scale, B.tensor_scale,
        bias, epilogue);

    // Cleanup
    cudaFreeAsync(d_descs, stream);
    cudaFreeAsync(sfa_interleaved, stream);
    cudaFreeAsync(sfb_interleaved, stream);
    if (A_data_padded) cudaFreeAsync(A_data_padded, stream);
}

// ---------------------------------------------------------------------------
// Prepare/execute API
// ---------------------------------------------------------------------------

struct Fp4BlkScaledGemmPlan {
    uint8_t* A_data_padded;
    uint8_t* sfa_interleaved;
    uint8_t* sfb_interleaved;
    TmaDescriptor* d_descs;
    float tensor_scale_A;
    float tensor_scale_B;
    const half* bias;
    Fp4BlkScaledEpilogue epilogue;
    dim3 grid;
    dim3 block;
    size_t smem_size;
    int M, N, K;
};

Fp4BlkScaledGemmPlan* create_fp4_blkscaled_gemm_plan(
    const Fp4BlkScaledWeightTensor& A,
    const Fp4BlkScaledWeightTensor& B,
    int M, int N, int K,
    const half* bias,
    Fp4BlkScaledEpilogue epilogue
) {
    auto* plan = new Fp4BlkScaledGemmPlan();
    plan->M = M;
    plan->N = N;
    plan->K = K;
    plan->tensor_scale_A = A.tensor_scale;
    plan->tensor_scale_B = B.tensor_scale;
    plan->bias = bias;
    plan->epilogue = epilogue;
    plan->A_data_padded = nullptr;

    int sf_k = K / Config::SF_VEC_SIZE;
    int sf_cols = sf_k * 4;
    int M_pad = (M < Config::TILE_M) ? Config::TILE_M : M;
    int sfa_rows_pad = M_pad / 4;
    int sfb_rows = N / 4;

    // Interleave SFA
    cudaMalloc(&plan->sfa_interleaved, (size_t)sfa_rows_pad * sf_cols);
    cudaMemset(plan->sfa_interleaved, 0, (size_t)sfa_rows_pad * sf_cols);
    {
        int blocks = (M + 127) / 128;
        interleave_scales_kernel<<<blocks, 128>>>(
            A.block_scales, plan->sfa_interleaved, M, sf_k);
    }

    // Interleave SFB (transposed: src is [sf_k, N])
    cudaMalloc(&plan->sfb_interleaved, (size_t)sfb_rows * sf_cols);
    cudaMemset(plan->sfb_interleaved, 0, (size_t)sfb_rows * sf_cols);
    {
        int blocks = (N + 127) / 128;
        interleave_scales_transposed_kernel<<<blocks, 128>>>(
            B.block_scales, plan->sfb_interleaved, N, sf_k, N);
    }

    // Pad A if needed
    const uint8_t* A_data_tma = A.data;
    if (M < Config::TILE_M) {
        size_t data_bytes = (size_t)M_pad * K / 2;
        cudaMalloc(&plan->A_data_padded, data_bytes);
        cudaMemset(plan->A_data_padded, 0, data_bytes);
        cudaMemcpy(plan->A_data_padded, A.data, (size_t)M * K / 2,
                   cudaMemcpyDeviceToDevice);
        A_data_tma = plan->A_data_padded;
    }

    // TMA descriptors
    TmaDescriptor h_desc_A, h_desc_B, h_desc_SFA, h_desc_SFB;

    create_tma_desc_2d(&h_desc_A, A_data_tma, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                       M_pad, K / 2, Config::TILE_M, Config::TILE_K / 2, TmaSwizzle::B64);
    create_tma_desc_2d(&h_desc_B, B.data, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                       K, N / 2, Config::TILE_K, Config::TILE_N / 2, TmaSwizzle::B64);
    create_tma_desc_2d(&h_desc_SFA, plan->sfa_interleaved, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                       sfa_rows_pad, sf_cols, 32, Config::TILE_K / 4, TmaSwizzle::NONE);
    create_tma_desc_2d(&h_desc_SFB, plan->sfb_interleaved, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                       sfb_rows, sf_cols, 32, Config::TILE_K / 4, TmaSwizzle::NONE);

    cudaMalloc(&plan->d_descs, 4 * sizeof(TmaDescriptor));
    cudaMemcpy(plan->d_descs,     &h_desc_A,   sizeof(TmaDescriptor), cudaMemcpyHostToDevice);
    cudaMemcpy(plan->d_descs + 1, &h_desc_B,   sizeof(TmaDescriptor), cudaMemcpyHostToDevice);
    cudaMemcpy(plan->d_descs + 2, &h_desc_SFA, sizeof(TmaDescriptor), cudaMemcpyHostToDevice);
    cudaMemcpy(plan->d_descs + 3, &h_desc_SFB, sizeof(TmaDescriptor), cudaMemcpyHostToDevice);

    plan->grid = dim3((N + Config::TILE_N - 1) / Config::TILE_N,
                      (M + Config::TILE_M - 1) / Config::TILE_M);
    plan->block = dim3(Config::BLOCK_SIZE);
    plan->smem_size = sizeof(Fp4BlkScaledSmemLayout);

    cudaFuncSetAttribute(gemm_fp4_blkscaled_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, plan->smem_size);

    cudaDeviceSynchronize();
    return plan;
}

void execute_fp4_blkscaled_gemm(Fp4BlkScaledGemmPlan* plan, half* C, cudaStream_t stream) {
    cudaLaunchConfig_t launch_config = {};
    launch_config.gridDim = plan->grid;
    launch_config.blockDim = plan->block;
    launch_config.dynamicSmemBytes = plan->smem_size;
    launch_config.stream = stream;

    cudaLaunchAttribute launch_attrs[1];
    launch_attrs[0].id = cudaLaunchAttributeClusterDimension;
    launch_attrs[0].val.clusterDim.x = 1;
    launch_attrs[0].val.clusterDim.y = 1;
    launch_attrs[0].val.clusterDim.z = 1;
    launch_config.attrs = launch_attrs;
    launch_config.numAttrs = 1;

    cudaLaunchKernelEx(&launch_config, gemm_fp4_blkscaled_kernel,
        plan->d_descs, plan->d_descs + 1, plan->d_descs + 2, plan->d_descs + 3,
        C, plan->M, plan->N, plan->K,
        plan->tensor_scale_A, plan->tensor_scale_B,
        plan->bias, plan->epilogue);
}

void destroy_fp4_blkscaled_gemm_plan(Fp4BlkScaledGemmPlan* plan) {
    if (!plan) return;
    if (plan->A_data_padded) cudaFree(plan->A_data_padded);
    if (plan->sfa_interleaved) cudaFree(plan->sfa_interleaved);
    if (plan->sfb_interleaved) cudaFree(plan->sfb_interleaved);
    if (plan->d_descs) cudaFree(plan->d_descs);
    delete plan;
}

// ---------------------------------------------------------------------------
// Persistent kernel host launch
// ---------------------------------------------------------------------------

void launch_gemm_fp4_blkscaled_persistent(
    const Fp4BlkScaledWeightTensor& A,
    const Fp4BlkScaledWeightTensor& B,
    half* C,
    int M, int N, int K,
    const half* bias,
    Fp4BlkScaledEpilogue epilogue,
    cudaStream_t stream
) {
    int sf_k = K / Config::SF_VEC_SIZE;
    int sf_cols = sf_k * 4;
    int sfb_rows = N / 4;

    int M_pad = (M < Config::TILE_M) ? Config::TILE_M : M;
    int sfa_rows_pad = M_pad / 4;

    // Interleave scales (same as non-persistent)
    uint8_t* sfa_interleaved = nullptr;
    uint8_t* sfb_interleaved = nullptr;
    cudaMalloc(&sfa_interleaved, (size_t)sfa_rows_pad * sf_cols);
    cudaMalloc(&sfb_interleaved, (size_t)sfb_rows * sf_cols);
    cudaMemsetAsync(sfa_interleaved, 0, (size_t)sfa_rows_pad * sf_cols, stream);
    cudaMemsetAsync(sfb_interleaved, 0, (size_t)sfb_rows * sf_cols, stream);

    {
        int blocks = (M + 127) / 128;
        interleave_scales_kernel<<<blocks, 128, 0, stream>>>(
            A.block_scales, sfa_interleaved, M, sf_k);
    }
    {
        int blocks = (N + 127) / 128;
        interleave_scales_transposed_kernel<<<blocks, 128, 0, stream>>>(
            B.block_scales, sfb_interleaved, N, sf_k, N);
    }

    // Pad A data if M < TILE_M
    const uint8_t* A_data_tma = A.data;
    uint8_t* A_data_padded = nullptr;
    if (M < Config::TILE_M) {
        size_t data_bytes = (size_t)M_pad * K / 2;
        cudaMalloc(&A_data_padded, data_bytes);
        cudaMemsetAsync(A_data_padded, 0, data_bytes, stream);
        cudaMemcpyAsync(A_data_padded, A.data, (size_t)M * K / 2,
                        cudaMemcpyDeviceToDevice, stream);
        A_data_tma = A_data_padded;
    }

    // TMA descriptors (same as non-persistent)
    TmaDescriptor h_desc_A, h_desc_B, h_desc_SFA, h_desc_SFB;

    create_tma_desc_2d(&h_desc_A, A_data_tma, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                       M_pad, K / 2, Config::TILE_M, Config::TILE_K / 2, TmaSwizzle::B64);
    create_tma_desc_2d(&h_desc_B, B.data, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                       K, N / 2, Config::TILE_K, Config::TILE_N / 2, TmaSwizzle::B64);
    create_tma_desc_2d(&h_desc_SFA, sfa_interleaved, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                       sfa_rows_pad, sf_cols, 32, Config::TILE_K / 4, TmaSwizzle::NONE);
    create_tma_desc_2d(&h_desc_SFB, sfb_interleaved, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                       sfb_rows, sf_cols, 32, Config::TILE_K / 4, TmaSwizzle::NONE);

    TmaDescriptor* d_descs;
    cudaMalloc(&d_descs, 4 * sizeof(TmaDescriptor));
    cudaMemcpyAsync(d_descs,     &h_desc_A,   sizeof(TmaDescriptor), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_descs + 1, &h_desc_B,   sizeof(TmaDescriptor), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_descs + 2, &h_desc_SFA, sizeof(TmaDescriptor), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_descs + 3, &h_desc_SFB, sizeof(TmaDescriptor), cudaMemcpyHostToDevice, stream);

    // Tile grid dimensions
    int grid_m = (M_pad + Config::TILE_M - 1) / Config::TILE_M;
    int grid_n = (N + Config::TILE_N - 1) / Config::TILE_N;
    int total_tiles = grid_m * grid_n;

    // Atomic tile counter (initialized to 0)
    int* d_tile_counter = nullptr;
    cudaMalloc(&d_tile_counter, sizeof(int));
    cudaMemsetAsync(d_tile_counter, 0, sizeof(int), stream);

    // Persistent kernel: launch one CTA per SM. Each CTA processes multiple tiles
    // via atomic counter scheduling. Query device for SM count.
    int device_id = 0;
    cudaGetDevice(&device_id);
    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_id);

    // Don't launch more CTAs than tiles
    int num_ctas = (total_tiles < num_sms) ? total_tiles : num_sms;

    dim3 grid(num_ctas);
    dim3 block(Config::BLOCK_SIZE);
    size_t smem_size = sizeof(Fp4BlkScaledSmemPersistent);

    cudaFuncSetAttribute(gemm_fp4_blkscaled_persistent_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    cudaLaunchConfig_t launch_config = {};
    launch_config.gridDim = grid;
    launch_config.blockDim = block;
    launch_config.dynamicSmemBytes = smem_size;
    launch_config.stream = stream;

    cudaLaunchAttribute launch_attrs[1];
    launch_attrs[0].id = cudaLaunchAttributeClusterDimension;
    launch_attrs[0].val.clusterDim.x = 1;
    launch_attrs[0].val.clusterDim.y = 1;
    launch_attrs[0].val.clusterDim.z = 1;
    launch_config.attrs = launch_attrs;
    launch_config.numAttrs = 1;

    cudaLaunchKernelEx(&launch_config, gemm_fp4_blkscaled_persistent_kernel,
        d_descs, d_descs + 1, d_descs + 2, d_descs + 3,
        C, M, N, K,
        A.tensor_scale, B.tensor_scale,
        bias, epilogue,
        d_tile_counter, total_tiles, grid_n);

    // Cleanup
    cudaFreeAsync(d_tile_counter, stream);
    cudaFreeAsync(d_descs, stream);
    cudaFreeAsync(sfa_interleaved, stream);
    cudaFreeAsync(sfb_interleaved, stream);
    if (A_data_padded) cudaFreeAsync(A_data_padded, stream);
}

}  // namespace blaze
