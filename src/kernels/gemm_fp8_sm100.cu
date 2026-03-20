/**
 * gemm_fp8_sm100.cu — Production FP8 GEMM kernel for SM100 (Blackwell).
 *
 * Architecture: Warp-specialized, 2-stage double-buffered TMA pipeline.
 *   Warp 0:    TMA Producer — issues async TMA loads for A, B tiles
 *   Warps 1-2: MMA Consumer — issue tcgen05.mma, accumulate in TMEM
 *   Warp 3:    Epilogue     — TMEM → SMEM → global with optional fusion
 *
 * Tile: 128×128×64 (M×N×K per CTA)
 * Inputs: E4M3 (FP8)
 * Accumulator: FP32 in TMEM
 * Output: FP16
 */

#include "gemm/fp8_gemm_sm100.cuh"
#include "gemm/tmem_utils.cuh"
#include "gemm/tma_utils.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdio>

namespace blaze {

using Config = Fp8GemmConfig;

// Warp role assignment
enum WarpRole : int {
    TMA_PRODUCER = 0,
    MMA_CONSUMER_0 = 1,
    MMA_CONSUMER_1 = 2,
    EPILOGUE = 3,
};

/**
 * Shared memory layout for double-buffered pipeline.
 *
 * Buffer 0: [A0 | B0]
 * Buffer 1: [A1 | B1]
 * C staging: [C]
 * mbarriers: [mbar_load[2] | mbar_mma[2]]
 */
struct SmemLayout {
    __nv_fp8_e4m3 A[Config::PIPELINE_STAGES][Config::TILE_M * Config::TILE_K];
    __nv_fp8_e4m3 B[Config::PIPELINE_STAGES][Config::TILE_K * Config::TILE_N];
    float C[Config::TILE_M * Config::TILE_N];
    uint64_t mbar_load[Config::PIPELINE_STAGES];  // TMA completion barriers
    uint64_t mbar_mma[Config::PIPELINE_STAGES];   // MMA completion barriers
};

/**
 * TMA Producer warp: Loads A and B tiles asynchronously.
 *
 * Double-buffered: while MMA consumes stage j, TMA loads stage (j+1)%2.
 */
__device__ __forceinline__
void tma_producer(
    SmemLayout* smem,
    const TmaDescriptor* desc_A,
    const TmaDescriptor* desc_B,
    int bm,    // Block row offset in output
    int bn,    // Block col offset in output
    int K,     // Full K dimension
    int lane_id
) {
    const int num_k_tiles = K / Config::TILE_K;

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int stage = k_tile % Config::PIPELINE_STAGES;
        int k_offset = k_tile * Config::TILE_K;

        if (lane_id == 0) {
            // Set expected transaction bytes on the load barrier
            uint32_t tx_bytes_A = Config::SMEM_A_BYTES;
            uint32_t tx_bytes_B = Config::SMEM_B_BYTES;
            mbarrier_expect_tx(&smem->mbar_load[stage], tx_bytes_A + tx_bytes_B);

            // Issue TMA loads for A tile (bm, k_offset) and B tile (k_offset, bn)
            tma_load_2d(smem->A[stage], desc_A, k_offset, bm, &smem->mbar_load[stage]);
            tma_load_2d(smem->B[stage], desc_B, bn, k_offset, &smem->mbar_load[stage]);
        }

        // Wait for MMA to finish with this stage's buffer (if not first iteration)
        if (k_tile >= Config::PIPELINE_STAGES) {
            mbarrier_wait(&smem->mbar_mma[stage], (k_tile / Config::PIPELINE_STAGES) & 1);
        }
    }
}

/**
 * MMA Consumer warps: Execute tcgen05.mma instructions.
 *
 * Each iteration:
 *   1. Wait for TMA to complete loading this stage
 *   2. Issue tcgen05.mma (A from SMEM, B from SMEM, accumulate in TMEM)
 *   3. Signal MMA completion for buffer reuse
 */
__device__ __forceinline__
void mma_consumer(
    SmemLayout* smem,
    tmem_addr_t tmem_addr,
    int K,
    int warp_id
) {
    const int num_k_tiles = K / Config::TILE_K;

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int stage = k_tile % Config::PIPELINE_STAGES;

        // Wait for TMA to finish loading this stage
        mbarrier_wait(&smem->mbar_load[stage], (k_tile / Config::PIPELINE_STAGES) & 1);

        // Issue tcgen05.mma: FP8 inputs from SMEM, FP32 accumulator in TMEM
        uint32_t smem_a_addr = static_cast<uint32_t>(
            __cvta_generic_to_shared(smem->A[stage]));
        uint32_t smem_b_addr = static_cast<uint32_t>(
            __cvta_generic_to_shared(smem->B[stage]));

        // tcgen05.mma with E4M3 input format
        // The accumulate flag is 1 for all iterations (TMEM accumulates across K tiles)
        asm volatile(
            "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, 0;\n"
            :
            : "r"(tmem_addr), "l"((uint64_t)smem_a_addr),
              "l"((uint64_t)smem_b_addr), "r"(1)
            : "memory"
        );

        __syncwarp();

        // Signal that MMA is done with this buffer (producer can reuse)
        if (warp_id == WarpRole::MMA_CONSUMER_0) {
            mbarrier_arrive(&smem->mbar_mma[stage]);
        }
    }
}

/**
 * Epilogue warp: Transfer results from TMEM → SMEM → global memory.
 * Optionally fuses bias addition and/or activation.
 */
__device__ __forceinline__
void epilogue(
    SmemLayout* smem,
    tmem_addr_t tmem_addr,
    half* C_global,
    int bm, int bn, int M, int N,
    const half* bias,
    GemmEpilogue epilogue_op,
    int tid
) {
    int local_tid = tid - (WarpRole::EPILOGUE * 32);  // Thread within epilogue warp

    // Store TMEM → SMEM via tcgen05.ld (all warp threads participate)
    tmem_store_to_smem(smem->C, tmem_addr, Config::TILE_N);
    __syncwarp();

    // Apply epilogue and store to global memory
    // Each thread in the epilogue warp handles a chunk of the output tile
    const int elems_per_thread = (Config::TILE_M * Config::TILE_N) / 32;

    for (int i = 0; i < elems_per_thread; i++) {
        int idx = local_tid + i * 32;
        int row = idx / Config::TILE_N;
        int col = idx % Config::TILE_N;
        int global_row = bm + row;
        int global_col = bn + col;

        if (global_row < M && global_col < N) {
            float val = smem->C[idx];

            // Epilogue fusion
            switch (epilogue_op) {
                case GemmEpilogue::BIAS:
                    val += __half2float(bias[global_col]);
                    break;
                case GemmEpilogue::SILU:
                    val = val / (1.0f + expf(-val));  // SiLU: x * sigmoid(x)
                    break;
                case GemmEpilogue::BIAS_SILU:
                    val += __half2float(bias[global_col]);
                    val = val / (1.0f + expf(-val));
                    break;
                case GemmEpilogue::NONE:
                default:
                    break;
            }

            C_global[global_row * N + global_col] = __float2half(val);
        }
    }
}

/**
 * Main FP8 GEMM kernel.
 */
__cluster_dims__(1, 1, 1)
__global__ void __launch_bounds__(Config::BLOCK_SIZE)
gemm_fp8_kernel(
    const TmaDescriptor* __restrict__ desc_A,
    const TmaDescriptor* __restrict__ desc_B,
    half* __restrict__ C,
    int M, int N, int K,
    const half* __restrict__ bias,
    GemmEpilogue epilogue_op
) {
    extern __shared__ char smem_raw[];
    SmemLayout* smem = reinterpret_cast<SmemLayout*>(smem_raw);

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // CTA output tile position
    const int bm = blockIdx.y * Config::TILE_M;
    const int bn = blockIdx.x * Config::TILE_N;

    // Initialize mbarriers (thread 0 only)
    if (tid == 0) {
        for (int s = 0; s < Config::PIPELINE_STAGES; s++) {
            mbarrier_init(&smem->mbar_load[s], 1);  // 1 = TMA transaction-based
            mbarrier_init(&smem->mbar_mma[s], 1);   // 1 = MMA consumer arrival
        }
    }
    __syncthreads();

    // Allocate TMEM for accumulator
    __shared__ uint32_t smem_tmem_addr;
    tmem_addr_t tmem_addr;
    if (warp_id == 0) {
        tmem_alloc(&smem_tmem_addr, Config::TMEM_COLUMNS);
    }
    __syncthreads();
    tmem_addr = smem_tmem_addr;

    // Warp-specialized execution
    switch (warp_id) {
        case WarpRole::TMA_PRODUCER:
            tma_producer(smem, desc_A, desc_B, bm, bn, K, lane_id);
            break;

        case WarpRole::MMA_CONSUMER_0:
        case WarpRole::MMA_CONSUMER_1:
            mma_consumer(smem, tmem_addr, K, warp_id);
            break;

        case WarpRole::EPILOGUE:
            // Wait for all MMA iterations to complete
            __syncthreads();
            epilogue(smem, tmem_addr, C, bm, bn, M, N, bias, epilogue_op, tid);
            break;
    }

    // Deallocate TMEM (MANDATORY)
    __syncthreads();
    if (warp_id == 0) {
        tmem_dealloc(tmem_addr, Config::TMEM_COLUMNS);
    }
}

// --- Host launch function ---

void launch_gemm_fp8(
    const __nv_fp8_e4m3* A,
    const __nv_fp8_e4m3* B,
    half* C,
    int M, int N, int K,
    const half* bias,
    GemmEpilogue epilogue,
    cudaStream_t stream
) {
    // Create TMA descriptors
    TmaDescriptor h_desc_A, h_desc_B;
    create_tma_desc_2d(&h_desc_A, A, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                       M, K, Config::TILE_M, Config::TILE_K, TmaSwizzle::B128);
    create_tma_desc_2d(&h_desc_B, B, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                       K, N, Config::TILE_K, Config::TILE_N, TmaSwizzle::B128);

    // Copy descriptors to device
    TmaDescriptor *d_desc_A, *d_desc_B;
    cudaMalloc(&d_desc_A, sizeof(TmaDescriptor));
    cudaMalloc(&d_desc_B, sizeof(TmaDescriptor));
    cudaMemcpyAsync(d_desc_A, &h_desc_A, sizeof(TmaDescriptor),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_desc_B, &h_desc_B, sizeof(TmaDescriptor),
                    cudaMemcpyHostToDevice, stream);

    // Grid and block dimensions
    dim3 grid((N + Config::TILE_N - 1) / Config::TILE_N,
              (M + Config::TILE_M - 1) / Config::TILE_M);
    dim3 block(Config::BLOCK_SIZE);

    // Shared memory
    size_t smem_size = sizeof(SmemLayout);
    cudaFuncSetAttribute(gemm_fp8_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    // Cluster launch required for tcgen05 instructions
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

    cudaLaunchKernelEx(&launch_config, gemm_fp8_kernel,
        d_desc_A, d_desc_B, C, M, N, K, bias, epilogue);

    // Free device descriptors (in practice, cache these)
    cudaFreeAsync(d_desc_A, stream);
    cudaFreeAsync(d_desc_B, stream);
}

}  // namespace blaze
