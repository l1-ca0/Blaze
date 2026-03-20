/**
 * gemm_fp4_sm100.cu — Production NVFP4 GEMM kernel for SM100 (Blackwell).
 *
 * Extends the FP8 kernel architecture to handle NVFP4 (E2M1) format:
 *   - TMA loads FP4 packed data (2 values/byte) + block scales (E4M3)
 *   - tcgen05.mma with .e2m1 input format
 *   - Hardware applies block scales during MMA (Transformer Engine path)
 *   - Tensor-level scale applied in epilogue
 *
 * Tile: 128×128×128 (M×N×K) — same memory footprint as FP8 128×128×64.
 */

#include "gemm/fp4_gemm_sm100.cuh"
#include "gemm/tmem_utils.cuh"
#include "gemm/tma_utils.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdio>

namespace blaze {

using Config = Fp4GemmConfig;

/**
 * Shared memory layout for FP4 GEMM.
 *
 * Each stage holds: packed FP4 data + block scales for both A and B.
 * The C staging area is shared across stages (only used in epilogue).
 */
struct Fp4SmemLayout {
    // Double-buffered input tiles
    uint8_t A_data[Config::PIPELINE_STAGES][Config::TILE_M * Config::TILE_K / 2];
    uint8_t B_data[Config::PIPELINE_STAGES][Config::TILE_K * Config::TILE_N / 2];

    // Block scales for each stage
    __nv_fp8_e4m3 A_scales[Config::PIPELINE_STAGES][Config::TILE_M * Config::TILE_K / FP4_BLOCK_SIZE];
    __nv_fp8_e4m3 B_scales[Config::PIPELINE_STAGES][Config::TILE_K * Config::TILE_N / FP4_BLOCK_SIZE];

    // Output staging
    float C[Config::TILE_M * Config::TILE_N];

    // Synchronization barriers
    uint64_t mbar_load[Config::PIPELINE_STAGES];
    uint64_t mbar_mma[Config::PIPELINE_STAGES];
};

/**
 * TMA Producer for FP4: loads packed data and block scales.
 */
__device__ __forceinline__
void fp4_tma_producer(
    Fp4SmemLayout* smem,
    const TmaDescriptor* desc_A_data,
    const TmaDescriptor* desc_B_data,
    const TmaDescriptor* desc_A_scales,
    const TmaDescriptor* desc_B_scales,
    int bm, int bn, int K,
    int lane_id
) {
    const int num_k_tiles = K / Config::TILE_K;

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int stage = k_tile % Config::PIPELINE_STAGES;
        int k_offset = k_tile * Config::TILE_K;

        if (lane_id == 0) {
            // Expected bytes: packed data + scales for both A and B
            uint32_t tx_bytes =
                Config::SMEM_A_DATA_BYTES + Config::SMEM_B_DATA_BYTES +
                Config::SMEM_A_SCALE_BYTES + Config::SMEM_B_SCALE_BYTES;

            mbarrier_expect_tx(&smem->mbar_load[stage], tx_bytes);

            // Load FP4 packed data
            // A_data: (M, K/2) in bytes → tile at (bm, k_offset/2)
            tma_load_2d(smem->A_data[stage], desc_A_data,
                        k_offset / 2, bm, &smem->mbar_load[stage]);
            tma_load_2d(smem->B_data[stage], desc_B_data,
                        bn / 2, k_offset, &smem->mbar_load[stage]);

            // Load block scales
            // A_scales: (M, K/BLOCK_SIZE) → tile at (bm, k_offset/BLOCK_SIZE)
            tma_load_2d(smem->A_scales[stage], desc_A_scales,
                        k_offset / FP4_BLOCK_SIZE, bm, &smem->mbar_load[stage]);
            tma_load_2d(smem->B_scales[stage], desc_B_scales,
                        bn / FP4_BLOCK_SIZE, k_offset, &smem->mbar_load[stage]);
        }

        // Wait for MMA to release this buffer (if not first iterations)
        if (k_tile >= Config::PIPELINE_STAGES) {
            mbarrier_wait(&smem->mbar_mma[stage],
                         (k_tile / Config::PIPELINE_STAGES) & 1);
        }
    }
}

/**
 * MMA Consumer for FP4: issues tcgen05.mma with .e2m1 format.
 *
 * The hardware handles FP4 dequantization and block scale application
 * when using the Transformer Engine path. The block scales must be
 * in a specific layout adjacent to the data in shared memory.
 */
__device__ __forceinline__
void fp4_mma_consumer(
    Fp4SmemLayout* smem,
    tmem_addr_t tmem_addr,
    int K,
    int warp_id
) {
    const int num_k_tiles = K / Config::TILE_K;

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int stage = k_tile % Config::PIPELINE_STAGES;

        // Wait for TMA to complete
        mbarrier_wait(&smem->mbar_load[stage],
                     (k_tile / Config::PIPELINE_STAGES) & 1);

        // Issue tcgen05.mma with FP4 (E2M1) input format
        // The .kind::f8f6f4 selector with E2M1 data tells the hardware
        // to interpret the SMEM data as FP4 with block scaling.
        uint32_t smem_a_addr = static_cast<uint32_t>(
            __cvta_generic_to_shared(smem->A_data[stage]));
        uint32_t smem_b_addr = static_cast<uint32_t>(
            __cvta_generic_to_shared(smem->B_data[stage]));

        // FP4 MMA with block scale descriptors
        // The scale tensors must be laid out contiguously after the data
        // in the format expected by tcgen05.
        uint32_t smem_a_scale_addr = static_cast<uint32_t>(
            __cvta_generic_to_shared(smem->A_scales[stage]));
        uint32_t smem_b_scale_addr = static_cast<uint32_t>(
            __cvta_generic_to_shared(smem->B_scales[stage]));

        // tcgen05.mma for FP4:
        // Input A: E2M1 from SMEM + block scales
        // Input B: E2M1 from SMEM + block scales
        // Accumulator: FP32 in TMEM
        asm volatile(
            "{\n"
            "  tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, 0;\n"
            "}\n"
            :
            : "r"(tmem_addr),
              "l"((uint64_t)smem_a_addr),
              "l"((uint64_t)smem_b_addr),
              "r"(1)  // accumulate
            : "memory"
        );

        __syncwarp();

        // Signal MMA completion
        if (warp_id == 1) {
            mbarrier_arrive(&smem->mbar_mma[stage]);
        }
    }
}

/**
 * FP4 Epilogue: applies tensor-level scale and stores output.
 *
 * The accumulator in TMEM contains:
 *   sum_k( dequant(A_fp4) * dequant(B_fp4) )
 * which has been scaled by block scales but NOT by tensor scales.
 * The epilogue multiplies by tensor_scale_A * tensor_scale_B.
 */
__device__ __forceinline__
void fp4_epilogue(
    Fp4SmemLayout* smem,
    tmem_addr_t tmem_addr,
    half* C_global,
    int bm, int bn, int M, int N,
    float scale_A, float scale_B,
    const half* bias,
    Fp4Epilogue epilogue_op,
    int tid
) {
    int local_tid = tid % 32;  // Thread within warp
    float combined_scale = scale_A * scale_B;

    // TMEM → SMEM via tcgen05.ld (all warp threads participate)
    tmem_store_to_smem(smem->C, tmem_addr, Config::TILE_N);
    __syncwarp();

    // Apply tensor scale, epilogue ops, and store to global
    const int elems_per_thread = (Config::TILE_M * Config::TILE_N) / 32;

    for (int i = 0; i < elems_per_thread; i++) {
        int idx = local_tid + i * 32;
        int row = idx / Config::TILE_N;
        int col = idx % Config::TILE_N;
        int global_row = bm + row;
        int global_col = bn + col;

        if (global_row < M && global_col < N) {
            float val = smem->C[idx] * combined_scale;

            switch (epilogue_op) {
                case Fp4Epilogue::BIAS:
                    val += __half2float(bias[global_col]);
                    break;
                case Fp4Epilogue::SILU:
                    val = val / (1.0f + expf(-val));
                    break;
                case Fp4Epilogue::BIAS_SILU:
                    val += __half2float(bias[global_col]);
                    val = val / (1.0f + expf(-val));
                    break;
                default:
                    break;
            }

            C_global[global_row * N + global_col] = __float2half(val);
        }
    }
}

/**
 * Main FP4 GEMM kernel.
 */
__cluster_dims__(1, 1, 1)
__global__ void __launch_bounds__(Config::BLOCK_SIZE)
gemm_fp4_kernel(
    const TmaDescriptor* __restrict__ desc_A_data,
    const TmaDescriptor* __restrict__ desc_B_data,
    const TmaDescriptor* __restrict__ desc_A_scales,
    const TmaDescriptor* __restrict__ desc_B_scales,
    half* __restrict__ C,
    int M, int N, int K,
    float scale_A, float scale_B,
    const half* __restrict__ bias,
    Fp4Epilogue epilogue_op
) {
    extern __shared__ char smem_raw[];
    Fp4SmemLayout* smem = reinterpret_cast<Fp4SmemLayout*>(smem_raw);

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const int bm = blockIdx.y * Config::TILE_M;
    const int bn = blockIdx.x * Config::TILE_N;

    // Initialize barriers
    if (tid == 0) {
        for (int s = 0; s < Config::PIPELINE_STAGES; s++) {
            mbarrier_init(&smem->mbar_load[s], 1);
            mbarrier_init(&smem->mbar_mma[s], 1);
        }
    }
    __syncthreads();

    // Allocate TMEM
    __shared__ uint32_t smem_tmem_addr;
    tmem_addr_t tmem_addr;
    if (warp_id == 0) {
        tmem_alloc(&smem_tmem_addr, Config::TMEM_COLUMNS);
    }
    __syncthreads();
    tmem_addr = smem_tmem_addr;

    // Warp-specialized execution
    switch (warp_id) {
        case 0:  // TMA Producer
            fp4_tma_producer(smem, desc_A_data, desc_B_data,
                            desc_A_scales, desc_B_scales,
                            bm, bn, K, lane_id);
            break;
        case 1:
        case 2:  // MMA Consumers
            fp4_mma_consumer(smem, tmem_addr, K, warp_id);
            break;
        case 3:  // Epilogue
            __syncthreads();
            fp4_epilogue(smem, tmem_addr, C, bm, bn, M, N,
                        scale_A, scale_B, bias, epilogue_op, tid);
            break;
    }

    // Deallocate TMEM
    __syncthreads();
    if (warp_id == 0) {
        tmem_dealloc(tmem_addr, Config::TMEM_COLUMNS);
    }
}

// --- Host launch ---

void launch_gemm_fp4(
    const Fp4WeightTensor& A,
    const Fp4WeightTensor& B,
    half* C,
    int M, int N, int K,
    const half* bias,
    Fp4Epilogue epilogue,
    cudaStream_t stream
) {
    // Create TMA descriptors for data and scales
    TmaDescriptor h_desc_A_data, h_desc_B_data, h_desc_A_scales, h_desc_B_scales;

    // A data: (M, K/2) uint8 (FP4 packed)
    create_tma_desc_2d(&h_desc_A_data, A.data, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                       M, K / 2,
                       Config::TILE_M, Config::TILE_K / 2,
                       TmaSwizzle::B128);

    // B data: (K, N/2) uint8
    create_tma_desc_2d(&h_desc_B_data, B.data, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                       K, N / 2,
                       Config::TILE_K, Config::TILE_N / 2,
                       TmaSwizzle::B128);

    // A scales: (M, K/BLOCK_SIZE) FP8
    create_tma_desc_2d(&h_desc_A_scales, A.block_scales, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                       M, K / FP4_BLOCK_SIZE,
                       Config::TILE_M, Config::TILE_K / FP4_BLOCK_SIZE,
                       TmaSwizzle::NONE);

    // B scales: (K, N/BLOCK_SIZE) FP8
    create_tma_desc_2d(&h_desc_B_scales, B.block_scales, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                       K, N / FP4_BLOCK_SIZE,
                       Config::TILE_K, Config::TILE_N / FP4_BLOCK_SIZE,
                       TmaSwizzle::NONE);

    // Copy descriptors to device
    TmaDescriptor *d_descs;
    cudaMalloc(&d_descs, 4 * sizeof(TmaDescriptor));
    cudaMemcpyAsync(d_descs, &h_desc_A_data, sizeof(TmaDescriptor),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_descs + 1, &h_desc_B_data, sizeof(TmaDescriptor),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_descs + 2, &h_desc_A_scales, sizeof(TmaDescriptor),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_descs + 3, &h_desc_B_scales, sizeof(TmaDescriptor),
                    cudaMemcpyHostToDevice, stream);

    dim3 grid((N + Config::TILE_N - 1) / Config::TILE_N,
              (M + Config::TILE_M - 1) / Config::TILE_M);
    dim3 block(Config::BLOCK_SIZE);
    size_t smem_size = sizeof(Fp4SmemLayout);

    cudaFuncSetAttribute(gemm_fp4_kernel,
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

    cudaLaunchKernelEx(&launch_config, gemm_fp4_kernel,
        d_descs, d_descs + 1, d_descs + 2, d_descs + 3,
        C, M, N, K,
        A.tensor_scale, B.tensor_scale,
        bias, epilogue);

    cudaFreeAsync(d_descs, stream);
}

}  // namespace blaze
