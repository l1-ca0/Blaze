/**
 * gemm_fp4_sm100.cu — NVFP4 GEMM for SM100 (Blackwell).
 *
 * Computes C[M,N] = A[M,K] × B[K,N] where both A and B are in NVFP4
 * format (E2M1 data + E4M3 block scales + FP32 tensor scale).
 * FP32 TMEM accumulator, FP16 output. Tensor scales applied in epilogue.
 *
 * Warp-specialized, 2-stage double-buffered pipeline:
 *   Warp 0: MMA consumer  (tcgen05.mma.kind::f8f6f4, E2M1 × E2M1)
 *   Warp 1: TMA producer  (async bulk loads for packed data; global loads for scales)
 *   Warp 2: Idle
 *   Warp 3: Epilogue      (TMEM → SMEM → global store with tensor scale)
 *
 * Tile: 128×128×128 (M×N×K). K_PER_MMA=64, 2 inner iterations per tile.
 * Packed data loaded via TMA; block scales loaded via warp-cooperative
 * global memory reads (scale tiles are too narrow for TMA's 16-byte minimum).
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

struct __align__(128) Fp4SmemLayout {
    alignas(8) uint64_t mbar_load[Config::PIPELINE_STAGES];
    alignas(8) uint64_t mbar_mma[Config::PIPELINE_STAGES];
    uint32_t tmem_addr;
    char _pad[128 - (2 * 16 + 4)];

    uint8_t A_data[Config::PIPELINE_STAGES][Config::TILE_M * Config::TILE_K / 2];
    uint8_t B_data[Config::PIPELINE_STAGES][Config::TILE_K * Config::TILE_N / 2];
    __nv_fp8_e4m3 A_scales[Config::PIPELINE_STAGES][Config::SMEM_A_SCALE_BYTES];
    __nv_fp8_e4m3 B_scales[Config::PIPELINE_STAGES][Config::SMEM_B_SCALE_BYTES];
    float C[Config::TILE_M * Config::TILE_N];
};

__device__ __forceinline__
void fp4_tma_producer(
    Fp4SmemLayout* smem,
    const TmaDescriptor* desc_A_data,
    const TmaDescriptor* desc_B_data,
    const __nv_fp8_e4m3* global_A_scales,
    const __nv_fp8_e4m3* global_B_scales,
    int bm, int bn, int M, int N, int K, int lane_id
) {
    const int num_k_tiles = K / Config::TILE_K;
    const int a_scale_stride = K / FP4_BLOCK_SIZE;
    const int b_scale_stride = N / FP4_BLOCK_SIZE;

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int stage = k_tile % Config::PIPELINE_STAGES;
        int k_offset = k_tile * Config::TILE_K;

        if (k_tile >= Config::PIPELINE_STAGES) {
            mbarrier_wait(&smem->mbar_mma[stage],
                         ((k_tile / Config::PIPELINE_STAGES) + 1) & 1);
        }

        if (lane_id == 0) {
            uint32_t tx_bytes = Config::SMEM_A_DATA_BYTES + Config::SMEM_B_DATA_BYTES;
            mbarrier_expect_tx(&smem->mbar_load[stage], tx_bytes);

            tma_load_2d(smem->A_data[stage], desc_A_data,
                        k_offset / 2, bm, &smem->mbar_load[stage]);
            tma_load_2d(smem->B_data[stage], desc_B_data,
                        bn / 2, k_offset, &smem->mbar_load[stage]);
        }

        // Warp-cooperative load of A block scales from global memory.
        {
            const int a_scale_col_start = k_offset / FP4_BLOCK_SIZE;
            const int total_a = Config::SMEM_A_SCALE_BYTES;
            for (int i = lane_id; i < total_a; i += 32) {
                int row = i / Config::A_SCALE_COLS;
                int col = i % Config::A_SCALE_COLS;
                int global_idx = (bm + row) * a_scale_stride + (a_scale_col_start + col);
                smem->A_scales[stage][i] = global_A_scales[global_idx];
            }
        }

        // Warp-cooperative load of B block scales from global memory.
        {
            const int b_scale_col_start = bn / FP4_BLOCK_SIZE;
            const int total_b = Config::SMEM_B_SCALE_BYTES;
            for (int i = lane_id; i < total_b; i += 32) {
                int row = i / Config::B_SCALE_COLS;
                int col = i % Config::B_SCALE_COLS;
                int global_idx = (k_offset + row) * b_scale_stride + (b_scale_col_start + col);
                smem->B_scales[stage][i] = global_B_scales[global_idx];
            }
        }
    }
}

/** MMA consumer: issues tcgen05.mma across K tiles with double-buffered SMEM. */
__device__ __forceinline__
void fp4_mma_consumer(Fp4SmemLayout* smem, tmem_addr_t tmem_addr, int K) {
    const int num_k_tiles = K / Config::TILE_K;
    constexpr int K_PER_MMA = 64;
    constexpr int NUM_K_ITERS = Config::TILE_K / K_PER_MMA;

    uint32_t idesc = make_idesc_f8f6f4(5, 5, Config::TILE_M, Config::TILE_N);
    bool first_mma = true;

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int stage = k_tile % Config::PIPELINE_STAGES;
        mbarrier_wait(&smem->mbar_load[stage], (k_tile / Config::PIPELINE_STAGES) & 1);

        uint32_t smem_a_base = static_cast<uint32_t>(__cvta_generic_to_shared(smem->A_data[stage]));
        uint32_t smem_b_base = static_cast<uint32_t>(__cvta_generic_to_shared(smem->B_data[stage]));

        for (int ki = 0; ki < NUM_K_ITERS; ki++) {
            asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

            uint32_t smem_a_addr = smem_a_base + ki * (K_PER_MMA / 2);
            uint32_t smem_b_addr = smem_b_base + ki * (K_PER_MMA / 2) * Config::TILE_N;

            uint64_t desc_a = make_smem_desc(smem_a_addr, Config::TILE_K / 2, 4);
            uint64_t desc_b = make_smem_desc(smem_b_addr, Config::TILE_N / 2, 4);

            if (elect_one_sync()) {
                if (first_mma) {
                    asm volatile(
                        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, 0;\n"
                        : : "r"(tmem_addr), "l"(desc_a), "l"(desc_b), "r"(idesc) : "memory");
                } else {
                    asm volatile(
                        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, 1;\n"
                        : : "r"(tmem_addr), "l"(desc_a), "l"(desc_b), "r"(idesc) : "memory");
                }
            }
            first_mma = false;

            asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
        }

        __syncwarp();
        if (elect_one_sync()) {
            mbarrier_arrive(&smem->mbar_mma[stage]);
        }
    }
}

__device__ __forceinline__
void fp4_epilogue(
    Fp4SmemLayout* smem, tmem_addr_t tmem_addr,
    half* C_global, int bm, int bn, int M, int N,
    float scale_A, float scale_B,
    const half* bias, Fp4Epilogue epilogue_op, int tid
) {
    int local_tid = tid % 32;
    float combined_scale = scale_A * scale_B;
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

__cluster_dims__(1, 1, 1)
__global__ void __launch_bounds__(Config::BLOCK_SIZE)
gemm_fp4_kernel(
    const TmaDescriptor* __restrict__ desc_A_data,
    const TmaDescriptor* __restrict__ desc_B_data,
    const __nv_fp8_e4m3* __restrict__ global_A_scales,
    const __nv_fp8_e4m3* __restrict__ global_B_scales,
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

    if (tid == 0) {
        for (int s = 0; s < Config::PIPELINE_STAGES; s++) {
            mbarrier_init(&smem->mbar_load[s], 1);
            mbarrier_init(&smem->mbar_mma[s], 1);
        }
    }
    __syncthreads();

    tmem_addr_t tmem_addr;
    if (warp_id == 0) {
        tmem_alloc(&smem->tmem_addr, Config::TMEM_COLUMNS);
    }
    __syncthreads();
    tmem_addr = smem->tmem_addr;

    switch (warp_id) {
        case 0:
            fp4_mma_consumer(smem, tmem_addr, K);
            break;
        case 1:
            fp4_tma_producer(smem, desc_A_data, desc_B_data,
                            global_A_scales, global_B_scales,
                            bm, bn, M, N, K, lane_id);
            break;
        case 2:
        case 3:
            break;
    }

    __syncthreads();
    tmem_store_to_smem_warp(smem->C, tmem_addr, Config::TILE_N, warp_id);
    __syncthreads();

    if (warp_id == 3) {
        fp4_epilogue(smem, tmem_addr, C, bm, bn, M, N,
                    scale_A, scale_B, bias, epilogue_op, tid);
    }

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
    const uint8_t* A_data_tma = A.data;
    const __nv_fp8_e4m3* A_scales_ptr = A.block_scales;
    uint8_t* A_data_padded = nullptr;
    __nv_fp8_e4m3* A_scales_padded = nullptr;
    int M_tma = M;

    if (M < Config::TILE_M) {
        M_tma = Config::TILE_M;
        size_t data_bytes = (size_t)M_tma * K / 2;
        size_t scale_count = (size_t)M_tma * K / FP4_BLOCK_SIZE;
        cudaMalloc(&A_data_padded, data_bytes);
        cudaMalloc(&A_scales_padded, scale_count * sizeof(__nv_fp8_e4m3));
        cudaMemsetAsync(A_data_padded, 0, data_bytes, stream);
        cudaMemsetAsync(A_scales_padded, 0, scale_count * sizeof(__nv_fp8_e4m3), stream);
        cudaMemcpyAsync(A_data_padded, A.data, (size_t)M * K / 2,
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(A_scales_padded, A.block_scales,
                        (size_t)M * K / FP4_BLOCK_SIZE * sizeof(__nv_fp8_e4m3),
                        cudaMemcpyDeviceToDevice, stream);
        A_data_tma = A_data_padded;
        A_scales_ptr = A_scales_padded;
    }

    // TMA descriptors for packed data only; scales loaded via global memory.
    TmaDescriptor h_desc_A_data, h_desc_B_data;

    create_tma_desc_2d(&h_desc_A_data, A_data_tma, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                       M_tma, K / 2, Config::TILE_M, Config::TILE_K / 2, TmaSwizzle::B64);
    create_tma_desc_2d(&h_desc_B_data, B.data, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                       K, N / 2, Config::TILE_K, Config::TILE_N / 2, TmaSwizzle::B64);

    TmaDescriptor *d_descs;
    cudaMalloc(&d_descs, 2 * sizeof(TmaDescriptor));
    cudaMemcpyAsync(d_descs,     &h_desc_A_data, sizeof(TmaDescriptor), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_descs + 1, &h_desc_B_data, sizeof(TmaDescriptor), cudaMemcpyHostToDevice, stream);

    dim3 grid((N + Config::TILE_N - 1) / Config::TILE_N,
              (M + Config::TILE_M - 1) / Config::TILE_M);
    dim3 block(Config::BLOCK_SIZE);
    size_t smem_size = sizeof(Fp4SmemLayout);

    cudaError_t err = cudaFuncSetAttribute(gemm_fp4_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "FP4 GEMM: cudaFuncSetAttribute failed: %s (smem=%zu)\n",
                cudaGetErrorString(err), smem_size);
        return;
    }

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
        d_descs, d_descs + 1, A_scales_ptr, B.block_scales,
        C, M, N, K, A.tensor_scale, B.tensor_scale, bias, epilogue);

    cudaFreeAsync(d_descs, stream);
    if (A_data_padded) cudaFreeAsync(A_data_padded, stream);
    if (A_scales_padded) cudaFreeAsync(A_scales_padded, stream);
}

}  // namespace blaze
