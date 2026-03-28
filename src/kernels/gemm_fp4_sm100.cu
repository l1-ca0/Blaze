/**
 * gemm_fp4_sm100.cu — NVFP4 GEMM for SM100 (Blackwell).
 *
 * Computes C[M,N] = A[M,K] × B[K,N] where both A and B are in NVFP4
 * format (E2M1 data + E4M3 block scales + FP32 tensor scale).
 * FP32 TMEM accumulator, FP16 output. Tensor scales applied in epilogue.
 *
 * Warp-specialized, 2-stage double-buffered pipeline:
 *   Warp 0: MMA consumer  (tcgen05.mma.kind::f8f6f4, E2M1 × E2M1)
 *   Warp 1: TMA producer  (async bulk loads for A_data + B_data)
 *   Warp 2: A_scale loader (vectorized uint2 global loads for A block scales)
 *   Warp 3: B_scale loader (vectorized uint2 global loads for B block scales)
 *   All warps: Epilogue   (TMEM → registers → global, no SMEM staging)
 *
 * Tile: 128×128×128 (M×N×K). K_PER_MMA=64, 2 inner iterations per tile.
 * Packed data loaded via TMA with B64 swizzle; block scales loaded
 * concurrently by dedicated warps via vectorized global loads.
 *
 * Synchronization: mbar_load init=3 (3 arrivals required):
 *   - Warp 1: mbarrier_expect_tx (implicit arrival, counts as 1) + TMA loads
 *   - Warp 2: mbarrier_arrive after A_scales written to SMEM
 *   - Warp 3: mbarrier_arrive after B_scales written to SMEM
 * Barrier flips only when all 3 arrivals AND all TMA bytes are received.
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

// SMEM layout — scales loaded by dedicated warps 2/3 concurrently with TMA.
struct __align__(128) Fp4SmemLayout {
    alignas(8) uint64_t mbar_load[Config::PIPELINE_STAGES];
    alignas(8) uint64_t mbar_mma[Config::PIPELINE_STAGES];
    uint32_t tmem_addr;
    char _pad[128 - (2 * 16 + 4)];

    uint8_t A_data[Config::PIPELINE_STAGES][Config::TILE_M * Config::TILE_K / 2];
    uint8_t B_data[Config::PIPELINE_STAGES][Config::TILE_K * Config::TILE_N / 2];
    __nv_fp8_e4m3 A_scales[Config::PIPELINE_STAGES][Config::SMEM_A_SCALE_BYTES];
    __nv_fp8_e4m3 B_scales[Config::PIPELINE_STAGES][Config::SMEM_B_SCALE_BYTES];
    // No float C[] — epilogue reads TMEM directly to registers
};

/**
 * TMA producer (warp 1): loads A_data and B_data via TMA only.
 * Scale loading is handled by warps 2 and 3 concurrently.
 * mbarrier_expect_tx counts as 1 implicit arrival (no explicit arrive needed).
 */
__device__ __forceinline__
void fp4_tma_producer(
    Fp4SmemLayout* smem,
    const TmaDescriptor* desc_A_data,
    const TmaDescriptor* desc_B_data,
    int bm, int bn, int K, int lane_id
) {
    const int num_k_tiles = K / Config::TILE_K;

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
        // No explicit mbarrier_arrive — expect_tx is the implicit arrival for warp 1.
    }
}

/**
 * A_scale loader (warp 2): vectorized uint2 global loads for A block scales.
 * 128 rows × 8 bytes/row = 1024 bytes per stage. 32 threads × 4 iters.
 * Arrives on mbar_load after SMEM writes complete.
 */
__device__ __forceinline__
void fp4_a_scale_loader(
    Fp4SmemLayout* smem,
    const __nv_fp8_e4m3* global_A_scales,
    int bm, int K, int lane_id
) {
    const int num_k_tiles = K / Config::TILE_K;
    const int a_scale_stride = K / FP4_BLOCK_SIZE;

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int stage = k_tile % Config::PIPELINE_STAGES;
        int k_offset = k_tile * Config::TILE_K;

        if (k_tile >= Config::PIPELINE_STAGES) {
            mbarrier_wait(&smem->mbar_mma[stage],
                         ((k_tile / Config::PIPELINE_STAGES) + 1) & 1);
        }

        const int a_scale_col_start = k_offset / FP4_BLOCK_SIZE;
        for (int r = lane_id; r < Config::TILE_M; r += 32) {
            *reinterpret_cast<uint2*>(&smem->A_scales[stage][r * Config::A_SCALE_COLS]) =
                *reinterpret_cast<const uint2*>(
                    &global_A_scales[(bm + r) * a_scale_stride + a_scale_col_start]);
        }
        __syncwarp();

        if (lane_id == 0) {
            mbarrier_arrive(&smem->mbar_load[stage]);
        }
    }
}

/**
 * B_scale loader (warp 3): vectorized uint2 global loads for B block scales.
 * 128 rows × 8 bytes/row = 1024 bytes per stage. 32 threads × 4 iters.
 * Arrives on mbar_load after SMEM writes complete.
 */
__device__ __forceinline__
void fp4_b_scale_loader(
    Fp4SmemLayout* smem,
    const __nv_fp8_e4m3* global_B_scales,
    int bn, int N, int K, int lane_id
) {
    const int num_k_tiles = K / Config::TILE_K;
    const int b_scale_stride = N / FP4_BLOCK_SIZE;
    const int b_scale_col_start = bn / FP4_BLOCK_SIZE;

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int stage = k_tile % Config::PIPELINE_STAGES;
        int k_offset = k_tile * Config::TILE_K;

        if (k_tile >= Config::PIPELINE_STAGES) {
            mbarrier_wait(&smem->mbar_mma[stage],
                         ((k_tile / Config::PIPELINE_STAGES) + 1) & 1);
        }

        for (int r = lane_id; r < Config::TILE_K; r += 32) {
            *reinterpret_cast<uint2*>(&smem->B_scales[stage][r * Config::B_SCALE_COLS]) =
                *reinterpret_cast<const uint2*>(
                    &global_B_scales[(k_offset + r) * b_scale_stride + b_scale_col_start]);
        }
        __syncwarp();

        if (lane_id == 0) {
            mbarrier_arrive(&smem->mbar_load[stage]);
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

/**
 * Direct TMEM → register → global epilogue for FP4 GEMM with vectorized stores.
 * All 4 warps participate. Each warp reads its own 32-row TMEM slice.
 * Packs 8 halves into uint4 for 16-byte aligned stores (8× less L2 traffic).
 */
__device__ __forceinline__
void fp4_epilogue_direct(
    tmem_addr_t tmem_addr,
    half* C_global, int bm, int bn, int M, int N,
    float scale_A, float scale_B,
    const half* bias, Fp4Epilogue epilogue_op,
    int warp_id, int lane_id
) {
    asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

    float combined_scale = scale_A * scale_B;
    int row = warp_id * 32 + lane_id;
    int global_row = bm + row;

    for (int cg = 0; cg < Config::TILE_N / 8; cg++) {
        float vals[8];
        tmem_load_8xf32(vals, tmem_addr, cg * 8);
        tmem_wait_ld();

        if (global_row < M) {
            int global_col_base = bn + cg * 8;

            // Apply combined scale and epilogue to all 8 values
            for (int c = 0; c < 8; c++) vals[c] *= combined_scale;

            switch (epilogue_op) {
                case Fp4Epilogue::BIAS:
                    for (int c = 0; c < 8; c++)
                        vals[c] += __half2float(bias[global_col_base + c]);
                    break;
                case Fp4Epilogue::SILU:
                    for (int c = 0; c < 8; c++)
                        vals[c] = vals[c] / (1.0f + expf(-vals[c]));
                    break;
                case Fp4Epilogue::BIAS_SILU:
                    for (int c = 0; c < 8; c++) {
                        vals[c] += __half2float(bias[global_col_base + c]);
                        vals[c] = vals[c] / (1.0f + expf(-vals[c]));
                    }
                    break;
                default:
                    break;
            }

            // Pack 8 halves into uint4 for vectorized 16-byte store
            half h[8];
            for (int c = 0; c < 8; c++) h[c] = __float2half(vals[c]);

            if (global_col_base + 7 < N) {
                *reinterpret_cast<uint4*>(&C_global[global_row * N + global_col_base]) =
                    *reinterpret_cast<uint4*>(h);
            } else {
                for (int c = 0; c < 8; c++) {
                    if (global_col_base + c < N)
                        C_global[global_row * N + global_col_base + c] = h[c];
                }
            }
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
            mbarrier_inval(&smem->mbar_load[s]);
            mbarrier_inval(&smem->mbar_mma[s]);
        }
        for (int s = 0; s < Config::PIPELINE_STAGES; s++) {
            // mbar_load: 3 arrivals = expect_tx (warp 1) + arrive (warp 2) + arrive (warp 3)
            mbarrier_init(&smem->mbar_load[s], 3);
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

    // K-loop: all 4 warps have dedicated roles
    switch (warp_id) {
        case 0:
            fp4_mma_consumer(smem, tmem_addr, K);
            break;
        case 1:
            fp4_tma_producer(smem, desc_A_data, desc_B_data,
                            bm, bn, K, lane_id);
            break;
        case 2:
            fp4_a_scale_loader(smem, global_A_scales, bm, K, lane_id);
            break;
        case 3:
            fp4_b_scale_loader(smem, global_B_scales, bn, N, K, lane_id);
            break;
    }

    // Wait for MMA completion, then all warps do TMEM → register → global
    __syncthreads();
    fp4_epilogue_direct(tmem_addr, C, bm, bn, M, N,
                       scale_A, scale_B, bias, epilogue_op,
                       warp_id, lane_id);

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

// ---------------------------------------------------------------------------
// Prepare/execute API
// ---------------------------------------------------------------------------

struct Fp4GemmPlan {
    uint8_t* A_data_padded;           // Padded A data if M < TILE_M (else nullptr)
    __nv_fp8_e4m3* A_scales_padded;   // Padded A scales if M < TILE_M (else nullptr)
    const __nv_fp8_e4m3* A_scales_ptr;
    const __nv_fp8_e4m3* B_scales_ptr;
    TmaDescriptor* d_descs;           // Device TMA descriptors [A_data, B_data]
    float scale_A;
    float scale_B;
    const half* bias;
    Fp4Epilogue epilogue;
    dim3 grid;
    dim3 block;
    size_t smem_size;
    int M, N, K;
};

Fp4GemmPlan* create_fp4_gemm_plan(
    const Fp4WeightTensor& A,
    const Fp4WeightTensor& B,
    int M, int N, int K,
    const half* bias,
    Fp4Epilogue epilogue
) {
    auto* plan = new Fp4GemmPlan();
    plan->M = M;
    plan->N = N;
    plan->K = K;
    plan->scale_A = A.tensor_scale;
    plan->scale_B = B.tensor_scale;
    plan->B_scales_ptr = B.block_scales;
    plan->bias = bias;
    plan->epilogue = epilogue;

    // Pad M if smaller than tile
    const uint8_t* A_data_tma = A.data;
    plan->A_scales_ptr = A.block_scales;
    plan->A_data_padded = nullptr;
    plan->A_scales_padded = nullptr;
    int M_tma = M;

    if (M < Config::TILE_M) {
        M_tma = Config::TILE_M;
        size_t data_bytes = (size_t)M_tma * K / 2;
        size_t scale_count = (size_t)M_tma * K / FP4_BLOCK_SIZE;
        cudaMalloc(&plan->A_data_padded, data_bytes);
        cudaMalloc(&plan->A_scales_padded, scale_count * sizeof(__nv_fp8_e4m3));
        cudaMemset(plan->A_data_padded, 0, data_bytes);
        cudaMemset(plan->A_scales_padded, 0, scale_count * sizeof(__nv_fp8_e4m3));
        cudaMemcpy(plan->A_data_padded, A.data, (size_t)M * K / 2,
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(plan->A_scales_padded, A.block_scales,
                   (size_t)M * K / FP4_BLOCK_SIZE * sizeof(__nv_fp8_e4m3),
                   cudaMemcpyDeviceToDevice);
        A_data_tma = plan->A_data_padded;
        plan->A_scales_ptr = plan->A_scales_padded;
    }

    // TMA descriptors for packed data only
    TmaDescriptor h_desc_A_data, h_desc_B_data;
    create_tma_desc_2d(&h_desc_A_data, A_data_tma, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                       M_tma, K / 2, Config::TILE_M, Config::TILE_K / 2, TmaSwizzle::B64);
    create_tma_desc_2d(&h_desc_B_data, B.data, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                       K, N / 2, Config::TILE_K, Config::TILE_N / 2, TmaSwizzle::B64);

    cudaMalloc(&plan->d_descs, 2 * sizeof(TmaDescriptor));
    cudaMemcpy(plan->d_descs,     &h_desc_A_data, sizeof(TmaDescriptor), cudaMemcpyHostToDevice);
    cudaMemcpy(plan->d_descs + 1, &h_desc_B_data, sizeof(TmaDescriptor), cudaMemcpyHostToDevice);

    plan->grid = dim3((N + Config::TILE_N - 1) / Config::TILE_N,
                      (M + Config::TILE_M - 1) / Config::TILE_M);
    plan->block = dim3(Config::BLOCK_SIZE);
    plan->smem_size = sizeof(Fp4SmemLayout);

    cudaFuncSetAttribute(gemm_fp4_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, plan->smem_size);

    cudaDeviceSynchronize();
    return plan;
}

void execute_fp4_gemm(Fp4GemmPlan* plan, half* C, cudaStream_t stream) {
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

    cudaLaunchKernelEx(&launch_config, gemm_fp4_kernel,
        plan->d_descs, plan->d_descs + 1,
        plan->A_scales_ptr, plan->B_scales_ptr,
        C, plan->M, plan->N, plan->K,
        plan->scale_A, plan->scale_B,
        plan->bias, plan->epilogue);
}

void destroy_fp4_gemm_plan(Fp4GemmPlan* plan) {
    if (!plan) return;
    if (plan->A_data_padded) cudaFree(plan->A_data_padded);
    if (plan->A_scales_padded) cudaFree(plan->A_scales_padded);
    cudaFree(plan->d_descs);
    delete plan;
}

}  // namespace blaze
