/**
 * gemm_fp8_sm100.cu — FP8 GEMM for SM100 (Blackwell).
 *
 * Computes C[M,N] = A[M,K] × B[K,N] with E4M3 inputs, FP32 TMEM
 * accumulator, and FP16 output. Optional bias/activation epilogue.
 *
 * Warp-specialized, 2-stage double-buffered TMA pipeline:
 *   Warp 0: MMA consumer  (tcgen05.mma.kind::f8f6f4, E4M3 × E4M3)
 *   Warp 1: TMA producer  (async bulk tensor loads for A, B)
 *   Warp 2-3: Idle during K-loop
 *   All warps: Epilogue   (TMEM → registers → global, no SMEM staging)
 *
 * Tile: 128×128×64 (M×N×K). K_PER_MMA=32, 2 inner iterations per tile.
 *
 * Key optimization: SMEM C staging buffer eliminated. Each warp reads its
 * own 32-row TMEM slice directly to registers and writes to global memory.
 * This saves 64 KB SMEM, increasing occupancy from 12.5% to ~43%.
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

enum WarpRole : int {
    MMA_CONSUMER = 0,
    TMA_PRODUCER = 1,
    IDLE = 2,
    EPILOGUE = 3,
};

// SMEM layout WITHOUT the 64 KB C staging buffer.
// Only holds A/B tiles (double-buffered) + mbarriers + tmem_addr.
struct __align__(128) SmemLayout {
    alignas(8) uint64_t mbar_load[Config::PIPELINE_STAGES];
    alignas(8) uint64_t mbar_mma[Config::PIPELINE_STAGES];
    uint32_t tmem_addr;
    char _pad[128 - (2 * 16 + 4)];

    __nv_fp8_e4m3 A[Config::PIPELINE_STAGES][Config::TILE_M * Config::TILE_K];
    __nv_fp8_e4m3 B[Config::PIPELINE_STAGES][Config::TILE_K * Config::TILE_N];
    // No float C[] — epilogue reads TMEM directly to registers
};

__device__ __forceinline__
void tma_producer(
    SmemLayout* smem,
    const TmaDescriptor* desc_A,
    const TmaDescriptor* desc_B,
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
            mbarrier_expect_tx(&smem->mbar_load[stage],
                              Config::SMEM_A_BYTES + Config::SMEM_B_BYTES);
            tma_load_2d(smem->A[stage], desc_A, k_offset, bm, &smem->mbar_load[stage]);
            tma_load_2d(smem->B[stage], desc_B, bn, k_offset, &smem->mbar_load[stage]);
        }
    }
}

/** MMA consumer: issues tcgen05.mma across K tiles with double-buffered SMEM. */
__device__ __forceinline__
void mma_consumer(SmemLayout* smem, tmem_addr_t tmem_addr, int K) {
    const int num_k_tiles = K / Config::TILE_K;
    constexpr int K_PER_MMA = 32;
    constexpr int NUM_K_ITERS = Config::TILE_K / K_PER_MMA;

    uint32_t idesc = make_idesc_f8f6f4(0, 0, Config::TILE_M, Config::TILE_N);
    bool first_mma = true;

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int stage = k_tile % Config::PIPELINE_STAGES;
        mbarrier_wait(&smem->mbar_load[stage], (k_tile / Config::PIPELINE_STAGES) & 1);

        uint32_t smem_a_base = static_cast<uint32_t>(__cvta_generic_to_shared(smem->A[stage]));
        uint32_t smem_b_base = static_cast<uint32_t>(__cvta_generic_to_shared(smem->B[stage]));

        for (int ki = 0; ki < NUM_K_ITERS; ki++) {
            asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

            uint32_t smem_a_addr = smem_a_base + ki * K_PER_MMA;
            uint32_t smem_b_addr = smem_b_base + ki * K_PER_MMA * Config::TILE_N;

            uint64_t desc_a = make_smem_desc(smem_a_addr, Config::TILE_K * 1, 4);
            uint64_t desc_b = make_smem_desc(smem_b_addr, Config::TILE_N * 1, 2);

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
 * Direct TMEM → register → global epilogue with vectorized stores.
 * Each warp reads its own 32-row TMEM slice and writes to global memory.
 *
 * Key optimization: packs 8 FP16 values into a uint4 (16-byte) store.
 * Without this, individual half writes cause 94% excessive L2 sectors
 * (2/32 bytes per sector = 6.25% utilization). With uint4 stores,
 * utilization improves to 16/32 = 50% — an 8× reduction in L2 traffic.
 *
 * All 4 warps participate equally in the epilogue.
 */
__device__ __forceinline__
void epilogue_direct(
    tmem_addr_t tmem_addr,
    half* C_global, int bm, int bn, int M, int N,
    const half* bias, GemmEpilogue epilogue_op,
    int warp_id, int lane_id
) {
    // Make TMEM data (from async MMA) visible to this warp's tcgen05.ld
    asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

    int row = warp_id * 32 + lane_id;
    int global_row = bm + row;

    // Each warp reads its 32 rows from TMEM, 8 columns at a time
    for (int cg = 0; cg < Config::TILE_N / 8; cg++) {
        float vals[8];
        tmem_load_8xf32(vals, tmem_addr, cg * 8);
        tmem_wait_ld();

        if (global_row < M) {
            int global_col_base = bn + cg * 8;

            // Apply epilogue to all 8 values
            switch (epilogue_op) {
                case GemmEpilogue::BIAS:
                    for (int c = 0; c < 8; c++)
                        vals[c] += __half2float(bias[global_col_base + c]);
                    break;
                case GemmEpilogue::SILU:
                    for (int c = 0; c < 8; c++)
                        vals[c] = vals[c] / (1.0f + expf(-vals[c]));
                    break;
                case GemmEpilogue::BIAS_SILU:
                    for (int c = 0; c < 8; c++) {
                        vals[c] += __half2float(bias[global_col_base + c]);
                        vals[c] = vals[c] / (1.0f + expf(-vals[c]));
                    }
                    break;
                case GemmEpilogue::NONE:
                default:
                    break;
            }

            // Pack 8 halves into uint4 for vectorized 16-byte store
            // Address is 16-byte aligned: bn is TILE_N-aligned (128), cg*8*2=cg*16
            half h[8];
            for (int c = 0; c < 8; c++) h[c] = __float2half(vals[c]);

            if (global_col_base + 7 < N) {
                // Full 8-column group: single 16-byte aligned store
                *reinterpret_cast<uint4*>(&C_global[global_row * N + global_col_base]) =
                    *reinterpret_cast<uint4*>(h);
            } else {
                // Edge case: partial column group at N boundary
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
    const int bm = blockIdx.y * Config::TILE_M;
    const int bn = blockIdx.x * Config::TILE_N;

    if (tid == 0) {
        for (int s = 0; s < Config::PIPELINE_STAGES; s++) {
            mbarrier_inval(&smem->mbar_load[s]);
            mbarrier_inval(&smem->mbar_mma[s]);
        }
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

    // K-loop: warp 0 does MMA, warp 1 does TMA, warps 2-3 idle
    switch (warp_id) {
        case WarpRole::MMA_CONSUMER:
            mma_consumer(smem, tmem_addr, K);
            break;
        case WarpRole::TMA_PRODUCER:
            tma_producer(smem, desc_A, desc_B, bm, bn, K, lane_id);
            break;
        case WarpRole::IDLE:
        case WarpRole::EPILOGUE:
            break;
    }

    // Wait for MMA completion, then all warps do TMEM → register → global
    __syncthreads();
    epilogue_direct(tmem_addr, C, bm, bn, M, N, bias, epilogue_op, warp_id, lane_id);

    // Ensure all warps finished reading TMEM before deallocation
    __syncthreads();
    if (warp_id == 0) {
        tmem_dealloc(tmem_addr, Config::TMEM_COLUMNS);
    }
}

// --- Host launch ---

void launch_gemm_fp8(
    const __nv_fp8_e4m3* A,
    const __nv_fp8_e4m3* B,
    half* C,
    int M, int N, int K,
    const half* bias,
    GemmEpilogue epilogue,
    cudaStream_t stream
) {
    const __nv_fp8_e4m3* A_tma = A;
    __nv_fp8_e4m3* A_padded = nullptr;
    int M_tma = M;
    if (M < Config::TILE_M) {
        M_tma = Config::TILE_M;
        cudaMalloc(&A_padded, (size_t)M_tma * K * sizeof(__nv_fp8_e4m3));
        cudaMemsetAsync(A_padded, 0, (size_t)M_tma * K * sizeof(__nv_fp8_e4m3), stream);
        cudaMemcpyAsync(A_padded, A, (size_t)M * K * sizeof(__nv_fp8_e4m3),
                        cudaMemcpyDeviceToDevice, stream);
        A_tma = A_padded;
    }

    TmaDescriptor h_desc_A, h_desc_B;
    create_tma_desc_2d(&h_desc_A, A_tma, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                       M_tma, K, Config::TILE_M, Config::TILE_K, TmaSwizzle::B64);
    create_tma_desc_2d(&h_desc_B, B, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                       K, N, Config::TILE_K, Config::TILE_N, TmaSwizzle::B128);

    TmaDescriptor *d_desc_A, *d_desc_B;
    cudaMalloc(&d_desc_A, sizeof(TmaDescriptor));
    cudaMalloc(&d_desc_B, sizeof(TmaDescriptor));
    cudaMemcpyAsync(d_desc_A, &h_desc_A, sizeof(TmaDescriptor), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_desc_B, &h_desc_B, sizeof(TmaDescriptor), cudaMemcpyHostToDevice, stream);

    dim3 grid((N + Config::TILE_N - 1) / Config::TILE_N,
              (M + Config::TILE_M - 1) / Config::TILE_M);
    dim3 block(Config::BLOCK_SIZE);
    size_t smem_size = sizeof(SmemLayout);

    cudaError_t err = cudaFuncSetAttribute(gemm_fp8_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "FP8 GEMM: cudaFuncSetAttribute failed: %s (smem=%zu)\n",
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

    cudaLaunchKernelEx(&launch_config, gemm_fp8_kernel,
        d_desc_A, d_desc_B, C, M, N, K, bias, epilogue);

    cudaFreeAsync(d_desc_A, stream);
    cudaFreeAsync(d_desc_B, stream);
    if (A_padded) cudaFreeAsync(A_padded, stream);
}

// ---------------------------------------------------------------------------
// Prepare/execute API
// ---------------------------------------------------------------------------

struct Fp8GemmPlan {
    __nv_fp8_e4m3* A_padded;   // Padded copy if M < TILE_M (else nullptr)
    const __nv_fp8_e4m3* A_tma; // Points to A or A_padded
    TmaDescriptor* d_desc_A;
    TmaDescriptor* d_desc_B;
    const half* bias;
    GemmEpilogue epilogue;
    dim3 grid;
    dim3 block;
    size_t smem_size;
    int M, N, K;
};

Fp8GemmPlan* create_fp8_gemm_plan(
    const __nv_fp8_e4m3* A,
    const __nv_fp8_e4m3* B,
    int M, int N, int K,
    const half* bias,
    GemmEpilogue epilogue
) {
    auto* plan = new Fp8GemmPlan();
    plan->M = M;
    plan->N = N;
    plan->K = K;
    plan->bias = bias;
    plan->epilogue = epilogue;

    // Pad M if smaller than tile
    plan->A_tma = A;
    plan->A_padded = nullptr;
    int M_tma = M;
    if (M < Config::TILE_M) {
        M_tma = Config::TILE_M;
        cudaMalloc(&plan->A_padded, (size_t)M_tma * K * sizeof(__nv_fp8_e4m3));
        cudaMemset(plan->A_padded, 0, (size_t)M_tma * K * sizeof(__nv_fp8_e4m3));
        cudaMemcpy(plan->A_padded, A, (size_t)M * K * sizeof(__nv_fp8_e4m3),
                   cudaMemcpyDeviceToDevice);
        plan->A_tma = plan->A_padded;
    }

    // TMA descriptors
    TmaDescriptor h_desc_A, h_desc_B;
    create_tma_desc_2d(&h_desc_A, plan->A_tma, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                       M_tma, K, Config::TILE_M, Config::TILE_K, TmaSwizzle::B64);
    create_tma_desc_2d(&h_desc_B, B, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                       K, N, Config::TILE_K, Config::TILE_N, TmaSwizzle::B128);

    cudaMalloc(&plan->d_desc_A, sizeof(TmaDescriptor));
    cudaMalloc(&plan->d_desc_B, sizeof(TmaDescriptor));
    cudaMemcpy(plan->d_desc_A, &h_desc_A, sizeof(TmaDescriptor), cudaMemcpyHostToDevice);
    cudaMemcpy(plan->d_desc_B, &h_desc_B, sizeof(TmaDescriptor), cudaMemcpyHostToDevice);

    plan->grid = dim3((N + Config::TILE_N - 1) / Config::TILE_N,
                      (M + Config::TILE_M - 1) / Config::TILE_M);
    plan->block = dim3(Config::BLOCK_SIZE);
    plan->smem_size = sizeof(SmemLayout);

    cudaFuncSetAttribute(gemm_fp8_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, plan->smem_size);

    cudaDeviceSynchronize();
    return plan;
}

void execute_fp8_gemm(Fp8GemmPlan* plan, half* C, cudaStream_t stream) {
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

    cudaLaunchKernelEx(&launch_config, gemm_fp8_kernel,
        plan->d_desc_A, plan->d_desc_B, C,
        plan->M, plan->N, plan->K, plan->bias, plan->epilogue);
}

void destroy_fp8_gemm_plan(Fp8GemmPlan* plan) {
    if (!plan) return;
    if (plan->A_padded) cudaFree(plan->A_padded);
    cudaFree(plan->d_desc_A);
    cudaFree(plan->d_desc_B);
    delete plan;
}

}  // namespace blaze
