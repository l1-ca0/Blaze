/**
 * gemm_mixed_sm100.cu — Mixed-precision GEMM for SM100 (Blackwell).
 *
 * Primary inference kernel: FP16 activations × NVFP4 weights.
 * The host launch converts FP16 activations to E4M3, then the kernel
 * computes C[M,N] = A_e4m3[M,K] × B_fp4[K,N] with FP32 TMEM
 * accumulator and FP16 output. Tensor scale applied in epilogue.
 *
 * Warp-specialized, 2-stage double-buffered pipeline:
 *   Warp 0: MMA consumer  (tcgen05.mma.kind::f8f6f4, E4M3 × E2M1)
 *   Warp 1: TMA producer  (A + B_data via TMA)
 *   Warp 2: B_scale loader (vectorized uint2 global loads for B block scales)
 *   Warp 3: Idle during K-loop
 *   All warps: Epilogue   (TMEM → registers → global, no SMEM staging)
 *
 * Tile: 128×128×64 (M×N×K). K_PER_MMA=32, 2 inner iterations per tile.
 * A (E4M3) and B packed data loaded via TMA with B64 swizzle;
 * B block scales loaded concurrently by warp 2 via vectorized global loads.
 *
 * Synchronization: mbar_load init=2 (2 arrivals required):
 *   - Warp 1: mbarrier_expect_tx (implicit arrival, counts as 1) + TMA loads
 *   - Warp 2: mbarrier_arrive after B_scales written to SMEM
 * Barrier flips only when both arrivals AND all TMA bytes are received.
 */

#include "gemm/mixed_gemm_sm100.cuh"
#include "gemm/tmem_utils.cuh"
#include "gemm/tma_utils.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdio>

namespace blaze {

using Config = MixedGemmConfig;

// SMEM layout — B_scales loaded by warp 2 concurrently with TMA.
struct __align__(128) MixedSmemLayout {
    alignas(8) uint64_t mbar_load[Config::PIPELINE_STAGES];
    alignas(8) uint64_t mbar_mma[Config::PIPELINE_STAGES];
    uint32_t tmem_addr;
    char _pad[128 - (2 * Config::PIPELINE_STAGES * 8 + 4)];

    __nv_fp8_e4m3 A[Config::PIPELINE_STAGES][Config::TILE_M * Config::TILE_K];
    uint8_t B_data[Config::PIPELINE_STAGES][Config::TILE_K * Config::TILE_N / 2];
    __nv_fp8_e4m3 B_scales[Config::PIPELINE_STAGES][Config::SMEM_B_SCALE_BYTES];
    // No float C[] — epilogue reads TMEM directly to registers
};

/**
 * TMA producer (warp 1): loads A and B_data via TMA only.
 * B_scale loading is handled by warp 2 concurrently.
 * mbarrier_expect_tx counts as 1 implicit arrival (no explicit arrive needed).
 */
__device__ __forceinline__
void mixed_tma_producer(
    MixedSmemLayout* smem,
    const TmaDescriptor* desc_A,
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
            uint32_t tx_bytes = Config::SMEM_A_BYTES + Config::SMEM_B_DATA_BYTES;
            mbarrier_expect_tx(&smem->mbar_load[stage], tx_bytes);

            tma_load_2d(smem->A[stage], desc_A,
                        k_offset, bm, &smem->mbar_load[stage]);
            tma_load_2d(smem->B_data[stage], desc_B_data,
                        bn / 2, k_offset, &smem->mbar_load[stage]);
        }
        // No explicit mbarrier_arrive — expect_tx is the implicit arrival for warp 1.
    }
}

/**
 * B_scale loader (warp 2): vectorized uint2 global loads for B block scales.
 * 64 rows × 8 bytes/row = 512 bytes per stage. 32 threads × 2 iters.
 * Arrives on mbar_load after SMEM writes complete.
 */
__device__ __forceinline__
void mixed_b_scale_loader(
    MixedSmemLayout* smem,
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
            *reinterpret_cast<uint2*>(&smem->B_scales[stage][r * Config::SCALE_COLS]) =
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
void mixed_mma_consumer(MixedSmemLayout* smem, tmem_addr_t tmem_addr, int K) {
    const int num_k_tiles = K / Config::TILE_K;
    constexpr int K_PER_MMA = 32;
    constexpr int NUM_K_ITERS = Config::TILE_K / K_PER_MMA;

    uint32_t idesc = make_idesc_f8f6f4(0, 5, Config::TILE_M, Config::TILE_N);  // E4M3 × E2M1
    bool first_mma = true;

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int stage = k_tile % Config::PIPELINE_STAGES;
        mbarrier_wait(&smem->mbar_load[stage], (k_tile / Config::PIPELINE_STAGES) & 1);

        uint32_t smem_a_base = static_cast<uint32_t>(__cvta_generic_to_shared(smem->A[stage]));
        uint32_t smem_b_base = static_cast<uint32_t>(__cvta_generic_to_shared(smem->B_data[stage]));

        for (int ki = 0; ki < NUM_K_ITERS; ki++) {
            asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

            uint32_t smem_a_addr = smem_a_base + ki * K_PER_MMA;
            uint32_t smem_b_addr = smem_b_base + ki * K_PER_MMA * (Config::TILE_N / 2);

            uint64_t desc_a = make_smem_desc(smem_a_addr, Config::TILE_K, 4);      // B64 swizzle
            uint64_t desc_b = make_smem_desc(smem_b_addr, Config::TILE_N / 2, 4);  // B64 swizzle

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
 * Direct TMEM → register → global epilogue for mixed GEMM with vectorized stores.
 * All 4 warps participate. Each warp reads its own 32-row TMEM slice.
 * Packs 8 halves into uint4 for 16-byte aligned stores (8× less L2 traffic).
 */
__device__ __forceinline__
void mixed_epilogue_direct(
    tmem_addr_t tmem_addr,
    half* C_global, int bm, int bn, int M, int N,
    float weight_scale, const half* bias, const half* residual,
    MixedEpilogue epilogue_op,
    int warp_id, int lane_id
) {
    asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

    int row = warp_id * 32 + lane_id;
    int global_row = bm + row;

    for (int cg = 0; cg < Config::TILE_N / 8; cg++) {
        float vals[8];
        tmem_load_8xf32(vals, tmem_addr, cg * 8);
        tmem_wait_ld();

        if (global_row < M) {
            int global_col_base = bn + cg * 8;

            // Apply weight scale and epilogue to all 8 values
            for (int c = 0; c < 8; c++) vals[c] *= weight_scale;

            switch (epilogue_op) {
                case MixedEpilogue::BIAS:
                    for (int c = 0; c < 8; c++)
                        vals[c] += __half2float(bias[global_col_base + c]);
                    break;
                case MixedEpilogue::SILU:
                    for (int c = 0; c < 8; c++)
                        vals[c] = vals[c] / (1.0f + expf(-vals[c]));
                    break;
                case MixedEpilogue::BIAS_SILU:
                    for (int c = 0; c < 8; c++) {
                        vals[c] += __half2float(bias[global_col_base + c]);
                        vals[c] = vals[c] / (1.0f + expf(-vals[c]));
                    }
                    break;
                case MixedEpilogue::RESIDUAL_ADD:
                    for (int c = 0; c < 8; c++)
                        vals[c] += __half2float(residual[global_row * N + global_col_base + c]);
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
gemm_mixed_kernel(
    const TmaDescriptor* __restrict__ desc_A,
    const TmaDescriptor* __restrict__ desc_B_data,
    const __nv_fp8_e4m3* __restrict__ global_B_scales,
    half* __restrict__ C,
    int M, int N, int K,
    float weight_scale,
    const half* __restrict__ bias,
    const half* __restrict__ residual,
    MixedEpilogue epilogue_op
) {
    extern __shared__ char smem_raw[];
    MixedSmemLayout* smem = reinterpret_cast<MixedSmemLayout*>(smem_raw);

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
            mbarrier_init(&smem->mbar_load[s], 2);
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
            mixed_mma_consumer(smem, tmem_addr, K);
            break;
        case 1:
            mixed_tma_producer(smem, desc_A, desc_B_data, bm, bn, K, lane_id);
            break;
        case 2:
            mixed_b_scale_loader(smem, global_B_scales, bn, N, K, lane_id);
            break;
        case 3:
            break;
    }

    // Wait for MMA completion, then all warps do TMEM → register → global
    __syncthreads();
    mixed_epilogue_direct(tmem_addr, C, bm, bn, M, N,
                         weight_scale, bias, residual, epilogue_op,
                         warp_id, lane_id);

    __syncthreads();
    if (warp_id == 0) {
        tmem_dealloc(tmem_addr, Config::TMEM_COLUMNS);
    }
}

/** Element-wise FP16 → E4M3 conversion (nearest rounding). */
__global__ void fp16_to_e4m3_kernel(
    const half* __restrict__ in,
    __nv_fp8_e4m3* __restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __nv_fp8_e4m3(in[i]);
    }
}

// --- Host launch ---

void launch_gemm_mixed(
    const half* A,
    const Fp4WeightTensor& B,
    half* C,
    int M, int N, int K,
    const half* bias,
    const half* residual,
    MixedEpilogue epilogue,
    cudaStream_t stream
) {
    // Convert FP16 activations → E4M3
    __nv_fp8_e4m3* A_e4m3;
    cudaMalloc(&A_e4m3, (size_t)M * K * sizeof(__nv_fp8_e4m3));
    {
        int threads = 256;
        int blocks = ((M * K) + threads - 1) / threads;
        fp16_to_e4m3_kernel<<<blocks, threads, 0, stream>>>(A, A_e4m3, M * K);
    }

    // Pad M if smaller than tile
    const __nv_fp8_e4m3* A_tma = A_e4m3;
    __nv_fp8_e4m3* A_padded = nullptr;
    int M_tma = M;
    if (M < Config::TILE_M) {
        M_tma = Config::TILE_M;
        cudaMalloc(&A_padded, (size_t)M_tma * K * sizeof(__nv_fp8_e4m3));
        cudaMemsetAsync(A_padded, 0, (size_t)M_tma * K * sizeof(__nv_fp8_e4m3), stream);
        cudaMemcpyAsync(A_padded, A_e4m3, (size_t)M * K * sizeof(__nv_fp8_e4m3),
                        cudaMemcpyDeviceToDevice, stream);
        A_tma = A_padded;
    }

    // TMA descriptors for A and B packed data only; B scales loaded via global memory.
    TmaDescriptor h_desc_A, h_desc_B_data;
    create_tma_desc_2d(&h_desc_A, A_tma, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                       M_tma, K, Config::TILE_M, Config::TILE_K, TmaSwizzle::B64);
    create_tma_desc_2d(&h_desc_B_data, B.data, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                       K, N / 2, Config::TILE_K, Config::TILE_N / 2, TmaSwizzle::B64);

    TmaDescriptor* d_descs;
    cudaMalloc(&d_descs, 2 * sizeof(TmaDescriptor));
    cudaMemcpyAsync(d_descs,     &h_desc_A,      sizeof(TmaDescriptor), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_descs + 1, &h_desc_B_data, sizeof(TmaDescriptor), cudaMemcpyHostToDevice, stream);

    dim3 grid((N + Config::TILE_N - 1) / Config::TILE_N,
              (M + Config::TILE_M - 1) / Config::TILE_M);
    dim3 block(Config::BLOCK_SIZE);
    size_t smem_size = sizeof(MixedSmemLayout);

    cudaError_t err = cudaFuncSetAttribute(gemm_mixed_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Mixed GEMM: cudaFuncSetAttribute failed: %s (smem=%zu)\n",
                cudaGetErrorString(err), smem_size);
        cudaFreeAsync(A_e4m3, stream);
        if (A_padded) cudaFreeAsync(A_padded, stream);
        cudaFreeAsync(d_descs, stream);
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

    cudaLaunchKernelEx(&launch_config, gemm_mixed_kernel,
        d_descs, d_descs + 1, B.block_scales,
        C, M, N, K, B.tensor_scale, bias, residual, epilogue);

    cudaFreeAsync(d_descs, stream);
    cudaFreeAsync(A_e4m3, stream);
    if (A_padded) cudaFreeAsync(A_padded, stream);
}

// ---------------------------------------------------------------------------
// Prepare/execute API
// ---------------------------------------------------------------------------

struct MixedGemmPlan {
    __nv_fp8_e4m3* A_e4m3;      // Pre-converted activations
    __nv_fp8_e4m3* A_padded;    // Padded copy if M < TILE_M (else nullptr)
    const __nv_fp8_e4m3* B_scales;  // Pointer to B block scales (not owned)
    TmaDescriptor* d_descs;     // Device TMA descriptors [A, B_data]
    float weight_scale;
    const half* bias;
    const half* residual;
    MixedEpilogue epilogue;
    dim3 grid;
    dim3 block;
    size_t smem_size;
    int M, N, K;
};

MixedGemmPlan* create_mixed_gemm_plan(
    const half* A,
    const Fp4WeightTensor& B,
    int M, int N, int K,
    const half* bias,
    const half* residual,
    MixedEpilogue epilogue
) {
    auto* plan = new MixedGemmPlan();
    plan->M = M;
    plan->N = N;
    plan->K = K;
    plan->weight_scale = B.tensor_scale;
    plan->B_scales = B.block_scales;
    plan->bias = bias;
    plan->residual = residual;
    plan->epilogue = epilogue;

    // Convert FP16 activations → E4M3
    cudaMalloc(&plan->A_e4m3, (size_t)M * K * sizeof(__nv_fp8_e4m3));
    {
        int threads = 256;
        int blocks = ((M * K) + threads - 1) / threads;
        fp16_to_e4m3_kernel<<<blocks, threads>>>(A, plan->A_e4m3, M * K);
    }

    // Pad M if smaller than tile
    const __nv_fp8_e4m3* A_tma = plan->A_e4m3;
    plan->A_padded = nullptr;
    int M_tma = M;
    if (M < Config::TILE_M) {
        M_tma = Config::TILE_M;
        cudaMalloc(&plan->A_padded, (size_t)M_tma * K * sizeof(__nv_fp8_e4m3));
        cudaMemset(plan->A_padded, 0, (size_t)M_tma * K * sizeof(__nv_fp8_e4m3));
        cudaMemcpy(plan->A_padded, plan->A_e4m3, (size_t)M * K * sizeof(__nv_fp8_e4m3),
                   cudaMemcpyDeviceToDevice);
        A_tma = plan->A_padded;
    }

    // TMA descriptors for A and B packed data only
    TmaDescriptor h_desc_A, h_desc_B_data;
    create_tma_desc_2d(&h_desc_A, A_tma, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                       M_tma, K, Config::TILE_M, Config::TILE_K, TmaSwizzle::B64);
    create_tma_desc_2d(&h_desc_B_data, B.data, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                       K, N / 2, Config::TILE_K, Config::TILE_N / 2, TmaSwizzle::B64);

    cudaMalloc(&plan->d_descs, 2 * sizeof(TmaDescriptor));
    cudaMemcpy(plan->d_descs,     &h_desc_A,      sizeof(TmaDescriptor), cudaMemcpyHostToDevice);
    cudaMemcpy(plan->d_descs + 1, &h_desc_B_data, sizeof(TmaDescriptor), cudaMemcpyHostToDevice);

    plan->grid = dim3((N + Config::TILE_N - 1) / Config::TILE_N,
                      (M + Config::TILE_M - 1) / Config::TILE_M);
    plan->block = dim3(Config::BLOCK_SIZE);
    plan->smem_size = sizeof(MixedSmemLayout);

    cudaFuncSetAttribute(gemm_mixed_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, plan->smem_size);

    cudaDeviceSynchronize();
    return plan;
}

void execute_mixed_gemm(MixedGemmPlan* plan, half* C, cudaStream_t stream) {
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

    cudaLaunchKernelEx(&launch_config, gemm_mixed_kernel,
        plan->d_descs, plan->d_descs + 1, plan->B_scales,
        C, plan->M, plan->N, plan->K, plan->weight_scale,
        plan->bias, plan->residual, plan->epilogue);
}

void destroy_mixed_gemm_plan(MixedGemmPlan* plan) {
    if (!plan) return;
    cudaFree(plan->A_e4m3);
    if (plan->A_padded) cudaFree(plan->A_padded);
    cudaFree(plan->d_descs);
    delete plan;
}

// ---------------------------------------------------------------------------

void launch_fused_gate_up(
    const half* x,
    const Fp4WeightTensor& W_gate,
    const Fp4WeightTensor& W_up,
    half* output,
    int M, int N, int K,
    cudaStream_t stream
) {
    half *d_gate, *d_up;
    cudaMalloc(&d_gate, M * N * sizeof(half));
    cudaMalloc(&d_up, M * N * sizeof(half));

    launch_gemm_mixed(x, W_gate, d_gate, M, N, K,
                      nullptr, nullptr, MixedEpilogue::SILU, stream);
    launch_gemm_mixed(x, W_up, d_up, M, N, K,
                      nullptr, nullptr, MixedEpilogue::NONE, stream);

    // TODO: element-wise gate * up (define in silu.cu)
    cudaMemcpyAsync(output, d_gate, M * N * sizeof(half),
                    cudaMemcpyDeviceToDevice, stream);

    cudaFreeAsync(d_gate, stream);
    cudaFreeAsync(d_up, stream);
}

}  // namespace blaze
