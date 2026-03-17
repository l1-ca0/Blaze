#pragma once
/**
 * tmem_utils.cuh — TMEM allocation/deallocation wrappers for SM100 (Blackwell).
 *
 * Tensor Memory (TMEM) is 256 KB/SM dedicated accumulator storage on Blackwell.
 * Layout: 128 rows × 512 columns × 32 bits.
 * TMEM addresses are per-SM, allocated/freed with tcgen05.alloc / tcgen05.dealloc.
 *
 * Key rules:
 *   - tcgen05.dealloc is MANDATORY before kernel exit (hardware will hang otherwise)
 *   - Allocation is per-CTA (cta_group::1) or per-TPC pair (cta_group::2)
 *   - TMEM cannot be accessed with regular register instructions
 *   - Data moves to/from TMEM via tcgen05.cp, tcgen05.ld, tcgen05.st
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

namespace blaze {

using tmem_addr_t = uint32_t;

// Number of FP32 values per TMEM column (128 rows × 1 column × 32b = 128 floats)
constexpr int TMEM_ROWS = 128;
constexpr int TMEM_COLS_TOTAL = 512;
constexpr int TMEM_SIZE_PER_SM_BYTES = 256 * 1024;  // 256 KB

/**
 * Allocate TMEM columns for a single-CTA group.
 *
 * tcgen05.alloc writes the TMEM base address to shared memory (not a register).
 * The caller must provide a pointer to a __shared__ uint32_t variable.
 * Must be called by ALL threads in a warp (warp-collective, .sync.aligned).
 *
 * @param smem_tmem_addr Pointer to a __shared__ uint32_t where the TMEM address will be written
 * @param num_columns Number of TMEM columns to allocate.
 *                    Each column holds 128 FP32 values (128 rows).
 *                    For a TILE_M×TILE_N accumulator: num_columns = TILE_N
 */
__device__ __forceinline__
void tmem_alloc(uint32_t* smem_tmem_addr, uint32_t num_columns) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_tmem_addr));
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
        :
        : "r"(smem_addr), "r"(num_columns)
    );
}

/**
 * Allocate TMEM columns for a 2-CTA cooperative group (across TPC pair).
 * Used for larger tile sizes requiring cross-CTA cooperation.
 */
__device__ __forceinline__
void tmem_alloc_cta2(uint32_t* smem_tmem_addr, uint32_t num_columns) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_tmem_addr));
    asm volatile(
        "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;\n"
        :
        : "r"(smem_addr), "r"(num_columns)
    );
}

/**
 * Deallocate TMEM columns. MANDATORY before kernel exit.
 * Failure to deallocate causes hardware hangs.
 *
 * @param addr The TMEM base address returned by tmem_alloc
 * @param num_columns Must match the allocation size
 */
__device__ __forceinline__
void tmem_dealloc(tmem_addr_t addr, uint32_t num_columns) {
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
        :
        : "r"(addr), "r"(num_columns)
    );
}

__device__ __forceinline__
void tmem_dealloc_cta2(tmem_addr_t addr, uint32_t num_columns) {
    asm volatile(
        "tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;\n"
        :
        : "r"(addr), "r"(num_columns)
    );
}

/**
 * Load 8 FP32 values from TMEM → registers (per-lane).
 *
 * tcgen05.ld syntax: tcgen05.ld.sync.aligned.32x32b.x{N}.b32 {regs}, [tmem_addr]
 * TMEM addressing: taddr + (row_group << 16) + col
 *   - row_group: selects which 32-row group (0..3 for 128 rows)
 *   - col: starting column index
 *
 * Each lane in the warp receives one row's worth of data.
 * Lane i gets data for row = row_group * 32 + i.
 * Must be followed by tcgen05.wait::ld before using results.
 */
__device__ __forceinline__
void tmem_load_8xf32(float out[8], tmem_addr_t taddr, int row_group, int col) {
    uint32_t addr = taddr + (row_group << 16) + col;
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n"
        : "=f"(out[0]), "=f"(out[1]), "=f"(out[2]), "=f"(out[3]),
          "=f"(out[4]), "=f"(out[5]), "=f"(out[6]), "=f"(out[7])
        : "r"(addr)
    );
}

__device__ __forceinline__
void tmem_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory");
}

/**
 * Transfer a 128-row × TILE_N-column accumulator from TMEM → shared memory.
 *
 * Must be called by all 32 threads of a warp. Each lane handles one row
 * per 32-row group. Uses tcgen05.ld.32x32b.x8 (32 rows × 8 cols per load).
 *
 * @param smem_out  Destination shared memory buffer [128 × tile_n] floats, row-major
 * @param taddr     TMEM base address from alloc
 * @param tile_n    Number of columns (must be multiple of 8)
 */
__device__ __forceinline__
void tmem_store_to_smem(float* smem_out, tmem_addr_t taddr, int tile_n) {
    int lane = threadIdx.x % 32;
    for (int rg = 0; rg < 4; rg++) {
        for (int cg = 0; cg < tile_n / 8; cg++) {
            float vals[8];
            tmem_load_8xf32(vals, taddr, rg, cg * 8);
            tmem_wait_ld();

            int row = rg * 32 + lane;
            for (int c = 0; c < 8; c++) {
                smem_out[row * tile_n + cg * 8 + c] = vals[c];
            }
        }
    }
}

/**
 * Copy from shared memory → TMEM (e.g. to zero-init accumulator).
 * tcgen05.cp syntax: tcgen05.cp.cta_group::{N}.sync.aligned.shared::cta.b32 [tmem], [smem]
 *
 * @param tmem_addr Destination TMEM address
 * @param smem_addr Source shared memory address (via __cvta_generic_to_shared)
 */
__device__ __forceinline__
void tmem_copy_from_smem(tmem_addr_t tmem_addr, uint32_t smem_addr) {
    asm volatile(
        "tcgen05.cp.cta_group::1.sync.aligned.shared::cta.b32 [%0], [%1];\n"
        :
        : "r"(tmem_addr), "r"(smem_addr)
        : "memory"
    );
}

/**
 * Broadcast tmem_addr from thread 0 to all threads in warp.
 */
__device__ __forceinline__
tmem_addr_t tmem_broadcast_addr(tmem_addr_t addr) {
    return __shfl_sync(0xFFFFFFFF, addr, 0);
}

/**
 * RAII-style TMEM guard. Ensures deallocation on scope exit.
 * Use in kernels to prevent TMEM leak from early returns.
 */
struct TmemGuard {
    tmem_addr_t addr;
    uint32_t num_columns;
    bool is_warp0;

    /**
     * Must be called by all threads in warp 0 (alloc/dealloc are warp-collective).
     *
     * @param smem_tmem_addr Pointer to a __shared__ uint32_t for alloc output
     * @param cols Number of TMEM columns to allocate
     * @param warp_id Warp ID within block
     */
    __device__ TmemGuard(uint32_t* smem_tmem_addr, uint32_t cols, int warp_id)
        : num_columns(cols), is_warp0(warp_id == 0) {
        // All warp-0 threads must execute alloc (warp-collective)
        if (is_warp0) {
            tmem_alloc(smem_tmem_addr, cols);
        }
        __syncthreads();
        addr = *smem_tmem_addr;
    }

    __device__ ~TmemGuard() {
        __syncthreads();
        if (is_warp0) {
            tmem_dealloc(addr, num_columns);
        }
    }

    // Non-copyable
    __device__ TmemGuard(const TmemGuard&) = delete;
    __device__ TmemGuard& operator=(const TmemGuard&) = delete;
};

}  // namespace blaze
