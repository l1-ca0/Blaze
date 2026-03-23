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
 * TMEM addressing: taddr + col
 *   - The row selection is IMPLICIT: the hardware automatically reads the
 *     32-row slice belonging to the executing warp.
 *   - col: starting column index
 *
 * Each lane in the warp receives one row's worth of data.
 * Lane i gets data for row = (warp_id_in_warpgroup * 32) + i.
 * Must be followed by tcgen05.wait::ld before using results.
 */
__device__ __forceinline__
void tmem_load_8xf32(float out[8], tmem_addr_t taddr, int col) {
    uint32_t addr = taddr + col;
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
    asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    // tcgen05.ld is warp-scoped: each warp can only read its own 32 rows.
    // Row group is implicit (determined by hardware based on warp ID).
    int rg = warp_id;
    for (int cg = 0; cg < tile_n / 8; cg++) {
        float vals[8];
        tmem_load_8xf32(vals, taddr, cg * 8);
        tmem_wait_ld();

        int row = rg * 32 + lane;
        for (int c = 0; c < 8; c++) {
            smem_out[row * tile_n + cg * 8 + c] = vals[c];
        }
    }
}

/**
 * Per-warp TMEM → SMEM transfer. Each warp reads ONLY its own 32-row slice.
 *
 * TMEM is warp-scoped: warp i can only access rows [i*32, (i+1)*32).
 * All 4 warps must call this cooperatively to transfer the full 128-row tile.
 * Must be preceded by __syncthreads() to ensure MMA completion.
 *
 * The SMEM destination uses stride = tile_n + SMEM_C_PAD to avoid bank conflicts.
 * Without padding, all 32 lanes write to the same bank (31.8-way conflict)
 * because row * tile_n is always a multiple of 32 when tile_n is 128.
 * With +1 padding: lane * (tile_n+1) % 32 = lane, giving zero conflicts.
 *
 * @param smem_out    Destination shared memory buffer [128 × (tile_n + SMEM_C_PAD)] floats
 * @param taddr       TMEM base address from alloc
 * @param tile_n      Number of data columns (must be multiple of 8)
 * @param smem_stride Row stride in the SMEM buffer (tile_n + SMEM_C_PAD)
 * @param warp_id     Warp index within block (0-3)
 */
// Padding to eliminate SMEM bank conflicts in TMEM→SMEM transfer.
// 4 floats (16 bytes) aligns rows to 16B while fully breaking conflicts.
constexpr int SMEM_C_PAD = 4;

__device__ __forceinline__
void tmem_store_to_smem_warp(float* smem_out, tmem_addr_t taddr, int tile_n, int smem_stride, int warp_id) {
    // CRITICAL: Each warp must execute fence::after_thread_sync to make TMEM
    // data (written by async MMA) visible to its tcgen05.ld instructions.
    // Without this, only the MMA-issuing warp can see the TMEM writes.
    asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

    int lane = threadIdx.x % 32;
    // tcgen05.ld row selection is implicit — hardware reads the warp's own 32 rows.
    // warp_id is only used for computing the SMEM output offset.
    int row = warp_id * 32 + lane;
    for (int cg = 0; cg < tile_n / 8; cg++) {
        float vals[8];
        tmem_load_8xf32(vals, taddr, cg * 8);
        tmem_wait_ld();

        int base = row * smem_stride + cg * 8;
        for (int c = 0; c < 8; c++) {
            smem_out[base + c] = vals[c];
        }
    }
}

/** Backward-compatible overload: smem_stride defaults to tile_n (no padding). */
__device__ __forceinline__
void tmem_store_to_smem_warp(float* smem_out, tmem_addr_t taddr, int tile_n, int warp_id) {
    tmem_store_to_smem_warp(smem_out, taddr, tile_n, tile_n, warp_id);
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

// NOTE: tcgen05.commit is the proper way to wait for async MMA TMEM writes,
// but triggers ptxas ICE (C7907) in CUDA 13.1.80. Workaround: the MMA consumer
// warp does TMEM→SMEM itself after fence::before_thread_sync (which ensures
// MMA completion for the issuing warp), avoiding the need for tcgen05.commit.

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

/**
 * Build a 64-bit SMEM descriptor for tcgen05.mma operands (SM100 Blackwell format).
 *
 * Descriptor layout (from CUTLASS SmemDescriptor):
 *   Bits [0,14)  : start_address >> 4   (16-byte granularity)
 *   Bits [14,16) : reserved (0)
 *   Bits [16,30) : leading_byte_offset >> 4  (0 for swizzled modes)
 *   Bits [30,32) : reserved (0)
 *   Bits [32,46) : stride_byte_offset >> 4   (row stride in 16-byte units)
 *   Bits [46,48) : version = 1               (MUST be 1 for Blackwell)
 *   Bits [48,61) : reserved (0)
 *   Bits [61,64) : layout_type               (swizzle mode)
 *
 * Swizzle layout_type values:
 *   0 = SWIZZLE_NONE
 *   2 = SWIZZLE_128B
 *   4 = SWIZZLE_64B
 *   6 = SWIZZLE_32B
 *
 * @param smem_addr         Shared memory address (from __cvta_generic_to_shared)
 * @param stride_bytes      Row stride in bytes (e.g., TILE_K * elem_size for A)
 * @param layout_type       Swizzle mode (0=NONE, 2=128B, 4=64B, 6=32B)
 */
__device__ __forceinline__
uint64_t make_smem_desc(uint32_t smem_addr, uint32_t stride_bytes, uint32_t layout_type) {
    uint64_t desc = 0;
    desc |= ((uint64_t)(smem_addr >> 4) & 0x3FFF);              // bits [0,14): base addr / 16
    // LBO = 0 for swizzled modes (bits [16,30) stay 0)
    desc |= ((uint64_t)(stride_bytes >> 4) & 0x3FFF) << 32;     // bits [32,46): SBO / 16
    desc |= (1ULL << 46);                                        // bits [46,48): version = 1
    desc |= ((uint64_t)(layout_type & 0x7)) << 61;              // bits [61,64): swizzle
    return desc;
}

/**
 * Elect one thread within a warp (for tcgen05.mma which is warp-collective
 * but only one thread issues the instruction).
 */
__device__ __forceinline__
bool elect_one_sync() {
    uint32_t pred;
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  elect.sync _|p, 0xFFFFFFFF;\n"
        "  selp.b32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(pred)
    );
    return pred != 0;
}

/**
 * Build the 32-bit instruction descriptor (idesc) for tcgen05.mma.kind::f8f6f4.
 *
 * Bit layout (from CUTLASS InstrDescriptor):
 *   Bits [0,2)   : sparse_id2 (0 for dense)
 *   Bit  [2]     : sparse_flag (0 for dense)
 *   Bit  [3]     : saturate (0)
 *   Bits [4,6)   : c_format (0=F16, 1=F32, 2=S32)
 *   Bits [7,10)  : a_format (0=E4M3, 1=E5M2, 3=E2M3, 4=E3M2, 5=E2M1)
 *   Bits [10,13) : b_format (same encoding)
 *   Bit  [13]    : a_negate (0)
 *   Bit  [14]    : b_negate (0)
 *   Bit  [15]    : a_major (0=K-major, 1=MN-major)
 *   Bit  [16]    : b_major (0=K-major, 1=MN-major)
 *   Bits [17,23) : n_dim = N >> 3
 *   Bits [24,29) : m_dim = M >> 4
 *   Bits [30,32) : max_shift (0)
 */
__device__ __host__ __forceinline__
uint32_t make_idesc_f8f6f4(
    uint32_t a_format,   // 0=E4M3, 1=E5M2, 5=E2M1
    uint32_t b_format,
    uint32_t tile_m,     // Must be multiple of 16
    uint32_t tile_n      // Must be multiple of 8
) {
    uint32_t idesc = 0;
    idesc |= (1u << 4);                        // c_format = F32
    idesc |= ((a_format & 0x7) << 7);          // a_format
    idesc |= ((b_format & 0x7) << 10);         // b_format
    idesc |= (((tile_n >> 3) & 0x3F) << 17);   // n_dim
    idesc |= (((tile_m >> 4) & 0x1F) << 24);   // m_dim
    return idesc;
}

}  // namespace blaze
