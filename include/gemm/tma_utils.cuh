#pragma once
/**
 * tma_utils.cuh — TMA (Tensor Memory Accelerator) descriptor and operation helpers.
 *
 * TMA provides hardware-accelerated async tensor loads from global memory to shared
 * memory, handling address calculation, boundary checks, and data format conversion.
 *
 * On Blackwell, TMA supports:
 *   - 1D through 5D tensor descriptors
 *   - Async bulk copy (global → shared, shared → global)
 *   - Swizzle modes for bank-conflict-free SMEM access
 *   - Integration with mbarrier for synchronization
 *
 * TMA descriptors are 128-byte objects created on the host (or device), describing
 * the tensor layout in global memory and the box (tile) shape for each transfer.
 */

#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <cstdio>

namespace blaze {

// Maximum TMA descriptor size (128 bytes per CUDA spec)
constexpr int TMA_DESC_SIZE = 128;

/**
 * TMA tensor descriptor wrapper.
 * Encapsulates the opaque CUtensorMap used by TMA hardware.
 */
struct TmaDescriptor {
    alignas(64) char data[TMA_DESC_SIZE];
};

/**
 * Swizzle modes for shared memory layout.
 * Controls how addresses are permuted to avoid bank conflicts.
 */
enum class TmaSwizzle : uint32_t {
    NONE   = 0,
    B32    = 1,   //  32-byte swizzle
    B64    = 2,   //  64-byte swizzle
    B128   = 3,   // 128-byte swizzle (most common for GEMM)
};

/**
 * Create a 2D TMA descriptor for a row-major matrix.
 *
 * This sets up the tensor map that TMA hardware uses to translate
 * tile coordinates into global memory addresses and perform async copies.
 *
 * @param desc        Output descriptor (must be 64-byte aligned)
 * @param global_ptr  Pointer to the matrix in global/HBM memory
 * @param format      Element format (CU_TENSOR_MAP_DATA_TYPE_*)
 * @param rows        Number of rows in the global matrix
 * @param cols        Number of columns in the global matrix
 * @param box_rows    Tile height (rows per TMA load)
 * @param box_cols    Tile width (cols per TMA load)
 * @param swizzle     Shared memory swizzle mode
 */
inline void create_tma_desc_2d(
    TmaDescriptor* desc,
    const void* global_ptr,
    CUtensorMapDataType format,
    uint64_t rows,
    uint64_t cols,
    uint32_t box_rows,
    uint32_t box_cols,
    TmaSwizzle swizzle = TmaSwizzle::B128
) {
    // Element size in bytes
    uint32_t elem_size;
    switch (format) {
        case CU_TENSOR_MAP_DATA_TYPE_FLOAT16:  elem_size = 2; break;
        case CU_TENSOR_MAP_DATA_TYPE_BFLOAT16: elem_size = 2; break;
        case CU_TENSOR_MAP_DATA_TYPE_FLOAT32:  elem_size = 4; break;
        case CU_TENSOR_MAP_DATA_TYPE_UINT8:    elem_size = 1; break;  // FP8/FP4 packed
        default: elem_size = 2; break;
    }

    // Global dimensions (in elements).
    // Callers must ensure the allocation is at least box-sized in each dimension.
    uint64_t global_dim[2] = {cols, rows};

    // Global strides (in bytes). For row-major: stride[0] is the row stride.
    // TMA uses strides starting from dimension 1; dimension 0 stride is implicit.
    uint64_t global_strides[1] = {cols * elem_size};

    // Box (tile) dimensions (in elements)
    uint32_t box_dim[2] = {box_cols, box_rows};

    // Element strides (1 = dense)
    uint32_t elem_strides[2] = {1, 1};

    // Swizzle mapping
    CUtensorMapSwizzle cu_swizzle;
    switch (swizzle) {
        case TmaSwizzle::NONE: cu_swizzle = CU_TENSOR_MAP_SWIZZLE_NONE; break;
        case TmaSwizzle::B32:  cu_swizzle = CU_TENSOR_MAP_SWIZZLE_32B;  break;
        case TmaSwizzle::B64:  cu_swizzle = CU_TENSOR_MAP_SWIZZLE_64B;  break;
        case TmaSwizzle::B128: cu_swizzle = CU_TENSOR_MAP_SWIZZLE_128B; break;
        default: cu_swizzle = CU_TENSOR_MAP_SWIZZLE_NONE; break;
    }

    // OOB fill with NaN/zero is only valid for float types.
    // For integer types (UINT8 used by FP8/FP4), must use FILL_NONE.
    // Padded globalDim ensures the box always fits, so OOB won't occur.
    bool is_float = (format == CU_TENSOR_MAP_DATA_TYPE_FLOAT16 ||
                     format == CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 ||
                     format == CU_TENSOR_MAP_DATA_TYPE_FLOAT32 ||
                     format == CU_TENSOR_MAP_DATA_TYPE_FLOAT64 ||
                     format == CU_TENSOR_MAP_DATA_TYPE_TFLOAT32);
    CUtensorMapFloatOOBfill oob_fill = is_float
        ? CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
        : CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

    CUresult result = cuTensorMapEncodeTiled(
        reinterpret_cast<CUtensorMap*>(desc),
        format,
        2,                                // 2D tensor
        const_cast<void*>(global_ptr),
        global_dim,
        global_strides,
        box_dim,
        elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        cu_swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        oob_fill
    );

    if (result != CUDA_SUCCESS) {
        const char* err_str;
        cuGetErrorString(result, &err_str);
        fprintf(stderr, "TMA descriptor creation failed: %s\n", err_str);
        exit(EXIT_FAILURE);
    }
}

/**
 * Issue an async TMA load from global memory → shared memory.
 *
 * The load is asynchronous — use mbarrier to synchronize.
 * Only one thread per CTA (typically thread 0 of the producer warp) should issue this.
 *
 * @param smem_addr  Shared memory destination (generic → shared conversion needed)
 * @param desc       TMA descriptor (in constant/global memory)
 * @param coord_x    Column coordinate in the global tensor
 * @param coord_y    Row coordinate in the global tensor
 * @param mbar_addr  mbarrier address for completion notification
 */
__device__ __forceinline__
void tma_load_2d(
    void* smem_ptr,
    const TmaDescriptor* desc,
    int32_t coord_x,
    int32_t coord_y,
    uint64_t* mbar_ptr
) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint64_t mbar_addr = static_cast<uint64_t>(__cvta_generic_to_shared(mbar_ptr));

    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];\n"
        :
        : "r"(smem_addr), "l"(desc), "r"(coord_x), "r"(coord_y), "l"(mbar_addr)
        : "memory"
    );
}

/**
 * Initialize an mbarrier for TMA synchronization.
 *
 * @param mbar_ptr  Pointer to mbarrier in shared memory
 * @param count     Expected number of arriving threads/transactions
 */
__device__ __forceinline__
void mbarrier_init(uint64_t* mbar_ptr, uint32_t count) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    asm volatile(
        "mbarrier.init.shared.b64 [%0], %1;\n"
        :
        : "r"(smem_addr), "r"(count)
    );
}

/**
 * Set the expected transaction count on an mbarrier (for TMA).
 *
 * TMA uses transaction-based completion: the mbarrier tracks bytes
 * rather than thread arrivals.
 *
 * @param mbar_ptr   Pointer to mbarrier in shared memory
 * @param tx_bytes   Expected total bytes from all TMA loads on this barrier
 */
__device__ __forceinline__
void mbarrier_expect_tx(uint64_t* mbar_ptr, uint32_t tx_bytes) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    asm volatile(
        "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n"
        :
        : "r"(smem_addr), "r"(tx_bytes)
    );
}

/**
 * Wait on an mbarrier phase.
 *
 * Blocks until the mbarrier's transaction count reaches zero
 * (all TMA loads completed).
 *
 * @param mbar_ptr  Pointer to mbarrier in shared memory
 * @param phase     Phase bit to wait on (alternates 0/1 for double buffering)
 */
__device__ __forceinline__
void mbarrier_wait(uint64_t* mbar_ptr, uint32_t phase) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  WAIT_LOOP_%=:\n"
        "  mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
        "  @!p bra WAIT_LOOP_%=;\n"
        "}\n"
        :
        : "r"(smem_addr), "r"(phase)
    );
}

/**
 * Invalidate an mbarrier, clearing all internal state including TX tracking.
 *
 * Must be called before mbarrier_init when reusing a barrier that previously
 * had expect_tx arrivals (e.g., in persistent kernel tile loops).
 * After invalidation, the barrier must be reinitialized before use.
 */
__device__ __forceinline__
void mbarrier_inval(uint64_t* mbar_ptr) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    asm volatile(
        "mbarrier.inval.shared.b64 [%0];\n"
        :
        : "r"(smem_addr)
        : "memory"
    );
}

/**
 * Arrive at an mbarrier (non-TMA thread arrival).
 */
__device__ __forceinline__
void mbarrier_arrive(uint64_t* mbar_ptr) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    asm volatile(
        "mbarrier.arrive.shared.b64 _, [%0];\n"
        :
        : "r"(smem_addr)
    );
}

/**
 * Compute the TMA tile bytes for a 2D tile.
 *
 * @param box_rows  Number of rows in the tile
 * @param box_cols  Number of columns in the tile
 * @param elem_size Bytes per element
 * @return Total bytes in the TMA transfer
 */
constexpr uint32_t tma_tile_bytes(uint32_t box_rows, uint32_t box_cols, uint32_t elem_size) {
    return box_rows * box_cols * elem_size;
}

}  // namespace blaze
