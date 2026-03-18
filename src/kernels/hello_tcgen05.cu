/**
 * hello_tcgen05.cu — Minimal Blackwell SM100 bringup kernel.
 *
 * Purpose: Verify the tcgen05 TMEM alloc/dealloc lifecycle on real hardware.
 * If this kernel runs, the entire tcgen05 instruction pipeline 
 * (PTX assembly, compiler, driver, GPU) is confirmed working.
 *
 * What this kernel does:
 *   1. Allocates 32 columns of Tensor Memory (TMEM) via tcgen05.alloc
 *   2. Reads back the TMEM base address (written to shared memory by hardware)
 *   3. Deallocates the TMEM via tcgen05.dealloc
 *   4. Writes a completion marker to global memory
 *
 * Key SM100 (Blackwell) constraints exercised here:
 *   - tcgen05.alloc/dealloc are warp-collective (.sync.aligned):
 *     ALL 32 threads in warp 0 must execute them together
 *   - tcgen05.alloc writes the result to shared memory, not a register
 *     (the [smem_addr] operand uses .shared::cta address space)
 *   - tcgen05.dealloc is MANDATORY before kernel exit (hardware hangs otherwise)
 *   - Cluster launch is required even for cta_group::1 — the tcgen05 ISA
 *     relies on cluster scheduling infrastructure on SM100
 *   - Minimum TMEM allocation is 32 columns (must be power of 2, max 512)
 *
 * Architecture: 128 threads = 4 warps. Only warp 0 drives TMEM operations.
 * Launch: 1 block, 128 threads, cluster dims 1×1×1 via cudaLaunchKernelEx.
 */

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

using tmem_addr_t = uint32_t;

/**
 * Minimal TMEM lifecycle test.
 * 128 threads (4 warps). Warp 0 drives alloc/dealloc.
 * Both instructions are warp-collective (.sync.aligned).
 */
__cluster_dims__(1, 1, 1)
__global__ void kernel_hello_tcgen05(float* output) {
    __shared__ uint32_t smem_tmem_addr;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;

    // Step 1: Allocate TMEM — full warp 0 must participate
    // Minimum allocation is 32 columns (must be power of 2, range: 32-512)
    uint32_t num_columns = 32;
    if (warp_id == 0) {
        uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem_tmem_addr));
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
            :
            : "r"(smem_addr), "r"(num_columns)
        );
    }
    __syncthreads();

    // Read alloc result
    tmem_addr_t tmem_addr = smem_tmem_addr;

    // Record alloc result before attempting dealloc
    // Note: TMEM address 0 is VALID (it's the base of the TMEM address space)
    if (tid == 0) {
        output[0] = static_cast<float>(tmem_addr);
        output[1] = 1.0f;  // If we get here, alloc succeeded
    }
    __syncthreads();

    // Step 2: Deallocate TMEM — full warp 0 must participate
    if (warp_id == 0) {
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
            :
            : "r"(tmem_addr), "r"(num_columns)
        );
    }
    __syncthreads();

    if (tid == 0) {
        output[2] = 42.0f;  // Completion marker
    }
}

int main() {
    printf("=== Blaze: tcgen05 Hardware Bringup ===\n\n");

    int device;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("SMs: %d\n", prop.multiProcessorCount);
    printf("\n");

    if (prop.major != 10 || prop.minor != 0) {
        fprintf(stderr, "ERROR: SM100 required. Found: %d.%d\n", prop.major, prop.minor);
        return EXIT_FAILURE;
    }

    float* d_output;
    float h_output[3] = {0};
    CHECK_CUDA(cudaMalloc(&d_output, 3 * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_output, 0, 3 * sizeof(float)));

    printf("Launching hello_tcgen05 kernel (1 block, 128 threads, cluster 1x1x1)...\n");

    // tcgen05 instructions require cluster launch infrastructure on SM100.
    // Even with cta_group::1 (single CTA), we must use cudaLaunchKernelEx
    // with explicit cluster dimensions instead of the <<<>>> syntax.
    // This tells the scheduler to set up the cluster context that tcgen05
    // instructions expect at the hardware level.
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(1, 1, 1);
    config.blockDim = dim3(128, 1, 1);
    config.dynamicSmemBytes = 0;
    config.stream = 0;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 1;  // 1 CTA per cluster (matches cta_group::1)
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    cudaError_t launch_err = cudaLaunchKernelEx(&config, kernel_hello_tcgen05, d_output);
    if (launch_err != cudaSuccess) {
        fprintf(stderr, "Launch FAILED: %s\n", cudaGetErrorString(launch_err));
        cudaFree(d_output);
        return EXIT_FAILURE;
    }

    cudaError_t sync_err = cudaDeviceSynchronize();
    cudaError_t last_err = cudaGetLastError();

    if (sync_err != cudaSuccess) {
        fprintf(stderr, "\nKernel FAILED: %s\n", cudaGetErrorString(sync_err));
        fprintf(stderr, "Last error: %s\n", cudaGetErrorString(last_err));

        // Try to read output anyway (alloc result may have been written)
        cudaMemcpy(h_output, d_output, 3 * sizeof(float), cudaMemcpyDeviceToHost);
        printf("  TMEM addr (may be stale): 0x%08x\n", static_cast<uint32_t>(h_output[0]));
        printf("  Alloc success: %.0f\n", h_output[1]);
        printf("  Completion: %.0f\n", h_output[2]);

        printf("\nTry: compute-sanitizer ./build/hello_tcgen05\n");
        cudaFree(d_output);
        return EXIT_FAILURE;
    }

    CHECK_CUDA(cudaMemcpy(h_output, d_output, 3 * sizeof(float), cudaMemcpyDeviceToHost));

    printf("\nResults:\n");
    printf("  TMEM address: 0x%08x\n", static_cast<uint32_t>(h_output[0]));
    printf("  Alloc success: %s\n", h_output[1] == 1.0f ? "YES" : "NO");
    printf("  Kernel completed: %s\n", h_output[2] == 42.0f ? "YES" : "NO");

    if (h_output[1] == 1.0f && h_output[2] == 42.0f) {
        printf("\n=== SUCCESS: tcgen05 TMEM lifecycle verified! ===\n");
    } else {
        printf("\n=== FAILURE ===\n");
    }

    CHECK_CUDA(cudaFree(d_output));
    return (h_output[2] == 42.0f) ? EXIT_SUCCESS : EXIT_FAILURE;
}
