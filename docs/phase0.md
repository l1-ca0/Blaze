# Phase 0: Hardware Bringup

Verify the SM100 (Blackwell) `tcgen05` instruction pipeline end-to-end on real B200 hardware, and establish a correct GEMM baseline.

Phase 0 answers two questions:
1. Can we execute `tcgen05` PTX instructions on the B200?
2. Is our matrix indexing correct?

## Hardware & Toolchain

| Spec | Value |
|------|-------|
| GPU | NVIDIA B200 (GB202) |
| Compute Capability | 10.0 (SM100) |
| SMs | 148 |
| TMEM per SM | 256 KB (128 rows × 512 columns × 32 bits) |
| HBM3e | 192 GB, 8 TB/s |
| Tensor Core Peak (FP16 dense) | ~2250 TFLOPS |
| CUDA Toolkit | 13.1.80 |
| Driver | 590.44.01 |

Build target: `sm_100a` (the `a` suffix enables architecture-accelerated features: `tcgen05`, TMEM, TMA).

### Build

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=100a
make -j$(nproc)
```

---

## Experiment 1: TMEM Lifecycle (`hello_tcgen05`)

Source: [`src/kernels/hello_tcgen05.cu`](../src/kernels/hello_tcgen05.cu)

A minimal kernel that allocates Tensor Memory, reads back the address, and deallocates — no computation.

### How It Works

Blackwell introduces Tensor Memory (TMEM), a per-SM scratchpad dedicated to tensor core accumulators. Managing it requires two new PTX instructions:

```
tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [smem_addr], num_columns;
tcgen05.dealloc.cta_group::1.sync.aligned.b32 tmem_addr, num_columns;
```

Both are warp-collective (`.sync.aligned`) — all 32 threads in the warp must execute them together. The alloc instruction writes the resulting TMEM base address to shared memory (not a register), which other threads then read via `__syncthreads()`.

The kernel is structured as 128 threads (4 warps), with only warp 0 driving the TMEM operations:

```cpp
__cluster_dims__(1, 1, 1)
__global__ void kernel_hello_tcgen05(float* output) {
    __shared__ uint32_t smem_tmem_addr;
    const int warp_id = threadIdx.x / 32;

    // Allocate 32 TMEM columns (minimum, must be power of 2)
    if (warp_id == 0) {
        uint32_t smem_addr = static_cast<uint32_t>(
            __cvta_generic_to_shared(&smem_tmem_addr));
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
            : : "r"(smem_addr), "r"(32));
    }
    __syncthreads();

    tmem_addr_t tmem_addr = smem_tmem_addr;
    // ... record results ...

    // Mandatory deallocation before kernel exit
    if (warp_id == 0) {
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
            : : "r"(tmem_addr), "r"(32));
    }
}
```

The kernel is launched via `cudaLaunchKernelEx` with explicit cluster dimensions — `tcgen05` instructions require the cluster scheduling infrastructure even for single-CTA operations.

### Run

```bash
./build/hello_tcgen05
```

### Results

```
Device: NVIDIA B200
Compute Capability: 10.0
SMs: 148

Results:
  TMEM address: 0x00000000
  Alloc success: YES
  Kernel completed: YES

=== SUCCESS: tcgen05 TMEM lifecycle verified! ===
```

TMEM address `0x00000000` is valid — it's the base of the TMEM address space.

### Nsight Compute Profile

```bash
./scripts/profile.sh tmem ./build/hello_tcgen05
```

```
Metric                                                             Value
-----------------------------------------------------------------  --------
smsp__inst_executed.sum                                                 322
sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed              0%
sm__pipe_tma_cycles_active.avg.pct_of_peak_sustained_elapsed             0%
smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed        0%
sm__throughput.avg.pct_of_peak_sustained_elapsed                      0.01%
```

322 instructions total. Zero tensor core and TMA activity — expected, since this kernel only exercises alloc/dealloc, not MMA.

---

## Experiment 2: Naive FP16 GEMM (`gemm_fp16_naive`)

Source: [`src/kernels/gemm_fp16_naive.cu`](../src/kernels/gemm_fp16_naive.cu)

Computes `C[4096×4096] = A[4096×4096] × B[4096×4096]` in FP16 using CUDA cores (not tensor cores), while exercising TMEM alloc/dealloc across 1024 concurrent CTAs. Validates correctness against cuBLAS.

### Kernel Architecture

| Parameter | Value |
|-----------|-------|
| Problem size | M=N=K=4096 |
| Tile size | 128 × 128 × 16 (M × N × K) |
| Grid | 32 × 32 = 1024 CTAs |
| Block | 128 threads (4 warps) |
| SMEM per CTA | 8320 bytes (128B header + 4KB A tile + 4KB B tile) |
| Registers | 54 per thread |
| Accumulators | 128 FP32 values per thread (in registers) |
| TMEM allocation | 32 columns per CTA (lifecycle validation only) |

### Kernel Flow

Each CTA owns a 128×128 output tile and iterates over 256 K-tiles:

1. TMEM alloc — warp 0, 32 columns (lifecycle validation, not used for compute)
2. Tiled GEMM — all threads cooperatively load A/B tiles from global memory into shared memory, then compute via CUDA-core FMA with FP32 accumulation
3. Store — FP32 accumulators narrowed to FP16, written to global memory
4. TMEM dealloc — mandatory cleanup

The inner loop loads a 128×16 tile of A and a 16×128 tile of B into shared memory, then each thread accumulates its portion of the output:

```cpp
for (int tk = 0; tk < k; tk += TILE_K) {
    // Cooperative load: 128 threads load A tile and B tile into SMEM
    for (int i = tid; i < SMEM_A_SIZE; i += BLOCK_SIZE)
        smem_A[i] = A[...];
    for (int i = tid; i < SMEM_B_SIZE; i += BLOCK_SIZE)
        smem_B[i] = B[...];
    __syncthreads();

    // Each thread computes its 128 output elements
    for (int idx = 0; idx < ELEMS_PER_THREAD; idx++) {
        int row = (tid + idx * BLOCK_SIZE) / TILE_N;
        int col = (tid + idx * BLOCK_SIZE) % TILE_N;
        float sum = 0.0f;
        for (int kk = 0; kk < TILE_K; kk++)
            sum += __half2float(smem_A[row * TILE_K + kk])
                 * __half2float(smem_B[kk * TILE_N + col]);
        acc[idx] += sum;
    }
    __syncthreads();
}
```

### cuBLAS Reference

cuBLAS operates in column-major. To compute row-major `C = A × B`, we use the identity `C^T = B^T × A^T` — row-major data reinterpreted as column-major is implicitly transposed:

```cpp
cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K, &alpha,
            d_B, N,    // cuBLAS sees B^T(N,K)
            d_A, K,    // cuBLAS sees A^T(K,M)
            &beta, d_C_ref, N);
```

### Run

```bash
./build/gemm_fp16_naive
```

### Correctness Results

```
Performance:
  Average time: 44.054 ms
  Throughput:   3.12 TFLOPS (FP16, CUDA cores)

Correctness vs cuBLAS:
  Max absolute error:  0.750000
  Mean absolute error: 0.035081
  Max relative error:  0.317139
  Mean relative error: 0.003848
  Mismatched elements: 0 / 16777216 (atol=0.5, rtol=0.05)
  Status: PASS
```

### Error Analysis

Mean relative error of 0.4% confirms correctness. The max relative error (31.7%) and max absolute error (0.75) are expected artifacts of FP16 arithmetic:

- FP16 is not associative. Our kernel accumulates in 16-wide K-tiles; cuBLAS uses different tiling. Two correct implementations produce different results for near-zero output elements.
- FP16 precision is ~3 decimal digits. Over K=4096 multiply-adds, accumulated rounding differences are amplified.
- cuBLAS uses tensor-op math (`CUBLAS_TENSOR_OP_MATH`) with different intermediate precision than our FP32 CUDA-core accumulation.

The relative error metric uses `max(|ref|, |test|, 1.0)` as the denominator to avoid pathological inflation from near-zero values.

### Nsight Compute: Targeted Pipeline Metrics

```bash
./scripts/profile.sh tmem ./build/gemm_fp16_naive
```

**Our naive kernel:**

```
Metric                                                             Value
-----------------------------------------------------------------  -----------
smsp__inst_executed.sum                                            5,999,673,344
l1tex__t_set_accesses_pipe_lsu_mem_global_op_ld.sum                   50,331,648
sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed                   0%
sm__pipe_tma_cycles_active.avg.pct_of_peak_sustained_elapsed                  0%
smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed             0%
sm__throughput.avg.pct_of_peak_sustained_elapsed                          23.68%
l1tex__throughput.avg.pct_of_peak_sustained_elapsed                       14.58%
sm__warps_active.avg.pct_of_peak_sustained_elapsed                         5.92%
```

**cuBLAS (CUTLASS SM100 tensor-op kernel):**

```
Kernel: cutlass3x_sm100_tensorop_h256x256x16gemm_..._256x256x64_0_nnn_align8_2sm
Grid:   (16, 32, 1) × (256, 1, 1)

Metric                                                             Value
-----------------------------------------------------------------  -----------
smsp__inst_executed.sum                                              3,943,633
l1tex__t_set_accesses_pipe_lsu_mem_global_op_ld.sum                          0
sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed            76.29%
sm__pipe_tma_cycles_active.avg.pct_of_peak_sustained_elapsed            1.00%
smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed      76.24%
sm__throughput.avg.pct_of_peak_sustained_elapsed                       76.41%
l1tex__throughput.avg.pct_of_peak_sustained_elapsed                    38.12%
sm__warps_active.avg.pct_of_peak_sustained_elapsed                      8.14%
```

### Nsight Compute: Full Kernel Analysis (`--set full`)

```bash
./scripts/profile.sh ncu ./build/gemm_fp16_naive
```

The full profile (40 replay passes) reveals deeper architectural details.

**cuBLAS:**

| Category | Metric | Value |
|----------|--------|-------|
| Throughput | Compute (SM) | 75.76% |
| Throughput | Memory | 37.80% |
| Throughput | Duration | 178.40 µs |
| Compute | IPC Active | 0.22 inst/cycle |
| Compute | Highest pipeline: TMEM | 75.8% |
| Memory | DRAM Throughput | 8.31% |
| Memory | L2 Compression Ratio | 286.40% |
| Launch | Registers/Thread | 118 |
| Launch | Dynamic SMEM/Block | 231.42 KB |
| Launch | Cluster Size | 2 (2-SM cooperative) |
| Launch | Waves | 3.46 |
| Occupancy | Theoretical / Achieved | 12.50% / 9.88% |
| Occupancy | Limiter: shared memory | 1 block/SM |

**Our naive kernel:**

| Category | Metric | Value |
|----------|--------|-------|
| Throughput | Compute (SM) | 52.09% |
| Throughput | Memory | 33.64% |
| Throughput | Duration | 16.91 ms |
| Compute | IPC Active | 2.34 inst/cycle |
| Compute | Highest pipeline: FMA | 42.1% |
| Memory | Local memory: 94.05% of L1 sectors | register spills |
| Memory | Local load utilization | 1.0 of 32 bytes/sector |
| Memory | 97.90% local loads spill to L2 | |
| Launch | Registers/Thread | 54 |
| Launch | Dynamic SMEM/Block | 8.32 KB |
| Launch | Waves | 0.77 |
| Occupancy | Theoretical / Achieved | 56.25% / 6.25% |
| Occupancy | Limiter: registers | 9 blocks/SM |

### Profile Analysis

| Metric | Naive GEMM | cuBLAS | Ratio |
|--------|-----------|--------|-------|
| Duration | 16.91 ms | 178.40 µs | 95× |
| Instructions executed | 6.0 billion | 3.9 million | 1520× |
| IPC Active | 2.34 | 0.22 | 0.09× |
| Highest pipeline | FMA (42.1%) | TMEM (75.8%) | — |
| Registers/thread | 54 | 118 | — |
| SMEM/block | 8.32 KB | 231.42 KB | 28× |

1520× instruction gap. Each `tcgen05.mma` instruction retires an entire tile of multiply-accumulates, whereas CUDA-core FMA operates on individual element pairs. This single number captures why tensor cores exist.

TMEM is the bottleneck for cuBLAS — and that's ideal. The full profile identifies TMEM as the highest-utilized pipeline at 75.8%, encompassing `LDT(M)`, `STT(M)`, `UTCCP`, `UTCMMA`, and `UTCSHIFT` operations. Being bottlenecked on the tensor memory pipeline means compute is the limiter, not data movement.

IPC paradox: 2.34 vs 0.22. Our naive kernel issues 10× more instructions per cycle than cuBLAS, yet runs 95× slower. cuBLAS's low IPC reflects that each issued instruction (a `tcgen05.mma`) does orders of magnitude more useful work than a scalar FMA. High IPC on scalar instructions is not a substitute for hardware-accelerated matrix operations.

cuBLAS uses 231 KB SMEM per block. Each 2-SM cluster consumes nearly the full 256 KB available, limiting occupancy to 1 block per SM. Despite 12.5% theoretical occupancy, the kernel achieves 75.8% pipeline utilization through deep instruction-level parallelism within the few active warps.

Our `acc[]` array spills to local memory. 94% of all L1 traffic is local memory (register spills). The 128 FP32 accumulators per thread (512 bytes) don't fit in 54 registers, so the compiler spills them to thread-private global memory cached through L1. Each local load/store uses only 1 of 32 bytes per sector — a 32× bandwidth waste from strided access patterns. This is the dominant performance bottleneck in the naive kernel.

Grid doesn't fill the GPU. 1024 CTAs on 148 SMs yields only 0.77 waves. cuBLAS achieves 3.46 waves despite heavier per-block resources because 2-SM clusters spread work across the chip more efficiently.

TMA at 1%. cuBLAS loads data exclusively through the Tensor Memory Accelerator (zero L1 global loads), but TMA barely registers because loads complete asynchronously and overlap entirely with MMA compute. The kernel is compute-bound on the tensor pipe — the ideal operating point.

---

## Compilation

```
hello_tcgen05:    Used 12 registers, used 1 barriers, 4 bytes smem
gemm_fp16_naive:  Used 54 registers, used 1 barriers, 512 bytes cumulative stack size
```

The 512-byte stack in `gemm_fp16_naive` holds the 128 FP32 accumulators (128 × 4 = 512 bytes) that spill from registers. The full ncu profile confirms this: 94% of L1 traffic is local memory from these spilled accumulators.

---

## Files

| File | Purpose |
|------|---------|
| `src/kernels/hello_tcgen05.cu` | TMEM lifecycle bringup kernel |
| `src/kernels/gemm_fp16_naive.cu` | Naive GEMM + TMEM validation vs cuBLAS |
| `include/gemm/tmem_utils.cuh` | TMEM alloc/dealloc/load/store wrappers |
| `include/gemm/tma_utils.cuh` | TMA descriptor setup, mbarrier helpers |
| `scripts/profile.sh` | Nsight Compute/Systems profiling harness |
| `CMakeLists.txt` | Build system (sm_100a, cuBLAS) |
