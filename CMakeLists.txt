cmake_minimum_required(VERSION 3.24)
project(blaze LANGUAGES CXX CUDA)

# Require CUDA 13.0+ for SM100 / tcgen05 support
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
# sm_100a = Blackwell accelerated (enables tcgen05, TMEM, TMA features)
set(CMAKE_CUDA_ARCHITECTURES 100a)

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v --maxrregcount=255 --extended-lambda")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -O0")

# Default to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Include paths
include_directories(${CMAKE_SOURCE_DIR}/include)

# Find CUDA toolkit (provides CUDA::cublas, CUDA::cuda_driver, etc.)
find_package(CUDAToolkit REQUIRED)

# --- Phase 0: Naive bringup kernels ---
add_executable(hello_tcgen05 src/kernels/hello_tcgen05.cu)

add_executable(gemm_fp16_naive src/kernels/gemm_fp16_naive.cu)
target_link_libraries(gemm_fp16_naive CUDA::cublas)

# --- Phase 1: Production GEMM kernels ---
add_library(blaze_gemm STATIC
    src/kernels/gemm_fp8_sm100.cu
    src/kernels/gemm_fp4_sm100.cu
    src/kernels/gemm_fp4_blkscaled_sm100.cu
    src/kernels/gemm_mixed_sm100.cu
)
target_link_libraries(blaze_gemm CUDA::cuda_driver)

# --- Tests ---
add_executable(test_gemm_naive tests/test_gemm_naive.cu)
target_link_libraries(test_gemm_naive CUDA::cublas)

add_executable(test_gemm tests/test_gemm.cu)
target_link_libraries(test_gemm blaze_gemm CUDA::cublas)

add_executable(test_sm100_primitives tests/test_sm100_primitives.cu)
target_link_libraries(test_sm100_primitives CUDA::cuda_driver)

# test_commit disabled: tcgen05.commit has "State space incorrect" error in CUDA 13.1.80
# add_executable(test_commit tests/test_commit.cu)
# target_link_libraries(test_commit CUDA::cuda_driver)

# --- Benchmarks ---
add_executable(bench_gemm benchmarks/bench_gemm.cu)
target_link_libraries(bench_gemm blaze_gemm CUDA::cublas)
