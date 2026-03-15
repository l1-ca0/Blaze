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
    src/kernels/gemm_mixed_sm100.cu
)
target_link_libraries(blaze_gemm CUDA::cuda_driver)

# --- Phase 2: Attention kernels ---
add_library(blaze_attention STATIC
    src/kernels/flash_prefill_sm100.cu
    src/kernels/flash_decode_sm100.cu
)

# --- Phase 4: Supporting ops ---
add_library(blaze_ops STATIC
    src/kernels/rmsnorm.cu
    src/kernels/rope.cu
    src/kernels/silu.cu
)

# --- Runtime library ---
add_library(blaze_runtime STATIC
    src/runtime/model_runner.cu
    src/runtime/kv_cache.cpp
    src/runtime/generate.cpp
    src/runtime/weight_loader.cpp
)
target_link_libraries(blaze_runtime blaze_gemm blaze_attention blaze_ops)

# --- Tests ---
add_executable(test_gemm_naive tests/test_gemm_naive.cu)
target_link_libraries(test_gemm_naive CUDA::cublas)

add_executable(test_gemm tests/test_gemm.cu)
target_link_libraries(test_gemm blaze_gemm CUDA::cublas)

add_executable(test_attention tests/test_attention.cu)
target_link_libraries(test_attention blaze_attention CUDA::cublas)

# --- Benchmarks ---
add_executable(bench_gemm benchmarks/bench_gemm.cu)
target_link_libraries(bench_gemm blaze_gemm CUDA::cublas)

add_executable(bench_attention benchmarks/bench_attention.cu)
target_link_libraries(bench_attention blaze_attention)

# --- Python bindings (pybind11) ---
find_package(Python3 COMPONENTS Interpreter Development)
find_package(pybind11 QUIET)
if(pybind11_FOUND)
    pybind11_add_module(blaze_ext src/runtime/python_bindings.cpp)
    target_link_libraries(blaze_ext PRIVATE blaze_runtime)
endif()
