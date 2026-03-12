# Blaze

**Blackwell-native LLM inference kernels for NVIDIA B200.**

Hand-written CUDA kernels targeting SM100 — `tcgen05` tensor cores, Tensor Memory (TMEM), TMA, and native FP4 compute — for high-performance large language model inference. No frameworks, no wrappers, just metal.

## Motivation

Blackwell's hardware can hit 96% of theoretical peak on tensor cores. The bottleneck in LLM inference isn't compute — it's everything around it: data movement, softmax throughput, warp scheduling, and precision management. Existing inference engines are still adapting Hopper-era kernels to Blackwell. Blaze starts from scratch on SM100 to close this gap.

## What This Project Is

A ground-up implementation of the core CUDA kernels needed for LLM inference on B200:

- **GEMM kernels** — FP4, FP8, and mixed-precision matrix multiply using `tcgen05.mma` with TMEM accumulation and TMA pipelining
- **Fused attention** — Prefill and decode kernels with online softmax, TMEM-resident accumulators, and split-KV parallelism
- **FP4 quantization** — Offline weight conversion to NVFP4 format (E2M1 data + E4M3 micro-block scaling)
- **End-to-end generation** — Llama-7B text generation using only the above custom kernels

## What This Project Is Not

A production serving engine. There's no batching, no paged KV-cache, no HTTP server. This is a kernel-level project focused on squeezing maximum performance out of Blackwell's new hardware primitives.

## Key Blackwell Features Exploited

| Feature | What It Does | How Blaze Uses It |
|---|---|---|
| **Tensor Memory (TMEM)** | 256 KB/SM dedicated accumulator storage | MMA accumulators live in TMEM, freeing registers for softmax/CUDA-core work in parallel |
| **`tcgen05.mma`** | Async MMA launched by a single thread | Deeper pipelines, lower register pressure vs Hopper's warpgroup model |
| **TMA** | Hardware-accelerated async tensor loads | Double-buffered HBM→SMEM transfers overlapped with compute |
| **Native FP4** | E2M1 precision with hardware support | 2× memory savings over FP8, direct tensor core execution |
| **8 TB/s HBM3e** | Memory bandwidth | Sub-millisecond decode latency at practical sequence lengths |

## Requirements

- NVIDIA B200 GPU (SM100, compute capability 10.0)
- CUDA Toolkit 13.0+
- CMake 3.24+
- Python 3.10+, PyTorch 2.5+

## Build

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=100 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## Status

**Work in progress.** Building out kernel implementations phase by phase.

- [ ] Phase 0 — `tcgen05` bringup and TMEM lifecycle validation
- [ ] Phase 1 — FP8/FP4/mixed GEMM kernels 
- [ ] Phase 2 — Fused prefill and decode attention
- [ ] Phase 3 — NVFP4 quantization pipeline
- [ ] Phase 4 — End-to-end Llama-7B generation

## References

- [tcgen05 for dummies](https://gau-nernst.github.io/tcgen05/) — Blackwell tensor core tutorial in plain CUDA C++
- [FlashAttention-4](https://www.together.ai/blog/flashattention-4) — Algorithm/kernel co-design for Blackwell's asymmetric hardware scaling
- [Reverse-engineering FA4](https://modal.com/blog/reverse-engineer-flash-attention-4) — Pipeline breakdown of the FA4 kernel
- [Microbenchmarking Blackwell](https://arxiv.org/abs/2512.02189) — First systematic characterization of B200 (TMEM, DE, tcgen05)
- [FP4 MoE kernel engineering](https://huggingface.co/blog/apsys/blackwell-nvfp4-comparison) — Practical NVFP4 optimization on B200
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) — Reference tcgen05 kernel structure

