#!/bin/bash
# Blaze environment verification script
# Checks: CUDA toolkit, SM100 GPU, Nsight Compute, Python deps
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

echo "=== Blaze Environment Check ==="
echo ""

# 1. CUDA Toolkit
echo "--- CUDA Toolkit ---"
if command -v nvcc &>/dev/null; then
    CUDA_VER=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
    if [ "$CUDA_MAJOR" -ge 13 ]; then
        pass "CUDA $CUDA_VER (>= 13.0 required)"
    else
        fail "CUDA $CUDA_VER found, but >= 13.0 required for SM100/tcgen05"
    fi
else
    fail "nvcc not found. Install CUDA Toolkit 13.0+"
fi

# 2. GPU Detection
echo ""
echo "--- GPU ---"
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
    echo "  GPU: $GPU_NAME"
    echo "  Compute Capability: $COMPUTE_CAP"
    if [[ "$COMPUTE_CAP" == "10.0" ]]; then
        pass "SM100 (Blackwell) GPU detected"
    else
        fail "Compute capability $COMPUTE_CAP detected; need 10.0 (SM100/B200)"
    fi
else
    fail "nvidia-smi not found. Is the NVIDIA driver installed?"
fi

# 3. Nsight Compute
echo ""
echo "--- Nsight Compute ---"
if command -v ncu &>/dev/null; then
    NCU_VER=$(ncu --version 2>/dev/null | head -1)
    pass "Nsight Compute found: $NCU_VER"
else
    warn "ncu (Nsight Compute) not found. Profiling will not be available."
fi

# 4. Nsight Systems
if command -v nsys &>/dev/null; then
    pass "Nsight Systems found"
else
    warn "nsys (Nsight Systems) not found. Timeline profiling will not be available."
fi

# 5. CMake
echo ""
echo "--- Build Tools ---"
if command -v cmake &>/dev/null; then
    CMAKE_VER=$(cmake --version | head -1 | sed 's/cmake version //')
    CMAKE_MAJOR=$(echo "$CMAKE_VER" | cut -d. -f1)
    CMAKE_MINOR=$(echo "$CMAKE_VER" | cut -d. -f2)
    if [ "$CMAKE_MAJOR" -ge 3 ] && [ "$CMAKE_MINOR" -ge 24 ]; then
        pass "CMake $CMAKE_VER (>= 3.24 required)"
    else
        fail "CMake $CMAKE_VER found, but >= 3.24 required"
    fi
else
    fail "cmake not found. Install CMake 3.24+"
fi

# 6. Python
echo ""
echo "--- Python ---"
if command -v python3 &>/dev/null; then
    PY_VER=$(python3 --version | sed 's/Python //')
    pass "Python $PY_VER"

    # Check key packages
    python3 -c "import torch; print(f'  PyTorch {torch.__version__}')" 2>/dev/null && pass "PyTorch available" || warn "PyTorch not installed"
    python3 -c "import safetensors" 2>/dev/null && pass "safetensors available" || warn "safetensors not installed"
    python3 -c "import sentencepiece" 2>/dev/null && pass "sentencepiece available" || warn "sentencepiece not installed"
else
    fail "python3 not found. Install Python 3.10+"
fi

echo ""
echo "=== Environment check complete ==="
