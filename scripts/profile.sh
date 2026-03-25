#!/bin/bash
# Blaze profiling harness using Nsight Compute and Nsight Systems
#
# Usage:
#   ./scripts/profile.sh ncu <executable> [args...]         — full kernel analysis (all kernels)
#   ./scripts/profile.sh nsys <executable> [args...]        — timeline trace
#   ./scripts/profile.sh tmem <executable> [args...]        — TMEM-focused metrics
#   ./scripts/profile.sh gemm <kernel> <shape> [--metric M] — profile specific GEMM kernel+shape
#
# GEMM profiling mode (Phase 1):
#   ./scripts/profile.sh gemm fp8 prefill2048_qkv
#   ./scripts/profile.sh gemm fp8 prefill2048_qkv --metric tmem
#   ./scripts/profile.sh gemm mixed decode_qkv
#   ./scripts/profile.sh gemm fp4 batch128_ffn_down
#
#   Kernel types: fp8, mixed, fp4, fp4_blkscaled
#   Available shapes (index 0-12):
#     0  decode_qkv           (M=1,    N=12288, K=4096)
#     1  decode_out_proj      (M=1,    N=4096,  K=4096)
#     2  decode_ffn_gate_up   (M=1,    N=22016, K=4096)
#     3  decode_ffn_down      (M=1,    N=4096,  K=11008)
#     4  batch32_qkv          (M=32,   N=12288, K=4096)
#     5  batch32_ffn_down     (M=32,   N=4096,  K=11008)
#     6  batch128_qkv         (M=128,  N=12288, K=4096)
#     7  batch128_ffn_gate_up (M=128,  N=22016, K=4096)
#     8  batch128_ffn_down    (M=128,  N=4096,  K=11008)
#     9  prefill512_qkv       (M=512,  N=12288, K=4096)
#     10 prefill512_ffn_down  (M=512,  N=4096,  K=11008)
#     11 prefill2048_qkv      (M=2048, N=12288, K=4096)
#     12 prefill2048_ffn_down (M=2048, N=4096,  K=11008)
#
set -e

TOOL=${1:-ncu}
shift || { echo "Usage: $0 {ncu|nsys|tmem|gemm} ..."; exit 1; }

REPORT_DIR="profiles"
mkdir -p "$REPORT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ---------------------------------------------------------------------------
# GEMM profiling mode
# ---------------------------------------------------------------------------
if [ "$TOOL" = "gemm" ]; then
    KERNEL_TYPE=${1:?"Usage: $0 gemm {fp8|mixed|fp4} SHAPE_NAME [--metric tmem]"}
    SHAPE_NAME=${2:?"Usage: $0 gemm {fp8|mixed|fp4} SHAPE_NAME [--metric tmem]"}
    shift 2
    METRIC_MODE="full"
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --metric) METRIC_MODE="$2"; shift 2 ;;
            *) echo "Unknown option: $1"; exit 1 ;;
        esac
    done

    BENCH_BIN="./build/bench_gemm"
    if [ ! -f "$BENCH_BIN" ]; then
        echo "Error: $BENCH_BIN not found. Run: cmake --build build"
        exit 1
    fi

    # Map kernel type to ncu kernel name
    case "$KERNEL_TYPE" in
        fp8)            NCU_KERNEL="gemm_fp8_kernel" ;;
        mixed)          NCU_KERNEL="gemm_mixed_kernel" ;;
        fp4)            NCU_KERNEL="gemm_fp4_kernel" ;;
        fp4_blkscaled)  NCU_KERNEL="gemm_fp4_blkscaled_kernel" ;;
        *) echo "Unknown kernel type: $KERNEL_TYPE (use fp8, mixed, fp4, fp4_blkscaled)"; exit 1 ;;
    esac

    # Map shape name to index (0-12)
    declare -A SHAPE_MAP=(
        [decode_qkv]=0           [decode_out_proj]=1
        [decode_ffn_gate_up]=2   [decode_ffn_down]=3
        [batch32_qkv]=4          [batch32_ffn_down]=5
        [batch128_qkv]=6         [batch128_ffn_gate_up]=7
        [batch128_ffn_down]=8
        [prefill512_qkv]=9       [prefill512_ffn_down]=10
        [prefill2048_qkv]=11     [prefill2048_ffn_down]=12
    )

    SHAPE_IDX=${SHAPE_MAP[$SHAPE_NAME]}
    if [ -z "$SHAPE_IDX" ]; then
        echo "Unknown shape: $SHAPE_NAME"
        echo "Available shapes:"
        for key in $(echo "${!SHAPE_MAP[@]}" | tr ' ' '\n' | sort); do
            echo "  $key (index ${SHAPE_MAP[$key]})"
        done
        exit 1
    fi

    # bench_gemm launches: 13 shapes × 110 iterations (10 warmup + 100 timed) per kernel type.
    # ncu --kernel-name filters by kernel name, so we only count launches of our target kernel.
    # Skip to: shape_index * 110 + 10 warmup = first timed iteration of the target shape.
    LAUNCHES_PER_SHAPE=110
    SKIP=$((SHAPE_IDX * LAUNCHES_PER_SHAPE + 10))

    REPORT_NAME="${REPORT_DIR}/${NCU_KERNEL}_${SHAPE_NAME}_${TIMESTAMP}"

    echo "=== Profiling $KERNEL_TYPE GEMM: $SHAPE_NAME (shape $SHAPE_IDX) ==="
    echo "    Kernel:     $NCU_KERNEL"
    echo "    Launch skip: $SKIP (shape $SHAPE_IDX × $LAUNCHES_PER_SHAPE + 10 warmup)"
    echo "    Metric mode: $METRIC_MODE"
    echo ""

    case "$METRIC_MODE" in
        full)
            ncu --set full \
                --kernel-name "$NCU_KERNEL" \
                --launch-count 1 \
                --launch-skip "$SKIP" \
                --export "$REPORT_NAME" \
                --force-overwrite \
                "$BENCH_BIN"
            echo ""
            echo "Report saved to ${REPORT_NAME}.ncu-rep"
            ;;
        tmem)
            ncu --metrics \
                sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed,\
smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tmem_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_elapsed,\
smsp__inst_executed.sum \
                --kernel-name "$NCU_KERNEL" \
                --launch-count 1 \
                --launch-skip "$SKIP" \
                "$BENCH_BIN"
            ;;
        summary)
            ncu --metrics \
                gpu__time_duration.sum,\
sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_elapsed,\
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio \
                --kernel-name "$NCU_KERNEL" \
                --launch-count 1 \
                --launch-skip "$SKIP" \
                "$BENCH_BIN"
            ;;
        *)
            echo "Unknown metric mode: $METRIC_MODE (use full, tmem, summary)"
            exit 1
            ;;
    esac
    exit 0
fi

# ---------------------------------------------------------------------------
# Generic profiling modes
# ---------------------------------------------------------------------------
EXECUTABLE=$1
shift || { echo "Usage: $0 {ncu|nsys|tmem|gemm} <executable> [args...]"; exit 1; }
BASENAME=$(basename "$EXECUTABLE")

case "$TOOL" in
    ncu)
        echo "=== Nsight Compute: Full kernel analysis ==="
        ncu --set full \
            --target-processes all \
            --export "${REPORT_DIR}/${BASENAME}_${TIMESTAMP}" \
            --force-overwrite \
            "$EXECUTABLE" "$@"
        echo "Report saved to ${REPORT_DIR}/${BASENAME}_${TIMESTAMP}.ncu-rep"
        ;;

    tmem)
        echo "=== Nsight Compute: SM100 tcgen05/TMEM metrics ==="
        ncu --metrics \
            sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed,\
smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tmem_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_set_accesses_pipe_lsu_mem_global_op_ld.sum,\
sm__warps_active.avg.pct_of_peak_sustained_elapsed,\
smsp__inst_executed.sum \
            --target-processes all \
            "$EXECUTABLE" "$@"
        ;;

    nsys)
        echo "=== Nsight Systems: Timeline trace ==="
        nsys profile \
            --output "${REPORT_DIR}/${BASENAME}_${TIMESTAMP}" \
            --force-overwrite true \
            --trace cuda,nvtx \
            "$EXECUTABLE" "$@"
        echo "Report saved to ${REPORT_DIR}/${BASENAME}_${TIMESTAMP}.nsys-rep"
        ;;

    *)
        echo "Unknown tool: $TOOL"
        echo "Usage: $0 {ncu|nsys|tmem|gemm} ..."
        exit 1
        ;;
esac
