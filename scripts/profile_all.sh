#!/bin/bash
# Blaze: Full profiling snapshot for all GEMM kernels.
#
# Produces a timestamped report with:
#   1. Clean benchmark timing (no ncu) for all 3 kernels × 13 shapes
#   2. ncu summary metrics for all 3 kernels × representative shapes
#
# Usage:
#   ./scripts/profile_all.sh                  — full snapshot (bench + ncu)
#   ./scripts/profile_all.sh bench            — benchmark timing only (fast, ~10s)
#   ./scripts/profile_all.sh ncu              — ncu metrics only (~15 min)
#   ./scripts/profile_all.sh ncu --all-shapes — ncu for all 39 kernel×shape combos (~45 min)
#
# Output saved to: profiles/snapshot_YYYYMMDD_HHMMSS/
#
set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="profiles/snapshot_${TIMESTAMP}"
mkdir -p "$OUTDIR"

BENCH_BIN="./build/bench_gemm"
if [ ! -f "$BENCH_BIN" ]; then
    echo "Error: $BENCH_BIN not found. Run: cmake --build build"
    exit 1
fi

MODE=${1:-all}
ALL_SHAPES=false
if [ "$MODE" = "ncu" ] && [ "$2" = "--all-shapes" ]; then
    ALL_SHAPES=true
fi

# Representative shapes: small M (launch-bound), medium M, large M (compute-bound)
REP_SHAPES=(decode_qkv batch128_qkv prefill2048_qkv)
ALL_SHAPE_LIST=(
    decode_qkv decode_out_proj decode_ffn_gate_up decode_ffn_down
    batch32_qkv batch32_ffn_down
    batch128_qkv batch128_ffn_gate_up batch128_ffn_down
    prefill512_qkv prefill512_ffn_down
    prefill2048_qkv prefill2048_ffn_down
)
KERNELS=(fp8 mixed fp4 fp4_blkscaled)

# Map kernel type to ncu kernel name
declare -A KERNEL_MAP=(
    [fp8]=gemm_fp8_kernel
    [mixed]=gemm_mixed_kernel
    [fp4]=gemm_fp4_kernel
    [fp4_blkscaled]=gemm_fp4_blkscaled_kernel
)

# Map shape name to index
declare -A SHAPE_MAP=(
    [decode_qkv]=0           [decode_out_proj]=1
    [decode_ffn_gate_up]=2   [decode_ffn_down]=3
    [batch32_qkv]=4          [batch32_ffn_down]=5
    [batch128_qkv]=6         [batch128_ffn_gate_up]=7
    [batch128_ffn_down]=8
    [prefill512_qkv]=9       [prefill512_ffn_down]=10
    [prefill2048_qkv]=11     [prefill2048_ffn_down]=12
)

LAUNCHES_PER_SHAPE=110

# -------------------------------------------------------------------------
# Step 1: Clean benchmark timing
# -------------------------------------------------------------------------
run_bench() {
    echo "=== Step 1: Clean benchmark (no ncu) ==="
    echo ""
    "$BENCH_BIN" 2>&1 | tee "$OUTDIR/bench.txt"
    echo ""
    echo "Benchmark saved to $OUTDIR/bench.txt"
    echo ""
}

# -------------------------------------------------------------------------
# Step 2: ncu summary metrics per kernel × shape
# -------------------------------------------------------------------------
run_ncu() {
    if [ "$ALL_SHAPES" = true ]; then
        SHAPES=("${ALL_SHAPE_LIST[@]}")
        echo "=== Step 2: ncu summary metrics (all 39 combos) ==="
    else
        SHAPES=("${REP_SHAPES[@]}")
        echo "=== Step 2: ncu summary metrics (3 kernels × ${#SHAPES[@]} representative shapes) ==="
    fi
    echo ""

    NCU_FILE="$OUTDIR/ncu_summary.txt"
    : > "$NCU_FILE"

    # Header
    printf "%-8s %-24s %10s %8s %8s %8s %8s %8s %10s\n" \
        "Kernel" "Shape" "Time(us)" "TC%" "SM%" "L1%" "DRAM%" "Warps%" "BarStall" \
        | tee -a "$NCU_FILE"
    printf "%s\n" "$(printf '%.0s-' {1..110})" | tee -a "$NCU_FILE"

    for kernel in "${KERNELS[@]}"; do
        NCU_KERNEL="${KERNEL_MAP[$kernel]}"
        for shape in "${SHAPES[@]}"; do
            SHAPE_IDX=${SHAPE_MAP[$shape]}
            SKIP=$((SHAPE_IDX * LAUNCHES_PER_SHAPE + 10))

            # Run ncu and capture output
            NCU_OUT=$(ncu --metrics \
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
                "$BENCH_BIN" 2>&1)

            # Parse metrics from ncu output
            TIME=$(echo "$NCU_OUT" | grep "gpu__time_duration.sum" | awk '{print $NF}')
            TC=$(echo "$NCU_OUT" | grep "sm__pipe_tc_cycles_active" | awk '{print $NF}')
            SM=$(echo "$NCU_OUT" | grep "sm__throughput" | awk '{print $NF}')
            L1=$(echo "$NCU_OUT" | grep "l1tex__throughput" | awk '{print $NF}')
            DRAM=$(echo "$NCU_OUT" | grep "dram__throughput" | awk '{print $NF}')
            WARPS=$(echo "$NCU_OUT" | grep "sm__warps_active" | awk '{print $NF}')
            BARSTALL=$(echo "$NCU_OUT" | grep "barrier_per_issue" | awk '{print $NF}')

            printf "%-8s %-24s %10s %8s %8s %8s %8s %8s %10s\n" \
                "$kernel" "$shape" "$TIME" "$TC" "$SM" "$L1" "$DRAM" "$WARPS" "$BARSTALL" \
                | tee -a "$NCU_FILE"
        done
    done

    echo ""
    echo "ncu summary saved to $NCU_FILE"
    echo ""
}

# -------------------------------------------------------------------------
# Run requested mode
# -------------------------------------------------------------------------
echo "Blaze profiling snapshot: $OUTDIR"
echo "Date: $(date)"
echo ""

case "$MODE" in
    bench) run_bench ;;
    ncu)   run_ncu ;;
    all)   run_bench; run_ncu ;;
    *)     echo "Usage: $0 [bench|ncu|all] [--all-shapes]"; exit 1 ;;
esac

echo "=== Snapshot complete: $OUTDIR ==="
