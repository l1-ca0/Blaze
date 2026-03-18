#!/bin/bash
# Blaze profiling harness using Nsight Compute and Nsight Systems
# Usage:
#   ./scripts/profile.sh ncu <executable> [args...]   — detailed kernel metrics
#   ./scripts/profile.sh nsys <executable> [args...]   — timeline trace
#   ./scripts/profile.sh tmem <executable> [args...]   — TMEM-focused metrics
set -e

TOOL=${1:-ncu}
shift || { echo "Usage: $0 {ncu|nsys|tmem} <executable> [args...]"; exit 1; }
EXECUTABLE=$1
shift || { echo "Usage: $0 {ncu|nsys|tmem} <executable> [args...]"; exit 1; }

REPORT_DIR="profiles"
mkdir -p "$REPORT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
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
        echo "Usage: $0 {ncu|nsys|tmem} <executable> [args...]"
        exit 1
        ;;
esac
