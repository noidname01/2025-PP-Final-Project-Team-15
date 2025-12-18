#!/bin/bash
# 1_time_profiling.sh - Benchmark solver performance across mesh sizes
# Runs the same case (clear: hot-cold plate) on all mesh sizes with all three solvers

# Note: Don't use set -e, we want to continue even if individual simulations fail

BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTCASES_DIR="$(dirname "$BENCHMARK_DIR")"
RESULTS_DIR="$BENCHMARK_DIR/results_time_profiling"

# Configuration
IC_TYPE="clear"           # Use hot-cold plate for consistency
DELTA_T="0.01"           # Safe timestep for all meshes
MESH_SIZES=("25x25x25" "50x50x50" "100x100x100" "150x150x150")
SOLVERS=("explicit" "implicit" "cpu")

echo "========================================"
echo "Time Profiling Benchmark"
echo "========================================"
echo "Case: $IC_TYPE (hot-cold plate)"
echo "Time step: $DELTA_T"
echo "Mesh sizes: ${MESH_SIZES[@]}"
echo "Solvers: ${SOLVERS[@]}"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"
SUMMARY_FILE="$RESULTS_DIR/timing_summary.txt"

# Initialize summary file
cat > "$SUMMARY_FILE" << EOF
========================================
Time Profiling Benchmark Results
========================================
Test case: $IC_TYPE
Time step: $DELTA_T
Date: $(date)

EOF

echo "Mesh Size | Solver   | Cells    | Time (s) | Performance" >> "$SUMMARY_FILE"
echo "----------|----------|----------|----------|------------" >> "$SUMMARY_FILE"

# Run benchmarks
for MESH in "${MESH_SIZES[@]}"; do
    MESH_DIR="$TESTCASES_DIR/mesh_$MESH"

    if [ ! -d "$MESH_DIR" ]; then
        echo "Warning: $MESH_DIR not found, skipping..."
        continue
    fi

    echo "=== Testing mesh_$MESH ==="

    # Extract cell count from blockMeshDict
    CELLS=$(echo $MESH | cut -d'x' -f1)
    TOTAL_CELLS=$((CELLS * CELLS * CELLS))

    for SOLVER in "${SOLVERS[@]}"; do
        echo "  Running $SOLVER solver..."

        cd "$MESH_DIR"

        # Clean previous results
        ./Allclean > /dev/null 2>&1

        # Run simulation without VTK export (allow failure)
        RUN_STATUS="SUCCESS"
        ./Allrun.sh "$SOLVER" "$IC_TYPE" "$DELTA_T" novtk > /dev/null 2>&1 || RUN_STATUS="FAILED"

        # Extract execution time from log
        if [ -f "output.log" ] && [ "$RUN_STATUS" = "SUCCESS" ]; then
            # Check for instability indicators (crashes, NaN, Inf)
            if grep -qi "core dumped\|segmentation fault\|FOAM FATAL ERROR" output.log; then
                RUN_STATUS="UNSTABLE"
            # Also check for NaN/Inf in temperature field
            elif grep -E "T.*=.*(nan|inf|-nan|-inf)" output.log > /dev/null 2>&1; then
                RUN_STATUS="UNSTABLE"
            fi

            # Look for "ExecutionTime" or "ClockTime" in the log
            EXEC_TIME=$(grep -i "ExecutionTime\|ClockTime" output.log | tail -1 | awk '{print $3}')

            if [ -z "$EXEC_TIME" ] || [ "$RUN_STATUS" = "UNSTABLE" ]; then
                EXEC_TIME="x"
                PERF="x"
                echo "    ⚠ Unstable/Failed (likely dt too large for explicit)"
            else
                # Calculate cells per second
                PERF=$(awk "BEGIN {printf \"%.0f\", $TOTAL_CELLS / $EXEC_TIME}")
            fi

            # Save log to results (even if failed)
            LOG_NAME="${MESH}_${SOLVER}_${IC_TYPE}.log"
            cp output.log "$RESULTS_DIR/$LOG_NAME"

            # Print and save result
            printf "%-9s | %-8s | %-8d | %-8s | %s\n" \
                "$MESH" "$SOLVER" "$TOTAL_CELLS" "$EXEC_TIME" "$PERF" | tee -a "$SUMMARY_FILE"
        else
            # Simulation failed completely
            echo "    ✗ Failed to run (check solver compilation)"
            printf "%-9s | %-8s | %-8d | x        | x\n" \
                "$MESH" "$SOLVER" "$TOTAL_CELLS" >> "$SUMMARY_FILE"
        fi

        echo ""
    done
done

# Add summary statistics
cat >> "$SUMMARY_FILE" << EOF

========================================
Summary
========================================

Performance comparison (cells/s):
EOF

# Calculate speedups
echo "" >> "$SUMMARY_FILE"
echo "Results saved to: $RESULTS_DIR" >> "$SUMMARY_FILE"

echo "========================================"
echo "Time Profiling Complete!"
echo "========================================"
echo "Results: $RESULTS_DIR"
echo "Summary: $SUMMARY_FILE"
echo ""
echo "View results:"
echo "  cat $SUMMARY_FILE"
echo "  ls $RESULTS_DIR/*.log"
echo ""
