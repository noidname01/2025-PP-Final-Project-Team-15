#!/bin/bash
# run_all_benchmarks.sh - Run all benchmark tests sequentially

set -e  # Exit on error

BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo "Running All Benchmarks"
echo "========================================"
echo "This will run three benchmark suites:"
echo "  1. Time profiling (all meshes, 3 solvers)"
echo "  2. Visualization (all cases, 3 solvers)"
echo "  3. Stability comparison (explicit vs implicit)"
echo ""
echo "This may take a while..."
echo ""

# Prompt user to continue
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Run each benchmark
cd "$BENCHMARK_DIR"

echo ""
echo "========================================"
echo "1/3: Time Profiling Benchmark"
echo "========================================"
bash 1_time_profiling.sh

echo ""
echo "========================================"
echo "2/3: Visualization Benchmark"
echo "========================================"
bash 2_visualization.sh

echo ""
echo "========================================"
echo "3/3: Stability Comparison Benchmark"
echo "========================================"
bash 3_stability_comparison.sh

echo ""
echo "========================================"
echo "All Benchmarks Complete!"
echo "========================================"
echo ""
echo "Results locations:"
echo "  1. Time profiling:      $BENCHMARK_DIR/results_time_profiling/"
echo "  2. Visualization:       $BENCHMARK_DIR/results_visualization/"
echo "  3. Stability:           $BENCHMARK_DIR/results_stability/"
echo ""
echo "View summaries:"
echo "  cat results_time_profiling/timing_summary.txt"
echo "  cat results_visualization/README.md"
echo "  cat results_stability/stability_summary.txt"
echo ""
