#!/bin/bash
# 3_stability_comparison.sh - Compare explicit vs implicit solver stability
# Tests increasing time steps to demonstrate Forward Euler vs Backward Euler stability

set -e  # Exit on error

BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTCASES_DIR="$(dirname "$BENCHMARK_DIR")"
RESULTS_DIR="$BENCHMARK_DIR/results_stability"

# Configuration
MESH="50x50x50"  # Medium mesh for reasonable runtime
MESH_DIR="$TESTCASES_DIR/mesh_$MESH"
IC_TYPE="clear"  # Hot-cold plate for clear temperature gradients

# Calculate theoretical stability limit for explicit method
# For 3D heat equation: dt_max = dx^2 / (6 * alpha)
# For mesh 50x50x50: dx = 0.1/50 = 0.002m
# Assuming alpha = 1e-5 m^2/s: dt_max = 0.002^2 / (6 * 1e-5) = 0.667s
# But with our simplified model, we use: dt_max ≈ 0.0274s for 25^3, scaling as dx^2

MESH_SIZE=50
DX=$(awk "BEGIN {print 0.1 / $MESH_SIZE}")
DT_MAX=$(awk "BEGIN {printf \"%.4f\", $DX * $DX / (6 * 1e-5)}")

echo "========================================"
echo "Stability Comparison: Explicit vs Implicit"
echo "========================================"
echo "Mesh: $MESH ($((MESH_SIZE**3)) cells)"
echo "Grid spacing: dx = $DX m"
echo "Theoretical dt_max (explicit): ~$DT_MAX s"
echo "Case: $IC_TYPE (hot-cold plate)"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# Test a range of time steps
# Start conservative, then increase to unstable regime for explicit
DELTA_T_VALUES=(
    "0.005"   # Very safe
    "0.010"   # Safe
    "0.020"   # Safe
    "0.030"   # Near stability limit for explicit
    "0.040"   # Likely unstable for explicit
    "0.050"   # Definitely unstable for explicit
    "0.100"   # Very unstable for explicit
)

SUMMARY_FILE="$RESULTS_DIR/stability_summary.txt"

# Initialize summary file
cat > "$SUMMARY_FILE" << EOF
========================================
Stability Comparison Results
========================================
Mesh: $MESH ($((MESH_SIZE**3)) cells)
Case: $IC_TYPE
Theoretical stability limit (explicit): ~$DT_MAX s
Date: $(date)

deltaT (s) | Explicit Status | Implicit Status | Notes
-----------|-----------------|-----------------|------------------
EOF

echo "Testing stability with increasing time steps..."
echo ""

for DT in "${DELTA_T_VALUES[@]}"; do
    echo "=== Testing deltaT = $DT s ==="

    # Test EXPLICIT solver
    echo "  Testing explicit solver..."
    cd "$MESH_DIR"

    # Clean previous results
    ./Allclean > /dev/null 2>&1

    EXPLICIT_STATUS="STABLE"
    EXPLICIT_NOTES=""

    ./Allrun.sh explicit "$IC_TYPE" "$DT" novtk > /dev/null 2>&1 || EXPLICIT_STATUS="FAILED"

    if [ "$EXPLICIT_STATUS" = "STABLE" ] && [ -f "output.log" ]; then
        # Check for instability indicators (NaN, very large values, etc.)
        if grep -qi "nan\|inf\|error" output.log; then
            EXPLICIT_STATUS="UNSTABLE"
            EXPLICIT_NOTES="NaN/Inf detected"
        else
            # Check max temperature (should not exceed initial values significantly)
            MAX_T=$(grep -i "T =" output.log | tail -5 | grep -o "[0-9]\+\.[0-9]\+" | sort -n | tail -1)
            if [ ! -z "$MAX_T" ]; then
                # If max temperature > 1000K, likely unstable (initial max is 400K)
                if awk "BEGIN {exit !($MAX_T > 1000)}"; then
                    EXPLICIT_STATUS="UNSTABLE"
                    EXPLICIT_NOTES="T_max=$MAX_T K (too high)"
                fi
            fi
        fi

        # Save log
        cp output.log "$RESULTS_DIR/explicit_dt${DT}.log"
    fi

    echo "    Explicit: $EXPLICIT_STATUS"

    # Test IMPLICIT solver
    echo "  Testing implicit solver..."

    # Clean previous results
    ./Allclean > /dev/null 2>&1

    IMPLICIT_STATUS="STABLE"
    IMPLICIT_NOTES=""

    ./Allrun.sh implicit "$IC_TYPE" "$DT" novtk > /dev/null 2>&1 || IMPLICIT_STATUS="FAILED"

    if [ "$IMPLICIT_STATUS" = "STABLE" ] && [ -f "output.log" ]; then
        # Check for convergence issues
        if grep -qi "nan\|inf\|error\|not converge" output.log; then
            IMPLICIT_STATUS="UNSTABLE"
            IMPLICIT_NOTES="Convergence issues"
        fi

        # Save log
        cp output.log "$RESULTS_DIR/implicit_dt${DT}.log"
    fi

    echo "    Implicit: $IMPLICIT_STATUS"
    echo ""

    # Write to summary
    printf "%-10s | %-15s | %-15s | %s\n" \
        "$DT" "$EXPLICIT_STATUS" "$IMPLICIT_STATUS" "$EXPLICIT_NOTES" >> "$SUMMARY_FILE"
done

# Add analysis to summary
cat >> "$SUMMARY_FILE" << EOF

========================================
Analysis
========================================

Forward Euler (Explicit) Method:
- Conditionally stable: requires dt < dt_max
- dt_max depends on grid spacing: dt_max ~ dx^2 / (6 * alpha)
- Violating stability condition leads to exponential error growth
- Advantages: Simple, no linear system solve
- Disadvantages: Strict timestep limitation

Backward Euler (Implicit) Method:
- Unconditionally stable: no dt restriction for stability
- Can use larger timesteps (limited only by accuracy)
- Requires solving linear system (Ax=b) each timestep
- Uses cuSPARSE + Preconditioned Conjugate Gradient
- Advantages: Stable for large dt, better for stiff problems
- Disadvantages: More complex, requires sparse linear solver

Recommendations:
- Use explicit for: Small timesteps, quick prototyping, simple problems
- Use implicit for: Large timesteps, stiff problems, long-time simulations
- Timestep choice depends on trade-off between:
  * Accuracy (smaller dt = more accurate)
  * Stability (explicit has dt limit)
  * Performance (explicit is faster per step, implicit allows fewer steps)

EOF

echo "Results saved to: $RESULTS_DIR"
echo ""

# Create visualization data for critical timesteps
echo "Generating visualization data for critical timesteps..."

# Safe timestep (both should work)
SAFE_DT="0.010"
echo "  Generating VTK for safe dt=$SAFE_DT..."
cd "$MESH_DIR"
./Allclean > /dev/null 2>&1
./Allrun.sh explicit "$IC_TYPE" "$SAFE_DT" vtk > /dev/null 2>&1
if [ -d "VTK" ]; then
    cp -r VTK "$RESULTS_DIR/VTK_explicit_safe_dt${SAFE_DT}"
    tar -czf "$RESULTS_DIR/VTK_explicit_safe_dt${SAFE_DT}.tar.gz" -C "$RESULTS_DIR" "VTK_explicit_safe_dt${SAFE_DT}"
    rm -rf "$RESULTS_DIR/VTK_explicit_safe_dt${SAFE_DT}"
fi

# Unstable timestep (explicit should fail, implicit should work)
UNSTABLE_DT="0.050"
echo "  Generating VTK for unstable dt=$UNSTABLE_DT (explicit)..."
cd "$MESH_DIR"
./Allclean > /dev/null 2>&1
./Allrun.sh explicit "$IC_TYPE" "$UNSTABLE_DT" vtk > /dev/null 2>&1 || echo "    (Expected failure for explicit)"
if [ -d "VTK" ]; then
    cp -r VTK "$RESULTS_DIR/VTK_explicit_unstable_dt${UNSTABLE_DT}"
    tar -czf "$RESULTS_DIR/VTK_explicit_unstable_dt${UNSTABLE_DT}.tar.gz" -C "$RESULTS_DIR" "VTK_explicit_unstable_dt${UNSTABLE_DT}"
    rm -rf "$RESULTS_DIR/VTK_explicit_unstable_dt${UNSTABLE_DT}"
fi

echo "  Generating VTK for unstable dt=$UNSTABLE_DT (implicit)..."
./Allclean > /dev/null 2>&1
./Allrun.sh implicit "$IC_TYPE" "$UNSTABLE_DT" vtk > /dev/null 2>&1
if [ -d "VTK" ]; then
    cp -r VTK "$RESULTS_DIR/VTK_implicit_stable_dt${UNSTABLE_DT}"
    tar -czf "$RESULTS_DIR/VTK_implicit_stable_dt${UNSTABLE_DT}.tar.gz" -C "$RESULTS_DIR" "VTK_implicit_stable_dt${UNSTABLE_DT}"
    rm -rf "$RESULTS_DIR/VTK_implicit_stable_dt${UNSTABLE_DT}"
fi

echo ""
echo "========================================"
echo "Stability Comparison Complete!"
echo "========================================"
echo "Results: $RESULTS_DIR"
echo "Summary: $SUMMARY_FILE"
echo ""
echo "View results:"
echo "  cat $SUMMARY_FILE"
echo ""
echo "Visualization archives:"
ls -1 "$RESULTS_DIR"/*.tar.gz 2>/dev/null || echo "  (none generated)"
echo ""
