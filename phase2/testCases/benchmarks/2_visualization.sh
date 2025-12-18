#!/bin/bash
# 2_visualization.sh - Generate visualization data for all solvers and cases
# Runs 4 different initial conditions on mesh_25x25x25 with all three solvers

set -e  # Exit on error

BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTCASES_DIR="$(dirname "$BENCHMARK_DIR")"
RESULTS_DIR="$BENCHMARK_DIR/results_visualization"

# Configuration
MESH="25x25x25"
MESH_DIR="$TESTCASES_DIR/mesh_$MESH"
DELTA_T="0.02"            # Safe timestep for visualization
WRITE_INTERVAL="0.25"     # Write every 0.25s for detailed visualization
IC_TYPES=("clear" "hotSphere" "checkerboard" "multipleHotSpots")
SOLVERS=("explicit" "implicit" "cpu")

echo "========================================"
echo "Visualization Benchmark"
echo "========================================"
echo "Mesh: $MESH"
echo "Time step: $DELTA_T"
echo "Write interval: $WRITE_INTERVAL s"
echo "Cases: ${IC_TYPES[@]}"
echo "Solvers: ${SOLVERS[@]}"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# Run all combinations
for IC in "${IC_TYPES[@]}"; do
    echo "=== Case: $IC ==="

    for SOLVER in "${SOLVERS[@]}"; do
        echo "  Running $SOLVER solver..."

        cd "$MESH_DIR"

        # Clean previous results
        ./Allclean > /dev/null 2>&1

        # Run simulation with VTK export
        ./Allrun.sh "$SOLVER" "$IC" "$DELTA_T" vtk "$WRITE_INTERVAL" > /dev/null 2>&1

        # Create descriptive name for this run
        RUN_NAME="${MESH}_${IC}_${SOLVER}"

        # Copy VTK directory with descriptive name
        if [ -d "VTK" ]; then
            cp -r VTK "$RESULTS_DIR/VTK_$RUN_NAME"
            echo "    ✓ VTK saved as: VTK_$RUN_NAME"

            # Compress VTK directory
            cd "$RESULTS_DIR"
            tar -czf "VTK_$RUN_NAME.tar.gz" "VTK_$RUN_NAME" 2>/dev/null
            echo "    ✓ Compressed as: VTK_$RUN_NAME.tar.gz"

            # Remove uncompressed directory to save space
            rm -rf "VTK_$RUN_NAME"

            cd "$MESH_DIR"
        else
            echo "    ERROR: VTK directory not found!"
        fi

        echo ""
    done
done

# Create README for the visualization results
cat > "$RESULTS_DIR/README.md" << 'EOF'
# Visualization Benchmark Results

This directory contains VTK output for all solver and initial condition combinations.

## File Naming Convention

Files are named: `VTK_{mesh}_{case}_{solver}.tar.gz`

- **mesh**: Grid size (25x25x25)
- **case**: Initial condition type
  - `clear`: Hot-cold plate (400K top / 300K bottom)
  - `hotSphere`: Hot sphere at center (500K)
  - `checkerboard`: Checkerboard temperature pattern
  - `multipleHotSpots`: Multiple hot regions
- **solver**: Solver type
  - `explicit`: GPU explicit (Forward Euler)
  - `implicit`: GPU implicit (Backward Euler + cuSPARSE)
  - `cpu`: CPU OpenFOAM solver

## Usage

### Extract and visualize:
```bash
# Extract a specific case
tar -xzf VTK_25x25x25_hotSphere_explicit.tar.gz

# Open in ParaView
paraview VTK_25x25x25_hotSphere_explicit/*.vtk
```

### Compare solvers for same case:
```bash
# Extract all versions of hotSphere case
tar -xzf VTK_25x25x25_hotSphere_explicit.tar.gz
tar -xzf VTK_25x25x25_hotSphere_implicit.tar.gz
tar -xzf VTK_25x25x25_hotSphere_cpu.tar.gz

# Open all in ParaView to compare
paraview VTK_25x25x25_hotSphere_*/*.vtk
```

## Contents

Total: 12 VTK archives (4 cases × 3 solvers)

EOF

# List all generated files
echo "" >> "$RESULTS_DIR/README.md"
echo "## Generated Files" >> "$RESULTS_DIR/README.md"
echo "" >> "$RESULTS_DIR/README.md"
ls -lh "$RESULTS_DIR"/*.tar.gz | awk '{print "- " $9 " (" $5 ")"}' >> "$RESULTS_DIR/README.md"

echo "========================================"
echo "Visualization Benchmark Complete!"
echo "========================================"
echo "Results: $RESULTS_DIR"
echo ""
echo "Generated archives:"
ls -1 "$RESULTS_DIR"/*.tar.gz
echo ""
echo "To extract and visualize:"
echo "  cd $RESULTS_DIR"
echo "  tar -xzf VTK_25x25x25_hotSphere_explicit.tar.gz"
echo "  paraview VTK_25x25x25_hotSphere_explicit/*.vtk"
echo ""
