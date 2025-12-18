#!/bin/bash
# Allrun.sh - Run simulation with VTK export

# Parse command line arguments
SOLVER_TYPE="${1:-explicit}"  # Default to explicit if not specified
IC_TYPE="${2:-clear}"          # Default to clear (uniform) if not specified
DELTA_T="${3:-0.020}"          # Default to 0.020 if not specified
VTK_EXPORT="${4:-vtk}"         # Default to vtk (can be "vtk" or "novtk")
WRITE_INTERVAL="${5:-1}"       # Default to 1 (write every 1 second)

echo "=========================================="
echo "Running OpenFOAM Heat Solver"
echo "=========================================="
echo "Mesh: $(basename $(pwd))"
echo "Solver: $SOLVER_TYPE (GPU explicit/implicit/cpu)"
echo "Initial condition: $IC_TYPE"
echo "Time step (deltaT): $DELTA_T"
echo "VTK export: $VTK_EXPORT"
echo "Write interval: $WRITE_INTERVAL s"
echo ""

# Source OpenFOAM if needed
if [ -z "$WM_PROJECT" ]; then
    echo "Sourcing OpenFOAM environment..."
    source $HOME/OpenFOAM/OpenFOAM-13/etc/bashrc 2>/dev/null || \
    source $HOME/2025-PP-Final-Project-Team-15/OpenFOAM-13/etc/bashrc 2>/dev/null
fi

# Step 1: Clean everything
echo "Step 1: Cleaning previous results..."
./Allclean
echo ""

# Step 2: Copy origin to 0 (if origin exists)
echo "Step 2: Setting up initial condition..."
if [ -d "origin" ]; then
    echo "  Copying origin → 0"
    cp -r origin 0
else
    echo "  Warning: origin directory not found, creating default 0/ directory"
    mkdir -p 0
    # Copy from constant/polyMesh if it exists, otherwise will be created by blockMesh
fi
echo ""

# Step 3: Generate mesh
echo "Step 3: Generating mesh with blockMesh..."
blockMesh > log.blockMesh 2>&1
if [ $? -ne 0 ]; then
    echo "ERROR: blockMesh failed! Check log.blockMesh"
    exit 1
fi
echo "  ✓ Mesh generated successfully"
echo ""

# Step 4: Apply initial condition
echo "Step 4: Applying initial condition ($IC_TYPE)..."

if [ -f "system/setFieldsDict_$IC_TYPE" ]; then
    cp "system/setFieldsDict_$IC_TYPE" system/setFieldsDict
    setFields > log.setFields 2>&1
    if [ $? -ne 0 ]; then
        echo "ERROR: setFields failed! Check log.setFields"
        exit 1
    fi
    echo "  ✓ Initial condition applied ($IC_TYPE)"
else
    echo "  Warning: system/setFieldsDict_$IC_TYPE not found"
    echo "  Using default initial condition from origin/T"
fi
echo ""

# Step 5: Update deltaT and writeInterval in controlDict
echo "Step 5: Updating controlDict parameters..."
sed -i "s/^deltaT.*$/deltaT          $DELTA_T;  \/\/ Updated by Allrun.sh/" system/controlDict
sed -i "s/^writeInterval.*$/writeInterval   $WRITE_INTERVAL;  \/\/ Updated by Allrun.sh/" system/controlDict
echo "  ✓ Set deltaT = $DELTA_T"
echo "  ✓ Set writeInterval = $WRITE_INTERVAL"
echo ""

# Step 6: Run solver
echo "Step 6: Running solver..."
case "$SOLVER_TYPE" in
    explicit)
        echo "  Using GPU explicit solver (Forward Euler)"
        # Make sure explicit library is selected
        # sed -i 's/^.*CUDA_LIB_DIR = .*/CUDA_LIB_DIR = libheatCUDA/' ../../Make/options
        # sed -i 's/^#.*CUDA_LIB_DIR = libheatCUDA$/CUDA_LIB_DIR = libheatCUDA/' ../../Make/options
        # (cd ../.. && wclean > /dev/null 2>&1 && wmake > log.wmake 2>&1)
        heatFoamCUDA_explicit > output.log 2>&1
        SOLVER_NAME="GPU Explicit"
        ;;
    implicit)
        echo "  Using GPU implicit solver (Backward Euler + cuSPARSE)"
        # Make sure implicit library is selected
        # sed -i 's/^.*CUDA_LIB_DIR = .*/CUDA_LIB_DIR = libheatCUDA_implicit/' ../../Make/options
        # sed -i 's/^#.*CUDA_LIB_DIR = libheatCUDA_implicit$/CUDA_LIB_DIR = libheatCUDA_implicit/' ../../Make/options
        # (cd ../.. && wclean > /dev/null 2>&1 && wmake > log.wmake 2>&1)
        heatFoamCUDA_implicit > output.log 2>&1
        SOLVER_NAME="GPU Implicit"
        ;;
    cpu)
        echo "  Using CPU solver (laplacianFoam)"
        laplacianFoam > output.log 2>&1
        SOLVER_NAME="CPU"
        ;;
    *)
        echo "ERROR: Unknown solver type '$SOLVER_TYPE'"
        echo "Usage: ./Allrun.sh [explicit|implicit|cpu] [initial_condition] [deltaT] [vtk|novtk] [writeInterval]"
        echo "Examples:"
        echo "  ./Allrun.sh explicit hotSphere 0.01 vtk 1          # With VTK, write every 1s"
        echo "  ./Allrun.sh implicit clear 0.005 novtk 0.5         # Skip VTK, write every 0.5s"
        echo "  ./Allrun.sh cpu checkerboard 0.02 vtk 2            # Write every 2s"
        exit 1
        ;;
esac

if [ $? -ne 0 ]; then
    echo "ERROR: Solver failed! Check output.log"
    tail -20 output.log
    exit 1
fi
echo "  ✓ Solver completed successfully"
echo ""

# Step 7: Convert all timesteps to VTK (optional)
if [ "$VTK_EXPORT" = "vtk" ]; then
    echo "Step 7: Converting results to VTK format..."
    foamToVTK -ascii > log.foamToVTK 2>&1
    if [ $? -ne 0 ]; then
        echo "ERROR: foamToVTK failed! Check log.foamToVTK"
        exit 1
    fi
    echo "  ✓ All timesteps converted to VTK/"

    # Move all log files to VTK directory
    echo "  Moving log files to VTK/..."
    mv -f log.* output.log VTK/ 2>/dev/null
    echo "  ✓ Logs saved to VTK/"
    echo ""
else
    echo "Step 7: Skipping VTK conversion (novtk mode)"
    echo ""
fi

# Summary
echo "=========================================="
echo "Simulation Complete!"
echo "=========================================="
echo "Solver: $SOLVER_NAME"
echo "Initial condition: $IC_TYPE"
echo "Time step (deltaT): $DELTA_T"
echo "Write interval: $WRITE_INTERVAL s"
echo "VTK export: $VTK_EXPORT"
echo ""
echo "Results:"
echo "  - Time directories: $(ls -d [0-9]* [0-9]*.[0-9]* 2>/dev/null | sort -n | tr '\n' ' ')"
if [ "$VTK_EXPORT" = "vtk" ]; then
    echo "  - VTK output: VTK/"
    echo "  - All logs: VTK/log.* and VTK/output.log"
    echo ""
    echo "To visualize in ParaView:"
    echo "  paraview VTK/*.vtk"
else
    echo "  - Logs: log.* and output.log"
    echo ""
    echo "To convert to VTK later:"
    echo "  foamToVTK -ascii"
fi
echo ""
