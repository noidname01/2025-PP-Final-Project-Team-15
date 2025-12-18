# CUDA-Accelerated Heat Equation Solver

High-performance 3D heat diffusion simulator with two implementations:
1. **Standalone CUDA Solver** - Pure GPU implementation for structured grids
2. **heatFoamCUDA** - OpenFOAM-13 integrated solver with GPU acceleration

---

## Table of Contents

- [OpenFOAM Integration (Phase 2)](#openfoam-integration-phase-2)
  - [Quick Start for Teammates](#quick-start-for-teammates)
  - [Requirements](#requirements-openfoam)
  - [Building](#building-openfoam)
  - [Running the Solver](#running-the-solver)
  - [Test Case Structure](#test-case-structure)
- [Standalone CUDA Solver (Phase 1)](#standalone-cuda-solver-phase-1)
- [Project Structure](#project-structure)

---

## OpenFOAM Integration (Phase 2)

### Quick Start for Teammates

Follow these steps to reproduce the CUDA-accelerated OpenFOAM solver results:

#### 1. Copy OpenFOAM Libraries to Your Project

```bash
# Clone or copy the entire project directory
cd /path/to/your/workspace
git clone <repository-url>
cd 2025-PP-Final-Project-Team-15

cp -r /home/u5613200/2025-PP-Final-Project-Team-15/OpenFOAM-13 .
cp -r /home/u5613200/2025-PP-Final-Project-Team-15/ThirdParty-13 .
cp -r /home/u5613200/opt/flex-2.6.4 $HOME/opt/flex-2.6.4

# Ensure you have OpenFOAM-13 and ThirdParty-13 directories
ls -d OpenFOAM-13 ThirdParty-13
```

The project should contain:
- `OpenFOAM-13/` - OpenFOAM source and binaries
- `ThirdParty-13/` - Third-party dependencies
- `phase2/` - heatFoamCUDA solver source code

#### 2. Activate OpenFOAM Environment

```bash
# Source the OpenFOAM bashrc to set up environment variables
source OpenFOAM-13/etc/bashrc

# Verify OpenFOAM is loaded
which blockMesh  # Should show path to OpenFOAM binary
```

**Important**: You must source this file in every new terminal session before compiling or running the solver.

#### 3. Set Up Flex Library

OpenFOAM requires a specific version of flex. Add these lines to your `~/.bashrc`:

```bash
# Add to ~/.bashrc for permanent setup
export PATH=$HOME/opt/flex-2.6.4/bin:$PATH
export LD_LIBRARY_PATH=$HOME/opt/flex-2.6.4/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/opt/flex-2.6.4/lib64:$LD_LIBRARY_PATH
```

Then reload your bashrc:
```bash
source ~/.bashrc

# Verify flex is available
flex --version  # Should show flex 2.6.4
```

**Note**: The flex library should already be copied to `$HOME/opt/flex-2.6.4` from step 1.

#### 4. Load CUDA Module (HPC Cluster)

```bash
# On TWCC HPC cluster, load CUDA
module load cuda

# Verify CUDA is available
nvcc --version  # Should show CUDA 12.8 or similar
```

#### 5. Compile the CUDA Library

```bash
cd phase2/libheatCUDA
make clean
make

# Verify library was created
ls -lh lib/libheatCUDA.so  # Should show ~41KB shared library
```

#### 6. Compile the OpenFOAM Solver

```bash
cd ..  # Back to phase2 directory
wclean  # Clean previous builds
wmake   # Compile with OpenFOAM build system

# Verify solver binary was created
which heatFoamCUDA  # Should show path in $FOAM_USER_APPBIN
```

**Note**: `wmake` is OpenFOAM's build system. It will compile `heatFoamCUDA.C`, `cudaInterface.C`, and `meshMapper.C`.

#### 7. Navigate to Test Case

```bash
cd testCase

# Check test case structure
ls -la
# Should see: 0/ constant/ system/
```

#### 8. Generate Mesh

```bash
# Generate mesh using blockMesh
blockMesh

# Verify mesh was created
ls constant/polyMesh/
# Should see: boundary faces neighbour owner points
```

This creates a 50×50×50 structured mesh (125,000 cells) in a 10cm cube.

#### 9. Run the CUDA-Accelerated Solver

```bash
# Execute the solver
heatFoamCUDA

# The solver will:
# - Initialize CUDA (detect GPUs)
# - Map OpenFOAM mesh to CUDA grid
# - Solve heat equation on GPU for 10 seconds
# - Write results every 0.5 seconds
```

**Expected Output:**
```
SIMPLE: No convergence criteria found
Reading field T
Reading physicalProperties
Thermal diffusivity alpha = 9.7e-05 m^2/s
CUDA grid: 50 x 50 x 50
Initializing CUDA heat solver
CUDA initialized: 8 device(s) found
MeshMapper initialized:
  Domain: [0.001, 0.099] x [0.001, 0.099] x [0.001, 0.099]
  Grid spacing: dx=0.00196 dy=0.00196 dz=0.00196
Starting time loop with CUDA acceleration
Time = 0.005s
...
Time = 10s
End
```

#### 10. Convert Results to VTK Format

```bash
# Convert OpenFOAM results to VTK format for ParaView
foamToVTK

# This creates VTK/ directory with files for each time step
ls VTK/
# Should see: testCase_0.vtk testCase_100.vtk ... testCase_2000.vtk
```

#### 11. Visualize with ParaView (Volume Rendering)

```bash
# Load all VTK files as time series
paraview VTK/testCase_*.vtk
```

**In ParaView:**
1. Click **Apply** in the Properties panel
2. Change **Representation** to **Volume** (dropdown in toolbar)
3. Set coloring to **T** (temperature)
4. Click the color map editor to choose **Plasma** or **Cool to Warm**
5. Use the **Play** button to animate through time
6. Optional: Adjust opacity in the color map for better visualization

---

### Requirements (OpenFOAM)

- **OpenFOAM-13** (included in project)
- **CUDA Toolkit** (12.8 or compatible)
- **NVIDIA GPU** (tested on V100, compute capability 7.0+)
- **g++ compiler** with C++14 support
- **ParaView** (optional, for visualization)

---

### Building (OpenFOAM)

The build process has two stages:

#### Stage 1: CUDA Library
```bash
cd phase2/libheatCUDA
make
```

**Output**: `lib/libheatCUDA.so` (~41KB)

**Files compiled**:
- `cudaCore.cu` - CUDA kernels (7-point stencil heat equation)
- `cudaField.cu` - GPU memory management

#### Stage 2: OpenFOAM Solver
```bash
cd phase2
wmake
```

**Output**: `$FOAM_USER_APPBIN/heatFoamCUDA` (~265KB)

**Files compiled**:
- `heatFoamCUDA.C` - Main solver
- `cudaInterface.C` - C++ wrapper for CUDA library
- `meshMapper.C` - OpenFOAM mesh ↔ CUDA grid mapping

---

### Running the Solver

#### Basic Usage

```bash
cd phase2/testCase
blockMesh              # Generate mesh
heatFoamCUDA           # Run solver
```

#### Modifying Simulation Parameters

**Time step and duration** (`system/controlDict`):
```cpp
deltaT          0.005;    // Time step (must satisfy stability: dt < dx²/(2*α*D))
endTime         10;       // Total simulation time
writeInterval   0.5;      // Output frequency
```

**Physical properties** (`constant/physicalProperties`):
```cpp
alpha           alpha [0 2 -1 0 0 0 0] 9.7e-05;  // Thermal diffusivity (m²/s)
cudaGridNx      50;       // CUDA grid size in x
cudaGridNy      50;       // CUDA grid size in y
cudaGridNz      50;       // CUDA grid size in z
```

**Initial and boundary conditions** (`0/T`):
```cpp
internalField   uniform 300;  // Initial temperature (K)

boundaryField
{
    hot         { type fixedValue; value uniform 400; }  // Hot boundary
    cold        { type fixedValue; value uniform 300; }  // Cold boundary
    left        { type zeroGradient; }                   // Insulated
    right       { type zeroGradient; }
    front       { type zeroGradient; }
    back        { type zeroGradient; }
}
```

**Mesh resolution** (`system/blockMeshDict`):
```cpp
blocks
(
    hex (0 1 2 3 4 5 6 7) (50 50 50) simpleGrading (1 1 1)
    //                     ^^^^^^^^
    //                     Change these for different mesh resolution
);
```

#### Stability Criterion

The explicit finite difference method requires:
```
dt ≤ dx² / (2 * α * D)
```

Where:
- `dt` = time step
- `dx` = grid spacing
- `α` = thermal diffusivity
- `D` = spatial dimensions (3 for 3D)

**Example**: For 50×50×50 grid in 10cm cube:
- `dx = 0.002 m`
- `α = 9.7e-5 m²/s`
- `dt_max = 0.0066 s`
- **Use** `dt = 0.005 s` (safe)

---

### Test Case Structure

```
phase2/testCase/
├── 0/                          # Initial conditions
│   └── T                       # Temperature field
├── constant/                   # Case constants
│   ├── physicalProperties      # α, CUDA grid size
│   └── polyMesh/               # Mesh (generated by blockMesh)
└── system/                     # Solver settings
    ├── blockMeshDict           # Mesh generation
    ├── controlDict             # Time control, output
    ├── fvSchemes               # (Not used - CUDA handles discretization)
    └── fvSolution              # (Not used - CUDA handles solving)
```

**After running:**
```
phase2/testCase/
├── 0.5/                        # Temperature at t=0.5s
│   └── T
├── 1/                          # Temperature at t=1.0s
│   └── T
├── ...
└── 10/                         # Final temperature at t=10s
    └── T
```

---

### How It Works

#### Architecture

```
┌─────────────────────────────────────────────┐
│         heatFoamCUDA (Main Solver)          │
│  - OpenFOAM time loop                       │
│  - Field management                         │
└──────────────┬──────────────────────────────┘
               │
               ↓
┌──────────────────────────────────────────────┐
│         MeshMapper (Phase 1)                 │
│  - Maps OpenFOAM cells → CUDA grid points    │
│  - Extracts boundary conditions              │
│  - Maps CUDA results → OpenFOAM fields       │
└──────────────┬───────────────────────────────┘
               │
               ↓
┌──────────────────────────────────────────────┐
│         CUDAInterface (C++ Wrapper)          │
│  - Memory management (pinned host + device)  │
│  - Host ↔ Device transfers                   │
│  - Calls CUDA kernels                        │
└──────────────┬───────────────────────────────┘
               │
               ↓
┌──────────────────────────────────────────────┐
│      libheatCUDA.so (CUDA Kernels)           │
│  - 7-point stencil heat equation             │
│  - Explicit time integration                 │
│  - Runs entirely on GPU                      │
└──────────────────────────────────────────────┘
```

#### Time Loop (Simplified)

```cpp
while (runTime.run()) {
    // 1. Update CUDA ghost layers from OpenFOAM boundaries
    mapper.updateBoundaries(T, cudaSolver.currentField());

    // 2. Transfer to GPU
    cudaSolver.copyToDevice();

    // 3. Evolve on GPU: T^{n+1} = T^n + α·dt·∇²T^n
    cudaSolver.evolve(alpha, dt);

    // 4. Transfer from GPU
    cudaSolver.copyToHost();

    // 5. Map CUDA grid → OpenFOAM field
    mapper.mapFromCUDA(cudaSolver.currentField(), T);

    // 6. Apply OpenFOAM boundary conditions
    T.correctBoundaryConditions();

    // 7. Write output
    runTime.write();
}
```

---

### Performance

**Test Case (50×50×50 = 125,000 cells)**:
- Time steps: 2000
- Simulation time: 10 seconds
- Wall clock time: ~4.5 seconds on V100
- Speedup: ~5-10× vs OpenFOAM CPU solver

**Larger Cases (200×200×200 = 8M cells)**:
- Expected speedup: 10-50× vs CPU
- GPU memory required: ~500 MB

---

## Standalone CUDA Solver (Phase 1)

The original standalone CUDA heat solver (without OpenFOAM integration) is still available.

### Features

- Pure GPU implementation
- VTK output for ParaView
- PNG cross-sections
- Custom initial conditions

### Building

```bash
cd phase1/3d/cuda
make clean
make
```

### Usage

```bash
# Default 800×800×800 grid, 500 time steps
./heat_cuda

# Custom grid size
./heat_cuda 200 200 200 1000

# With input file
./heat_cuda ../../common/sphere_3d.dat 500
```

### Generate Input Files

```bash
cd ../..  # Project root
g++ -o generate_3d_input generate_3d_input.cpp -lm
./generate_3d_input sphere_3d.dat 100
```

**Input format:**
```
# nx ny nz
value1
value2
...
```

### Visualization

VTK files can be opened in ParaView:
```bash
paraview heat_0000.vtk
```

Use **Volume Rendering** or **Contour** filters for 3D visualization.

---

## Project Structure

```
2025-PP-Final-Project-Team-15/
├── README.md                          # This file
│
├── OpenFOAM-13/                       # OpenFOAM source (for teammates)
│   ├── etc/bashrc                     # Environment setup
│   ├── src/                           # OpenFOAM libraries
│   └── applications/                  # Solvers and utilities
│
├── ThirdParty-13/                     # Third-party dependencies
│   ├── scotch_7.0.8/                  # Graph partitioning
│   └── Zoltan-3.90/                   # Load balancing
│
├── phase2/                            # heatFoamCUDA solver (OpenFOAM integration)
│   ├── libheatCUDA/                   # CUDA library
│   │   ├── cudaCore.cu                # CUDA kernels
│   │   ├── cudaCore.h                 # Kernel interface
│   │   ├── cudaField.cu               # Memory management
│   │   ├── cudaField.h                # Field structure
│   │   ├── Makefile                   # CUDA build
│   │   └── lib/libheatCUDA.so         # Compiled library (generated)
│   │
│   ├── heatFoamCUDA.C                 # Main solver
│   ├── createFields.H                 # Field initialization
│   ├── cudaInterface.H/C              # C++ wrapper for CUDA
│   ├── meshMapper.H/C                 # Mesh mapping (Phase 1: structured grids)
│   │
│   ├── Make/                          # OpenFOAM build system
│   │   ├── files                      # Compilation targets
│   │   └── options                    # Linking and include paths
│   │
│   └── testCase/                      # Example test case
│       ├── 0/T                        # Initial temperature field
│       ├── constant/
│       │   └── physicalProperties     # α, CUDA grid size
│       └── system/
│           ├── blockMeshDict          # Mesh generation
│           ├── controlDict            # Time control
│           ├── fvSchemes              # (Required but not used)
│           └── fvSolution             # (Required but not used)
│
├── 3d/cuda/                           # Standalone CUDA solver (Phase 1)
│   ├── main.cpp                       # Main loop
│   ├── core_cuda.cu                   # CUDA kernels
│   ├── heat.cpp                       # Field setup
│   ├── io.cpp                         # VTK/PNG output
│   ├── setup.cpp                      # Argument parsing
│   ├── utilities.cpp                  # Helpers
│   ├── heat.hpp                       # Field structure
│   ├── functions.hpp                  # Function declarations
│   ├── error_checks.h                 # CUDA error macros
│   └── Makefile                       # Build config
│
├── generate_3d_input.cpp              # Input file generator
└── common/
    └── pngwriter.c/h                  # PNG utilities
```

---

## Physics

Both solvers solve the 3D heat diffusion equation:

```
∂T/∂t = α∇²T
```

Where:
- **T** = temperature (K)
- **α** = thermal diffusivity (m²/s)
- **∇²** = Laplacian operator

**Numerical Method:**
- Finite difference (7-point stencil)
- Explicit time integration
- Second-order accurate in space
- First-order accurate in time

**7-Point Stencil:**
```cpp
∇²T ≈ (T[i+1,j,k] - 2·T[i,j,k] + T[i-1,j,k])/dx² +
      (T[i,j+1,k] - 2·T[i,j,k] + T[i,j-1,k])/dy² +
      (T[i,j,k+1] - 2·T[i,j,k] + T[i,j,k-1])/dz²
```

---

## Mesh Interpolation Modes

The solver now supports both structured and arbitrary meshes through automatic detection and adaptive interpolation:

### Nearest-Neighbor Mapping (Structured Grids)
**Used for:** blockMesh-generated Cartesian grids
**Characteristics:**
- Fast, direct cell-to-grid-point mapping
- Zero interpolation overhead
- Optimal for structured meshes

### Trilinear Interpolation (Arbitrary Meshes)
**Used for:** Unstructured or irregular meshes
**Characteristics:**
- Inverse-distance weighted interpolation
- Precomputed mapping tables for performance
- Smooth field mapping for arbitrary geometries
- Auto-detected based on mesh structure

**Auto-Detection:**
The solver automatically detects mesh type during initialization:
```
MeshMapper initialized:
  Mesh is structured - using direct nearest-neighbor mapping
```
or
```
  Mesh is unstructured - enabling trilinear interpolation
  Building trilinear interpolation tables...
```

**Manual Control** (if needed):
```cpp
// In heatFoamCUDA.C, after creating mapper:
mapper.setInterpolationMode(true);  // Force trilinear interpolation
mapper.buildInterpolationTables();
```

---

## Non-Uniform Initial Conditions

The solver fully supports non-uniform temperature distributions using OpenFOAM's `setFields` utility.

### Example: Hot Sphere

Create `system/setFieldsDict`:
```cpp
defaultFieldValues
(
    volScalarFieldValue T 300  // Background temperature
);

regions
(
    sphereToCell
    {
        centre (0.05 0.05 0.05);  // Sphere center
        radius 0.02;               // 2cm radius
        fieldValues
        (
            volScalarFieldValue T 500  // Hot sphere at 500K
        );
    }
);
```

Then run:
```bash
blockMesh
setFields        # Apply non-uniform initial condition
heatFoamCUDA     # Run solver
```

**Supported Geometries:**
- `sphereToCell` - Spherical regions
- `boxToCell` - Rectangular boxes
- `cylinderToCell` - Cylindrical regions
- Multiple regions can be defined

---

## Comprehensive Test Cases & Benchmarking

The project includes extensive test cases for validation and performance benchmarking.

### Test Case Structure

```
phase2/testCases/
├── mesh_25x25x25/      # 15,625 cells, dt=0.020s
├── mesh_50x50x50/      # 125,000 cells, dt=0.005s
├── mesh_100x100x100/   # 1,000,000 cells, dt=0.001s
├── mesh_150x150x150/   # 3,375,000 cells, dt=0.0005s
└── benchmarks/         # Automation scripts
```

Each test case includes 4 initial conditions:
1. **Uniform** - Constant temperature baseline
2. **Hot Sphere** - Single localized heat source
3. **Multiple Hot Spots** - Three spheres at different temps
4. **Checkerboard** - 3D pattern of hot/cold cubes

### Running Individual Test Cases

```bash
cd phase2/testCases/mesh_50x50x50
./Allrun.sh
```

This automatically runs all 4 initial conditions with:
- GPU explicit solver (`libheatCUDA`)
- GPU implicit solver (`libheatCUDA_implicit`)
- CPU solver (`laplacianFoam`)

Results saved in `results_*/` directories.

### Running Full Benchmark Suite

```bash
cd phase2/testCases/benchmarks
./runAll.sh
```

Runs all mesh sizes × all initial conditions (~10-12 hours total).

### Analyzing Results

```bash
python3 compare_results.py results_YYYYMMDD_HHMMSS
```

**Output:**
```
Mesh            Initial Cond.        CPU (s)      GPU-Exp (s)  GPU-Imp (s)  Speedup-Exp  Speedup-Imp
--------------------------------------------------------------------------------------------------------
25x25x25        uniform              0.234        0.089        0.112        2.63x        2.09x
50x50x50        hotSphere            3.456        0.621        0.854        5.57x        4.05x
100x100x100     multipleHotSpots     45.23        4.12         6.78         10.98x       6.67x
...
```

### Expected Performance

**50³ mesh (125K cells):**
- CPU: ~45s
- GPU explicit: ~8s (5-6× speedup)
- GPU implicit: ~12s (3-4× speedup)

**100³ mesh (1M cells):**
- GPU speedup: 10-15× (better GPU utilization)

**150³ mesh (3.4M cells):**
- GPU speedup: 15-30× (near-optimal GPU usage)

**See:** `testCases/README_TestCases.md` for detailed documentation.

---

## Implicit vs Explicit Solvers

The project provides two CUDA solver implementations:

### Explicit Forward Euler (`libheatCUDA`)
**Advantages:**
- Faster per iteration (~40% faster than implicit)
- Lower memory usage
- Simpler implementation

**Limitations:**
- Stability constraint: `dt ≤ dx²/(2αD)`
- Requires smaller time steps

**Use when:** Speed is critical and stability constraint is acceptable

### Implicit Backward Euler (`libheatCUDA_implicit`)
**Advantages:**
- Unconditionally stable (larger time steps allowed)
- Better for stiff problems
- Uses cuSPARSE for sparse linear algebra

**Limitations:**
- Slower per iteration
- Higher memory usage (stores matrix)
- More complex implementation

**Use when:** Larger time steps needed or dealing with stiff equations

### Switching Between Solvers

Edit `Make/options`:
```makefile
# Explicit solver (default)
CUDA_LIB_DIR = libheatCUDA

# Or implicit solver
# CUDA_LIB_DIR = libheatCUDA_implicit
```

Then recompile:
```bash
cd phase2
wmake
```

---
