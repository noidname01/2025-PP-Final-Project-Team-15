# CUDA-Accelerated Heat Equation Solver for OpenFOAM

GPU-accelerated 3D heat diffusion solver with two implementations:
- **GPU Explicit Solver** - Forward Euler method on CUDA
- **GPU Implicit Solver** - Backward Euler with cuSPARSE
- **Integrated with OpenFOAM-13** for mesh handling and I/O

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Environment Setup](#environment-setup)
- [Compilation](#compilation)
- [Usage](#usage)
- [Test Cases & Benchmarking](#test-cases--benchmarking)
- [Project Structure](#project-structure)
- [Physics & Numerical Methods](#physics--numerical-methods)

---

## Features

### GPU Explicit Solver (Forward Euler)
- ✓ Fast explicit time integration
- ✓ 7-point stencil finite difference
- ✓ CUDA kernels for heat diffusion
- ✓ Conditionally stable (dt < dt_max)

### GPU Implicit Solver (Backward Euler)
- ✓ Unconditionally stable (any dt)
- ✓ cuSPARSE sparse linear solver
- ✓ Preconditioned Conjugate Gradient
- ✓ Jacobi preconditioner

### OpenFOAM Integration
- ✓ Mesh mapping (structured ↔ unstructured)
- ✓ Boundary condition support (fixedValue, zeroGradient)
- ✓ Mixed boundary conditions on all 6 faces
- ✓ Non-uniform initial conditions via setFields
- ✓ VTK output for ParaView visualization

---

## Requirements

- **OpenFOAM-13** (for mesh and I/O)
- **CUDA Toolkit 12.x** (nvcc, cuSPARSE, cuBLAS)
- **NVIDIA GPU** (compute capability 7.0+, tested on V100)
- **GCC 9.x+** with C++14 support
- **Flex 2.6.4** (for OpenFOAM)
- **ParaView** (optional, for visualization)

---

## Environment Setup

### 1. Set up Flex Library

Add to `~/.bashrc`:
```bash
export PATH=$HOME/opt/flex-2.6.4/bin:$PATH
export LD_LIBRARY_PATH=$HOME/opt/flex-2.6.4/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/opt/flex-2.6.4/lib64:$LD_LIBRARY_PATH
```

Then reload:
```bash
source ~/.bashrc
flex --version  # Should show flex 2.6.4
```

### 2. Load OpenFOAM Environment

**For each new terminal session:**
```bash
source $HOME/OpenFOAM/OpenFOAM-13/etc/bashrc

# Verify
which blockMesh  # Should show OpenFOAM binary path
```

### 3. Load CUDA Module (HPC Cluster)

```bash
module load cuda
nvcc --version  # Should show CUDA 12.x
```

---

## Compilation

### Build CUDA Libraries

```bash
# Build explicit solver library
cd phase2/libheatCUDA_explicit
make clean && make

# Build implicit solver library
cd ../libheatCUDA_implicit
make clean && make
```

**Output:**
- `libheatCUDA_explicit/libheatCUDA.so`
- `libheatCUDA_implicit/libheatCUDA_implicit.so`

### Build OpenFOAM Solver

```bash
cd phase2
wclean
wmake
```

**Output:** `$FOAM_USER_APPBIN/heatFoamCUDA_explicit` or `heatFoamCUDA_implicit`

### Switch Between Solvers

The solver is selected at compile time. To change:

**For Explicit Solver:**
```bash
cd phase2
# Ensure Make/options links to libheatCUDA_explicit
wmake
# Produces: heatFoamCUDA_explicit
```

**For Implicit Solver:**
```bash
cd phase2
# Ensure Make/options links to libheatCUDA_implicit
wmake
# Produces: heatFoamCUDA_implicit
```

---

## Usage

### Quick Start

```bash
# Navigate to test case
cd phase2/testCases/mesh_25x25x25

# Run with default parameters
./Allrun.sh
```

This will:
1. Clean previous results
2. Copy initial conditions
3. Generate mesh
4. Apply initial condition
5. Update controlDict parameters
6. Run solver
7. Convert to VTK format

### Advanced Usage

```bash
./Allrun.sh [solver] [case] [deltaT] [vtk|novtk] [writeInterval]
```

**Parameters:**
- `solver`: `explicit`, `implicit`, or `cpu` (default: explicit)
- `case`: `clear`, `hotSphere`, `checkerboard`, `multipleHotSpots` (default: clear)
- `deltaT`: Time step in seconds (default: 0.020)
- `vtk|novtk`: Enable/disable VTK conversion (default: vtk)
- `writeInterval`: Output frequency in seconds (default: 1)

**Examples:**
```bash
# GPU explicit, hot sphere, dt=0.01s, with VTK, write every 0.5s
./Allrun.sh explicit hotSphere 0.01 vtk 0.5

# GPU implicit, clear case, dt=0.02s, no VTK (faster benchmarking)
./Allrun.sh implicit clear 0.02 novtk

# CPU solver for comparison
./Allrun.sh cpu checkerboard 0.01 vtk 1
```

### Initial Conditions

**clear** - Hot-cold plate configuration
- Top half: 400K (hot)
- Bottom half: 300K (cold)

**hotSphere** - Single hot sphere at center
- Background: 300K
- Sphere (r=2cm): 500K

**checkerboard** - 3D checkerboard pattern
- Alternating hot (400K) and cold (300K) cubes

**multipleHotSpots** - Three hot spheres
- Background: 300K
- Three spheres at different locations: 450K, 475K, 500K

### Visualization

```bash
# After running with vtk option
cd VTK
paraview *.vtk
```

**In ParaView:**
1. Click **Apply**
2. Select **Volume** rendering
3. Color by **T** (temperature)
4. Use **Play** to animate

---

## Test Cases & Benchmarking

### Available Test Cases

Located in `phase2/testCases/`:
- `mesh_25x25x25/` - 15,625 cells (small, fast testing)
- `mesh_50x50x50/` - 125,000 cells (medium)
- `mesh_100x100x100/` - 1,000,000 cells (large)
- `mesh_150x150x150/` - 3,375,000 cells (very large)

Each includes:
- Automated run scripts (`Allrun.sh`, `Allclean`)
- 4 initial condition configurations
- Support for all three solvers

### Benchmark Suite

Located in `phase2/testCases/benchmarks/`:

#### 1. Time Profiling (`1_time_profiling.sh`)
- Runs same case on all mesh sizes
- Tests all three solvers
- Measures execution time and cells/second
- No VTK output (pure performance)

**Output:** `results_time_profiling/timing_summary.txt`

#### 2. Visualization (`2_visualization.sh`)
- Generates VTK for all cases and solvers
- Uses mesh_25x25x25 (fast)
- Write interval: 0.25s (40 timesteps)
- Creates compressed archives with descriptive names

**Output:** `results_visualization/VTK_*.tar.gz` (12 files)

#### 3. Stability Comparison (`3_stability_comparison.sh`)
- Tests explicit vs implicit with increasing dt
- Demonstrates Forward Euler stability limit
- Shows Backward Euler unconditional stability
- Generates visualization for critical timesteps

**Output:** `results_stability/stability_summary.txt`

### Running Benchmarks

```bash
cd phase2/testCases/benchmarks

# Run all benchmarks
./run_all_benchmarks.sh

# Or run individually
./1_time_profiling.sh
./2_visualization.sh
./3_stability_comparison.sh
```

---

## Project Structure

```
phase2/
├── libheatCUDA_explicit/          # GPU explicit solver library
│   ├── heatSolver.cu              # CUDA kernels (Forward Euler)
│   ├── heatSolver.h
│   ├── Makefile
│   └── libheatCUDA.so             # Compiled library
│
├── libheatCUDA_implicit/          # GPU implicit solver library
│   ├── heatSolver_implicit.cu     # CUDA kernels (Backward Euler + cuSPARSE)
│   ├── heatSolver_implicit.h
│   ├── matrix.hpp                 # Sparse matrix (COO format)
│   ├── Makefile
│   └── libheatCUDA_implicit.so    # Compiled library
│
├── heatFoamCUDA.C                 # Main OpenFOAM solver
├── meshMapper.C/H                 # OpenFOAM ↔ CUDA grid mapping
├── createFields.H                 # Field initialization
│
├── Make/
│   ├── files                      # Source files to compile
│   └── options                    # Compiler/linker flags
│
└── testCases/
    ├── mesh_25x25x25/             # Test case (small)
    │   ├── origin/T               # Initial condition template
    │   ├── system/
    │   │   ├── blockMeshDict
    │   │   ├── controlDict
    │   │   ├── setFieldsDict_*    # 4 initial conditions
    │   ├── Allrun.sh              # Automated run script
    │   └── Allclean               # Cleanup script
    │
    ├── mesh_50x50x50/             # Medium test case
    ├── mesh_100x100x100/          # Large test case
    ├── mesh_150x150x150/          # Very large test case
    │
    └── benchmarks/
        ├── 1_time_profiling.sh    # Performance benchmark
        ├── 2_visualization.sh     # Generate VTK archives
        ├── 3_stability_comparison.sh  # Stability analysis
        └── run_all_benchmarks.sh  # Run all benchmarks
```

---

## Physics & Numerical Methods

### Heat Equation

```
∂T/∂t = α∇²T
```

Where:
- **T** = temperature (K)
- **α** = thermal diffusivity (m²/s)
- **∇²** = Laplacian operator

### Spatial Discretization

**Finite Difference Method (FDM)** with 7-point stencil:

```
∇²T ≈ (T[i±1,j,k] + T[i,j±1,k] + T[i,j,k±1] - 6·T[i,j,k]) / dx²
```

- Second-order accurate in space
- Structured grid (regular Cartesian)
- Efficient for GPU parallelization

### Temporal Discretization

#### Explicit (Forward Euler)
```
T^{n+1} = T^n + α·dt·∇²T^n
```

**Stability:** Conditionally stable, requires `dt ≤ dx²/(6α)`

**Advantages:**
- Fast per iteration (~40% faster)
- Low memory usage
- Simple implementation

#### Implicit (Backward Euler)
```
(I - α·dt·∇²)T^{n+1} = T^n
```

Solved via cuSPARSE: `Ax = b`
- **A**: Sparse coefficient matrix (COO format)
- **Solver**: Preconditioned Conjugate Gradient (PCG)
- **Preconditioner**: Jacobi

**Stability:** Unconditionally stable (any dt)

**Advantages:**
- Larger timesteps allowed
- Better for stiff problems
- No dt restriction from stability

### Accuracy Comparison

**GPU (FDM) vs CPU OpenFOAM (FVM):**
- ~2% difference in results
- Due to different discretization methods:
  - **FDM**: Point-based stencil
  - **FVM**: Control volume integration
- Both are correct numerical approximations

### Mesh Mapping

**OpenFOAM (unstructured) ↔ CUDA (structured grid)**

**Process:**
1. Extract OpenFOAM temperature field
2. Map to regular CUDA grid (nx × ny × nz)
3. Update boundary conditions on ghost layers
4. Solve on GPU
5. Map results back to OpenFOAM mesh
6. Continue OpenFOAM time loop

**Boundary Conditions:**
- `fixedValue` → Ghost layer = specified value
- `zeroGradient` → Ghost layer = adjacent cell

---

## Performance

### Expected Speedup (vs CPU OpenFOAM)

| Mesh Size | Cells | GPU Explicit | GPU Implicit |
|-----------|-------|--------------|--------------|
| 25³ | 15,625 | 2-3× | 1.5-2× |
| 50³ | 125,000 | 5-8× | 3-5× |
| 100³ | 1,000,000 | 10-15× | 6-10× |
| 150³ | 3,375,000 | 15-30× | 10-20× |

### Stability Limits

**Explicit Solver:**
- For 25³ mesh (dx=4mm): dt_max ≈ 0.027s
- For 50³ mesh (dx=2mm): dt_max ≈ 0.007s
- For 100³ mesh (dx=1mm): dt_max ≈ 0.002s

**Implicit Solver:**
- No timestep restriction from stability
- Limited only by accuracy requirements
- Can use 5-10× larger timesteps

### Recommendation

- **Use explicit** for: Small timesteps, maximum speed per iteration
- **Use implicit** for: Large timesteps, long-time simulations, stiff problems

---

## Compilation Flow

### Stage 1: CUDA Libraries

```bash
cd libheatCUDA_explicit
nvcc -c heatSolver.cu -o heatSolver.o -Xcompiler -fPIC -std=c++14
nvcc -shared heatSolver.o -o libheatCUDA.so -lcudart -lcublas
```

### Stage 2: OpenFOAM Solver

```bash
cd phase2
wmake  # Uses Make/files and Make/options
```

**Compiled:**
- `heatFoamCUDA.C` → Main solver
- `meshMapper.C` → Mesh mapping

**Linked:**
- OpenFOAM libraries (-lfiniteVolume, -lfvOptions)
- CUDA library (-lheatCUDA or -lheatCUDA_implicit)
- CUDA runtime (-lcudart, -lcusparse, -lcublas)

**Output:** Executable in `$FOAM_USER_APPBIN/`

---

## License

MIT License

## Authors

2025 Parallel Programming Final Project - Team 15
