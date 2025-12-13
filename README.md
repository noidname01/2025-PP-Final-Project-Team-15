# 3D Heat Equation Solver with CUDA

A high-performance 3D heat diffusion simulator using CUDA for GPU acceleration. Solves the heat equation using finite difference methods on a single GPU.

## Features

- **GPU Accelerated**: CUDA implementation optimized for NVIDIA V100 GPUs
- **3D Visualization**: VTK output format for ParaView visualization
- **Flexible Input**: 3D initial condition files
- **Memory Management**: Pinned memory for fast host-device transfers
- **Real-time Output**: PNG cross-sections and full 3D VTK files

## Requirements

- NVIDIA GPU with CUDA support (tested on V100, compute capability 7.0)
- CUDA Toolkit with nvcc compiler
- g++ compiler
- libpng (optional, for PNG output)
- ParaView (for 3D visualization)

## Building

### 1. Compile the Heat Solver

```bash
cd 3d/cuda
make clean
make
```

This creates the executable `heat_cuda`.

### 2. Compile the Input Generator

```bash
cd ../..  # Back to project root
g++ -o generate_3d_input generate_3d_input.cpp -lm
```

## Usage

### Generate a 3D Input File

```bash
# Default 100x100x100 grid
./generate_3d_input sphere.dat

# Custom cubic grid (200x200x200)
./generate_3d_input sphere.dat 200

# Custom dimensions (200x150x100)
./generate_3d_input sphere.dat 200 150 100
```

The generator creates a hot sphere (95°C) in the center surrounded by cool material (15°C).

### Run the Simulator

**With 3D input file:**
```bash
cd 3d/cuda
./heat_cuda ../../sphere.dat 500
```

**With default parameters:**
```bash
./heat_cuda
# Uses 800x800x800 grid, 500 time steps, generated initial conditions
```

**With custom grid:**
```bash
./heat_cuda <nx> <ny> <nz> <nsteps>
./heat_cuda 200 200 200 1000
```

### Input File Format

3D input files must have this format:
```
# nx ny nz
value1
value2
value3
...
```

Values are listed in row-major order: i varies fastest, then j, then k (total of nx×ny×nz values).

## Output Files

### PNG Files
- `heat_0000.png` - Initial middle z-slice
- `heat_XXXX.png` - Intermediate slices (every 15000 iterations)
- `heat_NNNN.png` - Final slice

### VTK Files (3D)
- `heat_0000.vtk` - Initial 3D field
- `heat_XXXX.vtk` - Intermediate 3D fields
- `heat_NNNN.vtk` - Final 3D field

## Visualization with ParaView

### Installation
Download ParaView from: https://www.paraview.org/download/

### Opening VTK Files
1. Launch ParaView
2. **File → Open** → Select `heat_XXXX.vtk`
3. Click **Apply** in Properties panel

### Visualization Techniques

#### **Volume Rendering** (Glowing 3D Effect)

Perfect for seeing the overall temperature distribution:

1. Open VTK file and click **Apply**
2. Change **Representation** (dropdown in toolbar) to **Volume**
3. Set coloring to **temperature** (dropdown next to Representation)
4. Click color map editor icon (colorful bar)
5. Choose color scheme:
   - **Plasma** or **Inferno** for glowing effects
   - **Cool to Warm** for scientific visualization
6. Switch to **Opacity** tab
7. Adjust opacity transfer function:
   - Make low temperatures transparent
   - Make high temperatures opaque
8. Optional: Enable **Shade** in Properties for depth perception

#### **Contour/Isosurface** (Constant Temperature Surfaces)

Shows beautiful spherical shells at different temperatures:

1. Open VTK file and click **Apply**
2. **Filters → Common → Contour**
3. Set **Contour By**: temperature
4. Add isovalues:
   - Click **+** button to add values
   - Example: 20, 40, 60, 80
   - Or use **Value Range** with 5-10 contours
5. Click **Apply**
6. Color by **temperature** (toolbar dropdown)
7. Choose color map: **Rainbow** or **Cool to Warm**
8. Optional: Adjust opacity to ~0.7 to see multiple layers

#### **Slice** (2D Cross-sections)

Examine specific planes through the volume:

1. Open VTK file and click **Apply**
2. **Filters → Common → Slice**
3. Configure slice:
   - **Slice Type**: Plane
   - **Origin**: Center of domain (e.g., 4, 4, 4 for default grid)
   - **Normal**: Choose axis
     - (1,0,0) for YZ plane
     - (0,1,0) for XZ plane
     - (0,0,1) for XY plane
4. Click **Apply**
5. Color by **temperature**
6. Check **Show Plane** to see slice position
7. Drag the plane interactively to explore

#### **Clip** (Cut-away View)

See the interior while keeping part of the exterior:

1. Open VTK file and click **Apply**
2. **Filters → Common → Clip**
3. Set **Clip Type**: Plane
4. Position plane through center
5. Click **Apply**
6. Color by **temperature**
7. Optional: Apply **Contour** to clipped result for combined effect

### Tips for Better Visualization

**Color Maps:**
- **Cool to Warm**: Blue → white → red (scientific standard)
- **Rainbow**: Blue → green → yellow → red (classic)
- **Plasma/Inferno**: Perceptually uniform, great for presentations
- **Jet**: Traditional but not perceptually uniform

**Saving Images:**
1. **File → Save Screenshot**
2. Choose resolution (1920×1080 or higher)
3. Format: PNG recommended
4. Optional: Enable **Transparent Background**

**Animation:**
1. Load all VTK files using wildcard: `heat_*.vtk`
2. ParaView recognizes them as time series
3. Use **Play** button to animate
4. **File → Save Animation** to export video

## Performance

**Default Configuration (800³ grid):**
- Grid points: 512 million
- Memory usage: ~4 GB
- Time per iteration: ~0.1s on V100
- Total runtime (500 steps): ~50 seconds

**Smaller Test (100³ grid):**
- Grid points: 1 million
- Memory usage: ~8 MB
- Time per iteration: ~0.001s
- Total runtime (500 steps): ~0.5 seconds

## Physics

The code solves the 3D heat diffusion equation:

```
∂T/∂t = α∇²T
```

Where:
- T = temperature
- α = thermal diffusivity (0.5)
- ∇² = Laplacian operator

**Numerical Method:**
- Finite difference (7-point stencil)
- Explicit time integration
- Second-order accurate in space

**Boundary Conditions:**
- Fixed temperatures on all 6 faces (Dirichlet)
- Default boundaries: 20°C on x-min/y-max/z-min faces, 35°C on x-max/y-min/z-max faces

## Troubleshooting

### CUDA Out of Memory
```
Error: cudaMalloc failed
```
**Solution**: Reduce grid size

### No VTK Files Generated
**Solution**: Check write permissions and disk space

### ParaView Shows Black/Empty Volume
**Solution**:
- Verify temperature data range (not all zeros)
- Adjust opacity transfer function
- Set color map to "temperature"

### Compilation Errors
```
nvcc: command not found
```
**Solution**: Install CUDA toolkit and add to PATH

## Example Workflow

```bash
# 1. Generate input file
./generate_3d_input test_sphere.dat 100

# 2. Run simulation
cd 3d/cuda
./heat_cuda ../../test_sphere.dat 1000

# 3. Visualize in ParaView
paraview heat_0000.vtk
# Use Volume rendering or Contour as described above

# 4. Animate all timesteps
paraview heat_*.vtk
# Click Play button to see evolution
```

## Project Structure

```
├── generate_3d_input.cpp    # Input file generator
├── 3d/cuda/
│   ├── main.cpp             # Main simulation loop
│   ├── core_cuda.cu         # CUDA kernels
│   ├── heat.cpp             # Field initialization
│   ├── io.cpp               # VTK and PNG output
│   ├── setup.cpp            # Argument parsing
│   ├── utilities.cpp        # Helper functions
│   ├── heat.hpp             # Field structure
│   ├── matrix.hpp           # 3D matrix template
│   ├── functions.hpp        # Function declarations
│   ├── error_checks.h       # CUDA error macros
│   └── Makefile             # Build configuration
└── common/
    └── pngwriter.c/h        # PNG utilities
```
