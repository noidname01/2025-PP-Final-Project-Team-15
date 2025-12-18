# Makefile for libheatCUDA_implicit - cuSPARSE-based implicit solver for OpenFOAM

NVCC = nvcc
CXX = g++

# GPU architecture (V100 = compute capability 7.0)
GPU_ARCH = -arch=sm_70

# Compilation flags
# Use C++11 ABI to match OpenFOAM-13
NVCCFLAGS = -O3 $(GPU_ARCH) -DPINNED_MEMORY --compiler-options '-fPIC -D_GLIBCXX_USE_CXX11_ABI=0'
CXXFLAGS = -O3 -fPIC -DPINNED_MEMORY -D_GLIBCXX_USE_CXX11_ABI=0

# Output library
LIB_DIR = lib
LIB = $(LIB_DIR)/libheatCUDA.so

# Source files (note: using cudaCore_sparse.cu instead of cudaCore.cu)
CUDA_SRCS = cudaCore_sparse.cu cudaField.cu
CUDA_OBJS = $(CUDA_SRCS:.cu=.o)

# Libraries: cuSPARSE and cuBLAS for implicit solver
CUDA_LIBS = -lcudart -lcusparse -lcublas

# Build targets
all: $(LIB)

$(LIB): $(CUDA_OBJS)
	@mkdir -p $(LIB_DIR)
	$(NVCC) -shared $(NVCCFLAGS) $(CUDA_OBJS) -o $@ $(CUDA_LIBS)
	@echo "Library built: $@"
	@echo "Using cuSPARSE implicit solver with Jacobi preconditioner"

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f *.o $(LIB)
	rm -rf $(LIB_DIR)

.PHONY: all clean
