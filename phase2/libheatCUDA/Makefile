# Makefile for libheatCUDA - CUDA heat solver library for OpenFOAM integration

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

# Source files
CUDA_SRCS = cudaCore.cu cudaField.cu
CUDA_OBJS = $(CUDA_SRCS:.cu=.o)

# Build targets
all: $(LIB)

$(LIB): $(CUDA_OBJS)
	@mkdir -p $(LIB_DIR)
	$(NVCC) -shared $(NVCCFLAGS) $(CUDA_OBJS) -o $@ -lcudart
	@echo "Library built: $@"

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f *.o $(LIB)
	rm -rf $(LIB_DIR)

.PHONY: all clean
