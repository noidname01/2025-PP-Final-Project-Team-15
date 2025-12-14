ifeq ($(COMP),)
COMP=gnu
endif

COMMONDIR=../../common

ifeq ($(COMP),pgi)
CXX=g++
CC=gcc
NVCC=nvcc
GPU_ARCH=-arch=sm_70
GPU_MEMORY=-DPINNED_MEMORY
NVCCFLAGS=-g -O3 $(GPU_ARCH) -I$(COMMONDIR) $(GPU_MEMORY)
CCFLAGS=-g -O3 -I$(COMMONDIR) $(GPU_MEMORY) -DNDEBUG
LDFLAGS=
LIBS=-lcudart
endif

ifeq ($(COMP),gnu)
CXX=g++
CC=gcc
NVCC=nvcc
GPU_ARCH=-arch=sm_70
GPU_MEMORY=
GPU_MEMORY=-DPINNED_MEMORY
# GPU_MEMORY=-DUNIFIED_MEMORY
NVCCFLAGS=-g -O3 $(GPU_ARCH) -I$(COMMONDIR) $(GPU_MEMORY)
CCFLAGS=-g -O3 -march=native -I$(COMMONDIR) $(GPU_MEMORY) -DNDEBUG
LDFLAGS=
LIBS=-lcudart
endif

ifeq ($(COMP),intel)
CXX=g++
CC=gcc
NVCC=nvcc
GPU_ARCH=-arch=sm_70
GPU_MEMORY=-DPINNED_MEMORY
NVCCFLAGS=-g -O3 $(GPU_ARCH) -I$(COMMONDIR) $(GPU_MEMORY)
CCFLAGS=-g -O3 -I$(COMMONDIR) $(GPU_MEMORY) -DNDEBUG
LDFLAGS=
LIBS=-lcudart
endif

EXE=heat_cuda
OBJS=main.o heat.o core.o core_cuda.o setup.o io.o utilities.o
OBJS_PNG=$(COMMONDIR)/pngwriter.o


all: $(EXE)


utilities.o: utilities.cpp heat.hpp
heat.o: heat.cpp heat.hpp matrix.hpp
core.o: core.cpp heat.hpp
core_cuda.o: core_cuda.cu heat.hpp
setup.o: setup.cpp heat.hpp
io.o: io.cpp heat.hpp matrix.hpp
main.o: main.cpp heat.hpp

$(OBJS_PNG): C_COMPILER := $(CC)
$(OBJS): C_COMPILER := $(CXX)

$(EXE): $(OBJS) $(OBJS_PNG)
	$(CXX) $(CCFLAGS) $(OBJS) $(OBJS_PNG) -o $@ $(LDFLAGS) $(LIBS)

%.o: %.cpp
	$(CXX) $(CCFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
		
.PHONY: clean
clean:
	-/bin/rm -f $(EXE) a.out *.o *.png *~
