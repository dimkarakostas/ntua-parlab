.PHONY: all clean query
.DEFAULT: all

EMU ?= 0
DEBUG ?= 1
TEST_CONV ?= 0
REGINFO ?= 1
CXX = gcc
GPU_CXX = nvcc
CUDA_PREFIX = /usr/local/cuda
MKDIR_P = mkdir -p
RM = /bin/rm
ECHO = /bin/echo

ifeq ($(EMU), 1)
    # emulation mode in clones
	nvcc_backend = /usr/bin/gcc-4.4
    # --compiler-options -fno-inline is a workaround for CUDA 2.3 to
    # work correctly with gcc-4.4
	nvcc_emu = -deviceemu --compiler-options -fno-inline
	nvcc_arch = 
else
    # Fermi GPUs in termis
	nvcc_backend = /usr/bin/gcc-4.6
	nvcc_emu = 
	nvcc_arch = -arch sm_20
endif

MAKE_CPPFLAGS = -D_XOPEN_SOURCE -D_BSD_SOURCE -DEPS=1e-4 \
	-I/usr/local/cuda/include \
    -I/usr/lib/gcc/x86_64-linux-gnu/4.4/include \
    -I/usr/lib/gcc/x86_64-linux-gnu/4.4/include-fixed
MAKE_CXXFLAGS = -Wall -fopenmp -std=c99
MAKE_GPU_CXXFLAGS = -ccbin $(nvcc_backend) $(nvcc_emu) $(nvcc_arch)
MAKE_LDFLAGS = -L/usr/local/cuda/lib64 -lcudart -lm -fopenmp

ifeq ($(DEBUG), 1)
	MAKE_CPPFLAGS += -D_DEBUG_
	MAKE_CXXFLAGS += -g -O0
	MAKE_GPU_CXXFLAGS += -g -O0
else
	MAKE_CPPFLAGS += -DNDEBUG -D_NOCHECK
	MAKE_CXXFLAGS += -O3
	MAKE_GPU_CXXFLAGS += -O3
endif

ifeq ($(TEST_CONV), 1)
	MAKE_GPU_CXXFLAGS += -D_CONV
endif

ifeq ($(REGINFO), 1)
	MAKE_GPU_CXXFLAGS += --ptxas-options=-v
endif

ifdef GPU_BLOCK_X
	MAKE_CPPFLAGS += -DGPU_BLOCK_DIM_X=$(GPU_BLOCK_X)
	MAKE_GPU_CPPFLAGS += -DGPU_BLOCK_DIM_X=$(GPU_BLOCK_X)
endif

ifdef GPU_BLOCK_Y
	MAKE_CPPFLAGS += -DGPU_BLOCK_DIM_Y=$(GPU_BLOCK_Y)
	MAKE_GPU_CPPFLAGS += -DGPU_BLOCK_DIM_Y=$(GPU_BLOCK_Y)
endif

ifdef GPU_TIME_STEP
	MAKE_CPPFLAGS += -DTIME_STEP=$(GPU_TIME_STEP)
	MAKE_GPU_CPPFLAGS += -DTIME_STEP=$(GPU_TIME_STEP)
endif

ifdef GPU_TILE_X
	MAKE_CPPFLAGS += -DGPU_TILE_DIM_X=$(GPU_TILE_X)
	MAKE_GPU_CPPFLAGS += -DGPU_TILE_DIM_X=$(GPU_TILE_X)
endif

ifdef GPU_TILE_Y
	MAKE_CPPFLAGS += -DGPU_TILE_DIM_Y=$(GPU_TILE_Y)
	MAKE_GPU_CPPFLAGS += -DGPU_TILE_DIM_Y=$(GPU_TILE_Y)
endif

CPU_CPPFLAGS = $(MAKE_CPPFLAGS) $(CPPFLAGS)
CPU_CXXFLAGS = $(MAKE_CXXFLAGS) $(CXXFLAGS)
CPU_LDFLAGS  = $(LDFLAGS) $(MAKE_LDFLAGS)
GPU_CPPFLAGS = $(MAKE_GPU_CPPFLAGS) $(CPPFLAGS)
GPU_CXXFLAGS = $(MAKE_GPU_CXXFLAGS) $(CXXFLAGS)

CPU_COMPILE = $(CXX) -c $(CPU_CXXFLAGS) $(CPU_CPPFLAGS)
GPU_COMPILE = $(GPU_CXX) -c $(GPU_CXXFLAGS) $(GPU_CPPFLAGS)

PROGRAMS = jacobi gpuinfo
jacobi_SOURCES = \
	alloc.c \
	cpu_kernels.c \
	error.c \
	helper.c \
	main.c \
	timer.c

jacobi_GPU_SOURCES = \
	gpu_kernel_naive.cu \
	gpu_kernel_shmem.cu \
	gpu_kernel_time_tiled_shmem.cu \
	gpu_util.cu

gpuinfo_SOURCES = gpu_info.c

jacobi_OBJECTS = $(jacobi_SOURCES:%.c=%.o)
jacobi_GPU_OBJECTS = $(jacobi_GPU_SOURCES:%.cu=%.o)
gpuinfo_OBJECTS = $(gpuinfo_SOURCES:%.c=%.o)

all: $(PROGRAMS)

DEPDIR = .deps
df = $(DEPDIR)/$(*F)
%.o: %.c
	$(MKDIR_P) $(DEPDIR)
	$(CPU_COMPILE) -MD -o $@ $<
	@cp $*.d $(df).P; \
    sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
	    -e '/^$$/ d' -e 's/$$/ :/' < $*.d >> $(df).P; \
	rm -f $*.d

%.o: %.cu
	$(MKDIR_P) $(DEPDIR)
	$(CPU_COMPILE) -xc -MM -MF $*.d $<
	@cp $*.d $(df).P; \
    sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
	    -e '/^$$/ d' -e 's/$$/ :/' < $*.d >> $(df).P; \
	rm -f $*.d
	$(GPU_COMPILE) -o $@ $<

-include $(jacobi_SOURCES:%.c=$(DEPDIR)/%.P)
-include $(jacobi_GPU_SOURCES:%.cu=$(DEPDIR)/%.P)
-include $(gpuinfo_SOURCES:%.c=$(DEPDIR)/%.P)

jacobi: $(jacobi_OBJECTS) $(jacobi_GPU_OBJECTS)
	$(CXX) -o $@ $(jacobi_OBJECTS) $(jacobi_GPU_OBJECTS) $(CPU_LDFLAGS)

gpuinfo: $(gpuinfo_OBJECTS)
	$(CXX) -o $@ $(gpuinfo_OBJECTS) $(CPU_LDFLAGS)

clean:
	$(RM) -rf $(DEPDIR)
	$(RM) -f $(PROGRAMS) $(jacobi_OBJECTS) $(jacobi_GPU_OBJECTS) $(gpuinfo_OBJECTS)

query: gpuinfo
	@$(ECHO) '\n' | ./gpuinfo

help:
	@$(ECHO) -e 'Mini help for using this Makefile:'; \
	$(ECHO) -e '\tQuery host GPU: make query'; \
	$(ECHO) -e '\tBuild for emulation mode: make EMU=1 [default: 0]'; \
	$(ECHO) -e '\tBuild release version: make DEBUG=0 [default: 1]'; \
	$(ECHO) -e '\tTo activate the convergence test: make TEST_CONV=1 [default: 0]'; \
	$(ECHO) -e '\tSet tile dimensions for GPU: make GPU_TILE_X=<dim> GPU_TILE_Y=<dim> [default: see kernel.h]'; \
	$(ECHO) -e '\tSet thread block dimensions for GPU: make GPU_BLOCK_X=<dim> GPU_BLOCK_Y=<dim> [default: see kernel.h]'; \
	$(ECHO) -e '\tSet the time step for the time-tiling kernel: make GPU_TIME_STEP=<t>'; \
	$(ECHO) -e '\tDisable printing of GPU register info: make REGINFO=0 [default: 1]'; \
	$(ECHO) -e 'You can also customly set CXX, CPPFLAGS, CXXFLAGS and LDFLAGS as usual.'; \
	$(ECHO) -e 'NOTE: Be careful to recompile your code from scratch when changing tile dimensions.'; \
	$(ECHO) -e 'NOTE: When building release version, results checking is disabled.';
