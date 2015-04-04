/*
 *  kernel.h -- 
 *
 *  Copyright (C) 2010-2014, Computing Systems Laboratory (CSLab)
 *  Copyright (C) 2014,      Athena Elafrou
 */ 
#ifndef KERNEL_H__
#define KERNEL_H__

#include "common.h"
#include <stddef.h>

#ifdef USE_DP_VALUES
typedef double REAL;
#else
typedef float REAL;
#endif

/* Number of Jacobi iterations */
#define T 256

/* In case you implement the convergence test */
#ifdef _CONV
#   define C 100
#   define e 0.000001
#endif

/* Time step for time-tiling implementation */
#ifndef TIME_STEP
#   define TIME_STEP 1
#endif

/* Thread block dimensions */
#ifndef GPU_BLOCK_DIM_X
#   define GPU_BLOCK_DIM_X 32
#endif
#ifndef GPU_BLOCK_DIM_Y
#   define GPU_BLOCK_DIM_Y 32
#endif

/* Tile dimensions */
#ifndef GPU_TILE_DIM_X
#   define GPU_TILE_DIM_X 32
#endif
#ifndef GPU_TILE_DIM_Y
#   define GPU_TILE_DIM_Y 32
#endif

BEGIN_C_DECLS__

typedef struct kernel_data_struct {
    REAL **A_prev;
    REAL **A;
    int N;
} kernel_data_t;

/* Kernel selection framework */
#define __MAKE_KERNEL_NAME(method, arch, name)  method ## arch ## name
#define MAKE_KERNEL_NAME(method, arch, name)    __MAKE_KERNEL_NAME(method, arch, name)
#define DECLARE_KERNEL(method, arch, name)                              \
    void MAKE_KERNEL_NAME(method, arch, name)(kernel_data_t *d)

typedef void (*kernel_fn_t)(kernel_data_t *d);
typedef struct {
    const char *descr;
    kernel_fn_t fn;
} kernel_t;

enum {
    JACOBI,
    ALGO_END
};

enum {
    CPU_SERIAL,
    CPU_PARALLEL,
    GPU_NAIVE,
    GPU_SHMEM,
    GPU_SHMEM_IMPROVED,
    GPU_TIME_TILED_SHMEM,
    KERNEL_END
};

DECLARE_KERNEL(jacobi, _cpu, _serial);
DECLARE_KERNEL(jacobi, _cpu, _omp);
DECLARE_KERNEL(jacobi, _gpu, _naive);
DECLARE_KERNEL(jacobi, _gpu, _shmem);
DECLARE_KERNEL(jacobi, _gpu, _shmem_improved);
DECLARE_KERNEL(jacobi, _gpu, _time_tiled_shmem);

static const kernel_t kernels[] = {
    {
        "Jacobi CPU serial",
        MAKE_KERNEL_NAME(jacobi, _cpu, _serial),
    },

    {
        "Jacobi CPU parallel",
        MAKE_KERNEL_NAME(jacobi, _cpu, _omp),
    },

    {
        "Jacobi GPU naive",
        MAKE_KERNEL_NAME(jacobi, _gpu, _naive),
    },

    {
        "Jacobi GPU shmem",
        MAKE_KERNEL_NAME(jacobi, _gpu, _shmem),
    },

    {
        "Jacobi GPU shmem improved",
        MAKE_KERNEL_NAME(jacobi, _gpu, _shmem_improved),
    },

    {
        "Jacobi GPU time-tiled shmem",
        MAKE_KERNEL_NAME(jacobi, _gpu, _time_tiled_shmem),
    }
};

END_C_DECLS__

#endif  /* KERNEL_H__ */
