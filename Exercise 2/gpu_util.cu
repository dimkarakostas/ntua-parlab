// -*- c++ -*-
/*
 *  gpu_util.cu -- GPU utility functions
 *
 *  Copyright (C) 2010-2013, Computing Systems Laboratory (CSLab)
 *  Copyright (C) 2010-2013, Vasileios Karakasis
 *  Copyright (C) 2014,      Athena Elafrou
 */ 

#include <cuda.h>
#include <stdio.h>
#include "gpu_util.h"

/* Initialize the CUDA runtime */
void gpu_init()
{
    cudaFree(0);
}

void *gpu_alloc(size_t count)
{
    void *ret;
    if (cudaMalloc(&ret, count) != cudaSuccess) {
        ret = NULL;
    }

    return ret;
}

void gpu_free(void *gpuptr)
{
    cudaFree(gpuptr);
}

int copy_to_gpu(const void *host, void *gpu, size_t count)
{
    if (cudaMemcpy(gpu, host, count, cudaMemcpyHostToDevice) != cudaSuccess)
        return -1;
    return 0;
}

int copy_from_gpu(void *host, const void *gpu, size_t count)
{
    if (cudaMemcpy(host, gpu, count, cudaMemcpyDeviceToHost) != cudaSuccess)
        return -1;
    return 0;
}

const char *gpu_get_errmsg(cudaError_t err)
{
    return cudaGetErrorString(err);
}

const char *gpu_get_last_errmsg()
{
    return gpu_get_errmsg(cudaGetLastError());
}
    
void allocate_data_on_gpu(float **dev_A_prev, float **dev_A, size_t N)
{
    // Allocate GPU buffers
    unsigned int size = N * N * sizeof(float);
    *dev_A_prev = (float *) gpu_alloc(size);
    if (!(*dev_A_prev))
        error(0, "gpu_alloc() failed: %s", gpu_get_last_errmsg());

    *dev_A = (float *) gpu_alloc(size);
    if (!(*dev_A)) {
        cudaFree(*dev_A_prev);
        error(0, "gpu_alloc() failed: %s", gpu_get_last_errmsg());
    }
}

void copy_data_from_cpu(float *gpu, float **host, size_t N)
{
    if (copy_to_gpu(host[0], gpu, N*N*sizeof(float)) < 0)
        error(0, "copy_to_gpu() failed: %s", gpu_get_last_errmsg());
}

void copy_data_to_cpu(float **host, const float *gpu, size_t N)
{
    if (copy_from_gpu(host[0], gpu, N*N*sizeof(float)) < 0)
        error(0, "copy_from_gpu() failed: %s", gpu_get_last_errmsg());
}
