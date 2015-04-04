// -*- c++ -*-
/*
 *  gpu_kernel_naive.cu -- Naive Jacobi GPU kernel.
 *
 *  Copyright (C) 2014, Computing Systems Laboratory (CSLab)
 *  Copyright (C) 2014, Athena Elafrou
 */ 

#include <stdio.h>
#include <cuda.h>
#include "error.h"
#include "gpu_util.h"
#include "kernel.h"
#include "timer.h"

#define GPU_KERNEL_NAME(name)   do_jacobi_gpu ## name

#define BLOCK_SIZE_x 192
#define BLOCK_SIZE_y 1

/*
 *  Naive GPU kernel: 
 *  Every thread updates a single matrix entry directly on global memory 
 *  with a 1-1 mapping of threads to matrix elements.
 */ 
__global__ void GPU_KERNEL_NAME(_naive)(REAL *input, REAL *output, int N)
{
	register int row = blockIdx.y * blockDim.y + threadIdx.y;
	register int column = blockIdx.x * blockDim.x + threadIdx.x;
	if (row == 0 || row == N-1 || column == 0 || column == N-1 || row >= N || column >= N)
		return;
	register int index = N*row + column;
	output[index] = (input[index - 1] + input[index + 1] + input[index - N] + input[index + N])/4.0;
}

void MAKE_KERNEL_NAME(jacobi, _gpu, _naive)(kernel_data_t *data)
{
    int N = data->N;
    REAL **A = data->A;
    REAL **A_prev = data->A_prev;
    REAL *dev_A_prev = NULL, *dev_A = NULL;
    xtimer_t compute_timer, transfer_timer;

    timer_clear(&transfer_timer);
    timer_start(&transfer_timer);
    allocate_data_on_gpu(&dev_A_prev, &dev_A, N);
    copy_data_from_cpu(dev_A, A, N);
    copy_data_from_cpu(dev_A_prev, A_prev, N);
    timer_stop(&transfer_timer);

	dim3 dimblock(BLOCK_SIZE_x, BLOCK_SIZE_y);
	dim3 dimgrid((N + BLOCK_SIZE_x - 1)/BLOCK_SIZE_x, (N + BLOCK_SIZE_y - 1)/BLOCK_SIZE_y);

    timer_clear(&compute_timer);
    timer_start(&compute_timer);

    for (int t = 0; t < T; t++) {
		GPU_KERNEL_NAME(_naive)<<<dimgrid, dimblock>>>(dev_A_prev, dev_A, N);
        REAL *tmp = dev_A_prev;
        dev_A_prev = dev_A;
        dev_A = tmp;
    }

    // Wait for last kernel to finish, so as to measure correctly the
    // computation and transfer times
    cudaThreadSynchronize();
    timer_stop(&compute_timer);
    double jacobi = timer_elapsed_time(&compute_timer);

    // Copy back results to host
    timer_start(&transfer_timer);
    copy_data_to_cpu(A_prev, dev_A_prev, N);
    timer_stop(&transfer_timer);
    printf("Transfer time:      %lf s\n", timer_elapsed_time(&transfer_timer));
    printf("Computation time:   %lf s\n", jacobi);
    // Performance is only correct when there is no convergece test 
    size_t size = N*N;
    printf("Jacobi performance: %lf Gflops/s\n", (T*size*4*1.e-9)/jacobi);
}
