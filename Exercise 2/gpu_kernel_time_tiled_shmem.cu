// -*- c++ -*-
/*
 *  gpu_kernel_time_tiled_shmem.cu -- Time-tiled Jacobi GPU kernel.
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

#define GPU_KERNEL_NAME(name)   do_jacobi_gpu_time_tiled ## name

#define BLOCK_SIZE_x 32
#define BLOCK_SIZE_y 16

# undef TIME_STEP
# define TIME_STEP 3

#define LOCAL_ROW_HIGH_LIMIT         BLOCK_SIZE_y - TIME_STEP
#define LOCAL_ROW_LOW_LIMIT          TIME_STEP - 1
#define LOCAL_COL_HIGH_LIMIT         BLOCK_SIZE_x - TIME_STEP
#define LOCAL_COL_LOW_LIMIT          TIME_STEP - 1
#define GLOBAL_HIGH_LIMIT            N - TIME_STEP - 1
#define GLOBAL_LOW_LIMIT             TIME_STEP

__global__ void GPU_KERNEL_NAME(_time_tiled_shmem)(REAL *in, REAL *out, int N)
{
	int row = blockIdx.y * (blockDim.y - 2 * TIME_STEP) + threadIdx.y;
	int col = blockIdx.x * (blockDim.x - 2 * TIME_STEP) + threadIdx.x;
	
	if (row >= N || col >= N)
		return;

	int local_row = threadIdx.y;
	int local_col = threadIdx.x;
	double local_var;
	__shared__ double local_matrix[BLOCK_SIZE_y][BLOCK_SIZE_x];
	int index = N*row + col;

	local_matrix[local_row][local_col] = in[index];
	__syncthreads();

	if (row == 0 || row == N-1 || col == 0 || col == N-1 || local_row == 0 || local_row == BLOCK_SIZE_y-1 || local_col == 0 || local_col == BLOCK_SIZE_x-1)
		return;
	
	for (int i = 0; i < TIME_STEP; i++) {
		local_var = (local_matrix[local_row][local_col-1] + local_matrix[local_row-1][local_col] + local_matrix[local_row][local_col+1] + local_matrix[local_row+1][local_col])/4.0;
		__syncthreads();
		local_matrix[local_row][local_col] = local_var;
		__syncthreads();
	}

	if (((local_row < LOCAL_ROW_HIGH_LIMIT) && (local_row > LOCAL_ROW_LOW_LIMIT) && (local_col < LOCAL_COL_HIGH_LIMIT) && (local_col > LOCAL_COL_LOW_LIMIT)) ||
		((local_col < LOCAL_COL_HIGH_LIMIT) && (local_col > LOCAL_COL_LOW_LIMIT) && (row < GLOBAL_LOW_LIMIT)) ||
		((local_row < LOCAL_ROW_HIGH_LIMIT) && (local_row > LOCAL_ROW_LOW_LIMIT) && (col < GLOBAL_LOW_LIMIT)) ||
		((local_col < LOCAL_COL_HIGH_LIMIT) && (local_col > LOCAL_COL_LOW_LIMIT) && (row > GLOBAL_HIGH_LIMIT)) ||
		((local_row < LOCAL_ROW_HIGH_LIMIT) && (local_row > LOCAL_ROW_LOW_LIMIT) && (col > GLOBAL_HIGH_LIMIT)) ||
		((col < GLOBAL_LOW_LIMIT)  && (row < GLOBAL_LOW_LIMIT)) ||
		((col > GLOBAL_HIGH_LIMIT) && (row < GLOBAL_LOW_LIMIT)) ||
		((col < GLOBAL_LOW_LIMIT)  && (row > GLOBAL_HIGH_LIMIT)) ||
		((col > GLOBAL_HIGH_LIMIT) && (row > GLOBAL_HIGH_LIMIT)))
			out[index] = local_matrix[local_row][local_col];
}

void MAKE_KERNEL_NAME(jacobi, _gpu, _time_tiled_shmem)(kernel_data_t *data)
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
	dim3 dimgrid(((N - TIME_STEP - 1) + (BLOCK_SIZE_x - 2*TIME_STEP - 1))/(BLOCK_SIZE_x - 2*TIME_STEP), ((N - TIME_STEP - 1) + (BLOCK_SIZE_y - 2*TIME_STEP - 1))/(BLOCK_SIZE_y - 2*TIME_STEP));

    timer_clear(&compute_timer);
    timer_start(&compute_timer);

# undef T
# define T 255
    for (int t = 0; t < T; t+=TIME_STEP) {
		GPU_KERNEL_NAME(_time_tiled_shmem)<<<dimgrid, dimblock>>>(dev_A_prev, dev_A, N);
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
