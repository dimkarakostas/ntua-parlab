// -*- c++ -*-
/*
 *  gpu_kernel_shmem.cu -- Simple and improved Jacobi GPU kernels that
 *                         use shared memory.
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

#define BLOCK_SIZE_x 128
#define BLOCK_SIZE_y 2

#define iBLOCK_SIZE_x 128
#define iBLOCK_SIZE_y 4

#define TILE_SIZE_x 4
#define TILE_SIZE_y 2

__global__ void GPU_KERNEL_NAME(_shmem)(REAL *in, REAL *out, int N)
{
	int local_row = threadIdx.y + 1;
	int local_col = threadIdx.x + 1;
	__shared__ REAL local_matrix[BLOCK_SIZE_y + 2][BLOCK_SIZE_x + 2];
	int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int index = N*row + col;
	if (row >= N || col >= N)
		return;
	local_matrix[local_row][local_col] = in[index];
	if (row == N-1 || col == N-1)
		return;
	if (local_row == 1)
		local_matrix[local_row - 1][local_col] = in[N * (row - 1) + col];
	if (local_col == 1)
		local_matrix[local_row][local_col - 1] = in[N * row + col - 1];
	if (local_row == BLOCK_SIZE_y)
		local_matrix[local_row + 1][local_col] = in[N * (row + 1) + col];
	if (local_col == BLOCK_SIZE_x)
		local_matrix[local_row][local_col + 1] = in[N * row + col + 1];
	__syncthreads();
	out[index] = (local_matrix[local_row][local_col-1] + local_matrix[local_row-1][local_col] + local_matrix[local_row][local_col+1] + local_matrix[local_row+1][local_col])/4.0;
}

/*
 *  Improved GPU kernel that uses shared memory.
 */
__global__ void GPU_KERNEL_NAME(_shmem_improved)(REAL *in, REAL *out, int N)
{
    int index, local_row, local_col;
	__shared__ REAL local_matrix[TILE_SIZE_y * iBLOCK_SIZE_y + 2][TILE_SIZE_x * iBLOCK_SIZE_x + 2];

	int tile_threads_x = blockDim.x * TILE_SIZE_x;
	int tile_threads_y = blockDim.y * TILE_SIZE_y;
	int tile_zero_row = blockIdx.y * tile_threads_y;
	int next_tile_first_row = tile_zero_row + 1 + tile_threads_y; 
	int tile_zero_col = blockIdx.x * tile_threads_x;
	int next_tile_first_col = tile_zero_col + 1 + tile_threads_x;

	local_row = threadIdx.y + 1;
	for (int row = tile_zero_row + local_row; ((row < N) && (row < tile_zero_row + 1 + tile_threads_y)); row += blockDim.y) {
		local_col = threadIdx.x + 1;
		for (int col = tile_zero_col + local_col; ((col < N) && (col < tile_zero_col + 1 + tile_threads_x)); col += blockDim.x) {
			index = N*row + col;
			local_matrix[local_row][local_col] = in[index];
			if (row == N - 1 || col == N - 1) {
				local_col += blockDim.x;
				continue;
			}
			if (row == tile_zero_row + 1) {
				local_matrix[local_row - 1][local_col] = in[N * tile_zero_row + col];
			}
			if (col == tile_zero_col + 1) {
				local_matrix[local_row][local_col - 1] = in[N * row + tile_zero_col];
			}
			if (row == next_tile_first_row - 1) {
				local_matrix[local_row + 1][local_col] = in[N * next_tile_first_row + col];
			}
			if (col == next_tile_first_col - 1) {
				local_matrix[local_row][local_col + 1] = in[N * row + next_tile_first_col];
			}
			local_col += blockDim.x;
		}
		local_row += blockDim.y;
	}
	__syncthreads();

	local_row = threadIdx.y + 1;
	for (int row = tile_zero_row + local_row; ((row < N - 1) && (row < tile_zero_row + 1 + tile_threads_y)); row += blockDim.y) {
		local_col = threadIdx.x + 1;
		for (int col = tile_zero_col + local_col; ((col < N - 1) && (col < tile_zero_col + 1 + tile_threads_x)); col += blockDim.x) {
			index = N*row + col;
			out[index] = (local_matrix[local_row][local_col-1] + local_matrix[local_row-1][local_col] + local_matrix[local_row][local_col+1] + local_matrix[local_row+1][local_col])/4.0;
			local_col += blockDim.x;
		}
		local_row += blockDim.y;
	}
}

void MAKE_KERNEL_NAME(jacobi, _gpu, _shmem)(kernel_data_t *data)
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
		GPU_KERNEL_NAME(_shmem)<<<dimgrid, dimblock>>>(dev_A_prev, dev_A, N);
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

void MAKE_KERNEL_NAME(jacobi, _gpu, _shmem_improved)(kernel_data_t *data)
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

	dim3 dimblock(iBLOCK_SIZE_x, iBLOCK_SIZE_y);
	dim3 dimgrid((N - 2 + TILE_SIZE_x*iBLOCK_SIZE_x - 1)/(TILE_SIZE_x * iBLOCK_SIZE_x), (N - 2 + TILE_SIZE_y*iBLOCK_SIZE_y - 1)/(TILE_SIZE_y * iBLOCK_SIZE_y));

    timer_clear(&compute_timer);
    timer_start(&compute_timer);

    for (int t = 0; t < T; t++) {
		GPU_KERNEL_NAME(_shmem_improved)<<<dimgrid, dimblock>>>(dev_A_prev, dev_A, N);
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
