/*
 *  helper.c -- Helper functions.
 *
 *  Copyright (C) 2014, Computing Systems Laboratory (CSLab)
 *  Copyright (C) 2014, Athena Elafrou
 */ 

#include "helper.h"

static int matrix_equals(kernel_data_t *test, kernel_data_t *orig)
{
    for (int i = 0; i < orig->N; i++) {
        for (int j = 0; j < orig->N; j++) {
            if (ABS(orig->A_prev[i][j] - test->A_prev[i][j]) > EPS) {
	      		printf("element in position (%d,%d) differs\n", i, j);
                return 0;
            }
        }
    }
    return 1;
}

void report_results(xtimer_t *timer, size_t size)
{
    double elapsed_time = timer_elapsed_time(timer);
    printf("Total elapsed time: %lf s\n", elapsed_time);
    printf("Total performance:  %lf Gflops/s\n", T*size*4*1.e-9 / elapsed_time);
}

void check_result(kernel_data_t *test, kernel_data_t *orig)
{
    printf("Checking ... ");
    fflush(stdout);
    if (!matrix_equals(test, orig)) {
        printf("FAILED\n");
        fprintf(stderr, "Matrices not equal!\n");
        exit(EXIT_FAILURE);
    } else {
        printf("PASSED\n");
    }
}

kernel_data_t *data_create_CPU(int N)
{
    kernel_data_t *ret = (kernel_data_t *) malloc(sizeof(*ret));
    if (ret) {
        ret->N = N;
        ret->A_prev = (REAL **) calloc_2d(N, N, sizeof(REAL));
        if (!ret->A_prev) {
            free(ret);
            ret = NULL;
        }

        ret->A = (REAL **) calloc_2d(N, N, sizeof(REAL));
        if (!ret->A) {
            free(ret);
            ret = NULL;
        }
    }

    return ret;
}

void data_free_CPU(kernel_data_t *d)
{
    if (d) {
        if (d->A_prev)
            free_2d((void **) d->A_prev);
        if (d->A)
            free_2d((void **) d->A);
    }
    
    free(d);
}

void data_init(kernel_data_t *data)
{
    int N = data->N;

	// Initialize halo region
	for (int i = 0; i < N; i++) {
		data->A_prev[i][0] = data->A[i][0] = 0.01*(i+1)+0.001;
		data->A_prev[i][N-1] = data->A[i][N-1] = 0.01*(i+1)+0.001*N;
	}

	for (int j = 1; j < (N-1); j++) {
		data->A_prev[0][j] = data->A[0][j] = 0.01+0.001*(j+1);
		data->A_prev[N-1][j] = data->A[N-1][j] = 0.01*N+0.001*(j+1);
	}
}

void data_copy(kernel_data_t *dst, kernel_data_t *src)
{
    copy_2d((void **) dst->A_prev, (const void **) src->A_prev, src->N, src->N,
            sizeof(**src->A_prev));
    copy_2d((void **) dst->A, (const void **) src->A, src->N, src->N,
            sizeof(**src->A));
}
