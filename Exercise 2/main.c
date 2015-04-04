// -*- c++ -*-
/*
 *  main.c -- Jacobi front-end program.
 *
 *  Copyright (C) 2014, Computing Systems Laboratory (CSLab)
 *  Copyright (C) 2014, Athena Elafrou
 */ 

#include <stdlib.h>
#include <stdio.h>
#include "alloc.h"
#include "error.h"
#include "helper.h"
#include "gpu_util.h"
#include "kernel.h"
#include "timer.h"

// Helper functions
static void print_usage()
{
    printf("Usage: [KERNEL=<kernel_no>] %s <dim>\n", program_name);
    printf("KERNEL defaults to 0\n");
    printf("Available kernels [id:descr]:\n");
    int i, j;
    for (i = 0; i < ALGO_END; ++i)
        for (j = 0; j < KERNEL_END; ++j)
            printf("\t%zd:%s\n", i*KERNEL_END+j, kernels[i*KERNEL_END+j].descr);
}

int main(int argc, char **argv)
{
    set_program_name(argv[0]);
    if (argc < 2) {
        warning(0, "too few arguments");
        print_usage();
        exit(EXIT_FAILURE);
    }

    int N = atoi(argv[1]);
    if (!N)
        error(0, "invalid argument: %s", argv[1]);

    char *kernel_env = getenv("KERNEL");
    int kernel_id = 0;
    if (kernel_env)
        kernel_id = atoi(kernel_env);

    // Initialize runtime, so as not to pay the cost at the first call
    if (kernel_id > CPU_PARALLEL) {
        printf("Initializing CUDA runtime ... ");
        fflush(stdout);
        gpu_init();
        printf("DONE\n");
    }

    kernel_data_t *data = NULL;
    data = data_create_CPU(N);
    data_init(data);

#ifndef _NOCHECK
    kernel_data_t *check_data = NULL;
    check_data = data_create_CPU(N);
    data_copy(check_data, data);
    kernels[0].fn(check_data);
#endif

    // Run and time the selected kernel
    printf("Launching kernel:   %s\n", kernels[kernel_id].descr);
    fflush(stdout);
    xtimer_t timer;
    timer_clear(&timer);
    timer_start(&timer);
    kernels[kernel_id].fn(data);
    timer_stop(&timer);

    report_results(&timer, N*N);
#ifndef _NOCHECK
    check_result(data, check_data);
#endif

    // Cleanup
    data_free_CPU(data);
#ifndef _NOCHECK
    data_free_CPU(check_data);
#endif

    return EXIT_SUCCESS;
}
