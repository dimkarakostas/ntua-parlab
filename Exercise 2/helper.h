/*
 *  helper.h -- Helper functions.
 *
 *  Copyright (C) 2014, Computing Systems Laboratory (CSLab)
 *  Copyright (C) 2014, Athena Elafrou
 */ 

#ifndef HELPER_H__
#define HELPER_H__

#include <stdlib.h>
#include <stdio.h>
#include "alloc.h"
#include "kernel.h"
#include "timer.h"

// Data allocation and initialization
kernel_data_t *data_create_CPU(int N);
void data_init(kernel_data_t *d);
void data_copy(kernel_data_t *dst, kernel_data_t *src);
void data_free_CPU(kernel_data_t *d);

void check_result(kernel_data_t *test, kernel_data_t *orig);
void report_results(xtimer_t *timer, size_t size);

#endif  /* HELPER_H__ */
