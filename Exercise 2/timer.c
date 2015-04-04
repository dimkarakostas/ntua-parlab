/*
 *  timer.c -- Timer routines
 *
 *  Copyright (C) 2010-2012, Computing Systems Laboratory (CSLab)
 *  Copyright (C) 2010-2012, Vasileios Karakasis
 */ 
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

void timer_clear(xtimer_t *timer)
{
    timerclear(&timer->elapsed_time);
    timerclear(&timer->timestamp);
}

void timer_start(xtimer_t *timer)
{
    if (gettimeofday(&timer->timestamp, NULL) < 0) {
        perror("gettimeofday failed");
        exit(1);
    }
}

void timer_stop(xtimer_t *timer)
{
    struct timeval t_stop;
    struct timeval t_interval;
    if (gettimeofday(&t_stop, NULL) < 0) {
        perror("gettimeofday failed");
        exit(1);
    }

    timersub(&t_stop, &timer->timestamp, &t_interval);
    timeradd(&timer->elapsed_time, &t_interval, &timer->elapsed_time);
}

double timer_elapsed_time(xtimer_t *timer)
{
    return (timer->elapsed_time.tv_sec +
            timer->elapsed_time.tv_usec / (double) USEC_PER_SEC);
}

