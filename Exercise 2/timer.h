/*
 *  timer.h -- Timer interface
 *
 *  Copyright (C) 2010-2012, Computing Systems Laboratory (CSLab)
 *  Copyright (C) 2010-2012, Vasileios Karakasis
 */ 
#ifndef TIMER_H__
#define TIMER_H__

#include <sys/time.h>
#include "common.h"

struct xtimer {
    struct timeval elapsed_time;
    struct timeval timestamp;
};

typedef struct xtimer   xtimer_t;

BEGIN_C_DECLS__

void timer_clear(xtimer_t *timer);
void timer_start(xtimer_t *timer);
void timer_stop(xtimer_t *timer);
double timer_elapsed_time(xtimer_t *timer);

END_C_DECLS__

#endif  /* TIMER_H__ */
