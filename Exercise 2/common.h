/*
 *  common.h -- Basic definitions and declarations
 *
 *  Copyright (C) 2010-2014, Computing Systems Laboratory (CSLab)
 *  Copyright (C) 2010-2014, Vasileios Karakasis
 */ 
#ifndef COMMON_H__
#define COMMON_H__

#include <stddef.h>
#define USEC_PER_SEC    1000000L

#undef  BEGIN_C_DECLS__
#undef  END_C_DECLS__

#if defined(__cplusplus) || defined(__CUDACC__)
#   define BEGIN_C_DECLS__  extern "C" {
#   define END_C_DECLS__    }
#else
#   define BEGIN_C_DECLS__
#   define END_C_DECLS__
#endif  /* __cplusplus || __CUDACC__ */

#define MIN(a, b)   ((a) < (b) ? (a) : (b))
#define ABS(a)      ((a) < 0 ? -(a) : (a))

#endif  /* COMMON_H__ */
