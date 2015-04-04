/*
 *  error.h -- Error-handling routines
 *
 *  Copyright (C) 2010-2012, Computing Systems Laboratory (CSLab)
 *  Copyright (C) 2010-2012, Vasileios Karakasis
 */ 

#ifndef ERROR_H__
#define ERROR_H__

#include "common.h"

BEGIN_C_DECLS__

extern char *program_name;
void set_program_name(char *argv0);

void warning(int errnum, const char *fmt, ...);
void error(int errnum, const char *fmt, ...);
void fatal(int errnum, const char *fmt, ...);

END_C_DECLS__

#endif  /* ERROR_H__ */
