/*
 * This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2010-2011 University of Utah and the Flux Group.
 * All rights reserved.
 *
 * Log functions
 *
 */
#ifndef __KGPU_LOG_H__
#define __KGPU_LOG_H__

#define KGPU_LOG_INFO  1
#define KGPU_LOG_DEBUG 2
#define KGPU_LOG_ALERT 3
#define KGPU_LOG_ERROR 4
#define KGPU_LOG_PRINT 5

extern void kgpu_generic_log(int level, const char *module, const char *filename,
			int lineno, const char *fmt, ...);
extern int kgpu_log_level;

#define kgpu_do_log(level, module, ...) \
    kgpu_generic_log(level, module, __FILE__, __LINE__, ##__VA_ARGS__)

#endif
