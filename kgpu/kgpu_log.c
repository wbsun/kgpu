/*
 * This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2010-2011 University of Utah and the Flux Group.
 * All rights reserved.
 *
 * Log functions used by both kernel and user space.
 */
#include "kgpu.h"

#ifndef __KERNEL__

#include <stdio.h>
#include <stdarg.h>

#define printk printf
#define vprintk vprintf

#else

#include <linux/kernel.h>
#include <linux/module.h>

#endif /* __KERNEL__ */

int kgpu_log_level = KGPU_LOG_ALERT;

void
kgpu_generic_log(int level, const char *module, const char *filename,
	    int lineno, const char *fmt, ...)
{
    va_list args;
    
    if (level < kgpu_log_level)
	return;
    
    switch(level) {
    case KGPU_LOG_INFO:
	printk("[%s] %s::%d INFO: ", module, filename, lineno);
	break;
    case KGPU_LOG_DEBUG:
	printk("[%s] %s::%d DEBUG: ", module, filename, lineno);
	break;
    case KGPU_LOG_ALERT:
	printk("[%s] %s::%d ALERT: ", module, filename, lineno);
	break;
    case KGPU_LOG_ERROR:
	printk("[%s] %s::%d ERROR: ", module, filename, lineno);
	break;
    case KGPU_LOG_PRINT:
	printk("[%s] %s::%d: ", module, filename, lineno);
	break;
    default:
	break;
    }
    
    va_start(args, fmt);	
    vprintk(fmt, args);
    va_end(args);
}

#ifdef __KERNEL__

EXPORT_SYMBOL_GPL(kgpu_generic_log);
EXPORT_SYMBOL_GPU(kgpu_log_level);

#endif /* __KERNEL__ */
