/*
 * This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2010-2011 University of Utah and the Flux Group.
 * All rights reserved.
 *
 * Common functions used by both kernel and user space.
 */
#include "kgpu.h"

#ifndef __KERNEL__

#include <stdio.h>
#include <stdarg.h>

#define printk printf
#define vprintk vprintf

#endif /* __KERNEL__ */

int kgpu_log_level = KGPU_LOG_ALERT;

void kgpu_log(int level, const char *fmt, ...)
{
    va_list args;
    
    if (level < kgpu_log_level)
	return;
    
    switch(level) {
    case KGPU_LOG_INFO:
	printk("[kgpu] INFO: ");
	break;
    case KGPU_LOG_DEBUG:
	printk("[kgpu] DEBUG: ");
	break;
    case KGPU_LOG_ALERT:
	printk("[kgpu] ALERT: ");
	break;
    case KGPU_LOG_ERROR:
	printk("[kgpu] ERROR: ");
	break;
    case KGPU_LOG_PRINT:
	printk("[kgpu]: ");
	break;
    default:
	break;
    }
    
    va_start(args, fmt);	
    vprintk(fmt, args);
    va_end(args);
}
