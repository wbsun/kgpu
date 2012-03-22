/*
 * This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2012 University of Utah and the Flux Group.
 * All rights reserved.
 *
 * Header for KGPU runtime: userspace helper and kernel module.
 *
 */
#ifndef __RUNTIME_H__
#define __RUNTIME_H__

#include "kgpu.h"


typedef struct {
	u64 uva;
	u64 size;
} ku_meminfo;

typedef struct {
	int id;
	int errcode;
} ku_response;


#if defined __KERNEL__ || defined __KGPU__


/* Default 1GB GPU memory */
#define KGPU_BUF_SIZE (1<<30)
#define KGPU_MMAP_SIZE (1<<30)



#endif /* __KERNEL__ || __KGPU__ */


#endif
