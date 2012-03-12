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
	u32 size;
	int id;
	char service_name[KGPU_SERVICE_NAME_SIZE];
	
	u64 in;
	u64 out;
	u64 data;
	u64 insize;
	u64 outsize;
	u64 datasize;

	int *depon;
	int *depby;
	u32 ndepon;
	u32 ndepby;

	u8 deplevel;

	u8 splittable;
	u8 mergeable;

        s8 device;
} ku_request;

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


/* ioctl */
#include <linux/ioctl.h>

#define KGPU_IOC_MAGIC 'g'

#define KGPU_IOC_SET_GPU_BUFS \
    _IOW(KGPU_IOC_MAGIC, 1, struct kgpu_gpu_mem_info[KGPU_BUF_NR])
#define KGPU_IOC_GET_GPU_BUFS \
    _IOR(KGPU_IOC_MAGIC, 2, struct kgpu_gpu_mem_info[KGPU_BUF_NR])
#define KGPU_IOC_SET_STOP     _IO(KGPU_IOC_MAGIC, 3)
#define KGPU_IOC_GET_REQS     _IOR(KGPU_IOC_MAGIC, 4, 

#define KGPU_IOC_MAXNR 4

#include "kgpu_log.h"

#endif /* __KERNEL__ || __KGPU__ */


#endif
