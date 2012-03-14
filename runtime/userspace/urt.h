/*
 * This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2012 University of Utah and the Flux Group.
 * All rights reserved.
 */
/*
 * KGPU userspace runtime header
 */
#ifndef __URT_H__
#define __URT_H__

#include "../utils.h"
#include <cuda.h>

#define STREAM_NR 16

typedef struct {
	int id;
	ku_meminfo buf;         /* device mem for regular allocation */
	ku_meminfo mmbuf;       /* device mem for m-mapping */
	int workload;

	cudaStream_t streams[STREAM_NR];
	u8 streamuses[STREAM_NR];
} gpu_device;


void init_runtime(void);
void finit_runtime(void);

gpu_device *get_gpu_device(int);
int get_nr_gpu_device(void);
gpu_device *get_current_gpu(void);
void set_current_gpu(gpu_device*);

int alloc_gpu(kg_u_request *);
void free_gpu(kg_u_request *);
int alloc_stream(kg_u_request *);
void free_stream(kg_u_request *);
int alloc_devmem(kg_u_request *);
void free_devmem(kg_u_request *);


/* Request base, a hash table to maintain all active requests */
typedef struct {
	kg_u_request* (*get)(int id);
	kg_u_request* (*remove)(int id);
	kg_u_request* (*put)(kg_u_request* r);
} request_base_ops;

int init_request_base(void);
int finit_request_base(void);
request_base_ops* get_request_base_ops(void);

#include "service.h"

struct kg_service * lookup_service(const char *name);
int register_service(struct kg_service *s, void *libhandle);
int unregister_service(const char *name);
int load_service(const char *libpath);
int load_all_services(const char *libdir);
int unload_service(const char *name);
int unload_all_services();

#endif
