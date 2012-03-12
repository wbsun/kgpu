/*
 * This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2012 University of Utah and the Flux Group.
 * All rights reserved.
 */
#ifndef __SERVICE_H__
#define __SERVICE_H__

#include "kgpu.h"
#include <cuda.h>

struct kg_service;

struct _kg_u_request{
	u32 size;
	int id;

	/* Data info: */
	void *hin, *hout, *hdata;
	void *din, *dout, *ddata;
	u64 insize, outsize, datasize;

	int errcode;
	struct kg_service *s;

	/* Thread dimension info:  */
	int block_x, block_y;
	int grid_x, grid_y;

	int state;

	int *depon;
	int *depby;
	u32 ndepon;
	u32 ndepby;

	u8 deplevel;
	s8 device;

	u8 splittable;
	u8 mergeable;
	u8 splitted;
	u8 merged;

	int parent;
	int *children;
	u32 nchildren;

	cudaStream_t stream;
	int stream_id;	
};

typedef struct _kg_u_request kg_u_request;

struct kg_service {
	char name[KGPU_SERVICE_NAME_SIZE];
	int sid;

	/* Workflow interfaces */
	int (*compute_size)(kg_u_request *sreq);
	int (*launch)(kg_u_request *sreq);
	int (*prepare)(kg_u_request *sreq);
	int (*post)(kg_u_request *sreq);

	/* Return pointer to device function that does
	 * the service logic, but accepts a kg_u_request
	 * instead of full set of arguments.
	 * This is used to enable multi-service fusing.
 	 */
	void* (*device_function_pointer)();

	/* Request scheduling functions
	 */	
	/* Service scheduled interfaces: */
	int (*make_request)(kg_u_request *sreq);
	kg_u_request* (*take_request)(void);
	int (*done_request)(kg_u_request *sreq,
			    int (*general_done_request)(kg_u_request *sreq));

	/* General runtime scheduled interfaces: */
	int (*splittable)(kg_u_request *sreq);
	int (*mergeable)(kg_u_request *sreq);
	kg_u_request* (*do_merge)(kg_u_request *sreq);
	kg_u_request** (*do_split)(kg_u_request *sreq);
	int (*done_merged)(kg_u_request *sreq);
	int (*done_splitted)(kg_u_request *sreq);	
};

#define SERVICE_INIT "init_service"
#define SERVICE_FINIT "finit_service"
#define SERVICE_LIB_PREFIX "libsrv_"

typedef int (*fn_init_service)(
	void* libhandle,
	int (*reg_srv)(struct kg_service *, void*),
	int (*unreq_srv)(const char*));
typedef int (*fn_finit_service)(
    void* libhandle, int (*unreg_srv)(const char*));


#endif
