/* This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2010-2011 University of Utah and the Flux Group.
 * All rights reserved.
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "../../../kgpu/kgpu.h"
#include "../../../kgpu/gputils.h"

__global__ void empty_kernel(void)
{
}

static int empty_cs(struct kgpu_service_request *sr)
{
    sr->block_x = 1;
    sr->grid_x = 1;
    sr->block_y = 1;
    sr->grid_y = 1;
    return 0;
}

static int empty_launch(struct kgpu_service_request *sr)
{
    empty_kernel<<<dim3(sr->grid_x, sr->grid_y),
	dim3(sr->block_x, sr->block_y), 0, (cudaStream_t)(sr->stream)>>>();
    return 0;
}

static int empty_prepare(struct kgpu_service_request *sr)
{
    cudaStream_t s = (cudaStream_t)(sr->stream);
    csc( ah2dcpy( sr->din, sr->hin, sr->insize, s) );
    return 0;
}

static int empty_post(struct kgpu_service_request *sr)
{
    cudaStream_t s = (cudaStream_t)(sr->stream);
    csc( ad2hcpy( sr->hout, sr->dout, sr->outsize, s) );
    return 0;
}

static struct kgpu_service empty_srv;

extern "C" int init_service(void *lh, int (*reg_srv)(struct kgpu_service*, void*))
{
    printf("[libsrv_test] Info: init test service\n");
    
    sprintf(empty_srv.name, "empty_service");
    empty_srv.sid = 1;
    empty_srv.compute_size = empty_cs;
    empty_srv.launch = empty_launch;
    empty_srv.prepare = empty_prepare;
    empty_srv.post = empty_post;

    return reg_srv(&empty_srv, lh);
}

extern "C" int finit_service(void *lh, int (*unreg_srv)(const char*))
{
    printf("[libsrv_test] Info: finit test service\n");
    return unreg_srv(empty_srv.name);
}
