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


__global__ void inc_kernel(int *din, int *dout)
{
    int id = threadIdx.x +  blockIdx.x*blockDim.x;

    dout[id] = din[id]+1;    
}

int test_compute_size(struct kgpu_service_request *sr)
{
    sr->block_x = 32;
    sr->grid_x = sr->insize/256;
    sr->block_y = 1;
    sr->grid_y = 1;

    return 0;
}

int test_launch(struct kgpu_service_request *sr)
{
    printf("invoke kernel\n");
    inc_kernel<<<dim3(sr->grid_x, sr->grid_y), dim3(sr->block_x, sr->block_y), 0, (cudaStream_t)(sr->stream)>>>
	((int*)sr->din, (int*)sr->dout);
    printf("invoke done\n");
    return 0;
}

int test_prepare(struct kgpu_service_request *sr)
{
    cudaStream_t s = (cudaStream_t)(sr->stream);//get_stream(sr->stream_id);
    csc( ah2dcpy( sr->din, sr->hin, sr->insize, s) );
    return 0;
}

int test_post(struct kgpu_service_request *sr)
{
    cudaStream_t s = (cudaStream_t)(sr->stream);//get_stream(sr->stream_id);
    csc( ad2hcpy( sr->hout, sr->dout, sr->outsize, s) );
    return 0;
}

struct kgpu_service test_srv;

extern "C" int init_service(void *lh, int (*reg_srv)(struct kgpu_service*, void*))
{
    printf("[libsrv_test] Info: init test service\n");
    
    sprintf(test_srv.name, "test_service");
    test_srv.sid = 0;
    test_srv.compute_size = test_compute_size;
    test_srv.launch = test_launch;
    test_srv.prepare = test_prepare;
    test_srv.post = test_post;
    
    return reg_srv(&test_srv, lh);
}

extern "C" int finit_service(void *lh, int (*unreg_srv)(const char*))
{
    printf("[libsrv_test] Info: finit test service\n");
    return unreg_srv(test_srv.name);
}
