/* This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2010-2011 University of Utah and the Flux Group.
 * All rights reserved.
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "../../../kgpu/helper.h"
#include "../../../kgpu/gputils.h"


__global__ void inc_kernel(int *din, int *dout)
{
    int id = threadIdx.x +  blockIdx.x*blockDim.x;

    dout[id] = din[id]+1;    
}

int test_compute_size(struct service_request *sr)
{
    sr->block_x = 32;
    sr->grid_x = sr->kureq.insize/128;
    sr->block_y = 1;
    sr->grid_y = 1;

    return 0;
}

int test_launch(struct service_request *sr)
{
    inc_kernel<<<dim3(sr->grid_x, sr->grid_y), dim3(sr->block_x, sr->block_y), 0, (cudaStream_t)(sr->stream)>>>
	((int*)sr->dinput, (int*)sr->doutput);
    return 0;
}

int test_prepare(struct service_request *sr)
{
    cudaStream_t s = (cudaStream_t)(sr->stream);//get_stream(sr->stream_id);
    csc( ah2dcpy( sr->dinput, sr->kureq.input, sr->kureq.insize, s) );
    return 0;
}

int test_post(struct service_request *sr)
{
    cudaStream_t s = (cudaStream_t)(sr->stream);//get_stream(sr->stream_id);
    csc( ad2hcpy( sr->kureq.output, sr->doutput, sr->kureq.outsize, s) );
    return 0;
}

struct service test_srv;

extern "C" int init_service(void *lh, int (*reg_srv)(struct service*, void*))
{
    printf("init test service\n");
    
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
    printf("finit test service\n");
    return unreg_srv(test_srv.name);
}
