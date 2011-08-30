/*
 * This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2010-2011 University of Utah and the Flux Group.
 * All rights reserved.
 *
 */

#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include "helper.h"
#include "gputils.h"

extern "C" void init_gpu();
extern "C" void finit_gpu();

extern "C" void *alloc_pinned_mem(unsigned long size);
extern "C" void free_pinned_mem(void *p);

extern "C" int alloc_gpu_mem(struct kgpu_service_request *sreq);
extern "C" void free_gpu_mem(struct kgpu_service_request *sreq);
extern "C" int alloc_stream(struct kgpu_service_request *sreq);
extern "C" void free_stream(struct kgpu_service_request *sreq);

extern "C" int execution_finished(struct kgpu_service_request *sreq);
extern "C" int post_finished(struct kgpu_service_request *sreq);

extern "C" unsigned long get_stream(int sid);

#define MAX_STREAM_NR 8
static cudaStream_t streams[MAX_STREAM_NR];
static int streamuses[MAX_STREAM_NR];

static const dim3 default_block_size(32,1);
static const dim3 default_grid_size(512,1);

struct kgpu_gpu_mem_info devbuf;

void init_gpu()
{
    int i;

    // csc(cudaHostGetDevicePointer((void**)(&devbuf.uva),
				 // (void*)(hostbuf.uva), 0));
    // for (i=0; i< KGPU_BUF_NR; i++) {
    devbuf.uva = alloc_dev_mem(KGPU_BUF_SIZE);
    // }

    for (i=0; i<MAX_STREAM_NR; i++) {
        csc( cudaStreamCreate(&streams[i]) );
	streamuses[i] = 0;
    }
}

void finit_gpu()
{
    int i;

    // for (i=0; i<KGPU_BUF_NR; i++) {
    free_dev_mem(devbuf.uva);
    // }
    for (i=0; i<MAX_STREAM_NR; i++) {
	csc( cudaStreamDestroy(streams[i]));
    }
}

unsigned long get_stream(int stid)
{
    if (stid < 0 || stid >= MAX_STREAM_NR)
	return 0;
    else
	return (unsigned long)streams[stid];
}

void *alloc_pinned_mem(unsigned long size) {
    void *h;
    csc( cudaHostAlloc(&h, size, 0));//cudaHostAllocWriteCombined) );
    return h;
}

void free_pinned_mem(void* p) {
    csc( cudaFreeHost(p) );
}

static int __check_stream_done(cudaStream_t s)
{
    cudaError_t e = cudaStreamQuery(s);
    if (e == cudaSuccess) {
	return 1;
    } else if (e != cudaErrorNotReady)
	csc(e);

    return 0;
}

int execution_finished(struct kgpu_service_request *sreq)
{
    cudaStream_t s = (cudaStream_t)get_stream(sreq->stream_id);
    return __check_stream_done(s);
}

int post_finished(struct kgpu_service_request *sreq)
{
    cudaStream_t s = (cudaStream_t)get_stream(sreq->stream_id);
    return __check_stream_done(s);
}

/*
 * Allocation policy is simple here: copy what the kernel part does
 * for the GPU memory. This works because:
 *   - GPU memory and host memory are identical in size
 *   - Whenever a host memory region is allocated, the same-sized
 *     GPU memory must be used for its GPU computation.
 *   - The data field in ku_request also uses pinned memory but we
 *     won't allocate GPU memory for it cause it is just for
 *     service provider. This is fine since the data tend to be
 *     very tiny.
 */
int alloc_gpu_mem(struct kgpu_service_request *sreq)
{
    sreq->din = (void*)ADDR_REBASE(devbuf.uva, hostbuf.uva, sreq->hin);
    sreq->dout = (void*)ADDR_REBASE(devbuf.uva, hostbuf.uva, sreq->hout);
    sreq->ddata = (void*)ADDR_REBASE(devbuf.uva, hostbuf.uva, sreq->hdata);
    return 0;
}

void free_gpu_mem(struct kgpu_service_request *sreq)
{
    sreq->din = NULL;
    sreq->dout = NULL;
    sreq->ddata = NULL;
}

int alloc_stream(struct kgpu_service_request *sreq)
{
    int i;

    for (i=0; i<MAX_STREAM_NR; i++) {
	if (!streamuses[i]) {
	    streamuses[i] = 1;
	    sreq->stream_id = i;
	    sreq->stream = (unsigned long)(streams[i]);
	    return 0;
	}
    }
    return 1;
}

void free_stream(struct kgpu_service_request *sreq)
{
    if (sreq->stream_id >= 0 && sreq->stream_id < MAX_STREAM_NR) {
	streamuses[sreq->stream_id] = 0;
    }
}


int default_compute_size(struct kgpu_service_request *sreq)
{
    sreq->block_x = default_block_size.x;
    sreq->block_y = default_block_size.y;
    sreq->grid_x = default_grid_size.x;
    sreq->grid_y = default_grid_size.y;
    return 0;
}

int default_prepare(struct kgpu_service_request *sreq)
{
    cudaStream_t s = (cudaStream_t)get_stream(sreq->stream_id);
    csc( ah2dcpy( sreq->din, sreq->hin, sreq->insize, s) );
    return 0;
}

int default_post(struct kgpu_service_request *sreq)
{
    cudaStream_t s = (cudaStream_t)get_stream(sreq->stream_id);
    csc( ad2hcpy( sreq->hout, sreq->dout, sreq->outsize, s) );
    return 0;
}
