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

extern "C" void gpu_init();
extern "C" void gpu_finit();

extern "C" void *gpu_alloc_pinned_mem(unsigned long size);
extern "C" void gpu_free_pinned_mem(void *p);

extern "C" void gpu_pin_mem(void *p, size_t sz);
extern "C" void gpu_unpin_mem(void *p);

extern "C" int gpu_alloc_device_mem(struct kgpu_service_request *sreq);
extern "C" void gpu_free_device_mem(struct kgpu_service_request *sreq);
extern "C" int gpu_alloc_stream(struct kgpu_service_request *sreq);
extern "C" void gpu_free_stream(struct kgpu_service_request *sreq);

extern "C" int gpu_execution_finished(struct kgpu_service_request *sreq);
extern "C" int gpu_post_finished(struct kgpu_service_request *sreq);

extern "C" unsigned long gpu_get_stream(int sid);

#define MAX_STREAM_NR 8
static cudaStream_t streams[MAX_STREAM_NR];
static int streamuses[MAX_STREAM_NR];

static const dim3 default_block_size(32,1);
static const dim3 default_grid_size(512,1);

struct kgpu_gpu_mem_info devbuf;
struct kgpu_gpu_mem_info devbuf4vma;

void gpu_init()
{
    int i;

    devbuf.uva = alloc_dev_mem(KGPU_BUF_SIZE);
    devbuf4vma.uva = alloc_dev_mem(KGPU_BUF_SIZE);

    for (i=0; i<MAX_STREAM_NR; i++) {
        csc( cudaStreamCreate(&streams[i]) );
	streamuses[i] = 0;
    }
}

void gpu_finit()
{
    int i;

    free_dev_mem(devbuf.uva);
    free_dev_mem(devbuf4vma.uva);

    for (i=0; i<MAX_STREAM_NR; i++) {
	csc( cudaStreamDestroy(streams[i]));
    }
}

unsigned long gpu_get_stream(int stid)
{
    if (stid < 0 || stid >= MAX_STREAM_NR)
	return 0;
    else
	return (unsigned long)streams[stid];
}

void *gpu_alloc_pinned_mem(unsigned long size) {
    void *h;
    csc( cudaHostAlloc(&h, size, 0));//cudaHostAllocWriteCombined) );
    return h;
}

void gpu_free_pinned_mem(void* p) {
    csc( cudaFreeHost(p) );
}

void gpu_pin_mem(void *p, size_t sz)
{
    size_t rsz = round_up(sz, PAGE_SIZE);
    csc( cudaHostRegister(p, rsz, cudaHostRegisterPortable) );
}

void gpu_unpin_mem(void *p)
{
    csc( cudaHostUnregister(p) );
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

int gpu_execution_finished(struct kgpu_service_request *sreq)
{
    cudaStream_t s = (cudaStream_t)gpu_get_stream(sreq->stream_id);
    return __check_stream_done(s);
}

int gpu_post_finished(struct kgpu_service_request *sreq)
{
    cudaStream_t s = (cudaStream_t)gpu_get_stream(sreq->stream_id);
    return __check_stream_done(s);
}

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))

static int __merge_2ranges(
	unsigned long r1, unsigned long s1, unsigned long r2, unsigned long s2,
	unsigned long *e, unsigned long *s) 
{
    // r1   r2
    if (r1 < r2) {
        if (r1+s1 >= r2) {
            *e = r1;
            *s = max(r1+s1, r2+s2) - r1;
            return 1;
        }
        
        return 0;
    } else if (r1 == r2) {
        *e = r1;
        *s = max(s1, s2);
        return 1;
    } else {
        // r2  r1
        if (r2+s2 >= r1) {
            *e = r2;
            *s = max(r1+s1, r2+s2) - r2;
            return 1;
        }
        
        return 0;
    }
}


static int __merge_ranges(unsigned long ad[], unsigned long sz[], int n)
{
    int i;
    
    for (i=0; i<n; i++) {
    	ad[i] = round_down(ad[i], PAGE_SIZE);
    	sz[i] = round_up(sz[i], PAGE_SIZE);
    }
    
    switch(n) {
    	case 0:
    	    return 0;
    	case 1:
    	    return 1;
    	case 2:
    	    if (__merge_2ranges(ad[0], sz[0], ad[1], sz[1], &ad[0], &sz[0]))
    	        return 1;
    	    else
    	        return 2;
    	case 3:
    	    if (__merge_2ranges(ad[0], sz[0], ad[1], sz[1], &ad[0], &sz[0])) {
    	        if (__merge_2ranges(ad[0], sz[0], ad[2], sz[2], &ad[0], &sz[0])) {
    	            return 1;
    	        } else {
    	            ad[1] = ad[2];
    	            sz[1] = sz[2];
    	            return 2;
    	        }
    	    } else if (__merge_2ranges(ad[0], sz[0], ad[2], sz[2], &ad[0], &sz[0])) {
    	        if (__merge_2ranges(ad[0], sz[0], ad[1], sz[1], &ad[0], &sz[0])) {
    	            return 1;
    	        } else
    	            return 2;
    	    } else if (__merge_2ranges(ad[2], sz[2], ad[1], sz[1], &ad[1], &sz[1])) {
    	        if (__merge_2ranges(ad[0], sz[0], ad[1], sz[1], &ad[0], &sz[0]))
    	            return 1;
    	        else
    	            return 2;
    	    } else
    	        return 3;
    	   
    	default:
    	    return 0;
    }
    
    // should never reach here
    //return 0;
}

/*
 * A little bit old policy, but may give you a brief pic of what
 * K-U mm does.
 *
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
int gpu_alloc_device_mem(struct kgpu_service_request *sreq)
{
    unsigned long pin_addr[3] = {0,0,0}, pin_sz[3] = {0,0,0};
    int npins = 0, i;
    
    if (ADDR_WITHIN(sreq->hin, hostbuf.uva, hostbuf.size))
	sreq->din =
	    (void*)ADDR_REBASE(devbuf.uva, hostbuf.uva, sreq->hin);
    else {
	sreq->din =
	    (void*)ADDR_REBASE(devbuf4vma.uva, hostvma.uva, sreq->hin);

	pin_addr[npins] = TO_UL(sreq->hin);
	pin_sz[npins] = sreq->insize;
	npins++;
    }

    if (ADDR_WITHIN(sreq->hout, hostbuf.uva, hostbuf.size))
	sreq->dout =
	    (void*)ADDR_REBASE(devbuf.uva, hostbuf.uva, sreq->hout);
    else {
	sreq->dout =
	    (void*)ADDR_REBASE(devbuf4vma.uva, hostvma.uva, sreq->hout);

	pin_addr[npins] = TO_UL(sreq->hout);
	pin_sz[npins] = sreq->outsize;
	npins++;
    }

    if (ADDR_WITHIN(sreq->hdata, hostbuf.uva, hostbuf.size))
	sreq->ddata =
	    (void*)ADDR_REBASE(devbuf.uva, hostbuf.uva, sreq->hdata);
    else if (ADDR_WITHIN(sreq->hdata, hostvma.uva, hostvma.size)){
	sreq->ddata =
	    (void*)ADDR_REBASE(devbuf4vma.uva, hostvma.uva, sreq->hdata);

	pin_addr[npins] = TO_UL(sreq->hdata);
	pin_sz[npins] = sreq->datasize;
	npins++;
    }
    
    npins = __merge_ranges(pin_addr, pin_sz, npins);
    for (i=0; i<npins; i++) {
    	gpu_pin_mem((void*)pin_addr[i], pin_sz[i]);
    }

    return 0;
}

void gpu_free_device_mem(struct kgpu_service_request *sreq)
{
    unsigned long pin_addr[3] = {0,0,0}, pin_sz[3] = {0,0,0};
    int npins = 0, i;
    
    sreq->din = NULL;
    sreq->dout = NULL;
    sreq->ddata = NULL;   
    
    if (ADDR_WITHIN(sreq->hin, hostvma.uva, hostvma.size)) {
    	pin_addr[npins] = TO_UL(sreq->hin);
	pin_sz[npins] = sreq->insize;
	npins++;
    }
    if (ADDR_WITHIN(sreq->hout, hostvma.uva, hostvma.size)) {
    	pin_addr[npins] = TO_UL(sreq->hout);
	pin_sz[npins] = sreq->outsize;
	npins++;
    }
    if (ADDR_WITHIN(sreq->hdata, hostvma.uva, hostvma.size)) {
    	pin_addr[npins] = TO_UL(sreq->hdata);
	pin_sz[npins] = sreq->datasize;
	npins++;
    }
    
    npins = __merge_ranges(pin_addr, pin_sz, npins);
    for (i=0; i<npins; i++) {
    	gpu_unpin_mem((void*)pin_addr[i]);
    }
}

int gpu_alloc_stream(struct kgpu_service_request *sreq)
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

void gpu_free_stream(struct kgpu_service_request *sreq)
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
    cudaStream_t s = (cudaStream_t)gpu_get_stream(sreq->stream_id);
    csc( ah2dcpy( sreq->din, sreq->hin, sreq->insize, s) );
    return 0;
}

int default_post(struct kgpu_service_request *sreq)
{
    cudaStream_t s = (cudaStream_t)gpu_get_stream(sreq->stream_id);
    csc( ad2hcpy( sreq->hout, sreq->dout, sreq->outsize, s) );
    return 0;
}
