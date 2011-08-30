/*
 * This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2010-2011 University of Utah and the Flux Group.
 * All rights reserved.
 *
 */
 
#ifndef __GPUTILS_H__
#define __GPUTILS_H__

#define csc(...) _cuda_safe_call(__VA_ARGS__, __FILE__, __LINE__)
static cudaError_t _cuda_safe_call(cudaError_t e, const char *file, int line) {
    if (e!=cudaSuccess) {
	fprintf(stderr, "kgpu Error: %s %d %s\n",
		file, line, cudaGetErrorString(e));
	cudaThreadExit();
	abort();
    }
    return e;
}


static void *alloc_dev_mem(unsigned long size) {
    void *h;
    csc( cudaMalloc(&h, size) );
    return h;
}

static void free_dev_mem(void *p) {
    csc( cudaFree(p) );
}

#define ah2dcpy(dst, src, sz, stream) \
    cudaMemcpyAsync((void*)(dst), (void*)(src), (sz), cudaMemcpyHostToDevice, (stream))

#define ad2hcpy(dst, src, sz, stream) \
    cudaMemcpyAsync((void*)(dst), (void*)(src), (sz), cudaMemcpyDeviceToHost, (stream))

#endif
