/* This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2010-2011 University of Utah and the Flux Group.
 * All rights reserved.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "../../../kgpu/gputils.h"

#define THREADS_PER_BLOCK 64

#include "dev.cu"

extern "C" void cuda_gen_syndrome(int disks, unsigned long dsize, void**dps)
{
    u8 *dd, *hd;
    int i;
    unsigned long tsz = dsize*disks;

    int ngrids = dsize/512;
    int nthreads = 512/sizeof(u64);

    struct timeval t1, t2, t3;
    float ta, tc;

    cudaEvent_t e;
    cudaEventCreate(&e);

    /* these ops should not be counted because
     * with kgpu, allocation memory is simply bitmap search
     * and fast.
     */
    csc(cudaMalloc(&dd, tsz));
    csc(cudaHostAlloc(&hd, tsz, 0));
    
    gettimeofday(&t1,NULL);

    /* frankly speaking, this could be avoided by letting raid456 module
     * use gpu buffer
     */
    for (i=0;i<disks-2;i++) {
	memcpy(hd+i*dsize, dps[i], dsize);
    }

    gettimeofday(&t2, NULL);
    csc(cudaMemcpy(dd, hd, tsz-2*dsize, cudaMemcpyHostToDevice));

    raid6_pq<<<ngrids, nthreads>>>((unsigned int)disks, dsize, (u8*)dd);
      
    csc(cudaMemcpy(
	    hd+tsz-2*dsize,
	    dd+tsz-2*dsize,
	    /*tsz*/2*dsize,
	    cudaMemcpyDeviceToHost));

    /* kgpu could avoid this with async execution */
    cudaEventRecord(e,0);
    cudaEventSynchronize(e);
    
    gettimeofday(&t3, NULL);
   
    ta = (t3.tv_sec-t1.tv_sec)*1000
	+ (t3.tv_usec-t1.tv_usec)/1000.0f;
    tc = (t3.tv_sec-t2.tv_sec)*1000
	+ (t3.tv_usec-t2.tv_usec)/1000.0f;
    printf("GPU PQ: all: %fms, c&c: %fms, data: %lu*%i\n",
	   ta, tc, dsize, disks);

    cudaEventDestroy(e);
    
    for(i=disks-2;i<disks;i++)
	memcpy(dps[i], hd+i*dsize, dsize);
    csc(cudaFree(dd));
    csc(cudaFreeHost(hd));
}
