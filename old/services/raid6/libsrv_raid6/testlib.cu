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

#define NSTREAM 8

extern "C" void cuda_gen_syndrome(int disks, unsigned long dsize, void**dps, int stride)
{
    u8 *dd, *hd;
    int i, j;
    unsigned long tsz = dsize*disks;

    int ngrids = dsize/(stride*THREADS_PER_BLOCK*sizeof(u64));
    int nthreads = THREADS_PER_BLOCK;
    if (!ngrids) {
    	ngrids = 1;
    	nthreads = dsize/(stride*sizeof(u64));
    }

    struct timeval t1, t2, t3;
    float ta, tc;

    cudaStream_t s[NSTREAM];
    cudaEvent_t e, st;
    cudaEventCreate(&e);
    cudaEventCreate(&st);

    for (i=0; i<NSTREAM; i++)
	csc(cudaStreamCreate(&s[i]));

    /* these ops should not be counted because
     * with kgpu, allocation memory is simply bitmap search
     * and fast.
     */
    csc(cudaMalloc(&dd, tsz*NSTREAM));
    csc(cudaHostAlloc(&hd, tsz*NSTREAM, 0));

    if (!dd || !hd) {
	printf("out of memory\n");
	if (dd) cudaFree(dd);
	if (hd) cudaFreeHost(hd);
	return;
    }

    memset(hd, 0, tsz*NSTREAM);
    
    gettimeofday(&t1,NULL);

    /* frankly speaking, this could be avoided by letting raid456 module
     * use gpu buffer.
     *
     * for testing: data not copied here
     */
     for(j=0; j<NSTREAM; j++)
     for (i=0;i<disks-2;i++) {
	   memcpy(hd+i*dsize, dps[i], dsize);
     }

    //cudaEventRecord(st, 0);
    
    gettimeofday(&t2, NULL);

    for (j=0; j<NSTREAM; j++) {
	csc(ah2dcpy((dd+j*tsz), (hd+j*tsz), (tsz-2*dsize), s[j]));

	raid6_pq_str<<<dim3(ngrids,1), dim3(nthreads,1), 0, s[j]>>>(
	    (unsigned int)disks, dsize, (u8*)(dd+j*tsz), stride);

	csc(ad2hcpy((hd+(j+1)*tsz-2*dsize), (dd+(j+1)*tsz-2*dsize), 2*dsize, s[j]));
    }
	/*csc(cudaMemcpy(
	    hd+tsz-2*dsize,
	    dd+tsz-2*dsize,
	    2*dsize,
	    cudaMemcpyDeviceToHost));*/

    /* kgpu could avoid this with async execution */
    //cudaThreadSynchronize();
    cudaEventRecord(e,0);
    cudaEventSynchronize(e);
    
    gettimeofday(&t3, NULL);
    
    //cudaThreadSynchronize();
   
    ta = (t3.tv_sec-t1.tv_sec)*1000
	+ (t3.tv_usec-t1.tv_usec)/1000.0f;
    tc = (t3.tv_sec-t2.tv_sec)*1000
	+ (t3.tv_usec-t2.tv_usec)/1000.0f;
    printf("GPU PQ: str: %3i, c&c: %8.3fms, data: %8lu*%i bw: %9.3fMB/s\n",
	   stride, tc/NSTREAM, dsize, disks, dsize*(disks-2)*NSTREAM/(tc*1000));

    cudaEventDestroy(e);
    cudaEventDestroy(st);
    
    for(i=0; i<NSTREAM; i++)
	csc(cudaStreamDestroy(s[i]));
    
    for (j=0;j<NSTREAM; j++)
    for(i=disks-2;i<disks;i++)
	memcpy(dps[i], hd+i*dsize, dsize);
    csc(cudaFree(dd));
    csc(cudaFreeHost(hd));
}
