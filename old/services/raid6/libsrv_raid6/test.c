/*
 * This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2010-2011 University of Utah and the Flux Group.
 * All rights reserved.
 */

#include <stdio.h>
#include <stdlib.h>

void cuda_gen_syndrome(int disks, unsigned long dsize, void **dps, int stride);

#define NDISKS 16
#define MAX_DSZ (1024*64)
#define MIN_DSZ (1024*4)
#define DSZ (1024*128)

int main()
{
    int i;
    size_t sz;
    void *dps[NDISKS];
    char *data = (char*)malloc(NDISKS*MAX_DSZ);

    for (i=0; i<NDISKS; i++)
	dps[i] = data+MAX_DSZ*i;

    printf("pre-init for CUDA ... \n");
    cuda_gen_syndrome(NDISKS, MAX_DSZ, dps, 2);

    printf("do testing...\n");
    for (sz = MIN_DSZ; sz <= MAX_DSZ; sz += MIN_DSZ)
    	//for (i=1; i<32; i++) {    	
	    cuda_gen_syndrome(NDISKS, sz, dps, 1);
	//}
    
    printf("done!\n");
    
    free(data);
    
    return 0;
}
