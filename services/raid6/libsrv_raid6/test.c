/*
 * This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2010-2011 University of Utah and the Flux Group.
 * All rights reserved.
 */

#include <stdio.h>
#include <stdlib.h>

void cuda_gen_syndrome(int disks, unsigned long dsize, void **dps);

#define NDISKS 8
#define DSZ (1024*128)

int main()
{
    int i;
    void *dps[NDISKS];
    char *data = (char*)malloc(NDISKS*DSZ);

    for (i=0; i<NDISKS; i++)
	dps[i] = data+DSZ*i;

    printf("do testing...\n");
    cuda_gen_syndrome(NDISKS, DSZ, dps);
    printf("done!\n");
    
    free(data);
    
    return 0;
}
