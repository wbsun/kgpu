/* This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2010-2011 University of Utah and the Flux Group.
 * All rights reserved.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define PAGE_SIZE 4096
#define NDISKS 6

char *data[NDISKS*PAGE_SIZE];

static void makedata(void)
{
    int i;

    for (i=0; i<NDISKS*PAGE_SIZE; i++)
	data[i] = rand();
}

