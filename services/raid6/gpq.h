/* This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2010-2011 University of Utah and the Flux Group.
 * All rights reserved.
 */
#ifndef __GPQ_H__
#define __GPQ_H__

struct raid6_pq_data {
    unsigned long dsize;
    unsigned int nr_d;
};

/*
 * Not that efficient, but can save some time because
 * we can allocate disk pointers statically.
 */
#define MAX_DISKS 50

long test_gpq(int disks, size_t dsize);
long test_cpq(int disks, size_t dsize);

#endif
