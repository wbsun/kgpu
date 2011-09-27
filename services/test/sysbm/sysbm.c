/* This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2010-2011 University of Utah and the Flux Group.
 * All rights reserved.
 */

#include <linux/module.h>
#include <linux/init.h>
#include <linux/types.h>
#include <linux/kernel.h>
#include <linux/list.h>
#include <linux/spinlock.h>
#include <linux/gfp.h>
#include <linux/kthread.h>
#include <linux/proc_fs.h>
#include <linux/mm.h>
#include <linux/mm_types.h>
#include <linux/string.h>
#include <linux/completion.h>
#include <linux/uaccess.h>
#include <asm/page.h>
#include <linux/timex.h>

#include "../../../kgpu/kgpu.h"

/* customized log function */
#define g_log(level, ...) kgpu_do_log(level, "sysbm", ##__VA_ARGS__)
#define dbg(...) g_log(KGPU_LOG_DEBUG, ##__VA_ARGS__)
#define prt(...) g_log(KGPU_LOG_PRINT, ##__VA_ARGS__)

#define MAX_MEM_SZ (32*1024*1024)
#define MIN_MEM_SZ (0)

#define BATCH_NR 10

int mycb(struct kgpu_request *req)
{
    struct completion *c = (struct completion*)req->kdata;
    complete(c);
    return 0;
}

static int __init minit(void)
{
    struct kgpu_request *rs[BATCH_NR];
    void *bufs[BATCH_NR];
    struct completion cs[BATCH_NR];

    int i;
    struct timeval t0, t1;
    long tt;
    unsigned long sz;

    memset(rs, 0, sizeof(struct kgpu_request*)*BATCH_NR);
    memset(bufs, 0, sizeof(void*)*BATCH_NR);
    
    prt("prepare for testing\n");

    for (i=0; i<BATCH_NR; i++) {
	rs[i] = kgpu_alloc_request();
	if (!rs[i]) {
	    g_log(KGPU_LOG_ERROR, "request %d null\n", i);
	    goto cleanup;
	}
	bufs[i] = kgpu_vmalloc(MAX_MEM_SZ);
	if (!bufs[i]) {
	    g_log(KGPU_LOG_ERROR, "buf %d null\n", i);
	    goto cleanup;
	}
	rs[i]->in = bufs[i];
	rs[i]->out = bufs[i];
	rs[i]->callback = mycb;
	init_completion(cs+i);
	rs[i]->kdata = (void*)(cs+i);
	rs[i]->kdatasize = sizeof(void*);
	strcpy(rs[i]->service_name, "empty_service");
	rs[i]->insize = PAGE_SIZE;
	rs[i]->outsize = PAGE_SIZE;
	rs[i]->udata = NULL;
	rs[i]->udatasize = 0;
    }

    prt("done allocations, start first test\n");

    kgpu_call_sync(rs[0]);

    prt("done first test for CUDA init\n");

    rs[0]->id = kgpu_next_request_id();

    for (sz=MIN_MEM_SZ; sz<=MAX_MEM_SZ; sz=(sz?sz<<1:PAGE_SIZE)) {
	for (i=0; i<BATCH_NR; i++) {
	    rs[i]->insize = sz;
	    rs[i]->outsize = sz;
	}
	
	do_gettimeofday(&t0);
	for (i=0; i<BATCH_NR; i++) {
	    kgpu_call_async(rs[i]);
	}

	for (i=0; i<BATCH_NR; i++)
	    wait_for_completion(cs+i);
	do_gettimeofday(&t1);

	tt = 1000000*(t1.tv_sec-t0.tv_sec) + 
			((long)(t1.tv_usec) - (long)(t0.tv_usec));
	tt /= BATCH_NR;

	printk("ASYNC SIZE: %10lu B, TIME: %10lu MS, OPS: %8lu, BW: %8lu MB/S\n",
	       sz, tt, 1000000/tt, sz/tt);

	for (i=0; i<BATCH_NR; i++) {
	    init_completion(cs+i);
	    rs[i]->id = kgpu_next_request_id();
	}	
    }

    prt("done async, start sync\n");
    for (sz=MIN_MEM_SZ; sz<=MAX_MEM_SZ; sz=(sz?sz<<1:PAGE_SIZE)) {
	rs[0]->insize = sz;
	rs[0]->outsize = sz;

	do_gettimeofday(&t0);
	kgpu_call_sync(rs[0]);
	do_gettimeofday(&t1);

	tt = 1000000*(t1.tv_sec-t0.tv_sec) + 
			((long)(t1.tv_usec) - (long)(t0.tv_usec));

	printk("SYNC  SIZE: %10lu B, TIME: %10lu MS, OPS: %8lu, BW: %8lu MB/S\n",
	       sz, tt, 1000000/tt, sz/tt);

	rs[0]->id = kgpu_next_request_id();
    }

    prt("done sync\n");

cleanup:
    for (i=0; i<BATCH_NR; i++) {
	if (rs[i]) kgpu_free_request(rs[i]);
	if (bufs[i]) kgpu_vfree(bufs[i]);
    }

    prt("done test\n");
    
    return 0;
}

static void __exit mexit(void)
{
    g_log(KGPU_LOG_PRINT, "unload\n");
}

module_init(minit);
module_exit(mexit);

MODULE_LICENSE("GPL");
