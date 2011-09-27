/*
 * Asynchronous RAID-6 recovery calculations ASYNC_TX API.
 * Copyright(c) 2009 Intel Corporation
 *
 * based on raid6recov.c:
 *   Copyright 2002 H. Peter Anvin
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program; if not, write to the Free Software Foundation, Inc., 51
 * Franklin St - Fifth Floor, Boston, MA 02110-1301 USA.
 *
 *
 * For KGPU modification:
 *
 * Copyright (c) 2010-2011 University of Utah and the Flux Group.
 * All rights reserved.
 *
 */
#include <linux/kernel.h>
#include <linux/interrupt.h>
#include <linux/dma-mapping.h>
#include <linux/raid/pq.h>
#include <linux/async_tx.h>

static struct dma_async_tx_descriptor *
async_sum_product(struct page *dest, struct page **srcs, unsigned char *coef,
		  size_t len, struct async_submit_ctl *submit)
{
    struct dma_chan *chan = async_tx_find_channel(submit, DMA_PQ,
						  &dest, 1, srcs, 2, len);
    struct dma_device *dma = chan ? chan->device : NULL;
    const u8 *amul, *bmul;
    u8 ax, bx;
    u8 *a, *b, *c;

    if (dma) {
	dma_addr_t dma_dest[2];
	dma_addr_t dma_src[2];
	struct device *dev = dma->dev;
	struct dma_async_tx_descriptor *tx;
	enum dma_ctrl_flags dma_flags = DMA_PREP_PQ_DISABLE_P;

	if (submit->flags & ASYNC_TX_FENCE)
	    dma_flags |= DMA_PREP_FENCE;
	dma_dest[1] = dma_map_page(dev, dest, 0, len, DMA_BIDIRECTIONAL);
	dma_src[0] = dma_map_page(dev, srcs[0], 0, len, DMA_TO_DEVICE);
	dma_src[1] = dma_map_page(dev, srcs[1], 0, len, DMA_TO_DEVICE);
	tx = dma->device_prep_dma_pq(chan, dma_dest, dma_src, 2, coef,
				     len, dma_flags);
	if (tx) {
	    async_tx_submit(chan, tx, submit);
	    return tx;
	}

	/* could not get a descriptor, unmap and fall through to
	 * the synchronous path
	 */
	dma_unmap_page(dev, dma_dest[1], len, DMA_BIDIRECTIONAL);
	dma_unmap_page(dev, dma_src[0], len, DMA_TO_DEVICE);
	dma_unmap_page(dev, dma_src[1], len, DMA_TO_DEVICE);
    }

    /* run the operation synchronously */
    async_tx_quiesce(&submit->depend_tx);
    amul = raid6_gfmul[coef[0]];
    bmul = raid6_gfmul[coef[1]];
    a = page_address(srcs[0]);
    b = page_address(srcs[1]);
    c = page_address(dest);

    while (len--) {
	ax    = amul[*a++];
	bx    = bmul[*b++];
	*c++ = ax ^ bx;
    }

    return NULL;
}

static struct dma_async_tx_descriptor *
async_mult(struct page *dest, struct page *src, u8 coef, size_t len,
	   struct async_submit_ctl *submit)
{
    struct dma_chan *chan = async_tx_find_channel(submit, DMA_PQ,
						  &dest, 1, &src, 1, len);
    struct dma_device *dma = chan ? chan->device : NULL;
    const u8 *qmul; /* Q multiplier table */
    u8 *d, *s;

    if (dma) {
	dma_addr_t dma_dest[2];
	dma_addr_t dma_src[1];
	struct device *dev = dma->dev;
	struct dma_async_tx_descriptor *tx;
	enum dma_ctrl_flags dma_flags = DMA_PREP_PQ_DISABLE_P;

	if (submit->flags & ASYNC_TX_FENCE)
	    dma_flags |= DMA_PREP_FENCE;
	dma_dest[1] = dma_map_page(dev, dest, 0, len, DMA_BIDIRECTIONAL);
	dma_src[0] = dma_map_page(dev, src, 0, len, DMA_TO_DEVICE);
	tx = dma->device_prep_dma_pq(chan, dma_dest, dma_src, 1, &coef,
				     len, dma_flags);
	if (tx) {
	    async_tx_submit(chan, tx, submit);
	    return tx;
	}

	/* could not get a descriptor, unmap and fall through to
	 * the synchronous path
	 */
	dma_unmap_page(dev, dma_dest[1], len, DMA_BIDIRECTIONAL);
	dma_unmap_page(dev, dma_src[0], len, DMA_TO_DEVICE);
    }

    /* no channel available, or failed to allocate a descriptor, so
     * perform the operation synchronously
     */
    async_tx_quiesce(&submit->depend_tx);
    qmul  = raid6_gfmul[coef];
    d = page_address(dest);
    s = page_address(src);

    while (len--)
	*d++ = qmul[*s++];

    return NULL;
}

static struct dma_async_tx_descriptor *
__2data_recov_4(int disks, size_t bytes, int faila, int failb,
		struct page **blocks, struct async_submit_ctl *submit)
{
    struct dma_async_tx_descriptor *tx = NULL;
    struct page *p, *q, *a, *b;
    struct page *srcs[2];
    unsigned char coef[2];
    enum async_tx_flags flags = submit->flags;
    dma_async_tx_callback cb_fn = submit->cb_fn;
    void *cb_param = submit->cb_param;
    void *scribble = submit->scribble;

    p = blocks[disks-2];
    q = blocks[disks-1];

    a = blocks[faila];
    b = blocks[failb];

    /* in the 4 disk case P + Pxy == P and Q + Qxy == Q */
    /* Dx = A*(P+Pxy) + B*(Q+Qxy) */
    srcs[0] = p;
    srcs[1] = q;
    coef[0] = raid6_gfexi[failb-faila];
    coef[1] = raid6_gfinv[raid6_gfexp[faila]^raid6_gfexp[failb]];
    init_async_submit(submit, ASYNC_TX_FENCE, tx, NULL, NULL, scribble);
    tx = async_sum_product(b, srcs, coef, bytes, submit);

    /* Dy = P+Pxy+Dx */
    srcs[0] = p;
    srcs[1] = b;
    init_async_submit(submit, flags | ASYNC_TX_XOR_ZERO_DST, tx, cb_fn,
		      cb_param, scribble);
    tx = async_xor(a, srcs, 0, 2, bytes, submit);

    return tx;

}

static struct dma_async_tx_descriptor *
__2data_recov_5(int disks, size_t bytes, int faila, int failb,
		struct page **blocks, struct async_submit_ctl *submit)
{
    struct dma_async_tx_descriptor *tx = NULL;
    struct page *p, *q, *g, *dp, *dq;
    struct page *srcs[2];
    unsigned char coef[2];
    enum async_tx_flags flags = submit->flags;
    dma_async_tx_callback cb_fn = submit->cb_fn;
    void *cb_param = submit->cb_param;
    void *scribble = submit->scribble;
    int good_srcs, good, i;

    good_srcs = 0;
    good = -1;
    for (i = 0; i < disks-2; i++) {
	if (blocks[i] == NULL)
	    continue;
	if (i == faila || i == failb)
	    continue;
	good = i;
	good_srcs++;
    }
    BUG_ON(good_srcs > 1);

    p = blocks[disks-2];
    q = blocks[disks-1];
    g = blocks[good];

    /* Compute syndrome with zero for the missing data pages
     * Use the dead data pages as temporary storage for delta p and
     * delta q
     */
    dp = blocks[faila];
    dq = blocks[failb];

    init_async_submit(submit, ASYNC_TX_FENCE, tx, NULL, NULL, scribble);
    tx = async_memcpy(dp, g, 0, 0, bytes, submit);
    init_async_submit(submit, ASYNC_TX_FENCE, tx, NULL, NULL, scribble);
    tx = async_mult(dq, g, raid6_gfexp[good], bytes, submit);

    /* compute P + Pxy */
    srcs[0] = dp;
    srcs[1] = p;
    init_async_submit(submit, ASYNC_TX_FENCE|ASYNC_TX_XOR_DROP_DST, tx,
		      NULL, NULL, scribble);
    tx = async_xor(dp, srcs, 0, 2, bytes, submit);

    /* compute Q + Qxy */
    srcs[0] = dq;
    srcs[1] = q;
    init_async_submit(submit, ASYNC_TX_FENCE|ASYNC_TX_XOR_DROP_DST, tx,
		      NULL, NULL, scribble);
    tx = async_xor(dq, srcs, 0, 2, bytes, submit);

    /* Dx = A*(P+Pxy) + B*(Q+Qxy) */
    srcs[0] = dp;
    srcs[1] = dq;
    coef[0] = raid6_gfexi[failb-faila];
    coef[1] = raid6_gfinv[raid6_gfexp[faila]^raid6_gfexp[failb]];
    init_async_submit(submit, ASYNC_TX_FENCE, tx, NULL, NULL, scribble);
    tx = async_sum_product(dq, srcs, coef, bytes, submit);

    /* Dy = P+Pxy+Dx */
    srcs[0] = dp;
    srcs[1] = dq;
    init_async_submit(submit, flags | ASYNC_TX_XOR_DROP_DST, tx, cb_fn,
		      cb_param, scribble);
    tx = async_xor(dp, srcs, 0, 2, bytes, submit);

    return tx;
}

static struct dma_async_tx_descriptor *
__2data_recov_n(int disks, size_t bytes, int faila, int failb,
		struct page **blocks, struct async_submit_ctl *submit)
{
    struct dma_async_tx_descriptor *tx = NULL;
    struct page *p, *q, *dp, *dq;
    struct page *srcs[2];
    unsigned char coef[2];
    enum async_tx_flags flags = submit->flags;
    dma_async_tx_callback cb_fn = submit->cb_fn;
    void *cb_param = submit->cb_param;
    void *scribble = submit->scribble;

    p = blocks[disks-2];
    q = blocks[disks-1];

    /* Compute syndrome with zero for the missing data pages
     * Use the dead data pages as temporary storage for
     * delta p and delta q
     */
    dp = blocks[faila];
    blocks[faila] = NULL;
    blocks[disks-2] = dp;
    dq = blocks[failb];
    blocks[failb] = NULL;
    blocks[disks-1] = dq;

    init_async_submit(submit, ASYNC_TX_FENCE, tx, NULL, NULL, scribble);
    tx = async_gen_syndrome(blocks, 0, disks, bytes, submit);

    /* Restore pointer table */
    blocks[faila]   = dp;
    blocks[failb]   = dq;
    blocks[disks-2] = p;
    blocks[disks-1] = q;

    /* compute P + Pxy */
    srcs[0] = dp;
    srcs[1] = p;
    init_async_submit(submit, ASYNC_TX_FENCE|ASYNC_TX_XOR_DROP_DST, tx,
		      NULL, NULL, scribble);
    tx = async_xor(dp, srcs, 0, 2, bytes, submit);

    /* compute Q + Qxy */
    srcs[0] = dq;
    srcs[1] = q;
    init_async_submit(submit, ASYNC_TX_FENCE|ASYNC_TX_XOR_DROP_DST, tx,
		      NULL, NULL, scribble);
    tx = async_xor(dq, srcs, 0, 2, bytes, submit);

    /* Dx = A*(P+Pxy) + B*(Q+Qxy) */
    srcs[0] = dp;
    srcs[1] = dq;
    coef[0] = raid6_gfexi[failb-faila];
    coef[1] = raid6_gfinv[raid6_gfexp[faila]^raid6_gfexp[failb]];
    init_async_submit(submit, ASYNC_TX_FENCE, tx, NULL, NULL, scribble);
    tx = async_sum_product(dq, srcs, coef, bytes, submit);

    /* Dy = P+Pxy+Dx */
    srcs[0] = dp;
    srcs[1] = dq;
    init_async_submit(submit, flags | ASYNC_TX_XOR_DROP_DST, tx, cb_fn,
		      cb_param, scribble);
    tx = async_xor(dp, srcs, 0, 2, bytes, submit);

    return tx;
}

/*
 * GPU RAID6 recovery
 */

#include <linux/list.h>
#include <linux/gfp.h>
#include <linux/string.h>
#include <linux/spinlock.h>
#include <linux/module.h>
#include <linux/init.h>
#include <linux/slab.h>
#include <linux/completion.h>
#include <linux/kthread.h>
#include <linux/wait.h>
#include <asm/atomic.h>
#include "../../../kgpu/kgpu.h"
#include "../r62_recov.h"

#define R62_REQUEST_WAIT   0
#define R62_REQUEST_HANDLE 1
#define R62_REQUEST_DONE   2
#define R62_REQUEST_ERROR  3
#define R62_REQUEST_SYNGEN 4

#define R62_WAIT_TIMEOUT (10)

struct r62_request {
    long id;
    int disks;
    size_t bytes;
    u8 **blocks;
    int faila, failb;
    struct list_head list;
    int status;
    struct completion c;
};

struct r62_data {
    struct list_head reqs;
    struct semaphore reqsem;
    spinlock_t reqlock;
    int nr;
    atomic_long_t seq;
    struct task_struct *kt;
    wait_queue_head_t ktwait;
};

static struct kmem_cache *r62_request_cache;
static struct r62_data r62dat;

static int r62_max_reqs = 16;
module_param(r62_max_reqs, int, 0444);
MODULE_PARM_DESC(r62_max_reqs,
		 "max request queue size before procssing, default 16 ");

static int no_merge = 0;
module_param(no_merge, int, 0444);
MODULE_PARM_DESC(no_merge,
		 "do not perform request merge, default 0 (No)");

static int use_cpu = 0;
module_param(use_cpu, int, 0444);
MODULE_PARM_DESC(use_cpu,
		 "use cpu 2data recovery, default 0 (No)");

static int use_sim = 0;
module_param(use_sim, int, 0444);
MODULE_PARM_DESC(use_cpu,
		 "use cpu simulation for GPU call, default 0 (No)");

static int batch_timeout = 6;
module_param(batch_timeout, int, 0444);
MODULE_PARM_DESC(batch_timeout,
		 "timeout for batching requests, default 10, in jiffies");

static int test = 0;
module_param(test, int, 0444);
MODULE_PARM_DESC(test,
		 "test performance when loading, default 0 (No)");


#define do_log(level, ...) kgpu_do_log(level, "r62_recov", ##__VA_ARGS__)
#define prnt(...) 
//do_log(KGPU_LOG_PRINT, ##__VA_ARGS__)

static void do_test(void);

static void r62_request_ctr(void *data)
{
    struct r62_request *r = (struct r62_request*)data;

    INIT_LIST_HEAD(&r->list);
    r->status = R62_REQUEST_WAIT;
    init_completion(&r->c);    
    r->id = atomic_long_inc_return(&r62dat.seq);    
}

static int sim_gpu_call(struct kgpu_request *r)
{
    struct r62_recov_data *data = (struct r62_recov_data*)r->udata;
    u8 *p, *q, *dp, *dq;
    u8 px, qx;
    const u8 *pbmul;
    const u8 *qmul;
	
    int i, j;
	
    p = (u8*)r->in;
    q = p+data->n*PAGE_SIZE;
    dp = (u8*)r->out;
    dq = dp+data->n*PAGE_SIZE;
	
    for (i=0; i<data->n; i++) {
	pbmul = raid6_gfmul[data->idx[i].pbidx];
	qmul  = raid6_gfmul[data->idx[i].qidx];
	for (j=0; j<PAGE_SIZE; j++) {
	    int id = j + i*PAGE_SIZE;
	    px = p[id] ^ dp[id];
	    qx = qmul[q[id] ^ dq[id]];
	    dq[id] = pbmul[px] ^ qx;
	    dp[id] = dq[id] ^ px;
	}
    }
	
    return 0;
}

static void process_r62_requests(struct list_head *reqs, int n, size_t tsz)
{
    u8 *p, *q, *dp, *dq;
    struct list_head *pos, *tmp;
    struct r62_request *r = NULL;
    int i, j;

    struct kgpu_request *greq = NULL;
    u8 *gbuf = NULL;
    struct r62_recov_data *data = NULL;

    size_t rsz = n<<(PAGE_SHIFT+1);
    size_t dsz = sizeof(struct r62_recov_data)+n*sizeof(struct r62_tbl);

    //do_log(KGPU_LOG_PRINT, "#req = %d\n", n);

    gbuf = (u8*)kgpu_vmalloc(2*rsz+round_up(dsz, PAGE_SIZE));
			     
    if (!gbuf) {
	do_log(KGPU_LOG_ERROR, "out of GPU mem\n");
	goto fail_out;
    }

    greq = kgpu_alloc_request();
    if (!greq) {
	do_log(KGPU_LOG_ERROR, "can't alloc GPU request\n");
	goto fail_out;
    }

    greq->in        = gbuf;
    greq->out       = gbuf + rsz + round_up(dsz, PAGE_SIZE);
    greq->insize    = 2*rsz+round_up(dsz, PAGE_SIZE);
    greq->outsize   = rsz;
    greq->udata     = gbuf + rsz;
    greq->udatasize = dsz;

    data = (struct r62_recov_data*)greq->udata;
    data->bytes = (size_t)(n*PAGE_SIZE);
    data->n = n;
    
    p  = (u8*)greq->in;
    q  = p + (n*PAGE_SIZE);
    dp = (u8*)greq->out;
    dq = dp + (n*PAGE_SIZE);

    j = 0;
    list_for_each_safe(pos, tmp, reqs) {
	u8 *tp, *tq, *tdp, *tdq;
	r = list_entry(pos, struct r62_request, list);

	tp = r->blocks[r->disks-2];
	tq = r->blocks[r->disks-1];

        tdp = r->blocks[r->faila];
	r->blocks[r->faila] = (void *)raid6_empty_zero_page;
	r->blocks[r->disks-2] = tdp;
	tdq = r->blocks[r->failb];
	r->blocks[r->failb] = (void *)raid6_empty_zero_page;
	r->blocks[r->disks-1] = tdq;
	
	for (i=0; i<r->disks; i++) {
	    if (!r->blocks[i])
		break;
	}
	
	if (i== r->disks)
	    raid6_call.gen_syndrome(r->disks, r->bytes, (void**)r->blocks);
	else {
	    prnt("NULL pointer at req %lu %d blk\n", r->id, i);
	}
	
	r->status = R62_REQUEST_SYNGEN;

	r->blocks[r->faila]   = tdp;
	r->blocks[r->failb]   = tdq;
	r->blocks[r->disks-2] = tp;
	r->blocks[r->disks-1] = tq;

	data->idx[j].pbidx = raid6_gfexi[r->failb - r->faila];
	data->idx[j].qidx  = raid6_gfinv[raid6_gfexp[r->faila]^raid6_gfexp[r->failb]];

	memcpy(p+(j<<PAGE_SHIFT),
	       tp,//page_address(r->blocks[r->disks-2]),
	       PAGE_SIZE);
	memcpy(q+(j<<PAGE_SHIFT),
	       tq,//page_address(r->blocks[r->disks-1]),
	       PAGE_SIZE);
	memcpy(dp+(j<<PAGE_SHIFT),
	       tdp,//page_address(r->blocks[r->faila]),
	       PAGE_SIZE);
	memcpy(dq+(j<<PAGE_SHIFT),
	       tdq,//page_address(r->blocks[r->failb]),
	       PAGE_SIZE);
	       
	prnt("handle req %lu %d\n", r->id, j);
	
	j++;
    }
    
    strcpy(greq->service_name, "r62_recov");
    
    prnt("submit GPU request %d\n", greq->id);
    
    if (use_sim? sim_gpu_call(greq):kgpu_call_sync(greq)) {
	do_log(KGPU_LOG_ERROR, "call gpu failed\n");
	goto fail_out;
    } else {
    	prnt("done GPU request %d\n", greq->id);
	j = 0;
	list_for_each_safe(pos, tmp, reqs) {
	    r = list_entry(pos, struct r62_request, list);
	    
	    memcpy(r->blocks[r->faila],
		   dp+(j<<PAGE_SHIFT), PAGE_SIZE);
	    memcpy(r->blocks[r->failb],
		   dq+(j<<PAGE_SHIFT), PAGE_SIZE);
		   
	    list_del(pos);
	    
	    r->status = R62_REQUEST_DONE;
	    prnt("to complete req %lu %d\n", r->id, j);
	    if (!no_merge)
		complete(&r->c);
	    j++;
	}
	goto free_out;
    }

fail_out:
    list_for_each_safe(pos, tmp, reqs) {
	r = list_entry(pos, struct r62_request, list);
	list_del(pos);
	r->status = R62_REQUEST_ERROR;
	if (likely(!no_merge))
	    complete(&r->c);
	prnt("done req %lu with error\n", r->id);
    }
    
free_out:
    if (gbuf) kgpu_vfree(gbuf);
    if (greq) kgpu_free_request(greq);
}

// with r62dat.reqlock on hold
static int take_off_requests(struct list_head *reqs)
{
    int n = 0;
    struct list_head *pos, *tmp;
    struct r62_request *ite;
    list_replace_init(&r62dat.reqs, reqs);

    list_for_each_safe(pos, tmp, reqs) {
	ite = list_entry(pos, struct r62_request, list);
	ite->status = R62_REQUEST_HANDLE;
	n++;
	prnt("off req %lu\n", ite->id);
    }

    r62dat.nr = 0;
    return n;
}

static int r62d(void *data)
{
    int n;
    struct list_head reqs;
	
    while(!no_merge) {
	wait_event_timeout(r62dat.ktwait,
			   r62dat.nr >= r62_max_reqs,
			   batch_timeout);
		
	spin_lock_irq(&r62dat.reqlock);
	n = take_off_requests(&reqs);
	spin_unlock_irq(&r62dat.reqlock);
		
	if (n) {
	    process_r62_requests(&reqs, n, n*PAGE_SIZE);
	}
		
	if (kthread_should_stop())
	    break;
    }
    return 0;
}

static void init_gpu_system(void)
{
    r62_request_cache = kmem_cache_create(
	"r62_request_cache",
	sizeof(struct r62_request), 0,
	SLAB_HWCACHE_ALIGN, r62_request_ctr);
    if (!r62_request_cache) {
	do_log(KGPU_LOG_ERROR, "can't creat request cache\n");
	return;
    }
    
    atomic_long_set(&r62dat.seq, 0);
    INIT_LIST_HEAD(&r62dat.reqs);
    r62dat.nr = 0;
    sema_init(&r62dat.reqsem, 1);
    spin_lock_init(&r62dat.reqlock);
    
    init_waitqueue_head(&r62dat.ktwait);
    
    r62dat.kt = kthread_run(r62d, NULL, "r62d");
    if (!r62dat.kt) {
    	do_log(KGPU_LOG_ERROR, "can't create kernel thread\n");
    }
}

static void finit_gpu_system(void)
{
    if (r62_request_cache)
	kmem_cache_destroy(r62_request_cache);
	
    if (r62dat.kt)
	kthread_stop(r62dat.kt);
}


static void gpu_async_raid6_2drecov(int disks, size_t bytes,
				    int faila, int failb,
				    u8 **blocks)
{ 
    struct r62_request *r = kmem_cache_alloc(r62_request_cache,
					     GFP_KERNEL);
    if (!r) {
	do_log(KGPU_LOG_ERROR, "out of memory for r62_request\n");
	return;
    }
    r->disks = disks;
    r->bytes = bytes;
    r->faila = faila;
    r->failb = failb;
    r->blocks = blocks;

    if (likely(!no_merge)) {
 
	spin_lock_irq(&r62dat.reqlock);
	list_add_tail(&r->list, &r62dat.reqs);
	r62dat.nr++;
	if (r62dat.nr >= r62_max_reqs) {    	
	    spin_unlock_irq(&r62dat.reqlock);
	    wake_up_all(&r62dat.ktwait);
	    prnt("reach max reqs\n");
	} else {
	    spin_unlock_irq(&r62dat.reqlock);
	    prnt("req %lu added\n", r->id);    	
	}
	
	wait_for_completion(&r->c);
    } else {
	struct list_head h;
	INIT_LIST_HEAD(&h);
	list_add_tail(&r->list, &h);
	process_r62_requests(&h, 1, PAGE_SIZE);
    }
    
    prnt("req %lu finished\n", r->id);
    kmem_cache_free(r62_request_cache, r);
}

static int __init r62_recov_module_init(void)
{
    init_gpu_system();
    if (test)
	do_test();
    return 0;
}

static void __exit r62_recov_module_exit(void)
{
    finit_gpu_system();
}

module_init(r62_recov_module_init);
module_exit(r62_recov_module_exit);
		    

/**
 * async_raid6_2data_recov - asynchronously calculate two missing data blocks
 * @disks: number of disks in the RAID-6 array
 * @bytes: block size
 * @faila: first failed drive index
 * @failb: second failed drive index
 * @blocks: array of source pointers where the last two entries are p and q
 * @submit: submission/completion modifiers
 */
struct dma_async_tx_descriptor *
async_raid6_2data_recov(int disks, size_t bytes, int faila, int failb,
			struct page **blocks, struct async_submit_ctl *submit)
{
    void *scribble = submit->scribble;
    int non_zero_srcs, i;

    BUG_ON(faila == failb);
    if (failb < faila)
	swap(faila, failb);

    /* if a dma resource is not available or a scribble buffer is not
     * available punt to the synchronous path.  In the 'dma not
     * available' case be sure to use the scribble buffer to
     * preserve the content of 'blocks' as the caller intended.
     */
    if (!async_dma_find_channel(DMA_PQ) || !scribble) {

	async_tx_quiesce(&submit->depend_tx);

	if (!use_cpu && bytes == PAGE_SIZE) {
	    // With our patch for raid5.c, blocks already on new allocated
	    // bufs. so we don't need re-aclloc new one.
	    //u8 **ptrs = kmalloc(sizeof(u8*)*disks, GFP_KERNEL);
	    u8 **ptrs = (u8**)blocks;
			    
	    for (i = 0; i < disks; i++)
		if (blocks[i] == NULL)
		    ptrs[i] = (u8 *) raid6_empty_zero_page;
		else
		    ptrs[i] = (u8*)page_address(blocks[i]);
			
	    prnt("gpu recov %d dsks, %lu bytes failed: %d %d\n", disks, bytes, faila, failb);
	    
	    gpu_async_raid6_2drecov(disks,
				    bytes,
				    faila,
				    failb,
				    ptrs);
	    //kfree(ptrs);
	} else {	    
	    void **ptrs = scribble ? scribble : (void **) blocks;
		
	    if (bytes != PAGE_SIZE) {
		prnt("non-page size %lu\n", bytes);
	    }
	    
	    for (i = 0; i < disks; i++)
		if (blocks[i] == NULL)
		    ptrs[i] = (void *) raid6_empty_zero_page;
		else
		    ptrs[i] = page_address(blocks[i]);
		
	    raid6_2data_recov(disks, bytes, faila, failb, ptrs);
	}
	    
	async_tx_sync_epilog(submit);
	    
	return NULL;
    }

    non_zero_srcs = 0;
    for (i = 0; i < disks-2 && non_zero_srcs < 4; i++)
	if (blocks[i])
	    non_zero_srcs++;
    switch (non_zero_srcs) {
    case 0:
    case 1:
	/* There must be at least 2 sources - the failed devices. */
	BUG();

    case 2:
	/* dma devices do not uniformly understand a zero source pq
	 * operation (in contrast to the synchronous case), so
	 * explicitly handle the special case of a 4 disk array with
	 * both data disks missing.
	 */
	return __2data_recov_4(disks, bytes, faila, failb, blocks, submit);
    case 3:
	/* dma devices do not uniformly understand a single
	 * source pq operation (in contrast to the synchronous
	 * case), so explicitly handle the special case of a 5 disk
	 * array with 2 of 3 data disks missing.
	 */
	return __2data_recov_5(disks, bytes, faila, failb, blocks, submit);
    default:
	return __2data_recov_n(disks, bytes, faila, failb, blocks, submit);
    }
}
EXPORT_SYMBOL_GPL(async_raid6_2data_recov);

#include <linux/async.h>

static int test_min_size = 1;
static int test_max_size = 1;
static int test_min_disks = 6;
static int test_max_disks = 8;

struct async_data {
    struct completion *c;
    void **pts;
    int disks;
};

void async_gpu(void *param, async_cookie_t cookie)
{
    struct async_data *d = (struct async_data*)param;
    gpu_async_raid6_2drecov(d->disks, PAGE_SIZE, 0, 1, (u8**)d->pts);
    complete(d->c);
}

void async_cpu(void *param, async_cookie_t cookie)
{
    struct async_data *d = (struct async_data*)param;
    raid6_2data_recov(d->disks, PAGE_SIZE, 0, 1, d->pts);
    complete(d->c);
}

static void do_test(void)
{
    int old_use_cpu = use_cpu;
    int old_use_sim = use_sim;

    int disks, size, i;

    struct timeval t0, t1;
    long t;

    struct async_data *ds = kmalloc(
	sizeof(struct async_data)*test_max_size, GFP_KERNEL);

    void **pgs = kmalloc(
	sizeof(void)*(test_max_disks*test_max_size), GFP_KERNEL|__GFP_ZERO);
    struct completion *cs = kmalloc(
	sizeof(struct completion)*test_max_size, GFP_KERNEL);
    if (!pgs) { do_log(KGPU_LOG_ERROR, "pgs no mem\n"); goto clean_out;}
    if (!cs) { do_log(KGPU_LOG_ERROR, "cs no mem\n"); goto clean_out;}
    if (!ds) { do_log(KGPU_LOG_ERROR, "ds no mem\n"); goto clean_out;}

    for (i=0; i<test_max_disks*test_max_size; i++) {
	pgs[i] = __get_free_page(GFP_KERNEL);
	if (!pgs[i]) {
	    do_log(KGPU_LOG_ERROR, "no page\n"); goto clean_out;
	}
    }

    printk("begin test\n");

    for (size = test_min_size; size <= test_max_size; size += test_min_size)
    {
	for (disks = test_min_disks; disks <= test_max_disks; disks += 2)
	{
	    printk("test %d %d\n", size, disks);
	    for (i=0; i<size; i++) {
		init_completion(cs+i);
	    }

	    //do_log(KGPU_LOG_PRINT, "do GPU test ...\n");

	    use_cpu = 0;
	    use_sim = 0;
	    do_gettimeofday(&t0);
	    for (i=0; i<size; i++) {
		ds[i].disks = disks;
		ds[i].c = cs+i;
		ds[i].pts = pgs+i*disks;

		//	async_schedule(async_gpu, ds);
	    }
	    //for (i=0; i<size; i++)
	    //	wait_for_completion(cs+i);
	    do_gettimeofday(&t1);

	    t = 1000000*(t1.tv_sec-t0.tv_sec) +
		((int)(t1.tv_usec) - (int)(t0.tv_usec));
	    printk("GPU %2d disks, %4d reqs, %8ldMB/s, %8ldus\n",
		   disks, size, (size*PAGE_SIZE*(disks-2))/t, t);

	    /* ----------- */
	    for (i=0; i<size; i++) {
		init_completion(cs+i);
	    }
	    //do_log(KGPU_LOG_PRINT, "do CPU test ...\n");

	    use_cpu = 1;
	    use_sim = 0;
	    do_gettimeofday(&t0);
	    for (i=0; i<size; i++) {
		ds[i].disks = disks;
		ds[i].c = cs+i;
		ds[i].pts = pgs+i*disks;

		async_schedule(async_cpu, ds);
	    }	
	    for (i=0; i<size; i++)
		wait_for_completion(cs+i);
	    do_gettimeofday(&t1);

	    t = 1000000*(t1.tv_sec-t0.tv_sec) +
		((int)(t1.tv_usec) - (int)(t0.tv_usec));
	    printk("CPU %2d disks, %4d reqs, %8ldMB/s, %8ldus\n",
		   disks, size, (size*PAGE_SIZE*(disks-2))/t, t);    
	}
    }


clean_out:
    if (pgs) {
	for (i=0; i<test_max_disks*test_max_size; i++) {
	    if (pgs[i]) free_page(pgs[i]);
	}
	kfree(pgs);
    }
    if (cs) kfree(cs);
    if (ds) kfree(ds);

    use_cpu = old_use_cpu;
    use_sim = old_use_sim;      
}

/**
 * async_raid6_datap_recov - asynchronously calculate a data and the 'p' block
 * @disks: number of disks in the RAID-6 array
 * @bytes: block size
 * @faila: failed drive index
 * @blocks: array of source pointers where the last two entries are p and q
 * @submit: submission/completion modifiers
 */
struct dma_async_tx_descriptor *
async_raid6_datap_recov(int disks, size_t bytes, int faila,
			struct page **blocks, struct async_submit_ctl *submit)
{
    struct dma_async_tx_descriptor *tx = NULL;
    struct page *p, *q, *dq;
    u8 coef;
    enum async_tx_flags flags = submit->flags;
    dma_async_tx_callback cb_fn = submit->cb_fn;
    void *cb_param = submit->cb_param;
    void *scribble = submit->scribble;
    int good_srcs, good, i;
    struct page *srcs[2];

    /* if a dma resource is not available or a scribble buffer is not
     * available punt to the synchronous path.  In the 'dma not
     * available' case be sure to use the scribble buffer to
     * preserve the content of 'blocks' as the caller intended.
     */
    if (!async_dma_find_channel(DMA_PQ) || !scribble) {
	void **ptrs = scribble ? scribble : (void **) blocks;

	async_tx_quiesce(&submit->depend_tx);
	for (i = 0; i < disks; i++)
	    if (blocks[i] == NULL)
		ptrs[i] = (void*)raid6_empty_zero_page;
	    else
		ptrs[i] = page_address(blocks[i]);

	raid6_datap_recov(disks, bytes, faila, ptrs);

	async_tx_sync_epilog(submit);

	return NULL;
    }

    good_srcs = 0;
    good = -1;
    for (i = 0; i < disks-2; i++) {
	if (i == faila)
	    continue;
	if (blocks[i]) {
	    good = i;
	    good_srcs++;
	    if (good_srcs > 1)
		break;
	}
    }
    BUG_ON(good_srcs == 0);

    p = blocks[disks-2];
    q = blocks[disks-1];

    /* Compute syndrome with zero for the missing data page
     * Use the dead data page as temporary storage for delta q
     */
    dq = blocks[faila];
    blocks[faila] = NULL;
    blocks[disks-1] = dq;

    /* in the 4-disk case we only need to perform a single source
     * multiplication with the one good data block.
     */
    if (good_srcs == 1) {
	struct page *g = blocks[good];

	init_async_submit(submit, ASYNC_TX_FENCE, tx, NULL, NULL,
			  scribble);
	tx = async_memcpy(p, g, 0, 0, bytes, submit);

	init_async_submit(submit, ASYNC_TX_FENCE, tx, NULL, NULL,
			  scribble);
	tx = async_mult(dq, g, raid6_gfexp[good], bytes, submit);
    } else {
	init_async_submit(submit, ASYNC_TX_FENCE, tx, NULL, NULL,
			  scribble);
	tx = async_gen_syndrome(blocks, 0, disks, bytes, submit);
    }

    /* Restore pointer table */
    blocks[faila]   = dq;
    blocks[disks-1] = q;

    /* calculate g^{-faila} */
    coef = raid6_gfinv[raid6_gfexp[faila]];

    srcs[0] = dq;
    srcs[1] = q;
    init_async_submit(submit, ASYNC_TX_FENCE|ASYNC_TX_XOR_DROP_DST, tx,
		      NULL, NULL, scribble);
    tx = async_xor(dq, srcs, 0, 2, bytes, submit);

    init_async_submit(submit, ASYNC_TX_FENCE, tx, NULL, NULL, scribble);
    tx = async_mult(dq, dq, coef, bytes, submit);

    srcs[0] = p;
    srcs[1] = dq;
    init_async_submit(submit, flags | ASYNC_TX_XOR_DROP_DST, tx, cb_fn,
		      cb_param, scribble);
    tx = async_xor(p, srcs, 0, 2, bytes, submit);

    return tx;
}
EXPORT_SYMBOL_GPL(async_raid6_datap_recov);

MODULE_AUTHOR("Dan Williams <dan.j.williams@intel.com>; Weibin Sun <wbsun@cs.utah.edu>");
MODULE_DESCRIPTION("asynchronous RAID-6 recovery api, with GPU");
MODULE_LICENSE("GPL");
