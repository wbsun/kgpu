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
#include "../../../kgpu/kgpu.h"
#include "../r62_recov.h"

#define R62_REQUEST_WAIT   0
#define R62_REQUEST_HANDLE 1
#define R62_REQUEST_DONE   2
#define R62_REQUEST_ERROR  3

#define R62_WAIT_TIMEOUT (HZ/100)

struct r62_request {
    int disks;
    size_t bytes;
    struct page **blocks;
    int faila, failb;
    struct list_head list;
    int status;
    struct completion c;
};

struct r62_data {
    struct list_head reqs;
    spinlock_t reqlock;
    int nr;
};

static struct kmem_cache *r62_request_cache;
static struct r62_data r62dat;

static int g_r62_max_reqs = 16;

module_param(g_r62_max_reqs, int, 16);
MODULE_PARM_DESC(g_r62_max_reqs, "max request queue size");

static void r62_request_ctr(void *data)
{
    struct r62_request *r = (struct r62_request*)data;

    INIT_LIST_HEAD(&r->list);
    r->status = R62_REQUEST_WAIT;
    init_completion(&r->c);
}

static void init_gpu_system(void)
{
    r62_request_cache = kmem_cache_create(
	"r62_request_cache",
	sizeof(struct r62_request), 0,
	SLAB_HWCACHE_ALIGN, r62_request_ctr);
    if (!r62_request_cache) {
	printk("[async_raid6_recov] Error: can't creat request cache\n");
	return;
    }

    spin_lock_init(&r62dat.reqlock);
    INIT_LIST_HEAD(&r62dat.reqs);
    r62dat.nr = 0;
}

static void finit_gpu_system(void)
{
    if (r62_request_cache)
	kmem_cache_destroy(r62_request_cache);
}

static void process_r62_requests(struct list_head *reqs, int n, size_t tsz)
{
    struct page *pp, *pq, *pdp, *pdq;
    u8 *p, *q, *dp, *dq;
    struct list_head *pos;
    struct r62_request *r = NULL;
    int i, j, pbidx, qidx;

    void **ptrs = NULL;

    struct kgpu_request *greq = NULL;
    u8 *gbuf = NULL;
    struct r62_recov_data *data = NULL;

    size_t rsz = n<<(PAGE_SHIFT+1);

    gbuf = (u8*)kgpu_vmalloc(2*rsz+PAGE_SIZE);
    if (!gbuf) {
	printk("[async_raid6_recov] Error: out of GPU mem\n");
	goto fail_out;
    }

    greq = kgpu_alloc_request();
    if (!greq) {
	printk("[async_raid6_recov] Error: can't alloc GPU request\n");
	goto fail_out;
    }

    greq->in        = gbuf;
    greq->out       = gbuf + rsz + PAGE_SIZE;
    greq->insize    = rsz;
    greq->outsize   = rsz;
    greq->udata     = gbuf + rsz;
    greq->udatasize = sizeof(int)*2;

    data = (struct r62_recov_data*)greq->udata;
    data->bytes = (size_t)(n<<PAGE_SHIFT);

    p  = gbuf;
    q  = p + (n<<PAGE_SHIFT);
    dp = q + (n<<PAGE_SHIFT);
    dq = dp + (n<<PAGE_SHIFT);

    j = 0;
    list_for_each(pos, reqs) {
	r = list_entry(pos, struct r62_request, list);

	if (!ptrs) {
	    ptrs = (void**)kmalloc(sizeof(void*)*r->disks, GFP_KERNEL);
	    if (!ptrs) {
		printk("[async_raid6_recov] Error: out of mem for ptrs\n");
		goto fail_out;
	    }
	}

	pp = r->blocks[r->disks-2];
	pq = r->blocks[r->disks-1];

	pdp = r->blocks[r->faila];
	r->blocks[r->faila] = virt_to_page((void *)raid6_empty_zero_page);
	r->blocks[r->disks-2] = pdp;
	pdq = r->blocks[r->failb];
	r->blocks[r->failb] = r->blocks[r->faila];
        r->blocks[r->disks-1] = pdq;

	for (i=0; i<r->disks; i++)
	    ptrs = page_address(r->blocks[i]);

	raid6_call.gen_syndrome(r->disks, r->bytes, ptrs);

	r->blocks[r->faila] = pdp;
	r->blocks[r->failb] = pdq;
	r->blocks[r->disks-2] = pp;
	r->blocks[r->disks-1] = pq;

	memcpy(p+(j<<PAGE_SHIFT),
	       page_address(r->blocks[r->disks-2]), PAGE_SIZE);
	memcpy(q+(j<<PAGE_SHIFT),
	       page_address(r->blocks[r->disks-1]), PAGE_SIZE);
	memcpy(dp+(j<<PAGE_SHIFT),
	       page_address(r->blocks[r->faila]), PAGE_SIZE);
	memcpy(dq+(j<<PAGE_SHIFT),
	       page_address(r->blocks[r->failb]), PAGE_SIZE);
	j++;
    }

    if (ptrs) kfree(ptrs);

    pbidx = raid6_gfexi[r->failb-r->faila];
    qidx  = raid6_gfinv[raid6_gfexp[r->faila]^raid6_gfexp[r->failb]];

    data->pbidx = pbidx;
    data->qidx = qidx;
    
    strcpy(greq->service_name, "r62_recov");

    if (kgpu_call_sync(greq)) {
	printk("[async_raid6_recov] Error: call gpu failed\n");
	goto fail_out;
    } else {
	j = 0;
	list_for_each(pos, reqs) {
	    r = list_entry(pos, struct r62_request, list);

	    memcpy(page_address(r->blocks[r->faila]),
		   dp+(j<<PAGE_SHIFT), PAGE_SIZE);
	    memcpy(page_address(r->blocks[r->failb]),
		   dq+(j<<PAGE_SHIFT), PAGE_SIZE);
	    j++;
	    r->status = R62_REQUEST_DONE;
	    complete(&r->c);
	}
	goto free_out;
    }

fail_out:
    list_for_each(pos, reqs) {
	r = list_entry(pos, struct r62_request, list);
	r->status = R62_REQUEST_ERROR;
	complete(&r->c);
    }
    
free_out:
    if (gbuf) kgpu_vfree(gbuf);
    if (greq) kgpu_free_request(greq);
}


/*
 * BIG ASSUMPTION: for all requests, faila == failb!
 * So use la or ra.
 */
static void gpu_async_raid6_2drecov(int disks, size_t bytes,
					int faila, int failb,
					struct page **blocks)
{
    struct list_head reqs, *pos;
    struct r62_request *ite;
    int n;
    size_t tsz;
    struct r62_request *r = kmem_cache_alloc(r62_request_cache,
					     GFP_KERNEL|__GFP_ZERO);
    if (!r) {
	printk("[async_raid6_recov] Error: out of memory for r62_request\n");
	return;
    }
    r->disks = disks;
    r->bytes = bytes;
    r->faila = faila;
    r->failb = failb;
    r->blocks = blocks;
    
    spin_lock(&r62dat.reqlock);
    list_add_tail(&r->list, &r62dat.reqs);
    r62dat.nr++;
    if (r62dat.nr >= g_r62_max_reqs) {
	goto do_process; // Yeah! I like GOTO!
    }
    spin_unlock(&r62dat.reqlock);

retry:
    if (wait_for_completion_interruptible_timeout(
	    &r->c, R62_WAIT_TIMEOUT))
    { // timeout	
	spin_lock(&r62dat.reqlock);
	if (r->status == R62_REQUEST_WAIT) {
	do_process:
	    n = 0;
	    tsz = 0;
	    reqs = r62dat.reqs;
	    reqs.next->prev = &reqs;
	    reqs.prev->next = &reqs;
	    INIT_LIST_HEAD(&r62dat.reqs);

	    list_for_each(pos, &reqs) {
		ite = list_entry(pos, struct r62_request, list);
		ite->status = R62_REQUEST_HANDLE;
		n++;
		tsz += ite->bytes;
	    }

	    r62dat.nr = 0;

	    spin_unlock(&r62dat.reqlock);
	    
	    process_r62_requests(&reqs, n, tsz);
	} else {
	    spin_unlock(&r62dat.reqlock);
	    goto retry;
	}
	    
    } else if (!completion_done(&r->c)) {
	goto retry;
    }

    kmem_cache_free(r62_request_cache, r);
}

static int __init r62_recov_module_init(void)
{
    init_gpu_system();
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

	pr_debug("%s: disks: %d len: %zu\n", __func__, disks, bytes);

	/* if a dma resource is not available or a scribble buffer is not
	 * available punt to the synchronous path.  In the 'dma not
	 * available' case be sure to use the scribble buffer to
	 * preserve the content of 'blocks' as the caller intended.
	 */
	if (!async_dma_find_channel(DMA_PQ) || !scribble) {

	    async_tx_quiesce(&submit->depend_tx);
	    
	    gpu_async_raid6_2drecov(disks,
				    bytes,
				    faila,
				    failb,
				    blocks);

	    async_tx_sync_epilog(submit);

	    return NULL;
	    		
	    /* void **ptrs = scribble ? scribble : (void **) blocks;
	    
	    async_tx_quiesce(&submit->depend_tx);
	    for (i = 0; i < disks; i++)
		if (blocks[i] == NULL)
		    ptrs[i] = (void *) raid6_empty_zero_page;
		else
		    ptrs[i] = page_address(blocks[i]);
	    
	    raid6_2data_recov(disks, bytes, faila, failb, ptrs);
	    
	    async_tx_sync_epilog(submit);
	    
	    return NULL; */
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

	pr_debug("%s: disks: %d len: %zu\n", __func__, disks, bytes);

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
