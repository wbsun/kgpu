/* This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2010-2011 University of Utah and the Flux Group.
 * All rights reserved.
 */
#include <linux/raid/pq.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/slab.h>
#include <linux/init.h>
#include "../../../kgpu/kgpu.h"
#include "../gpq.h"

/* customized log function */
#define gpq_log(level, ...) kgpu_do_log(level, "gpq", ##__VA_ARGS__)
#define dbg(...) gpq_log(KGPU_LOG_DEBUG, ##__VA_ARGS__)

static void raid6_gpq_gen_syndrome(int disks, size_t dsize, void **dps)
{
    size_t rsz = roundup(
	dsize*disks+sizeof(struct raid6_pq_data), PAGE_SIZE);

    struct raid6_pq_data *data;
    struct kgpu_req *req;
    struct kgpu_resp *resp;
    struct kgpu_buffer *buf;

    buf = alloc_gpu_buffer(rsz);
    if (unlikely(!buf)) {
	gpq_log(KGPU_LOG_ERROR, "GPU buffer allocation failed\n");
	return;
    }

    req = alloc_kgpu_request();
    resp = alloc_kgpu_response();
    if (unlikely(!req || !resp)) {
	gpq_log(KGPU_LOG_ERROR, "GPU request/response allocation failed\n");
	free_gpu_buffer(buf);
	return;
    }

    if (unlikely(dsize%PAGE_SIZE)) {
	gpq_log(KGPU_LOG_ERROR, "gpq only handle PAGE aligned memory\n");
	free_gpu_buffer(buf);
	free_kgpu_request(req);
	free_kgpu_response(resp);
	
	return;
    } else {
	int p=0, b;
	size_t cpsz;
	for (b=0; b<disks-2; b++) {
	    cpsz=0;
	    while (cpsz < dsize) {
		memcpy(__va(buf->pas[p]), (u8*)dps[b]+cpsz, PAGE_SIZE);
		p++;
		cpsz += PAGE_SIZE;
	    }
	}
    }

    strcpy(req->kureq.sname, "raid6_pq");
    req->kureq.input     = buf->va;
    req->kureq.output    = (u8*)(buf->va)+dsize*(disks-2);
    req->kureq.insize    = rsz;
    req->kureq.outsize   = dsize*2;
    req->kureq.data      = (u8*)(buf->va)+dsize*disks;
    req->kureq.datasize  = sizeof(struct raid6_pq_data);

    data = req->kureq.data;
    data->dsize = dsize;
    data->nr_d = disks;

    if (unlikely(call_gpu_sync(req, resp))) {
	gpq_log(KGPU_LOG_ERROR, "callgpu failed\n");
    } else {
	int p, b;
	size_t cpsz;
	p = dsize*(disks-2)/PAGE_SIZE;
	for (b=disks-2; b<disks; b++) {
	    cpsz = 0;
	    while (cpsz < dsize) {
		memcpy((u8*)dps[b]+cpsz, __va(buf->pas[p]), PAGE_SIZE);
		p++;
		cpsz += PAGE_SIZE;
	    }
	}
    }

    free_kgpu_request(req);
    free_kgpu_response(resp);
    free_gpu_buffer(buf);   
}

const struct raid6_calls raid6_gpq = {
    raid6_gpq_gen_syndrome,
    NULL,
    "gpq",
    0
};

static struct raid6_calls oldcall;
static int replace_global = 0;

module_param(replace_global, int, 0);
MODULE_PARM_DESC(replace_global, "replace global pq algorithm with gpq");

static int __init raid6_gpq_init(void)
{
    if (replace_global) {
	oldcall = raid6_call;
	raid6_call = raid6_gpq;
	gpq_log(KGPU_LOG_PRINT, "global pq algorithm replaced with gpq\n");
    }
    return 0;
}

static void __exit raid6_gpq_exit(void)
{
    if (replace_global) {
	raid6_call = oldcall;
    }
}

module_init(raid6_gpq_init);
module_exit(raid6_gpq_exit);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("gpq - GPU RAID6 PQ computing module");
MODULE_AUTHOR("Weibin Sun");
