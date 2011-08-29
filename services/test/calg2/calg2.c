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
#include <linux/uaccess.h>
#include <asm/page.h>

#include "../../../kgpu/kgpu.h"

/* customized log function */
#define g_log(level, ...) kgpu_do_log(level, "calg2", ##__VA_ARGS__)
#define dbg(...) g_log(KGPU_LOG_DEBUG, ##__VA_ARGS__)


int mycb(struct kgpu_request *req)
{
    g_log(KGPU_LOG_PRINT, "REQ ID: %d, RESP CODE: %d, %d\n",
	   req->id, req->errcode,
	   ((int*)(req->out))[0]);

    kgpu_vfree(req->in);
    kgpu_free_request(req);
    return 0;
}

static int __init minit(void)
{
    struct kgpu_request* req;
    char *buf;
    
    g_log(KGPU_LOG_PRINT, "loaded\n");

    req = kgpu_alloc_request();
    if (!req) {
	g_log(KGPU_LOG_ERROR, "request null\n");
	return 0;
    }
    
    buf = (char*)kgpu_vmalloc(PAGE_SIZE);
    if (!buf) {
        g_log(KGPU_LOG_ERROR, "buffer null\n");
	return 0;
    }

    req->in = buf;
    req->insize = 1024;
    req->out = buf;/*+1024;*/
    req->outsize = 1024;
    strcpy(req->service_name, "test_service");
    req->callback = mycb;

    ((int*)(buf))[0] = 100;

    kgpu_call_async(req);
    
    return 0;
}

static void __exit mexit(void)
{
    g_log(KGPU_LOG_PRINT, "unload\n");
}

module_init(minit);
module_exit(mexit);

MODULE_LICENSE("GPL");
