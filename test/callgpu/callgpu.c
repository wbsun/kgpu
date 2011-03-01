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

#include "../../kgpu/_kgpu.h"

int mycb(struct kgpu_req *req, struct kgpu_resp *resp)
{
    printk("[callgpu]: REQ ID: %d, RESP ID: %d, RESP CODE: %d\n",
	   req->kureq.id, resp->kuresp.id, resp->kuresp.errcode);
    free_gpu_buffer((struct kgpu_buffer*)(req->data));
    free_kgpu_request(req);
    free_kgpu_response(resp);
    return 0;
}

static int __init minit(void)
{
    struct kgpu_req* req;
    struct kgpu_resp* resp;
    struct kgpu_buffer *buf;
    
    printk("[callgpu]: loaded\n");

    req = alloc_kgpu_request();
    if (!req) {
	printk("[callgpu] Error: request null\n");
	return 0;
    }
    resp = alloc_kgpu_response();
    if (!resp) {
	printk("[callgpu] Error: response null\n");
	return 0;
    }
    buf = alloc_gpu_buffer();
    if (!buf) {
	printk("[callgpu] Error: buffer null\n");
	return 0;
    }
    req->data = buf;
    /*req->kureq.id = next_kgpu_request_id();*/
    resp->kuresp.id = req->kureq.id;

    req->kureq.input = buf->gb.addr;
    req->kureq.insize = 1024;
    req->kureq.output = req->kureq.input+1024;
    req->kureq.outsize = 1024;
    strcpy(req->kureq.sname, "some service");
    req->cb = mycb;

    call_gpu(req, resp);
    
    return 0;
}

static void __exit mexit(void)
{
    printk("[callgpu]: unload\n");
}

module_init(minit);
module_exit(mexit);

MODULE_LICENSE("GPL");
