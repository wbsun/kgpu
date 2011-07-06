/*
 * This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2010-2011 University of Utah and the Flux Group.
 * All rights reserved.
 *
 * KGPU k-u communication module.
 *
 * Weibin Sun
 */

#include <linux/module.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/list.h>
#include <linux/spinlock.h>
#include <linux/gfp.h>
#include <linux/kthread.h>
#include <linux/proc_fs.h>
#include <linux/mm.h>
#include <linux/string.h>
#include <linux/uaccess.h>
#include <linux/wait.h>
#include <linux/cdev.h>
#include <linux/fs.h>
#include <linux/poll.h>
#include <linux/slab.h>
#include <linux/bitmap.h>
#include <asm/atomic.h>
#include "kkgpu.h"

#define KGPU_MAJOR 0

struct kgpu_dev {
    struct cdev cdev;

    struct kgpu_mgmt_buffer mgmt_bufs[KGPU_BUF_NR];
    spinlock_t buflock;

    int rid_sequence;
    spinlock_t ridlock;

    struct list_head reqs;
    struct list_head resps;
    spinlock_t reqlock;
    spinlock_t resplock;
    wait_queue_head_t reqq;

    struct list_head rtdreqs;
    spinlock_t rtdreqlock;
};

static atomic_t kgpudev_av = ATOMIC_INIT(1);
static struct kgpu_dev kgpudev;
static int kgpu_major;

struct sync_call_data {
	wait_queue_head_t queue;
	void* olddata;
	ku_callback oldcb;
	int done;
};

static struct kmem_cache *kgpu_req_cache;
static struct kmem_cache *kgpu_resp_cache;
static struct kmem_cache *kgpu_allocated_buf_cache;

int call_gpu(struct kgpu_req *req, struct kgpu_resp *resp)
{
    req->resp = resp;
    
    spin_lock(&(kgpudev.reqlock));

    INIT_LIST_HEAD(&req->list);
    list_add_tail(&req->list, &(kgpudev.reqs));
    
    wake_up_interruptible(&(kgpudev.reqq));
    
    spin_unlock(&(kgpudev.reqlock));

    kgpu_log(KGPU_LOG_INFO, "call gpu %d\n", req->kureq.id);
    
    return 0;
}
EXPORT_SYMBOL_GPL(call_gpu);

static int sync_callback(struct kgpu_req *req, struct kgpu_resp *resp)
{
    struct sync_call_data *data = (struct sync_call_data*)
	req->data;
    
    data->done = 1;
    
    wake_up_interruptible(&data->queue);
    
    return 0;
}

int call_gpu_sync(struct kgpu_req *req, struct kgpu_resp *resp)
{
    struct sync_call_data *data
	= kmalloc(sizeof(struct sync_call_data), GFP_KERNEL);
    if (!data) {
	kgpu_log(KGPU_LOG_ERROR, "call_gpu_sync alloc mem failed\n");
	return 1;
    }
    
    req->resp = resp;
    
    data->olddata = req->data;
    data->oldcb = req->cb;
    data->done = 0;
    init_waitqueue_head(&data->queue);
    
    req->data = data;
    req->cb = sync_callback;
    
    spin_lock(&(kgpudev.reqlock));

    INIT_LIST_HEAD(&req->list);
    list_add_tail(&req->list, &(kgpudev.reqs));

    wake_up_interruptible(&(kgpudev.reqq));
    
    spin_unlock(&(kgpudev.reqlock));

    dbg("call gpu sync before %d\n", req->kureq.id);
    wait_event_interruptible(data->queue, (data->done==1));
    dbg("call gpu sync done %d\n", req->kureq.id);
    
    req->data = data->olddata;
    req->cb = data->oldcb;
    kfree(data);
    
    return 0;
}
EXPORT_SYMBOL_GPL(call_gpu_sync);

int next_kgpu_request_id(void)
{
    int rt = -1;
    
    spin_lock(&(kgpudev.ridlock));
    
    kgpudev.rid_sequence++;
    if (kgpudev.rid_sequence < 0)
	kgpudev.rid_sequence = 0;
    rt = kgpudev.rid_sequence;
    
    spin_unlock(&(kgpudev.ridlock));
    return rt;
}
EXPORT_SYMBOL_GPL(next_kgpu_request_id);

static void kgpu_req_constructor(void* data)
{
    struct kgpu_req *req = (struct kgpu_req*)data;
    if (req) {
	memset(req, 0, sizeof(struct kgpu_req));
	req->kureq.id = next_kgpu_request_id();
	INIT_LIST_HEAD(&req->list);
	req->kureq.sname[0] = 0;
    }
}

struct kgpu_req* alloc_kgpu_request(void)
{
    struct kgpu_req *req = kmem_cache_alloc(kgpu_req_cache, GFP_KERNEL);    
    return req;
}
EXPORT_SYMBOL_GPL(alloc_kgpu_request);

void free_kgpu_request(struct kgpu_req* req)
{
    kmem_cache_free(kgpu_req_cache, req);
}
EXPORT_SYMBOL_GPL(free_kgpu_request);

static void kgpu_resp_constructor(void *data)
{
    struct kgpu_resp *resp = (struct kgpu_resp*)data;
    if (resp) {
	memset(resp, 0, sizeof(struct kgpu_resp));
	resp->kuresp.errcode = KGPU_NO_RESPONSE;
	INIT_LIST_HEAD(&resp->list);
    }
}

struct kgpu_resp* alloc_kgpu_response(void)
{
    struct kgpu_resp *resp = kmem_cache_alloc(kgpu_resp_cache, GFP_KERNEL);
    return resp;
}
EXPORT_SYMBOL_GPL(alloc_kgpu_response);

void free_kgpu_response(struct kgpu_resp* resp)
{
    kmem_cache_free(kgpu_resp_cache, resp);
}
EXPORT_SYMBOL_GPL(free_kgpu_response);

static void kgpu_allocated_buf_constructor(void *data)
{
    struct kgpu_allocated_buffer *buf = (struct kgpu_allocated_buffer*)data;
    memset(buf, 0, sizeof(struct kgpu_allocated_buffer));
}

struct kgpu_buffer* alloc_gpu_buffer(unsigned long nbytes)
{
    int i;
    unsigned int req_nunits;
    struct kgpu_allocated_buffer *abuf;

    req_nunits = DIV_ROUND_UP(nbytes, KGPU_BUF_UNIT_SIZE);
    
    spin_lock(&(kgpudev.buflock));

    for (i=0; i<KGPU_BUF_NR; i++) {
	unsigned long idx = bitmap_find_next_zero_area(
	    kgpudev.mgmt_bufs[i].bitmap, kgpudev.mgmt_bufs[i].nunits,
	    0, req_nunits, 0);
	if (idx < kgpudev.mgmt_bufs[i].nunits) {
	    bitmap_set(kgpudev.mgmt_bufs[i].bitmap, idx, req_nunits);
	    spin_unlock(&(kgpudev.buflock));

	    abuf = kmem_cache_alloc(kgpu_allocated_buf_cache, GFP_KERNEL);
	    abuf->mgmt_buf_idx = i;
	    abuf->buf.npages = req_nunits*KGPU_BUF_NR_FRAMES_PER_UNIT;
	    abuf->buf.pas = kgpudev.mgmt_bufs[i].paddrs+(idx*KGPU_BUF_NR_FRAMES_PER_UNIT);
	    abuf->buf.va = (void*)((unsigned long)(kgpudev.mgmt_bufs[i].gb.addr)
				   +(idx*KGPU_BUF_UNIT_SIZE));
	    
	    return &(abuf->buf);
	}
    }

    spin_unlock(&(kgpudev.buflock));
    return NULL;
}
EXPORT_SYMBOL_GPL(alloc_gpu_buffer);

int free_gpu_buffer(struct kgpu_buffer *buf)
{
    int nr, idx;
    struct kgpu_allocated_buffer *abuf =
	container_of(buf, struct kgpu_allocated_buffer, buf);
    struct kgpu_mgmt_buffer *mbuf = &(kgpudev.mgmt_bufs[abuf->mgmt_buf_idx]);

    nr = buf->npages/KGPU_BUF_NR_FRAMES_PER_UNIT;
    idx = ((unsigned long)(buf->va) - (unsigned long)(mbuf->gb.addr))
	/KGPU_BUF_UNIT_SIZE;

    spin_lock(&(kgpudev.buflock));
    bitmap_clear(mbuf->bitmap, idx, nr);    
    spin_unlock(&(kgpudev.buflock));

    kmem_cache_free(kgpu_allocated_buf_cache, abuf);
    
    return 1;
}
EXPORT_SYMBOL_GPL(free_gpu_buffer);


/*
 * find request by id in the rtdreqs
 * offlist = 1: remove the request from the list
 * offlist = 0: keep the request in the list
 */
static struct kgpu_req* find_request(int id, int offlist)
{
    struct kgpu_req *pos, *n;

    spin_lock(&(kgpudev.rtdreqlock));
    
    list_for_each_entry_safe(pos, n, &(kgpudev.rtdreqs), list) {
	if (pos->kureq.id == id) {
	    if (offlist)
		list_del(&pos->list);
	    spin_unlock(&(kgpudev.rtdreqlock));
	    return pos;
	}
    }

    spin_unlock(&(kgpudev.rtdreqlock));

    return NULL;
}


int kgpu_open(struct inode *inode, struct file *filp)
{
    if (!atomic_dec_and_test(&kgpudev_av)) {
	atomic_inc(&kgpudev_av);
	return -EBUSY;
    }

    filp->private_data = &kgpudev;
    return 0;
}

int kgpu_release(struct inode *inode, struct file *file)
{
    atomic_set(&kgpudev_av, 1);
    return 0;
}

ssize_t kgpu_read(struct file *filp, char __user *buf, size_t c, loff_t *fpos)
{
    ssize_t ret = 0;
    struct list_head *r;
    struct kgpu_req *req = NULL;

    spin_lock(&(kgpudev.reqlock));
    while (list_empty(&(kgpudev.reqs))) {
	spin_unlock(&(kgpudev.reqlock));

	if (filp->f_flags & O_NONBLOCK)
	    return -EAGAIN;

	dbg("blocking read %s\n", current->comm);

	if (wait_event_interruptible(kgpudev.reqq, (!list_empty(&(kgpudev.reqs)))))
	    return -ERESTARTSYS;
	spin_lock(&(kgpudev.reqlock));
    }

    r = kgpudev.reqs.next;
    list_del(r);
    req = list_entry(r, struct kgpu_req, list);
    if (req) {
	memcpy/*copy_to_user*/(buf, &req->kureq, sizeof(struct ku_request));
	ret = c;/*sizeof(struct ku_request);*/

	dbg("one request read %s %d %ld\n",
	    req->kureq.sname, req->kureq.id, ret);
    }

    spin_unlock(&(kgpudev.reqlock));

    if (ret > 0 && req) {
	spin_lock(&(kgpudev.rtdreqlock));

	INIT_LIST_HEAD(&req->list);
	list_add_tail(&req->list, &(kgpudev.rtdreqs));

	spin_unlock(&(kgpudev.rtdreqlock));
    }
    
    dbg("%s read %lu return %ld\n",
	current->comm, c, ret);

    *fpos += ret;

    return ret;    
}

ssize_t kgpu_write(struct file *filp, const char __user *buf,
		   size_t count, loff_t *fpos)
{
    struct ku_response kuresp;
    struct kgpu_req *req;
    ssize_t ret = 0;
    size_t  realcount;
    
    if (count < sizeof(struct ku_response))
	ret = -EINVAL; /* Too small. */
    else
    {
	realcount = sizeof(struct ku_response);

	memcpy/*copy_from_user*/(&kuresp, buf, realcount);

	dbg("response ID: %d\n", kuresp.id);

	req = find_request(kuresp.id, 1);
	if (!req)
	{	    
	    dbg("no request found for %d\n", kuresp.id);
	    ret = -EFAULT; /* no request found */
	} else {
	    memcpy(&(req->resp->kuresp), &kuresp, realcount);

	    /*
	     * Different strategy should be applied here:
	     * #1 invoke the callback in the write syscall, like here.
	     * #2 add the resp into the resp-list in the write syscall
	     *    and return, a kernel thread will process the list
	     *    and invoke the callback.
	     *
	     * Currently, the first one is used because this can ensure
	     * the fast response. A kthread may have to sleep so that
	     * the response can't be processed ASAP.
	     */
	    req->cb(req, req->resp);
	    ret = realcount;
	    *fpos += ret;
	}
    }

    dbg("%s write %lu return %ld\n",
	current->comm, count, ret);

    return ret;
}

static int set_gpu_bufs(char __user *buf)
{
    int off=0, i, j;
    
    spin_lock(&(kgpudev.buflock));

    for (i=0; i<KGPU_BUF_NR; i++) {
	struct kgpu_mgmt_buffer *mbuf = &(kgpudev.mgmt_bufs[i]);
	copy_from_user(&mbuf->gb, buf+off, sizeof(struct gpu_buffer));

#if 0
	if (!kgpu_check_phy_consecutive((unsigned long)(mbuf->gb.addr),
				   mbuf->gb.size, PAGE_SIZE)) {
	    kgpu_log(KGPU_LOG_ERROR, "GPU buffer %p is not physically consecutive\n",
		mbuf->gb.addr);
	    return -EFAULT;
	}
#endif

	mbuf->npages = mbuf->gb.size/PAGE_SIZE;

	off += sizeof(struct gpu_buffer);
	if (!mbuf->paddrs)
	    mbuf->paddrs = kmalloc(sizeof(void*)*mbuf->npages, GFP_KERNEL);
	
	for (j=0; j<mbuf->npages; j++) {
	    mbuf->paddrs[j] =
		(void*)kgpu_virt2phy((unsigned long)(mbuf->gb.addr)
				     +j*PAGE_SIZE); 
	}

	mbuf->nunits = mbuf->npages/KGPU_BUF_NR_FRAMES_PER_UNIT;
	mbuf->bitmap = kmalloc(BITS_TO_LONGS(mbuf->nunits)*sizeof(long), GFP_KERNEL);
	bitmap_zero(mbuf->bitmap, mbuf->nunits);

	dbg("%p %p\n", mbuf->gb.addr, mbuf->paddrs[0]);
    }

    spin_unlock(&(kgpudev.buflock));
   
    return 0;
}

static int dump_gpu_bufs(char __user *buf)
{
    /* TODO dump kgpudev.mgmt_bufs' gb's to buf */
    return 0;
}

long kgpu_ioctl(struct file *filp,
	       unsigned int cmd, unsigned long arg)
{
    int err = 0;
    
    if (_IOC_TYPE(cmd) != KGPU_IOC_MAGIC)
	return -ENOTTY;
    if (_IOC_NR(cmd) > KGPU_IOC_MAXNR) return -ENOTTY;

    if (_IOC_DIR(cmd) & _IOC_READ)
	err = !access_ok(VERIFY_WRITE, (void __user *)arg, _IOC_SIZE(cmd));
    else if (_IOC_DIR(cmd) & _IOC_WRITE)
	err = !access_ok(VERIFY_READ, (void __user *)arg, _IOC_SIZE(cmd));
    if (err) return -EFAULT;

    switch (cmd) {
	
    case KGPU_IOC_SET_GPU_BUFS:
	err = set_gpu_bufs((char*)arg);
	break;
	
    case KGPU_IOC_GET_GPU_BUFS:
	err = dump_gpu_bufs((char*)arg);
	break;

    case KGPU_IOC_SET_STOP:
	/*TODO: stop all requests and this module */
	err = 0;
	break;

    default:
	err = -ENOTTY;
	break;
    }

    return err;
}

unsigned int kgpu_poll(struct file *filp, poll_table *wait)
{
    unsigned int mask = 0;
    
    spin_lock(&(kgpudev.reqlock));
    
    poll_wait(filp, &(kgpudev.reqq), wait);

    if (!list_empty(&(kgpudev.reqs))) 
	mask |= POLLIN | POLLRDNORM;

    mask |= POLLOUT | POLLWRNORM;

    spin_unlock(&(kgpudev.reqlock));

    return mask;
}

struct file_operations kgpu_ops =  {
    .owner          = THIS_MODULE,
    .read           = kgpu_read,
    .write          = kgpu_write,
    .poll           = kgpu_poll,
    .unlocked_ioctl = kgpu_ioctl,
    .open           = kgpu_open,
    .release        = kgpu_release,
};

int kgpu_init(void)
{
    int result;
    dev_t dev = 0;
    int devno;
    
    INIT_LIST_HEAD(&(kgpudev.reqs));
    INIT_LIST_HEAD(&(kgpudev.resps));
    INIT_LIST_HEAD(&(kgpudev.rtdreqs));
    
    spin_lock_init(&(kgpudev.reqlock));
    spin_lock_init(&(kgpudev.resplock));
    spin_lock_init(&(kgpudev.rtdreqlock));

    init_waitqueue_head(&(kgpudev.reqq));

    spin_lock_init(&(kgpudev.ridlock));
    spin_lock_init(&(kgpudev.buflock));

    kgpudev.rid_sequence = 0;

    kgpu_req_cache = kmem_cache_create(
	"kgpu_req_cache", sizeof(struct kgpu_req), 0,
	SLAB_HWCACHE_ALIGN, kgpu_req_constructor);
    if (!kgpu_req_cache) {
	kgpu_log(KGPU_LOG_ERROR, "can't create request cache\n");
	return -EFAULT;
    }
    kgpu_resp_cache = kmem_cache_create(
	"kgpu_resp_cache", sizeof(struct kgpu_resp), 0,
	SLAB_HWCACHE_ALIGN, kgpu_resp_constructor);
    if (!kgpu_resp_cache) {
	kgpu_log(KGPU_LOG_ERROR, "can't create response cache\n");
	kmem_cache_destroy(kgpu_req_cache);
	return -EFAULT;
    }

    kgpu_allocated_buf_cache = kmem_cache_create(
	"kgpu_allocated_buf_cache", sizeof(struct kgpu_allocated_buffer), 0,
	SLAB_HWCACHE_ALIGN, kgpu_allocated_buf_constructor);
    if (!kgpu_allocated_buf_cache) {
	kgpu_log(KGPU_LOG_ERROR, "can't create allocated buffer cache\n");
	kmem_cache_destroy(kgpu_allocated_buf_cache);
	return -EFAULT;
    }

    /* initialize buffer info */
    memset(kgpudev.mgmt_bufs, 0, sizeof(struct kgpu_mgmt_buffer)*KGPU_BUF_NR);

    /* alloc dev */	
    result = alloc_chrdev_region(&dev, 0, 1, KGPU_DEV_NAME);
    kgpu_major = MAJOR(dev);

    if (result < 0) {
        kgpu_log(KGPU_LOG_ERROR, "can't get major\n");
    } else {
	kgpu_log(KGPU_LOG_PRINT, "major %d\n", kgpu_major);
	devno = MKDEV(kgpu_major, 0);
	memset(&kgpudev.cdev, 0, sizeof(struct cdev));
	cdev_init(&kgpudev.cdev, &kgpu_ops);
	kgpudev.cdev.owner = THIS_MODULE;
	kgpudev.cdev.ops = &kgpu_ops;
	result = cdev_add(&kgpudev.cdev, devno, 1);
	if (result) {
	    kgpu_log(KGPU_LOG_ERROR, "can't add device %d", result);
	}
    }

    return result;
}
EXPORT_SYMBOL_GPL(kgpu_init);

void kgpu_cleanup(void)
{
    int i;
    dev_t devno = MKDEV(kgpu_major, 0);
    cdev_del(&kgpudev.cdev);

    unregister_chrdev_region(devno, 1);
    if (kgpu_req_cache)
	kmem_cache_destroy(kgpu_req_cache);
    if (kgpu_resp_cache)
	kmem_cache_destroy(kgpu_resp_cache);

    if (kgpu_allocated_buf_cache)
	kmem_cache_destroy(kgpu_allocated_buf_cache);

    /* clean up buffer info */
    for (i=0; i<KGPU_BUF_NR; i++) {
	kfree(kgpudev.mgmt_bufs[i].bitmap);
	kfree(kgpudev.mgmt_bufs[i].paddrs);
    }
    memset(kgpudev.mgmt_bufs, 0, sizeof(struct kgpu_mgmt_buffer)*KGPU_BUF_NR);
}
EXPORT_SYMBOL_GPL(kgpu_cleanup);

static int __init mod_init(void)
{
    kgpu_log(KGPU_LOG_PRINT, "KGPU loaded\n");
    return kgpu_init();
}

static void __exit mod_exit(void)
{
    kgpu_cleanup();
    kgpu_log(KGPU_LOG_PRINT, "KGPU unloaded\n");
}

module_init(mod_init);
module_exit(mod_exit);

MODULE_LICENSE("GPL");
