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


struct gpu_buffer {
    void *addr;
    unsigned long size;
};

#define gb.addr vaddr
#define gb.size bsize

struct kgpu_buffer {
    void *paddr;
    struct gpu_buffer gb;
};

#define KGPU_BUF_NR 4

static struct kgpu_buffer kgpu_bufs[KGPU_BUF_NR];


#define REQ_PROC_FILE "kgpureq"
#define RESP_PROC_FILE "kgpuresp"

static struct proc_dir_entry *kgpureqfs, *kgpurespfs;

struct kgpu_req;
struct kgpu_resp;

typedef int (*ku_callback)(struct kgpu_req *req,
			   struct kgpu_resp *resp);

struct ku_request {
    int id;
    int function;
    void *input;
    void *output;
    unsigned long insize;
    unsigned long outsize;
};

#define _req.id id
#define _req.function function
#define _req.input input
#define _req.output output
#define _req.insize insize
#define _req.outsize outsize

struct kgpu_req {
    struct list_head list;
    struct ku_request _req;
    struct kgpu_resp *resp;
    ku_callback cb;
};

struct ku_response {
    int id;
    int errno;
};

#define _resp.id id
#define _resp.errno errno

struct kgpu_resp {
    struct list_head list;
    struct ku_response _resp;
    struct kgpu_req *req;
};

static struct list_head reqs;
static struct list_head resps;
static spinlock_t reqlock;
static spinlock_t resplock;

static struct list_head rtdreqs;
static spinlock_t rtdreqlock;

static int reqfs_read(char *buf, char **bufloc,
		      off_t offset, int buflen,
		      int *eof, void *data)
{
    int ret;
    struct list_head *r;
    kgpu_req *req = NULL;
    ret = 0;

    spin_lock(&reqlock);

    if (!list_empty(&reqs)) {
	r = reqs.next;
	list_del(r);
	req = list_entry(r, kgpu_req, list);
	if (req) {
	    memcpy(buf, (char*)&(req->_req), sizeof(struct ku_request));
	    ret = sizeof(struct ku_request);
	}
    } else {
	ret = -EAGAIN;
    }

    spin_unlock(&reqlock);

    if (ret > 0 && req) {
	spin_lock(&rtdreqlock);

	INIT_LIST_HEAD(&req->list);
	list_add_tail(&req->list, &rtdreqs);

	spin_unlock(&rtdreqlock);
    }

    return ret;
}

static int reqfs_write(struct file *file, const char *buf,
		       unsigned long count, void *data)
{
}

static void init_queues()
{
    INIT_LIST_HEAD(&reqs);
    INIT_LIST_HEAD(&resps);
    spin_lock_init(&reqlock);
    spin_lock_init(&resplock);

    register_procfs();
}
