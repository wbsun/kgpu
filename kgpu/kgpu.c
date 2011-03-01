/*
 * KGPU k-u communication module.
 *
 * Weibin Sun
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
#include <linux/compiler.h>
#include <linux/wait.h>
#include <linux/cdev.h>
#include <linux/fs.h>
#include <linux/poll.h>
#include <asm/page.h>
#include <asm/page_types.h>
#include <asm/pgtable.h>
#include <asm/pgtable_types.h>
#include <asm/atomic.h>
#include "_kgpu.h"

#define KGPU_MAJOR 0

struct kgpu_dev {
    struct cdev cdev;

    struct kgpu_buffer bufs[KGPU_BUF_NR];
    int buf_uses[KGPU_BUF_NR];
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

static struct kgpu_buffer kgpu_bufs[KGPU_BUF_NR];
static int kgpu_buf_uses[KGPU_BUF_NR];
static spinlock_t buflock;

static int kgpu_rid_cnt = 0;
static spinlock_t ridlock;

static struct list_head reqs;
static struct list_head resps;
static spinlock_t reqlock;
static spinlock_t resplock;
static wait_queue_head_t reqq;

static struct list_head rtdreqs;
static spinlock_t rtdreqlock;

static struct cdev kgpudev;

static int kgpu_major;

static int bad_address(void *p)
{
    unsigned long dummy;
    return probe_kernel_address((unsigned long*)p, dummy);
}

/*
 * map any virtual address of the current process to its
 * physical one.
 */
static unsigned long kgpu_virt2phy(unsigned long vaddr)
{
    pgd_t *pgd = pgd_offset(current->mm, vaddr);
    pud_t *pud;
    pmd_t *pmd;
    pte_t *pte;

    /* to lock the page */
    struct page *pg;
    unsigned long paddr = 0;

    if (bad_address(pgd)) {
	printk(KERN_ALERT "[kgpu] Alert: bad address of pgd %p\n", pgd);
	goto bad;
    }
    if (pgd_none(*pgd) || pgd_bad(*pgd)) {
	printk(KERN_ALERT "[kgpu] Alert: pgd not present %lu\n", pgd_val(*pgd));
	goto out;
    }

    pud = pud_offset(pgd, vaddr);
    if (bad_address(pud)) {
	printk(KERN_ALERT "[kgpu] Alert: bad address of pud %p\n", pud);
	goto bad;
    }
    if (pud_none(*pud) || pud_bad(*pud)) {
	printk(KERN_ALERT "[kgpu] Alert: pud not present %lu\n", pud_val(*pud));
	goto out;
    }

    pmd = pmd_offset(pud, vaddr);
    if (bad_address(pmd)) {
	printk(KERN_ALERT "[kgpu] Alert: bad address of pmd %p\n", pmd);
	goto bad;
    }
    if (pmd_none(*pmd) || pmd_bad(*pmd)) {
	printk(KERN_ALERT "[kgpu] Alert: pmd not present %lu\n", pmd_val(*pmd));
	goto out;
    }

    pte = pte_offset_map/*kernel*/(pmd, vaddr);
    if (bad_address(pte)) {
	printk(KERN_ALERT "[kgpu] Alert: bad address of pte %p\n", pte);
	goto bad;
    }    
    if (!pte_present(*pte)) {
	printk(KERN_ALERT "[kgpu] Alert: pte not present %lu\n", pte_val(*pte));
	goto out;
    }

    pg = pte_page(*pte);
    paddr = (pte_val(*pte) & PHYSICAL_PAGE_MASK) | (vaddr&(PAGE_SIZE-1));

out:
    return paddr;
bad:
    printk(KERN_ALERT "[kgpu] Alert: Bad address\n");
    return 0;
}

int call_gpu(struct kgpu_req *req, struct kgpu_resp *resp)
{
    int dowake = 0;
    req->resp = resp;
    
    spin_lock(&reqlock);

    if (list_empty(&reqs))
	dowake = 1;

    INIT_LIST_HEAD(&req->list);
    list_add_tail(&req->list, &reqs);

    /*if (dowake)*/
	wake_up_interruptible(&reqq);
    
    spin_unlock(&reqlock);

    dbg("[kgpu] DEBUG: call gpu %d\n", req->kureq.id);
    
    return 0;
}
EXPORT_SYMBOL_GPL(call_gpu);

int next_kgpu_request_id(void)
{
    int rt = -1;
    
    spin_lock(&ridlock);
    
    kgpu_rid_cnt++;
    if (kgpu_rid_cnt < 0)
	kgpu_rid_cnt = 0;
    rt = kgpu_rid_cnt;
    
    spin_unlock(&ridlock);
    return rt;
}
EXPORT_SYMBOL_GPL(next_kgpu_request_id);

struct kgpu_req* alloc_kgpu_request(void)
{
    struct kgpu_req *req = kmalloc(sizeof(struct kgpu_req), GFP_KERNEL);
    if (req) {
	req->kureq.id = next_kgpu_request_id();
	INIT_LIST_HEAD(&req->list);
	req->kureq.sname[0] = 0;
    }
    return req;
}
EXPORT_SYMBOL_GPL(alloc_kgpu_request);

void free_kgpu_request(struct kgpu_req* req)
{
    kfree(req);
}
EXPORT_SYMBOL_GPL(free_kgpu_request);

struct kgpu_resp* alloc_kgpu_response(void)
{
    struct kgpu_resp *resp = kmalloc(sizeof(struct kgpu_resp), GFP_KERNEL);
    if (resp) {
	resp->kuresp.errcode = KGPU_NO_RESPONSE;
	INIT_LIST_HEAD(&resp->list);
    }
    return resp;
}
EXPORT_SYMBOL_GPL(alloc_kgpu_response);

void free_kgpu_response(struct kgpu_resp* resp)
{
    kfree(resp);
}
EXPORT_SYMBOL_GPL(free_kgpu_response);

struct kgpu_buffer* alloc_gpu_buffer(void)
{
    int i;
    spin_lock(&buflock);

    for (i=0; i<KGPU_BUF_NR; i++) {
	if (!kgpu_buf_uses[i]) {
	    kgpu_buf_uses[i] = 1;
	    spin_unlock(&buflock);
	    return &(kgpu_bufs[i]);
	}
    }

    spin_unlock(&buflock);
    return NULL;
}
EXPORT_SYMBOL_GPL(alloc_gpu_buffer);

int free_gpu_buffer(struct kgpu_buffer *buf)
{
    int i;

    spin_lock(&buflock);

    for (i=0; i<KGPU_BUF_NR; i++) {
	if (buf->paddr == kgpu_bufs[i].paddr) {
	    kgpu_buf_uses[i] = 0;
	    spin_unlock(&buflock);
	    return 0;
	}
    }

    spin_unlock(&buflock);
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

    spin_lock(&rtdreqlock);
    
    list_for_each_entry_safe(pos, n, &rtdreqs, list) {
	if (pos->kureq.id == id) {
	    if (offlist)
		list_del(&pos->list);
	    spin_unlock(&rtdreqlock);
	    return pos;
	}
    }

    spin_unlock(&rtdreqlock);

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

    spin_lock(&reqlock);
    while (list_empty(&reqs)) {
	spin_unlock(&reqlock);

	if (filp->f_flags & O_NONBLOCK)
	    return -EAGAIN;

	dbg("[kgpu] DEBUG: blocking read %s\n", current->comm);

	if (wait_event_interruptible(reqq, (!list_empty(&reqs))))
	    return -ERESTARTSYS;
	spin_lock(&reqlock);
    }

    r = reqs.next;
    list_del(r);
    req = list_entry(r, struct kgpu_req, list);
    if (req) {
	memcpy/*copy_to_user*/(buf, &req->kureq, sizeof(struct ku_request));
	ret = c;/*sizeof(struct ku_request);*/

	dbg("[kgpu] DEBUG: one request read %s %d %ld\n",
	    req->kureq.sname, req->kureq.id, ret);
    }

    spin_unlock(&reqlock);

    if (ret > 0 && req) {
	spin_lock(&rtdreqlock);

	INIT_LIST_HEAD(&req->list);
	list_add_tail(&req->list, &rtdreqs);

	spin_unlock(&rtdreqlock);
    }
    
    dbg("[kgpu] DEBUG: %s read %lu return %ld\n",
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

	dbg("[kgpu] DEBUG: response ID: %d\n", kuresp.id);

	req = find_request(kuresp.id, 1);
	if (!req)
	{	    
	    dbg("[kgpu] DEBUG: no request found for %d\n", kuresp.id);
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

    dbg("[kgpu] DEBUG: %s write %lu return %ld\n",
	current->comm, count, ret);

    return ret;
}


static int init_gpu_bufs(char __user *buf)
{
    int off=0, i;
    
    spin_lock(&buflock);

    for (i=0; i<KGPU_BUF_NR; i++) {
	copy_from_user(&(kgpu_bufs[i].gb), buf+off, sizeof(struct gpu_buffer));

	off += sizeof(struct gpu_buffer);
	kgpu_bufs[i].paddr =
	    (void*)kgpu_virt2phy((unsigned long)(kgpu_bufs[i].gb.addr));
	kgpu_buf_uses[i] = 0;

	dbg("[kgpu] DEBUG: %p %p %lu\n",
	    kgpu_bufs[i].gb.addr, kgpu_bufs[i].paddr, kgpu_bufs[i].gb.size);
    }

    spin_unlock(&buflock);
   
    return 0;
}

static int dump_gpu_bufs(char __user *buf)
{
    /* TODO dump kgpu_bufs' gb's to buf */
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
	err = init_gpu_bufs((char*)arg);
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
    
    spin_lock(&reqlock);
    
    poll_wait(filp, &reqq, wait);

    if (!list_empty(&reqs)) 
	mask |= POLLIN | POLLRDNORM;

    mask |= POLLOUT | POLLWRNORM;

    spin_unlock(&reqlock);

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
    
    INIT_LIST_HEAD(&reqs);
    INIT_LIST_HEAD(&resps);
    INIT_LIST_HEAD(&rtdreqs);
    
    spin_lock_init(&reqlock);
    spin_lock_init(&resplock);
    spin_lock_init(&rtdreqlock);

    init_waitqueue_head(&reqq);

    spin_lock_init(&ridlock);
    spin_lock_init(&buflock);

    result = alloc_chrdev_region(&dev, 0, 1, KGPU_DEV_NAME);
    kgpu_major = MAJOR(dev);

    if (result < 0) {
	printk("[kgpu] Error: can't get major\n");
    } else {
	printk("[kgpu] Info: major %d\n", kgpu_major);
	devno = MKDEV(kgpu_major, 0);
	memset(&kgpudev, 0, sizeof(struct cdev));
	cdev_init(&kgpudev, &kgpu_ops);
	kgpudev.owner = THIS_MODULE;
	kgpudev.ops = &kgpu_ops;
	result = cdev_add(&kgpudev, devno, 1);
	if (result) {
	    printk("[kgpu] Error: can't add device %d", result);
	}
    }

    return result;
}
EXPORT_SYMBOL_GPL(kgpu_init);

void kgpu_cleanup(void)
{
    dev_t devno = MKDEV(kgpu_major, 0);
    cdev_del(&kgpudev);

    unregister_chrdev_region(devno, 1);
}
EXPORT_SYMBOL_GPL(kgpu_cleanup);

static int __init mod_init(void)
{
    printk(KERN_INFO "[kgpu] KGPU loaded\n");
    return kgpu_init();
}

static void __exit mod_exit(void)
{
    kgpu_cleanup();
    printk(KERN_INFO "[kgpu] KGPU unloaded\n");
}

module_init(mod_init);
module_exit(mod_exit);

MODULE_LICENSE("GPL");
