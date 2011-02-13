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
#include <asm/uaccess.h>
#include <asm/page.h>
#include <asm/page_types.h>
#include <asm/pgtable.h>
#include <asm/pgtable_types.h>

struct gpu_buffer {
    void *addr;
    unsigned long size;
};


struct kgpu_buffer {
    void *paddr;
    struct gpu_buffer gb;
};

#define KGPU_BUF_NR 4

static struct kgpu_buffer kgpu_bufs[KGPU_BUF_NR];
static spinlock_t buflock;

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

struct kgpu_req {
    struct list_head list;
    struct ku_request kureq;
    struct kgpu_resp *resp;
    ku_callback cb;
};

struct ku_response {
    int id;
    int errno;
};

struct kgpu_resp {
    struct list_head list;
    struct ku_response kuresp;
    struct kgpu_req *req;
};

static struct list_head reqs;
static struct list_head resps;
static spinlock_t reqlock;
static spinlock_t resplock;

static struct list_head rtdreqs;
static spinlock_t rtdreqlock;


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
    unsigned long paddr;

    if (bad_address(pgd)) {
	printk(KERN_ALERT "[kgpu] Alert: bad address of pgd %p\n", pgd);
	goto bad;
    }
    if (!pgd_present(*pgd)) {
	printk(KERN_ALERT "[kgpu] Alert: pgd not present %lu\n", *pgd);
	goto out;
    }

    pud = pud_offset(pgd, vaddr);
    if (bad_address(pud)) {
	printk(KERN_ALERT "[kgpu] Alert: bad address of pud %p\n", pud);
	goto bad;
    }
    if (!pud_present(*pud) || pud_large(*pud)) {
	printk(KERN_ALERT "[kgpu] Alert: pud not present %lu\n", *pud);
	goto out;
    }

    pmd = pmd_offset(pud, vaddr);
    if (bad_address(pmd)) {
	printk(KERN_ALERT "[kgpu] Alert: bad address of pmd %p\n", pmd);
	goto bad;
    }
    if (!pmd_present(*pmd) || pmd_large(*pmd)) {
	printk(KERN_ALERT "[kgpu] Alert: pmd not present %lu\n", *md);
	goto out;
    }

    pte = pte_offset_kernel(pmd, vaddr);
    if (bad_address(pte)) {
	printk(KERN_ALERT "[kgpu] Alert: bad address of pte %p\n", pte);
	goto bad;
    }    
    if (!pte_present(*pte)) {
	printk(KERN_ALERT "[kgpu] Alert: pte not present %lu\n", *pte);
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

/*
 * Userspace reads a request.
 */
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
	    copy_to_user(buf, (char*)&(req->kureq), sizeof(struct ku_request));
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

/*
 * Userspace tells kernel the GPU buffers
 */
static int reqfs_write(struct file *file, const char *buf,
		       unsigned long count, void *data)
{
    int off = 0;
    int i;
    
    if (count < KGPU_BUF_NR*sizeof(struct gpu_buffer))
	return -EINVAL; /* too small */
    else
	count = KGPU_BUF_NR*sizeof(struct gpu_buffer);
    
    spin_lock(&buflock);

    for (i=0; i<KGPU_BUF_NR; i++) {
	copy_from_user(&(kgpu_bufs[i].gb), buf+offset, sizeof(struct gpu_buffer));
	offset += sizeof(struct gpu_buffer);
	kgpu_bufs[i].paddr = kgpu_virt2phy((unsigned long)(kgpu_bufs[i].gb.addr));
    }

    spin_unlock(&buflock);

    return count;
}

/*
 * find request by id in the rtdreqs
 * offlist = 1: remove the request from the list
 * offlist = 0: keep the request in the list
 */
static struct kgpu_req* find_request(int id, int offlist)
{
    struct kgpu_req *pos, *n;

    spin_lock(&rtdreqs);
    
    list_for_each_entry_safe(pos, n, &rtdreqs, list) {
	if (pos->id == id) {
	    if (offlist)
		list_del(pos->list);
	    spin_unlock(&rtdreqs);
	    return pos;
	}
    }

    spin_unlock(&rtdreqs);

    return NULL;
}

/*
 * Userspace sends response to kernel
 */
static int respfs_write(struct file *file, const char *buf,
			unsigned long count, void *data)
{
    int ret;
    struct ku_response kuresp;
    struct kgpu_req *req;
    
    if (count < sizeof(struct ku_response))
	return -EINVAL; /* Too small. */
    else
	count = sizeof(struct ku_response);

    copy_from_user(&kuresp, buf, count);

    req = find_request(kuresp.id, 1);
    if (!req)
	return -EFAULT; /* no request found */
    
    memcpy(&(req->resp->kuresp), &kuresp, count);

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

    return count;
}    

static void init_queues()
{
    INIT_LIST_HEAD(&reqs);
    INIT_LIST_HEAD(&resps);
    spin_lock_init(&reqlock);
    spin_lock_init(&resplock);

    register_procfs();
}
