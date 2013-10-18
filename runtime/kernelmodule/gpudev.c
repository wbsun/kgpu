/*
 * This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2012 University of Utah and the Flux Group.
 * All rights reserved.
 */

#include "kgpu.h"
#include "krt.h"

static struct gpudev_t {
	struct cdev cdev;
	struct class *cls;
	dev_t devno;
	int state;	
} gpudev;

static atomic_t gd_av = ATOMIC_INIT(1);

static int gd_open(struct inode *inode, struct file *filp)
{
	if (!atomic_dec_and_test(&gd_av)) {
		atomic_inc(&gd_av);
		return -EBUSY;
	}

	filp->private_data = &gpudev;
	return 0;
}

int kgpu_release(struct inode *inode, struct file *file)
{
	atomic_set(&kgpudev_av, 1);
	return 0;
}

static void fill_ku_request(struct kgpu_ku_request *kureq,
			    struct kgpu_request *req)
{
	kureq->id = req->id;
	memcpy(kureq->service_name, req->service_name, KGPU_SERVICE_NAME_SIZE);

	if (ADDR_WITHIN(req->in, kgpudev.gmpool.kva,
			kgpudev.gmpool.npages<<PAGE_SHIFT)) {
		kureq->in = (void*)ADDR_REBASE(kgpudev.gmpool.uva,
					       kgpudev.gmpool.kva,
					       req->in);
	} else {
		kureq->in = req->in;
	}

	if (ADDR_WITHIN(req->out, kgpudev.gmpool.kva,
			kgpudev.gmpool.npages<<PAGE_SHIFT)) {
		kureq->out = (void*)ADDR_REBASE(kgpudev.gmpool.uva,
						kgpudev.gmpool.kva,
						req->out);
	} else {
		kureq->out = req->out;
	}

	if (ADDR_WITHIN(req->udata, kgpudev.gmpool.kva,
			kgpudev.gmpool.npages<<PAGE_SHIFT)) {
		kureq->data = (void*)ADDR_REBASE(kgpudev.gmpool.uva,
						 kgpudev.gmpool.kva,
						 req->udata);
	} else {
		kureq->data = req->udata;
	}
    
	kureq->insize = req->insize;
	kureq->outsize = req->outsize;
	kureq->datasize = req->udatasize;
}

ssize_t kgpu_read(
	struct file *filp, char __user *buf, size_t c, loff_t *fpos)
{
	ssize_t ret = 0;
	struct list_head *r;
	struct _kgpu_request_item *item;

	spin_lock(&(kgpudev.reqlock));
	while (list_empty(&(kgpudev.reqs))) {
		spin_unlock(&(kgpudev.reqlock));

		if (filp->f_flags & O_NONBLOCK)
			return -EAGAIN;

		if (wait_event_interruptible(
			    kgpudev.reqq, (!list_empty(&(kgpudev.reqs)))))
			return -ERESTARTSYS;
		spin_lock(&(kgpudev.reqlock));
	}

	r = kgpudev.reqs.next;
	list_del(r);
	item = list_entry(r, struct _kgpu_request_item, list);
	if (item) {
		struct kgpu_ku_request kureq;
		fill_ku_request(&kureq, item->r);
	
		memcpy(buf, &kureq, sizeof(struct kgpu_ku_request));
		ret = c;
	}

	spin_unlock(&(kgpudev.reqlock));

	if (ret > 0 && item) {
		spin_lock(&(kgpudev.rtdreqlock));

		INIT_LIST_HEAD(&item->list);
		list_add_tail(&item->list, &(kgpudev.rtdreqs));

		spin_unlock(&(kgpudev.rtdreqlock));
	}
    
	*fpos += ret;

	return ret;    
}

ssize_t kgpu_write(struct file *filp, const char __user *buf,
		   size_t count, loff_t *fpos)
{
	struct kgpu_ku_response kuresp;
	struct _kgpu_request_item *item;
	ssize_t ret = 0;
	size_t  realcount;
    
	if (count < sizeof(struct kgpu_ku_response))
		ret = -EINVAL; /* Too small. */
	else
	{
		realcount = sizeof(struct kgpu_ku_response);

		memcpy/*copy_from_user*/(&kuresp, buf, realcount);

		item = find_request(kuresp.id, 1);
		if (!item)
		{	    
			ret = -EFAULT; /* no request found */
		} else {
			item->r->errcode = kuresp.errcode;
			if (unlikely(kuresp.errcode != 0)) {
				switch(kuresp.errcode) {
				case KGPU_NO_RESPONSE:
					kgpu_log(KGPU_LOG_ALERT,
						 "userspace helper doesn't give any response\n");
					break;
				case KGPU_NO_SERVICE:
					kgpu_log(KGPU_LOG_ALERT,
						 "no such service %s\n",
						 item->r->service_name);
					break;
				case KGPU_TERMINATED:
					kgpu_log(KGPU_LOG_ALERT,
						 "request is terminated\n"
						);
					break;
				default:
					kgpu_log(KGPU_LOG_ALERT,
						 "unknown error with code %d\n",
						 kuresp.id);
					break;		    
				}
			}

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
			item->r->callback(item->r);
			ret = count;/*realcount;*/
			*fpos += ret;
			kmem_cache_free(kgpu_request_item_cache, item);
		}
	}

	return ret;
}

static int clear_gpu_mempool(void)
{
	struct _kgpu_mempool *gmp = &kgpudev.gmpool;

	spin_lock(&kgpudev.gmpool_lock);
	if (gmp->pages)
		kfree(gmp->pages);
	if (gmp->bitmap)
		kfree(gmp->bitmap);
	if (gmp->alloc_sz)
		kfree(gmp->alloc_sz);

	vunmap((void*)gmp->kva);
	spin_unlock(&kgpudev.gmpool_lock);
	return 0;
}

static int set_gpu_mempool(char __user *buf)
{
	struct kgpu_gpu_mem_info gb;
	struct _kgpu_mempool *gmp = &kgpudev.gmpool;
	int i;
	int err=0;

	spin_lock(&(kgpudev.gmpool_lock));
    
	copy_from_user(&gb, buf, sizeof(struct kgpu_gpu_mem_info));

	/* set up pages mem */
	gmp->uva = (unsigned long)(gb.uva);
	gmp->npages = gb.size/PAGE_SIZE;
	if (!gmp->pages) {
		gmp->pages = kmalloc(sizeof(struct page*)*gmp->npages, GFP_KERNEL);
		if (!gmp->pages) {
			kgpu_log(KGPU_LOG_ERROR, "run out of memory for gmp pages\n");
			err = -ENOMEM;
			goto unlock_and_out;
		}
	}

	for (i=0; i<gmp->npages; i++)
		gmp->pages[i]= kgpu_v2page(
			(unsigned long)(gb.uva) + i*PAGE_SIZE
			);

	/* set up bitmap */
	gmp->nunits = gmp->npages/KGPU_BUF_NR_FRAMES_PER_UNIT;
	if (!gmp->bitmap) {
		gmp->bitmap = kmalloc(
			BITS_TO_LONGS(gmp->nunits)*sizeof(long), GFP_KERNEL);
		if (!gmp->bitmap) {
			kgpu_log(KGPU_LOG_ERROR, "run out of memory for gmp bitmap\n");
			err = -ENOMEM;
			goto unlock_and_out;
		}
	}    
	bitmap_zero(gmp->bitmap, gmp->nunits);

	/* set up allocated memory sizes */
	if (!gmp->alloc_sz) {
		gmp->alloc_sz = kmalloc(
			gmp->nunits*sizeof(u32), GFP_KERNEL);
		if (!gmp->alloc_sz) {
			kgpu_log(KGPU_LOG_ERROR,
				 "run out of memory for gmp alloc_sz\n");
			err = -ENOMEM;
			goto unlock_and_out;
		}
	}
	memset(gmp->alloc_sz, 0, gmp->nunits);

	/* set up kernel remapping */
	gmp->kva = (unsigned long)vmap(
		gmp->pages, gmp->npages, GFP_KERNEL, PAGE_KERNEL);
	if (!gmp->kva) {
		kgpu_log(KGPU_LOG_ERROR, "map pages into kernel failed\n");
		err = -EFAULT;
		goto unlock_and_out;
	}
    
unlock_and_out:
	spin_unlock(&(kgpudev.gmpool_lock));    

	return err;
}

static int dump_gpu_bufs(char __user *buf)
{
	/* TODO: dump gmpool's info to buf */
	return 0;
}

static int terminate_all_requests(void)
{
	/* TODO: stop receiving requests, set all reqeusts code to
	   KGPU_TERMINATED and call their callbacks */
	kgpudev.state = KGPU_TERMINATED;
	return 0;
}

static long kgpu_ioctl(struct file *filp,
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
		err = set_gpu_mempool((char*)arg);
		break;
	
	case KGPU_IOC_GET_GPU_BUFS:
		err = dump_gpu_bufs((char*)arg);
		break;

	case KGPU_IOC_SET_STOP:
		err = terminate_all_requests();
		break;

	default:
		err = -ENOTTY;
		break;
	}

	return err;
}

static unsigned int kgpu_poll(struct file *filp, poll_table *wait)
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

static void kgpu_vm_open(struct vm_area_struct *vma)
{
	// just let it go
}

static void kgpu_vm_close(struct vm_area_struct *vma)
{
	// nothing we can do now
}

static int kgpu_vm_fault(struct vm_area_struct *vma,
			 struct vm_fault *vmf)
{
	/*static struct page *p = NULL;
	  if (!p) {
	  p = alloc_page(GFP_KERNEL);
	  if (p) {
	  kgpu_log(KGPU_LOG_PRINT,
	  "first page fault, 0x%lX (0x%lX)\n",
	  TO_UL(vmf->virtual_address),
	  TO_UL(vma->vm_start));
	  vmf->page = p;
	  return 0;
	  } else {
	  kgpu_log(KGPU_LOG_ERROR,
	  "first page fault, kgpu mmap no page ");
	  }
	  }*/
	/* should never call this */
	kgpu_log(KGPU_LOG_ERROR,
		 "kgpu mmap area being accessed without pre-mapping 0x%lX (0x%lX)\n",
		 (unsigned long)vmf->virtual_address,
		 (unsigned long)vma->vm_start);
	vmf->flags |= VM_FAULT_NOPAGE|VM_FAULT_ERROR;
	return VM_FAULT_SIGBUS;
}

static struct vm_operations_struct kgpu_vm_ops = {
	.open  = kgpu_vm_open,
	.close = kgpu_vm_close,
	.fault = kgpu_vm_fault,
};

static void set_vm(struct vm_area_struct *vma)
{
	kgpudev.vm.vma = vma;
	kgpudev.vm.start = vma->vm_start;
	kgpudev.vm.end = vma->vm_end;
	kgpudev.vm.npages = (vma->vm_end - vma->vm_start)>>PAGE_SHIFT;
	kgpudev.vm.alloc_sz = kmalloc(
		sizeof(u32)*kgpudev.vm.npages, GFP_KERNEL);
	kgpudev.vm.bitmap = kmalloc(
		BITS_TO_LONGS(kgpudev.vm.npages)*sizeof(long), GFP_KERNEL);
	if (!kgpudev.vm.alloc_sz || !kgpudev.vm.bitmap) {
		kgpu_log(KGPU_LOG_ERROR,
			 "out of memory for vm's bitmap and records\n");
		if (kgpudev.vm.alloc_sz) kfree(kgpudev.vm.alloc_sz);
		if (kgpudev.vm.bitmap) kfree(kgpudev.vm.bitmap);
		kgpudev.vm.alloc_sz = NULL;
		kgpudev.vm.bitmap = NULL;
	};
	bitmap_zero(kgpudev.vm.bitmap, kgpudev.vm.npages);
	memset(kgpudev.vm.alloc_sz, 0, sizeof(u32)*kgpudev.vm.npages);
}

static void clean_vm(void)
{
	if (kgpudev.vm.alloc_sz)
		kfree(kgpudev.vm.alloc_sz);
	if (kgpudev.vm.bitmap)
		kfree(kgpudev.vm.bitmap);
}

static int kgpu_mmap(struct file *filp, struct vm_area_struct *vma)
{
	if (vma->vm_end - vma->vm_start != KGPU_MMAP_SIZE) {
		kgpu_log(KGPU_LOG_ALERT,
			 "mmap size incorrect from 0x$lX to 0x%lX with "
			 "%lu bytes\n", vma->vm_start, vma->vm_end,
			 vma->vm_end-vma->vm_start);
		return -EINVAL;
	}
	vma->vm_ops = &kgpu_vm_ops;
	vma->vm_flags |= VM_RESERVED;
	set_vm(vma);
	return 0;
}

static struct file_operations gpudev_ops =  {
	.owner          = THIS_MODULE,
	.read           = gd_read,
	.write          = gd_write,
	.poll           = gd_poll,
	.unlocked_ioctl = gd_ioctl,
	.open           = gd_open,
	.release        = gd_release,
	.mmap           = gd_mmap,
};

