/*
 * This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2012 University of Utah and the Flux Group.
 * All rights reserved.
 */
#include "kgpu.h"

#define COMM_DEV_READ_SIZE    1
#define COMM_DEV_READ_REQUEST 2

typedef struct commdev_t 
{
	struct cdev cdev;
	struct class *cls;
	dev_t devno;
	atomic_t available;
	ku_request_t *rbuf;
	u32 rbufsz;
	int state;
} commdev_t;

static commdev_t commdev;

static void build_ku_request(k_request_t *r)
{
	kg_k_service_t *ks;
	
	commdev.rbuf->size = sizeof(ku_request_t);
	commdev.rbuf->id = r->id;
	ks = get_kservice(r->orig->service_name);
	if (ks) {
		commdev.rbuf->s = ks->usrv;
	} else {
		commdev.rubf->s = NULL;
	}
	commdev.rbuf->hin = r->hin;
	commdev.rbuf->din = r->din;
	commdev.rbuf->hout = r->hout;
	commdev.rbuf->dout = r->dout;
	commdev.rbuf->hdata = r->hdata;
	commdev.rbuf->ddata = r->ddata;
	commdev.rbuf->insize = r->insize;
	commdev.rbuf->outsize = r->outsize;
	commdev.rbuf->datasize = r->datasize;
	
	commdev.rbuf->memflags = r->memflags;
	commdev.rbuf->device = r->device;
	commdev.rbuf->stream_idx = r->stream_idx;
	commdev.rbuf->start_time = r->start_time;
	commdev.rbuf->schedflags = r->schedflags;
	commdev.rbuf->segidx = r->segidx;
}

static void request_done(ku_response_t *resp)
{
	k_request_t *req;
	
	req = get_krequest(resp->id);
	if (!req) {
		krt_log(KGPU_LOG_ERROR, 
			"Can't find request by response ID: %d\n",
			resp->id);
	} else {
		krequest_done(req);
	}
}

static int cd_open(struct inode *inode, struct file *filp)
{
	if (!atomic_dec_and_test(&commdev.available)) {
		atomic_inc(&commdev.available);
		return -EBUSY;
	}

	filp->private_data = &commdev;
	return 0;
}

static int cd_release(struct inode *inode, struct file *file)
{
	atomic_set(&commdev.available, 1);
	return 0;
}

/* Read policy:
 *   Process: When reqeusts available, first of all, sizeof(u32) bytes will
 *            be read, then the request body will be read according to the
 *            size read before. 
 *            As a result, to read a request, two calls of read() are needed,
 *            first read() uses sizeof(u32) bytes buf to read the request size
 *            s, second read() uses s bytes buf to read the body.
 *    Notes:  When poll() ready, it means the two read() calls can be performed
 *            safely.
 *            Read() always blocks when reading the size of request. If size is
 *            available, request body must be available too.
 */
static ssize_t 
cd_read(struct file *filp, char __user *buf, size_t c, loff_t *fpos)
{
	ssize_t ret = 0;
	
	if (commdev.state == COMM_DEV_READ_SIZE) {
		k_request_t *r = NULL;
		while (!has_krequest()) {
			if (flip->f_flags & O_NONBLOCK)
				return -EAGAIN;
			if (wait_for_krequest())
				return -ERESTARTSYS;
		}
	
		r = fetch_next_krequest();
		build_ku_request(r);
		copy_to_user(
			buf, &commdev.rbuf->size, sizeof(commdev.rbuf->size));
		ret = c; /* sizeof(commdev.rbuf->size); */
		commdev.state = COMM_DEV_READ_REQUEST;
	} else {
		copy_to_user(buf, commdev.rbuf, commdev.rbuf->size);
		ret = c; /* commdev.rbuf->size; */
		commdev.state = COMM_DEV_READ_SIZE;
	}
	
	*fpos += ret;	
	return ret;
}

static ssize_t
cd_write(struct file *filp, const char __user *buf, size_t count, loff_t *fpos)
{
	struct ku_response_t resp;
	ssize_t ret = 0;
	
	if (count < sizeof(ku_response_t))
		ret = -EINVAL;
	else {
		copy_from_user(&resp, buf, sizeof(ku_response_t));
		request_done(&resp);
		ret = count;
		*fpos += ret;
	}
	
	return ret;
}

static long cd_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
	int err = 0;
	
	/* routing checks: */
	if (_IOC_TYPE(cmd) != KGPU_IOC_MAGIC)
		return -ENOTTY;
	if (_IOC_NR(cmd) > KGPU_IOC_MAXNR) return -ENOTTY;

	if (_IOC_DIR(cmd) & _IOC_READ)
		err = !access_ok(
			VERIFY_WRITE, (void __user *)arg, _IOC_SIZE(cmd));
	else if (_IOC_DIR(cmd) & _IOC_WRITE)
		err = !access_ok(
			VERIFY_READ, (void __user *)arg, _IOC_SIZE(cmd));
	if (err) return -EFAULT;
	
	switch (cmd) {
	case KGPU_IOC_SET_HOST_ALLOC_BUF:
		err = mm_set_host_alloc_buf((ku_meminfo_t*)arg);
		break;
	case KGPU_IOC_SET_HOST_MAP_BUF:
		err = mm_set_host_map_buf((ku_meminfo_t*)arg);
		break;
	case KGPU_IOC_SET_NR_GPU:
		err = set_nr_gpu((int*)arg);
		break;
	case KGPU_IOC_SET_DEV_BUF:
		err = mm_set_dev_buf((ku_devmeminfo_t*)arg);
		break;
	case KGPU_IOC_SET_USERVICE_INFO:
		err = set_uservice_info((ku_serviceinfo_t*)arg);
		break;
	case KGPU_IOC_SET_STOP:
		err = krt_stop();
		break;
	default:
		err = -ENOTTY;
		break;	
	}
	
	return err;
}

static unsigned int cd_poll(struct file *filp, poll_table *wait)
{
	unsigned int mask = 0;
	
	/* Check state first, because has_krequest needs lock */
	if (commdev.state == COMM_DEV_READ_REQUEST)
		mask |= POLLIN | POLLRDNORM;
	else if (has_krequest())
		mask |= POLLIN | POLLRDNORM;
	mask |= POLLOUT | POLLWRNORM;
	return mask;
}

/* MMAP support */
static  void cd_vm_open(struct vm_area_struct *vma)
{
	// nothing todo
}
static void cd_vm_close(struct vm_area_struct *vma)
{
	// nothing todo
}

/* mmap-ed memory area for k/u communication, things like stream states,
 * status updates and so on and so forth.
 */
static int cd_vm_fault(struct vm_area_struct *vma, struct vm_fault *vmf)
{
	krt_log(KGPU_LOG_ERROR, "mmap used but not supported yet!\n");
	vmf->flags |= VM_FAULT_NOPAGE | VM_FAULT_ERROR;
	return VM_FAULT_SIGBUS;
}

static struct vm_operations_struct cd_vm_ops = {
	.open  = cd_vm_open,
	.close = cd_vm_close,
	.fault = cd_vm_fault,
};

static int cd_mmap(struct file *filp, struct vm_area_struct *vma)
{
	if (vma->vm_end - vma->vm_start != KGPU_MMAP_SIZE) {
		krt_log(KGPU_LOG_ALERT,
			 "mmap size incorrect from 0x$lX to 0x%lX with "
			 "%lu bytes\n", vma->vm_start, vma->vm_end,
			 vma->vm_end-vma->vm_start);
		return -EINVAL;
	}
	vma->vm_ops = &cd_vm_ops;
	vma->vm_flags |= VM_RESERVED;
	set_vm(vma); // TODO: fix this by writing set_vm or similar
	return 0;
}

static struct file_operations cd_ops =  {
	.owner          = THIS_MODULE,
	.read           = cd_read,
	.write          = cd_write,
	.poll           = cd_poll,
	.unlocked_ioctl = cd_ioctl,
	.open           = cd_open,
	.release        = cd_release,
	.mmap           = cd_mmap,
};

int kucomm_init(void)
{
	int devno;
	int result = 0;
	
	commdev.state = COMM_DEV_READ_SIZE;
	commdev.available = ATOMIC_INIT(1);
	
	commdev.cls = class_create(THIS_MODULE, "KGPU_DEV_NAME");
	if (IS_ERR(commdev.cls)) {
		result = PTR_ERR(commdev.cls);
		krt_log(KGPU_LOG_ERROR, "Failed to create class for kucomm.\n");
		return result;
	}
	
	result = alloc_chrdev_region(&commdev.devno, 0, 1, KGPU_DEV_NAME);
	devno = MAJOR(commdev.devno);
	commdev.devno = MKDEV(devno, 0);
	
	if (result < 0) {
		krt_log(KGPU_LOG_ERROR, "Can't get device major code\n");
	} else {
		struct device *d;
		memset(&commdev.cdev, 0, sizeof(struct cdev));
		cdev_init(&commdev.cdev, &cd_ops);
		commdev.cdev.owner = THIS_MODULE;
		commdev.cdev.ops = &cd_ops;
		result = cdev_add(&commdev.cdev, commdev.devno, 1);
		if (result) {
			krt_log(KGPU_LOG_ERROR, 
				"Can't add kucomm device %d\n",
				result);
		} else {		
			d = device_create(
				commdev.cls, NULL, commdev.devno, NULL,
				KGPU_DEV_NAME);
			if (IS_ERR(d)) {
				krt_log(KGPU_LOG_ERROR, 
					"Failed to create kucomm device\n");
				result = PTR_ERR(d);
			} else {
				/* Put mem allocation at the end so that
				 * we don't need to free the allocated mem
				 * if something fails after allocation
				 */
				commdev.rbuf = kmalloc(PAGE_SIZE, GFP_KERNEL);
				if (!commdev.rbuf) {
					krt_log(KGPU_LOG_ERROR,
						"Out of memory when initializing kucomm\n");
					return -ENOMEM;
				}
				commdev.rbufsz = PAGE_SIZE;
			}
		}
		
	}
	return result;
}

void kucomm_finit(void)
{
	device_destroy(commdev.cls, commdev.devno);
	cdev_del(&commdev.cdev);
	class_destroy(commdev.cls);
	
	unregister_chrdev_region(commdev.devno, 1);
	kfree(commdev.rbuf);
}





















