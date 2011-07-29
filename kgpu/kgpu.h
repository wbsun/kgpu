/*
 * This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2010-2011 University of Utah and the Flux Group.
 * All rights reserved.
 *
 * Common header for userspace helper, kernel mode KGPU and KGPU clients
 *
 */

#ifndef __KGPU_H__
#define __KGPU_H__

struct gpu_buffer {
    void *addr;
    unsigned long size;
};

#define SERVICE_NAME_SIZE 32

struct ku_request {
    int id;
    char sname[SERVICE_NAME_SIZE];
    void *input;
    void *output;
    unsigned long insize;
    unsigned long outsize;
    void *data;
    unsigned long datasize;
};

/* kgpu's errno */
#define KGPU_OK 0
#define KGPU_NO_RESPONSE 1
#define KGPU_NO_SERVICE 2

struct ku_response {
    int id;
    int errcode;
};

/*
 * Only for kernel code or helper
 */
#if defined __KERNEL__ || defined __KGPU__

#define KGPU_BUF_NR 1
#define KGPU_BUF_SIZE (1024*1024*1024)

#define KGPU_DEV_NAME "kgpu"

/* ioctl */
#include <linux/ioctl.h>

#define KGPU_IOC_MAGIC 'g'

#define KGPU_IOC_SET_GPU_BUFS _IOW(KGPU_IOC_MAGIC, 1, struct gpu_buffer[KGPU_BUF_NR])
#define KGPU_IOC_GET_GPU_BUFS _IOR(KGPU_IOC_MAGIC, 2, struct gpu_buffer[KGPU_BUF_NR])
#define KGPU_IOC_SET_STOP     _IO(KGPU_IOC_MAGIC, 3)
#define KGPU_IOC_GET_REQS     _IOR(KGPU_IOC_MAGIC, 4, 

#define KGPU_IOC_MAXNR 4

#include "kgpu_log.h"

#endif /* __KERNEL__ || __KGPU__  */

/*
 * For helper and service providers
 */
#ifndef __KERNEL__

struct service;

struct service_request {
    struct ku_request kureq;
    struct service *s;
    int block_x, block_y;
    int grid_x, grid_y;
    int state;
    int errcode;
    int stream_id;
    unsigned long stream;
    void *dinput;
    void *doutput;
    void *data;
};

/* service request states: */
#define REQ_INIT 1
#define REQ_MEM_DONE 2
#define REQ_PREPARED 3
#define REQ_RUNNING 4
#define REQ_POST_EXEC 5
#define REQ_DONE 6

#include "service.h"

#endif /* no __KERNEL__ */

/*
 * For kernel code only
 */
#ifdef __KERNEL__

#include <linux/list.h>

struct kgpu_buffer {
    void *va;
    void **pas;
    unsigned int npages;
};

struct kgpu_req;
struct kgpu_resp;

typedef int (*ku_callback)(struct kgpu_req *req,
			   struct kgpu_resp *resp);

struct kgpu_req {
    struct list_head list;
    struct ku_request kureq;
    struct kgpu_resp *resp;
    ku_callback cb;
    void *data;
};

struct kgpu_resp {
    struct list_head list;
    struct ku_response kuresp;
    struct kgpu_req *req;
};

extern int call_gpu(struct kgpu_req*, struct kgpu_resp*);
extern int call_gpu_sync(struct kgpu_req*, struct kgpu_resp*);
extern int next_kgpu_request_id(void);
extern struct kgpu_req* alloc_kgpu_request(void);
extern struct kgpu_resp* alloc_kgpu_response(void);
extern struct kgpu_buffer* alloc_gpu_buffer(unsigned long nbytes);
extern int free_gpu_buffer(struct kgpu_buffer *);
extern void free_kgpu_response(struct kgpu_resp*);
extern void free_kgpu_request(struct kgpu_req*);

#endif /* __KERNEL__ */

#endif
