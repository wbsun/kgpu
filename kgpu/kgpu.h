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

#include <linux/ioctl.h>

struct gpu_buffer {
    void *addr;
    unsigned long size;
};

#define KGPU_BUF_NR 1
#define KGPU_BUF_SIZE (1024*1024*1024)

#define KGPU_DEV_NAME "kgpu"

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

#define KGPU_IOC_MAGIC 'g'

#define KGPU_IOC_SET_GPU_BUFS _IOW(KGPU_IOC_MAGIC, 1, struct gpu_buffer[KGPU_BUF_NR])
#define KGPU_IOC_GET_GPU_BUFS _IOR(KGPU_IOC_MAGIC, 2, struct gpu_buffer[KGPU_BUF_NR])
#define KGPU_IOC_SET_STOP     _IO(KGPU_IOC_MAGIC, 3)

#define KGPU_IOC_MAXNR 4


/* log stuff */
#define KGPU_LOG_INFO  1
#define KGPU_LOG_DEBUG 2
#define KGPU_LOG_ALERT 3
#define KGPU_LOG_ERROR 4
#define KGPU_LOG_PRINT 5

extern void kgpu_log(int level, const char *fmt, ...);
extern int kgpu_log_level;

/* shorthand for debug */
#define dbg(...) kgpu_log(KGPU_LOG_DEBUG, __VA_ARGS__)


#ifdef __KERNEL__

#include <linux/list.h>

/*
 * Kernel mode KGPU code
 */

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
