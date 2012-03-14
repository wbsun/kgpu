/*
 * This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2012 University of Utah and the Flux Group.
 * All rights reserved.
 *
 * Common header for userspace helper, kernel mode KGPU and KGPU clients
 *
 */
#ifndef __KGPU_H__
#define __KGPU_H__

#include "utils.h"

#define KGPU_SERVICE_NAME_SIZE 32

/* KGPU's errno */
#define KGPU_OK 0
#define KGPU_NO_RESPONSE 1
#define KGPU_NO_SERVICE 2
#define KGPU_TERMINATED 3

#define KGPU_DEV_NAME "kgpu"

/* Level of dependence  */
#define KGPU_DEP_ORDER 0   /* Just in order */
#define KGPU_DEP_DEVICE 1  /* Must be on same device + level ORDER */
#define KGPU_DEP_DATA 2    /* Shared data + level DEVICE */

/* Request type: */
#define KGPU_NORMAL_REQ 0
#define KGPU_DUMB_REQ   1

/* Special dependence IDs: */
#define KGPU_ID_ANY 0
#define KGPU_ID_ALLBEFORE ((int)(0x7fffffff))
#define KGPU_ID_ALLAFTER  ((int)(0x7ffffffe))
#define KGPU_ID_KMAX ((int)(0x7ffffffd));
#define KGPU_ID_UMAX ((int)(0x80000000));

/* Request struct and related ones used by kernel space: */
struct kg_request_t;

typedef int (*kg_request_callback_t)(struct kg_request_t*);

typedef struct kg_request_t {
	u32 size;
	int id;
	char service_name[KGPU_SERVICE_NAME_SIZE];

	u64 in, out, data;
	u64 insize, outsize, datasize;
	
	int *depon, *depby;
	u32 nrdepon, nrdepby;
	u8 deplevel;
    
	s8 device;

	u8 splittable;
	u8 mergeable;

	u64 start_time;

	void* cbdata;
	kg_request_callback_t callback;
} kg_request_t;

extern kg_request_t* kg_new_request(u32 ndepon, u32 ndepby);
extern void kg_free_request(kg_request_t* r);
extern int kg_request_id(void);

extern int kg_make_request(kg_request_t* r, int sync);

extern void* kg_vmalloc(u64 nbytes);
extern void  kg_vfree(void* p);

/* Request data passed through k/u boundary: */
typedef struct ku_request_t {
	u32 size;
	int id;
	char service_name[KGPU_SERVICE_NAME_SIZE];

	u64 in, out, data;
	u64 insize, outsize, datasize;

	int *depon, *depby;
	u32 nrdepon, nrdepby;
	u8 deplevel;

	s8 device;

	u64 start_time;

    	int parent;
	int *children;
	u32 nrchildren;

	u8 splitted;
	u8 merged;
} ku_request_t;
/* Hidden data:
 * ----------------------
 * |  ku_request_t  |  |
 * |----------------|  |
 * |     depon      |  V
 * |----------------| size
 * |     depby      |  ^
 * |----------------|  |
 * |    children    |  |
 * ----------------------
 */

#define KGPU_NON_SCHED  0
#define KGPU_SPLIT_SRC  1
#define KGPU_SPLIT_PROD 2
#define KGPU_MERGE_SRC  3
#define KGPU_MERGE_PROD 4
#define request_sched_type(r)						\
	(((r)->splitted && (r)->children)? KGPU_SPLIT_SRC :		\
	 (((r)->splitted && !(r)->children)? KGPU_SPLIT_PROD :		\
	  (((r)->merged && (r)->children)? KGPU_MERGE_PROD :		\
	   (((r)->merged && !(r)->children)? KGPU_MERGE_SRC :		\
	    KGPU_NON_SCHED))))
#define is_dumb_request(r)					\
	((request_sched_type((r)) == KGPU_SPLIT_SRC)		\
	 || (request_sched_type((r)) == KGPU_MERGE_SRC))

extern ku_request_t* alloc_ku_request(u32 nrdepon, u32 nrdepby, u32 nrchildren);
extern void free_ku_request(ku_request_t* r);

typedef struct ku_meminfo_t {
	u64 uva;
	u64 size;
} ku_meminfo_t;

typedef struct ku_response_t {
	int id;
	int errcode;
} ku_response_t;


/* Data for k/u memory information communication: */
#define KGPU_BUF_NR 1

/* Default 1GB GPU memory */
#define KGPU_BUF_SIZE (1<<30)
#define KGPU_MMAP_SIZE (1<<30)

/* ioctl */
#include <linux/ioctl.h>

#define KGPU_IOC_MAGIC 'g'
#define KGPU_IOC_SET_GPU_BUFS \
    _IOW(KGPU_IOC_MAGIC, 1, struct ku_meminfo_t[KGPU_BUF_NR])
#define KGPU_IOC_GET_GPU_BUFS \
    _IOR(KGPU_IOC_MAGIC, 2, struct ku_meminfo_t[KGPU_BUF_NR])
#define KGPU_IOC_SET_STOP     _IO(KGPU_IOC_MAGIC, 3)
#define KGPU_IOC_GET_REQS     _IOR(KGPU_IOC_MAGIC, 4, 

#define KGPU_IOC_MAXNR 4


#include <linux/rbtree.h>

/* Request maintained by kernel module: */
typdef struct k_request_t {
	u32 size;
	int id;

	u64 in, out, data;
	u64 insize, outsize, datasize;
	
	int *depon, *depby;
	u32 nrdepon, nrdepby;
	u8 deplevel;

	u64 start_time;

	int parent;
	int *children;
	u32 nrchildren;

	u8 splitted;
	u8 merged;
	
	kg_request_t *orig;
	struct list_head glist;
	struct list_head rqueue;
	struct rb_node   rbase;
} k_request_t;

extern k_request_t* alloc_k_request(u32 nrdepon, u32 nrdepby, u32 nrchildren);
extern void free_k_request(k_request_t* r);
extern void init_ku_request(ku_request_t* kur, k_request_t* kr);

/* Global shared request base, which stores all requests in k_request_t format: */
extern k_request_t* rbase_get_request(int id);
extern int rbase_put_request(k_request_t* r);
extern k_request_t* rbase_remove_request(int id);
extern void rbase_init(void);
extern void rbase_finit(void);

/* Code to be moved to rbase related file */
static k_request_t* r_search(struct rb_root *root, int id)
{
	struct rb_node *node = root->rb_node;

	while (node) {
		k_request_t *data = container_of(node, k_request_t, rbase);

		if (id < data->id)
			node = node->rb_left;
		else if (id > data->id)
			node = node->rb_right;
		else
			return data;
	}
	return NULL;
}

static int r_insert(struct rb_root *root, k_request_t* r)
{
	struct rb_node **new = &(root->rb_node), *parent = NULL;

	while (*new) {
		k_request_t *this = container_of(*new, k_request_t, rbase);

		parent = *new;
		if (r->id < this->id)
			new = &((*new)->rb_left);
		else if (r->id > this->id)
			new = &((*new)->rb_right);
		else
			return 1;
	}

	rb_link_node(&r->rbase, parent, new);
	rb_insert_color(&r->rbase, root);

	return 0;
}

/* Service provider, kernel part: */
typedef kg_k_service_t {
	char name[KGPU_SERVICE_NAME_SIZE];

	int (*make_request)(kg_request_t* r);
	
	struct rb_node ksreg;   /* Service registry */
} kg_k_service_t;

extern int kg_register_kservice(kg_k_service_t* s);
extern int kg_unregister_kservice(kg_k_service_t* s);

extern kg_k_service_t* ksreg_get_service(char *name);
extern int ksreq_put_service(kg_k_service_t* s);
extern kg_k_service_t* ksreg_remove_service(char* name);

extern void ksreg_init(void);
extern void ksreg_finit(void);


/* Userspace GPU info: */
#include <cuda.h>

#define STREAM_NR 16

typedef struct gpu_device_t{
	int id;
	ku_meminfo_t buf;         /* device mem for regular allocation */
	ku_meminfo_t mmbuf;       /* device mem for m-mapping */
	int workload;

	cudaStream_t streams[STREAM_NR];
	u8 streamuses[STREAM_NR];
} gpu_device_t;

extern void urt_init(void);
extern void urt_finit(void);

extern gpu_device_t* get_gpu_device(int);
extern int get_nr_gpu_device(void);
extern gpu_device_t* get_current_gpu(void);
extern void set_current_gpu(gpu_device_t*);

struct kg_u_service_t;

typedef struct kg_u_request_t {
	u32 size;
	int id;

	void *hin, *hout, *hdata;
	void *din, *dout, *ddata;
	u64 insize, outsize, datasize;

	int errcode;
	struct kg_u_service_t *s;

	int block_x, block_y, grid_x, grid_y;
	int state;

	int *depon, *depby, *children;
	u32 nrdepon, nrdepby, nrchildren;
	int parent;
	u8 deplevel;
	s8 device;
	u64 start_time;
	u8 splitted;
	u8 merged;

        int stream_idx;
	
	struct list_head glist;
	struct list_head rqueue;
	
} kg_u_request_t;

typedef struct kg_u_service_t {
	char name[KGPU_SERVICE_NAME_SIZE];
	int sid;

	/* Workflow interfaces */
	int (*compute_size)(kg_u_request_t *r);
	int (*launch)(kg_u_request_t *r);
	int (*prepare)(kg_u_request_t *r);
	int (*post)(kg_u_request_t *r);

	/* Return pointer to device function that does
	 * the service logic, but accepts a kg_u_request
	 * instead of full set of arguments.
	 * This is used to enable multi-service fusing.
 	 */
	void* (*device_function_pointer)(void);
} kg_u_service_t;


#define SERVICE_INIT "init_service"
#define SERVICE_FINIT "finit_service"
#define SERVICE_LIB_PREFIX "libsrv_"
#define SERVICE_KMOD_PREFIX "kmsrv_"

typedef int (*fn_init_service_t)(
	void* libhandle,
	int (*reg_srv)(kg_u_service_t *, void*),
	int (*unreq_srv)(const char*));
typedef int (*fn_finit_service_t)(
    void* libhandle, int (*unreg_srv)(const char*));


/* Userspace GPU resource management: */
extern int alloc_gpu(kg_u_request_t *r);
extern void free_gpu(kg_u_request_t *r);
extern int alloc_stream(kg_u_request_t *r);
extern void free_stream(kg_u_request_t *r);
extern int alloc_devmem(kg_u_request_t *r);
extern void free_devmem(kg_u_request_t *r);

extern void urbase_init(void);
extern voud urbase_finit(void);
extern kg_u_request_t* urbase_get_request(int id);
extern int urbase_put_request(kg_u_request_t *r);
extern kg_u_request_t* urbase_remove_request(kg_u_request_t* r);


extern int kg_register_uservice(kg_u_service_t* s);
extern int kg_unregister_uservice(kg_u_service_t* s);

extern kg_u_service_t* usreg_get_service(char *name);
extern int usreq_put_service(kg_u_service_t* s);
extern kg_u_service_t* usreg_remove_service(char* name);

extern void usreg_init(void);
extern void usreg_finit(void);

extern int load_service(const char *libpath, const char *kmpath);
extern int load_all_services(const char *srvdir);
extern int unload_service(const char *name);
extern int unload_all_services();

#endif
