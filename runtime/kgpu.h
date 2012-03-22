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

/* Request memory management flags: */
#define KGPU_RMM_IO_OVERLAP      0x00000001
#define KGPU_RMM_ALL_OVERLAP     0x00000002
#define KGPU_RMM_ALL_CONSECUTIVE 0x00000004
#define KGPU_RMM_NO_OUT_COPYBACK 0x00000008
#define KGPU_RMM_NO_IN_COPY      0x00000010
#define KGPU_RMM_NO_IN_ALLOC     0x00000020
#define KGPU_RMM_NO_OUT_ALLOC    0x00000040
#define KGPU_RMM_NO_DATA_ALLOC   0x00000080
#define KGPU_RMM_NO_DATA_COPY    0x00000100
#define KGPU_RMM_IO_CONSECUTIVE  0X00000200

/* Request scheduling info: */
#define KGPU_NON_SCHED     0x00000000
#define KGPU_SPLITTABLE    0x00000001
#define KGPU_MERGEABLE     0x00000002
#define KGPU_SPLITTED      0x00000004
#define KGPU_MERGED        0x00000008
#define KGPU_SPLIT_SRC     0x00000010
#define KGPU_SPLIT_PROD    0x00000020
#define KGPU_MERGE_SRC     0x00000040
#define KGPU_MERGE_PROD    0x00000080

/* Request struct and related ones used by kernel space: */
struct kg_request_t;

typedef int (*kg_request_callback_t)(struct kg_request_t*);

typedef struct kg_request_t {
	u32 size;
	int id;
	char service_name[KGPU_SERVICE_NAME_SIZE];

	u64 in, out, data;
	u64 insize, outsize, datasize;
	
	u32 memflags;
	
	int *depon, *depby;
	u32 nrdepon, nrdepby;
	int deponlevel;
	int depbylevel;
    
	int device;

	u32 schedflags;

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

extern u64 kg_alloc_mmap_area(u64 size);
extern void kg_free_mmap_area(u64 start);

extern void kg_unmap_area(u64 start);
extern int kg_map_page(struct page *p, u64 addr);

extern void* kg_map_pages(struct page **ps, int nr); 

/* Request data passed through k/u boundary: */
typedef struct ku_request_t {
	u32 size;
	int id;
	kg_u_service_t* s;

	u64 hin, hout, hdata;
	u64 din, dout, ddata;
	u64 insize, outsize, datasize;
	
	u32 memflags;

	int device;
	int stream_idx;

	u64 start_time;

	u32 schedflags;
	u32 segidx;
} ku_request_t;

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

typedef struct ku_devmeminfo_t {
	int device;
	u64 uva;
	u64 size;
} ku_devmeminfo_t;

typedef struct ku_serviceinfo_t {
	char name[KGPU_SERVICE_NAME_SIZE];
	kg_u_service_t *s;
} ku_serviceinfo_t;

/* Data for k/u memory information communication: */
#define KGPU_BUF_NR 2

/* Default 1GB GPU memory */
#define KGPU_BUF_SIZE (1<<30)
#define KGPU_MMAP_SIZE (1<<30)

/* ioctl */
#include <linux/ioctl.h>

/* Operations via ioctl:
 *  1 Set host buffer for normal allocation (struct ku_meminfo_t)
 *  2 Set host buffer for mem mapping (struct ku_meminfo_t)
 *  3 Set number of GPU devices (int)
 *  4 Set device buffer info (struct ku_devmeminfo_t)
 *  5 Set userspace service pointer (struct ku_serviceinfo_t)
 *  6 Stop
 */
#define KGPU_IOC_MAGIC 'g'
#define KGPU_IOC_SET_HOST_ALLOC_BUF                 \
    _IOW(KGPU_IOC_MAGIC, 1, struct ku_meminfo_t)
#define KGPU_IOC_SET_HOST_MAP_TYPE                  \
_IOW(KGPU_IOC_MAGIC, 2, int)
#define KGPU_IOC_SET_HOST_MAP_BUF                   \
    _IOW(KGPU_IOC_MAGIC, 3, struct ku_meminfo_t)
#define KGPU_IOC_SET_NR_GPU                         \
    _IOW(KGPU_IOC_MAGIC, 4, int)
#define KGPU_IOC_SET_DEV_BUF                        \
    _IOW(KGPU_IOC_MAGIC, 5, struct ku_devmeminfo_t)
#define KGPU_IOC_SET_USERVICE_INFO                  \
    _IOW(KGPU_IOC_MAGIC, 6, struct ku_serviceinfo_t)
#define KGPU_IOC_SET_STOP     _IO(KGPU_IOC_MAGIC, 7)

#define KGPU_IOC_MAXNR 7


#include <linux/rbtree.h>

struct k_request_t;

typedef int (*k_request_callback_t)(struct k_request_t*);

/* Request maintained by kernel module: */
typdef struct k_request_t {
	u32 size;
	int id;

	u64 hin, hout, hdata;
	u64 din, dout, ddata;
	u64 insize, outsize, datasize;
	
	u32 memflags;
	
	int device;
	int stream_idx;
	
	int *depon, *depby;
	u32 nrdepon, nrdepby;
	int deponlevel;
	int depbylevel

	u64 start_time;

	int parent;
	int *children;
	u32 nrchildren;

	u32 schedflags;
	u32 segidx;

	void *cbdata;
	k_request_callback_t callback;
	
	kg_request_t *orig;
	struct list_head glist;
	struct list_head rqueue;
	struct hlist_node hashnode;
} k_request_t;

extern k_request_t* alloc_k_request(u32 nrdepon, u32 nrdepby, u32 nrchildren);
extern void free_k_request(k_request_t* r);
extern void init_ku_request(ku_request_t* kur, k_request_t* kr);

extern int general_kreq_callback(struct k_request_t* r);

/* Request mgmt functions used by schedulers: */
extern int submit_krequest(k_request_t* r);
extern int alloc_gpu_resources(k_request_t* r);
extern int free_gpu_resources(k_request_t* r);
extern int put_krequest_wait(k_request_t* r);
extern int set_krequest_ready(k_request_t* r);
extern k_request_t* fetch_next_krequest(void);
extern int has_krequest(void);
extern void unfetch_krequest(k_request_t* r);
extern void krequest_done(k_request_t* r);

/* block until krequest available, return 1 for error, 0 for normal */
extern int wait_for_krequest(void);

extern int depby_done(k_request_t* r);
extern int depon_done(k_request_t* r);
extern int split_seg_done(k_request_t* r);
extern int merged_done(k_request_t* r);

/* Global shared request base, which stores all requests in k_request_t format: */
extern k_request_t* get_krequest(int id);
extern int put_krequest(k_request_t* r);
extern k_request_t* remove_krequest(int id);
extern void krbase_init(void);
extern void krbase_finit(void);

/* GPU resource management module functions */

/* External functions for ioctl commands that manipulate GPU resources */
extern int set_nr_gpu(int nrgpu);
extern int mm_set_host_alloc_buf(ku_meminfo_t* info);
extern int mm_set_host_map_buf(ku_meminfo_t* info);
extern int mm_set_host_map_type(int maptype);
extern int mm_set_dev_buf(ku_devmeminfo_t* info);
extern int mm_alloc_krequest_devmem(k_request_t *r);
extern int mm_free_krequest_devmem(k_request_t *r); 
extern int set_uservice_info(ku_serviceinfo_t* info);
extern int krt_stop(void);

/* Service provider, kernel part. 
 * Services include their userspace pointers too. This can simplify
 * userspace helper by avoiding complex management policy of services.
 * Userspace service pointers are notified via ioctl. This may be before
 * the registeration of kernel part. So both kservice registeration and 
 * uservice notification can add a new service object into service base,
 * they just fill up fields they know, and leave other fields as the
 * other one's job. 
 */
typedef kg_k_service_t {
	char name[KGPU_SERVICE_NAME_SIZE];
	kg_u_service_t *usrv;

	int (*make_request)(kg_request_t* r);
	
	struct rb_node srbnode;   /* Service db */	
} kg_k_service_t;

extern int kg_register_kservice(kg_k_service_t* s);
extern int kg_unregister_kservice(kg_k_service_t* s);

extern kg_k_service_t* get_kservice(char *name);
extern int put_kservice(kg_k_service_t* s);
extern kg_k_service_t* remove_kservice(char* name);

extern void ksbase_init(void);
extern void ksbase_finit(void);


/* Kernel space GPU resource */

#define KGPU_MMAP_MAP_SELF_DEVICE_VMA        0
#define KGPU_MMAP_MAP_MALLOC_VMA_NO_PRE_PIN  1
#define KGPU_MMAP_MAP_MALLOC_VMA_PRE_PIN_ALL 2
#define KGPU_MMAP_MAP_CUDA_VMA               3

extern int kgpu_mem_map_type;

#define KGPU_BUF_UNIT_SIZE PAGE_SIZE
#define KGPU_BUF_UNIT_SHIFT PAGE_SHIFT

typedef struct k_hostmem_pool_t {
	u64 uva;
	u64 kva;
	struct page **pages;
	u32 npages;
	u32 nunits;
	u64 *bitmap;
	u32 *alloc_sz;
	spinlock_t lock;
} k_hostmem_pool_t;

typedef struct k_hostmem_vma_t {
	struct vm_area_struct *vma;
	u64 start;
	u64 end;
	u32 npages;
	u32 *alloc_sz;
	u64 *bitmap;
	spinlock_t lock;
} k_hostmem_vma_t;

typedef struct k_devmem_pool_t {
	u64 start;
	u64 end;
	u64 size;
	u32 *alloc_sz;
	u32 nunits;
	u64 *bitmap;
	spinlock_t lock;
} k_devmem_pool_t;

typedef struct k_gpu_t {
	int id;
	int stream_uses[KGPU_NR_STREAM];
	int workload;
	k_devmem_pool_t devmem;
	spinlock_t lock;
} k_gpu_t;

extern void gpumm_init(void);
extern void gpumm_finit(void);

extern k_gpu_t* get_k_gpu(int id);

/* -------------------------------------------------------------------------- */
/* Out-dated info, need to change when writing urt part.                      */
/* -------------------------------------------------------------------------- */

/* Userspace GPU info: */
#include <cuda.h>

#define KGPU_NR_STREAM 16

typedef struct gpu_device_t{
	int id;
	ku_meminfo_t buf;         /* device mem for regular allocation */
	ku_meminfo_t mmbuf;       /* device mem for m-mapping */
	int workload;

	cudaStream_t streams[KGPU_NR_STREAM];
	u8 streamuses[KGPU_NR_STREAM];
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
	int deponlevel, depbylevel;
	int device;
	u64 start_time;
	u8 splitted;
	u8 merged;

        int stream_idx;
	
	struct list_head glist;
	struct list_head rqueue;
	struct hlist_node rbase;	
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
