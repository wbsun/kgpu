#ifndef ___KGPU_H__
#define ___KGPU_H__
#include "kgpu.h"

#ifndef _NDEBUG
#define dbg(...) printk(__VA_ARGS__)
#else
#define dbg(...)
#endif

#define KGPU_BUF_FRAME_SIZE (4*1024)
#define KGPU_BUF_FRAME_NR (KGPU_BUF_SIZE/KGPU_BUF_FRAME_SIZE)

struct kgpu_buffer {
    struct gpu_buffer gb;
    void **paddrs;
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
extern struct kgpu_buffer* alloc_gpu_buffer(void);
extern int free_gpu_buffer(struct kgpu_buffer *);
extern void free_kgpu_response(struct kgpu_resp*);
extern void free_kgpu_request(struct kgpu_req*);

#endif
