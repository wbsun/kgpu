#ifndef __KGPU_H__
#define __KGPU_H__

struct gpu_buffer {
    void *addr;
    unsigned long size;
};

struct kgpu_buffer {
    void *paddr;
    struct gpu_buffer gb;
};

#define KGPU_BUF_NR 4
#define KGPU_BUF_SISE (32*1024*1024)

#define REQ_PROC_FILE "kgpureq"
#define RESP_PROC_FILE "kgpuresp"

struct kgpu_req;
struct kgpu_resp;

typedef int (*ku_callback)(struct kgpu_req *req,
			   struct kgpu_resp *resp);

#define SERVICE_NAME_SIZE 32

struct ku_request {
    int id;
    char sname[SERVICE_NAME_SIZE];
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
    void *data;
};

/* kgpu's errno */
#define KGPU_OK 0
#define KGPU_NO_RESPONSE 1
#define KGPU_NO_SERVICE 2

struct ku_response {
    int id;
    int errno;
};

struct kgpu_resp {
    struct list_head list;
    struct ku_response kuresp;
    struct kgpu_req *req;
};


#endif
