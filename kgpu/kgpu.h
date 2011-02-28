#ifndef __KGPU_H__
#define __KGPU_H__

struct gpu_buffer {
    void *addr;
    unsigned long size;
};

#define KGPU_BUF_NR 4
#define KGPU_BUF_SIZE (32*1024*1024)

#define REQ_PROC_FILE "kgpureq"
#define RESP_PROC_FILE "kgpuresp"

#define SERVICE_NAME_SIZE 32

struct ku_request {
    int id;
    char sname[SERVICE_NAME_SIZE];
    void *input;
    void *output;
    unsigned long insize;
    unsigned long outsize;
};

/* kgpu's errno */
#define KGPU_OK 0
#define KGPU_NO_RESPONSE 1
#define KGPU_NO_SERVICE 2

struct ku_response {
    int id;
    int errcode;
};

#endif
