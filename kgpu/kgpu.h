#ifndef __KGPU_H__
#define __KGPU_H__

#include <linux/ioctl.h>

struct gpu_buffer {
    void *addr;
    unsigned long size;
};

#define KGPU_BUF_NR 128
#define KGPU_BUF_SIZE (4*1024*1024)

#define REQ_PROC_FILE "kgpureq"
#define RESP_PROC_FILE "kgpuresp"

#define KGPU_DEV_NAME "kgpu"

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

#define KGPU_IOC_MAGIC 'g'

#define KGPU_IOC_SET_GPU_BUFS _IOW(KGPU_IOC_MAGIC, 1, struct gpu_buffer[KGPU_BUF_NR])
#define KGPU_IOC_GET_GPU_BUFS _IOR(KGPU_IOC_MAGIC, 2, struct gpu_buffer[KGPU_BUF_NR])
#define KGPU_IOC_SET_STOP     _IO(KGPU_IOC_MAGIC, 3)

#define KGPU_IOC_MAXNR 4

#endif
