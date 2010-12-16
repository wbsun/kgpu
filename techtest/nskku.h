/*
 * Common types and constants needed by kernel, user and GPU.
 */

#ifndef __NSKKU_H__
#define __NSKKU_H__

#define NSK_MAX_TASK_FUNC_NR 8
#define NSK_MAX_REQ_NR 4

/* 8*1024*1024 */
#define NSK_MEM_SIZE 8388608

enum nsk_errno_t {
    NSK_ENONE,    // no error
    NSK_ENMEM,    // no enough memory
    NSK_ESYNC,    // synchronization error
};

enum nsk_task_state_t {
    NSK_TREADY,
    NSK_TSCHEDULED,
    NSK_TRUNNING,
    NSK_TSTOPPED,
    NSK_TNA,
    NSK_TWAIT,
};

typedef struct {
    int request_id;
    int taskfunc;
    /* both the data pointers(inputs/outputs) and the data are volatile */
    volatile void *inputs;
    volatile void *outputs;
    int insize;
    int outsize;
} nsk_request_t; // 32 bytes

typedef struct {
    int request_id;
    enum nsk_task_state_t state;
    enum nsk_errno_t errno;
    char padding[4];
} nsk_response_t; // 16 bytes

typedef int (*nsk_task_func_t)(nsk_request_t *req);

typedef struct {
    volatile int current;
    nsk_task_func_t task_funcs[NSK_MAX_TASK_FUNC_NR];
    volatile void *mems[3];
    nsk_request_t *requests;
    nsk_response_t *responses;    
    char padding[20];
} nsk_device_context_t; // 128 bytes

typedef struct {
    void *addr;
    unsigned int size; // no more than 4GB
} nsk_buf_info_t;

#define NSK_PROCFS_FILE "/proc/nsk"

#endif
