/*
 *  
 *  NSK - nsk.h
 */
#ifndef __NSK_H__
#define __NSK_H__

#define NSK_MAX_TASK_FUNC_NR 8
#define NSK_MAX_REQ_NR 8

enum nsk_errno_t {
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

typedef struct {
    volatile int current;
    nsk_taks_func_t task_funcs[NSK_MAX_TASK_FUNC_NR];
    volatile void *mems[3];
    volatile nsk_request_t *requests;
    volatile nsk_response_t *responses;    
    char padding[20];
} nsk_device_context_t;

typedef int (nsk_task_func_t*)(nsk_hd_request_t *req);

/* task functions on host side */
nsk_task_func_t h_task_funcs[NSK_MAX_TASK_FUNC_NR];

extern int *current;

extern volatile nsk_request_t *h_requests;
extern volatile nsk_response_t *h_responses;

void csc(cudaError_t e);

#endif
