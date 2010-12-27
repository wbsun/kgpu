/*
 *  
 *  NSK - nsk.h
 */
#ifndef __NSK_H__
#define __NSK_H__


#include "nskku.h"

#define GRIDS_X 32
#define BLOCKS_X 32

#define NOP_TASK 0

extern nsk_request_t *h_requests;
extern nsk_request_t *dh_requests;
extern nsk_response_t *h_responses;
extern nsk_response_t *dh_responses;
extern volatile int *h_current;
extern volatile int *h_taskdone;

extern volatile void* h_mems[4];
extern volatile void* dh_mems[3];
extern int devmemuses[3];
extern int hostmemuses[4];

#define SKERNEL 0
#define SH2D 1
#define SD2H 2
#define SCOM 1

void init_ku_context();
void init_hd_context();
void clean_hd_context();

int prepare_hd_task(nsk_request_t *kreq, int which);
void start_hd_task(int which);
int is_current_task_done();
void finish_task(int which, nsk_request_t *kreq);

void fill_tasks(nsk_task_func_t tfs[]);
void start_device_kernel();


#define USER_MODE_TEST

#ifdef USER_MODE_TEST
  #define umdbg(...) printf(__VA_ARGS__)
#else
  #define umdbg(...)
#endif


#endif
