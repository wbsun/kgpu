/*
 *  
 *  NSK - nsk.h
 */
#ifndef __NSK_H__
#define __NSK_H__

#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include "nskku.h"

#define GRIDS_X 32
#define BLOCKS_X 32

#define NOP_TASK 0

extern dim3 blockdim;
extern dim3 griddim;

/* error aware helpers for CUDA and syscall:
 *    csc: safe CUDA call
 *    ssce: safe sys call & exit if error
 *    sscp: safe sys call & pass if error
 *
 *    I know that naming sucks, but those functions are used so
 *    frequently that I just want to type a little bit fewer...
 */

void csc(cudaError_t e);
int ssc(int e, void (*panic)(int), int rt);
int ssce(int e);
int sscp(int e);

enum mem_mode_t {
    PINNED,
    PAGEABLE,
    MAPPED,
    WC,
};

void alloc_hdmem(void **pph, void **ppd, unsigned int size, mem_mode_t memMode);
void free_hdmem(void **pph, void **ppd, mem_mode_t memMode);

#define ALLOC_HDMEM(pph, ppd, sz, mm)\
    alloc_hdmem((void**)(pph), (void**)(ppd), (unsigned int)(sz), mm)

#define FREE_HDMEM(pph, ppd, mm) free_hdmem((void**)(pph), (void**)(ppd), mm)


void fill_tasks(nsk_device_context_t *dc);
void start_device_kernels(nsk_device_context_t *dc,
			  cudaStream_t smaster, cudaStream_t sslave);

void nsleep(long ns);

typedef struct {
    timespec start, stop;
} timer;

double ts2d(timespec *ts);
timespec get_timer_val(timer *tm);
void start_timer(timer *tm);
timespec stop_timer(timer *tm);

#endif
