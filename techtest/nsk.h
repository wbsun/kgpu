/*
 *  
 *  NSK - nsk.h
 */
#ifndef __NSK_H__
#define __NSK_H__

#include <time.h>

#include "nskku.h"


/* task functions on host side */
extern nsk_task_func_t h_task_funcs[NSK_MAX_TASK_FUNC_NR];

//extern int *current;

extern volatile nsk_request_t *h_requests;
extern volatile nsk_response_t *h_responses;

/* error aware helpers for CUDA and syscall:
 *    csc: safe CUDA call
 *    ssce: safe sys call & exit if error
 *    sscp: safe sys call & pass if error
 *
 *    I know that naming sucks, but those functions are used so
 *    frequently that I just want to type a little bit fewer...
 */
void csc(cudaError_t e);
int ssce(int e);
int sscp(int e);

void fill_tasks(nsk_device_context_t *dc);


static void nsleep(long ns)
{
    struct timespec tv;

    tv.tv_sec = ns/1000000000;
    tv.tv_nsec = ns%1000000000;

    nanosleep(&tv, NULL);
}

typedef struct {
    timespec start, stop;
} timer;

static double ts2d(timespec *ts)
{
    double d = ts->tv_sec;
    d += ts->tv_nsec/1000000000.0;
    return d;
}

static timespec get_timer_val(timer *tm)
{
    timespec temp;
    if ((tm->stop.tv_nsec - tm->start.tv_nsec)<0) {
	temp.tv_sec = tm->stop.tv_sec - tm->start.tv_sec-1;
	temp.tv_nsec = 1000000000+tm->stop.tv_nsec - tm->start.tv_nsec;
    } else {
	temp.tv_sec = tm->stop.tv_sec - tm->start.tv_sec;
	temp.tv_nsec = tm->stop.tv_nsec - tm->start.tv_nsec;
    }
    return temp;
}

static void start_timer(timer *tm)
{
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tm->start);
}

static timespec stop_timer(timer *tm)
{
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tm->stop);
    return get_timer_val(tm);
}

#endif
