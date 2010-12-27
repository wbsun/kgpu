#ifndef __HOST_UTILS_H__
#define __HOST_UTILS_H__

#include <time.h>

/* error aware helpers for CUDA and syscall:
 *    csc: safe CUDA call
 *    ssce: safe sys call & exit if error
 *    sscp: safe sys call & pass if error
 *
 *    I know that naming sucks, but those functions are used so
 *    frequently that I just want to type a little bit fewer...
 */
int ssc(int e, void (*panic)(int), int rt);
int ssce(int e);
int sscp(int e);

void nsleep(long ns);

typedef struct {
    timespec start, stop;
} timer;

double ts2d(timespec *ts);
timespec get_timer_val(timer *tm);
void start_timer(timer *tm);
timespec stop_timer(timer *tm);

volatile void* get_next_host_mem(int user);
void put_host_mem(volatile void* hostmem);

#endif
