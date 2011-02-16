#include "nsk.h"
#include "hostutils.h"
#include <stdio.h>

void nsleep(long ns)
{
    struct timespec tv;

    tv.tv_sec = ns/1000000000;
    tv.tv_nsec = ns%1000000000;

    nanosleep(&tv, NULL);
}

double ts2d(timespec *ts)
{
    double d = ts->tv_sec;
    d += ts->tv_nsec/1000000000.0;
    return d;
}

timespec get_timer_val(timer *tm)
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

void start_timer(timer *tm)
{
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tm->start);
}

timespec stop_timer(timer *tm)
{
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tm->stop);
    return get_timer_val(tm);
}

int ssc(int e, void (*panic)(int), int rt)
{
    if (e == -1) {
	perror("nsk Syscall error: ");
	if (panic)
	    panic(rt);
    }

    return e;
}

int ssce(int e)
{
    return ssc(e,exit,0);
}

int sscp(int e)
{
    return ssc(e,NULL,0);
}


volatile void* get_next_host_mem(int user)
{
    int i;

    for (i=0; i<4; i++) {
	if (hostmemuses[i] == -1) {
	    hostmemuses[i] = user;
	    return h_mems[i];
	}
    }
    
    return NULL;
}

void put_host_mem(volatile void* hostmem)
{
    int i;

    for (i=0; i<4; i++) {
	if (hostmem == h_mems[i]) {
	    hostmemuses[i] = -1;
	    return;
	}
    }
}


