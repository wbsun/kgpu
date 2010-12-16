#include <cuda.h>
#include "nsk.h"

void alloc_hdmem(void **pph, void **ppd, unsigned int size, mem_mode_t memMode)
{
    switch(memMode) {
    case PINNED:
	csc( cudaHostAlloc(pph, size, 0) );
	csc( cudaMalloc(ppd, size) );
	break;
    case PAGEABLE:
	*pph = malloc(size);
	csc( cudaMalloc(ppd, size) );
	break;
    case MAPPED:
	csc( cudaHostAlloc(pph, size, cudaHostAllocMapped) );
	csc( cudaHostGetDevicePointer(ppd, *pph, 0) );
	break;
    case WC:
	csc( cudaHostAlloc(pph, size,
			   cudaHostAllocMapped|cudaHostAllocWriteCombined) );
	csc( cudaHostGetDevicePointer(ppd, *pph, 0) );
    default:
	break;
    }
}

void free_hdmem(void **pph, void **ppd, mem_mode_t memMode)
{
    switch(memMode) {
    case PINNED:
    case MAPPED:
    case WC:
	csc(cudaFreeHost(*pph));
	break;
    case PAGEABLE:
	free(*pph);
    default:
	break;
    }

    *pph = NULL;
    csc(cudaFree(*ppd));
    *ppd = NULL;
}

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

void csc(cudaError_t e)
{
    if (e != cudaSuccess){
	printf("Error: %s\n", cudaGetErrorString(e));
	cudaThreadExit();
	exit(0);
    }
}

int _ssc(int e, void (panic*)(int), int rt)
{
    if (e == -1) {
	perror("Syscall error: ");
	if (panic)
	    panic(rt);
    }

    return 0;
}

int ssce(int e)
{
    return _scc(e,exit,0);
}

int sscp(int e)
{
    return _scc(e,NULL,0);
}
