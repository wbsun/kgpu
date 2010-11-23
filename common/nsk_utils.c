#include <cuda.h>
#include <stdio.h>

void csc(cudaError_t e)
{
    if (e != cudaSuccess){
	printf("Error: %s\n", cudaGetErrorString(e));
	cudaThreadExit();
	exit(0);
    }
}

static int _ssc(int e, void (panic*)(int), int rt)
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
