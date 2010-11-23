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
