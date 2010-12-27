#include <cuda.h>
#include <stdio.h>
#include "nsk.h"
#include "devutils.h"

int devmemuses[3];

volatile void* dh_mems[3];
volatile void *h_mems[4];

cudaStream_t ss[3];

void alloc_hdmem(void **pph, void **ppd, unsigned int size, mem_mode_t memMode)
{
    switch(memMode) {
    case PINNED:
	csc( cudaHostAlloc(pph, size, 0) );
	if (ppd != NULL)
	    csc( cudaMalloc(ppd, size) );
	break;
    case PAGEABLE:
	*pph = malloc(size);
	if (ppd != NULL)
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
    if (ppd != NULL)
    	csc(cudaFree(*ppd));
    *ppd = NULL;
}

void _csc(cudaError_t e, const char *file, int line)
{
    if (e != cudaSuccess){
	printf("nsk Error: %s %d %s\n", file, line, cudaGetErrorString(e));
	cudaThreadExit();
	exit(0);
    }
}

void init_hd_buffers()
{
    cudaStream_t s = ss[SKERNEL];
    // init device memory buffers uses:
    for (int i=0; i<3; i++)
	devmemuses[i] = -1; // nobody use
	
    // allocate memory buffers for data input and output
    for (int i=0; i<3; i++) {
	ALLOC_HDMEM(&(h_mems[i]), &(dh_mems[i]), NSK_MEM_SIZE, PINNED);
    }
    csc( cudaHostAlloc( (void**)&(h_mems[3]), NSK_MEM_SIZE, PINNED ) );

    for (int i=0; i<4; i++) {
	memset((void*)h_mems[i], 0, NSK_MEM_SIZE);
	if (i!= 3)
	    csc( h2d_cpy_a( dh_mems[i], h_mems[i], NSK_MEM_SIZE, s ) );
    }
    csc( cudaStreamSynchronize(s) );
}

void init_hd_streams()
{
    for (int i=0; i<3; i++)
	csc( cudaStreamCreate(&ss[i]) );
}

volatile void* get_next_device_mem(int user)
{
    int i;

    for (i=0; i<3; i++) {
	if (devmemuses[i] == -1) {
	    devmemuses[i] = user;
	    return dh_mems[i];
	}
    }
    
    return NULL;
}

void put_device_mem(volatile void* devmem)
{
    int i;

    for (i=0; i<3; i++) {
	if (devmem == dh_mems[i]) {
	    devmemuses[i] = -1;
	    return;
	}
    }
}
