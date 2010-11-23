#include <cuda.h>
#include "../common/nsk.h"
#include <string.h>

volatile nsk_request_t *h_requests;
volatile nsk_response_t *h_responses;

nsk_device_context_t *h_devctxt, *d_devctxt;



enum mem_mode_t {
    PINNED,
    PAGEABLE,
    MAPPED,
    WC,
};

void _alloc_mem(void **pph, void **ppd, unsigned int size, mem_mode_t memMode)
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

#define alloc_mem(pph, ppd, sz, mm)\
    _alloc_mem((void**)(pph), (void**)(ppd), (unsigned int)(sz), mm)

int init_context()
{
    alloc_mem(&h_devctxt, &d_devctxt, sizeof(nsk_device_context_t), PAGEABLE);
    memset((void*)h_devctxt, 0, sizeof(nsk_device_context_t));

    alloc_mem(&h_requests, &h_devctxt->requests,
	      sizeof(nsk_request_t)*NSK_MAX_REQ_NR, PINNED);
    alloc_mem(&h_responses, &h_devctxt->responses,
	      sizeof(nsk_response_t)*NSK_MAX_REQ_NR, PINNED);
}
