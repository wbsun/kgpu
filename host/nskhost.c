#include <cuda.h>
#include "../common/nsk.h"
#include <string.h>
#include "host.h"

volatile nsk_request_t *h_requests;
volatile nsk_response_t *h_responses;

nsk_device_context_t *h_devctxt, *d_devctxt;

volatile void *hd_mems[3];
volatile void *h_mems[4];

cudaStream_t smaster, sslave, sch2d, scd2h, sdh2d, sdd2h;

enum mem_mode_t {
    PINNED,
    PAGEABLE,
    MAPPED,
    WC,
};

static void _alloc_hdmem(void **pph, void **ppd, unsigned int size, mem_mode_t memMode)
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

static void _free_hdmem(void **pph, void **ppd, mem_mode_t memMode)
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


#define alloc_hdmem(pph, ppd, sz, mm)\
    _alloc_hdmem((void**)(pph), (void**)(ppd), (unsigned int)(sz), mm)

#define free_hdmem(pph, ppd, mm) _free_hdmem((void**)(pph), (void**)(ppd), mm)

static void _init_mem()
{
    int i;
    
    alloc_hdmem(&h_devctxt, &d_devctxt, sizeof(nsk_device_context_t), PAGEABLE);
    memset((void*)h_devctxt, 0, sizeof(nsk_device_context_t));

    alloc_hdmem(&h_requests, &h_devctxt->requests,
	      sizeof(nsk_request_t)*NSK_MAX_REQ_NR, PINNED);
    alloc_hdmem(&h_responses, &h_devctxt->responses,
	      sizeof(nsk_response_t)*NSK_MAX_REQ_NR, PINNED);

    for(i=0; i<3; i++)
	alloc_hdmem(&(h_mems[i]), &(h_devctxt->mems[i]), NSK_MEM_SIZE, PINNED);
    csc( cudaHostAlloc((void**)&(h_mems[3]), NSK_MEM_SIZE, PINNED));
}

static void _free_mem()
{
    int i;

    for(i=0; i<3; i++)
	free_hdmem(&(h_mems[i]), &(h_devctxt->mems[i]), PINNED);
    csc(cudaFreeHost((void*)h_mems[3]));

    free_hdmem(&h_responses, &h_devctxt->responses, PINNED);
    free_hdmem(&h_requests, &h_devctxt->requests, PINNED);
    free_hemem(&h_devctxt, &d_devctxt, PAGEABLE);
}


static void _init_nskk()
{
    nsk_buf_info_t bufs[4];
    int nskkfd;
    int i;

    for(i=0; i<4; i++) {
	bufs[i].addr = (void*)h_mems[i];
	bufs[i].size = NSK_MEM_SIZE;
    }
    
    nskkfd = scce(open(NSK_PROCFS_FILE, O_RDWR));
    scce(write(nskkfd, (void*)bufs, sizeof(nsk_buf_info_t)*4));
    close(nskkfd);    
}

static void _init_streams()
{
    csc(cudaStreamCreate(&smaster));
    csc(cudaStreamCreate(&sslave));
    csc(cudaStreamCreate(&sch2d));
    csc(cudaStreamCreate(&scd2h));
    csc(cudaStreamCreate(&sdh2d));
    csc(cudaStreamCreate(&sdd2h));
}

static void _destroy_streams()
{
    csc(cudaStreamDestroy(smaster));
    csc(cudaStreamDestroy(sslave));
    csc(cudaStreamDestroy(sch2d));
    csc(cudaStreamDestroy(scd2h));
    csc(cudaStreamDestroy(sdh2d));
    csc(cudaStreamDestroy(sdd2h));
}

static void _init_context()
{
    _init_mem();
    _init_nskk();

    fill_tasks(h_devctxt);
    _init_streams();
}

static void _cleanup_context()
{
}

static void _copy_context()
{
}
