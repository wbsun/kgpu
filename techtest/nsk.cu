#include <cuda.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "nsk.h"

nsk_task_func_t h_task_funcs[NSK_MAX_TASK_FUNC_NR];

nsk_request_t *h_requests;
volatile nsk_response_t *h_responses;
nsk_request_t *k_requests;
nsk_response_t *K_responses;

nsk_device_context_t *h_dc, *d_dc;

volatile void *d_mems[3];
volatile void *h_mems[4];

cudaStream_t smaster, sslave, sch2d, scd2h, sdh2d, sdd2h;
int current = 0;
int last = 0;
int next = 0;

enum mem_mode_t {
    PINNED,
    PAGEABLE,
    MAPPED,
    WC,
};

#define GIRDS_X 32
#define BLOCKS_X 32

#define NOP_TASK 0


dim3 blockdim = dim3(BLOCKS_X,1);
dim3 griddim = dim3(GRIDS_X,1);

__device__ volatile int slavesdone = 0;

__device__ int allslavesdone()
{
    return slavesdone == GRIDS_X;
}

__device__ void resetslavesdone()
{
    slavesdone = 0;
}

__device__ void slavedone()
{
    atomicAdd(&slavesdone,1);
    __threadfence();
}

__global__ void nskslave(nsk_device_context_t *dc)
{
    __shared__ nsk_request_t *req = NULL;
    __shared__ int goon = 1;
    __shared__ int current = 0;
    __shared__ int newtask = 0;

    while(goon){
	if (threadId.x == 0) {
	    if (current != dc->current) {
		current = dc->current;
		if (current == -1)
		    goon = 0;
		else {
		    req = dc->requests+current;
		    if (req->taskfunc != NOP_TASK)
			newtask = 1;
		}
		__threadfence_block();
	    }
	}

	__syncthreads();	
	if (newtask) {
	    dc->task_funcs[req->taskfunc](req);
	    __syncthreads();
	    //__threadfence_system(); // should in each task function for performance
	    if (threadIdx == 0) {
		slavedone();
		newtask = 0;
		__threadfence_block();
	    }
	}
    }
}


__global__ void nskmaster(
    nsk_device_context_t *dc)
{
    int i, current;
    nsk_response_t *resp;
    nsk_request_t *req;

    resetslavesdone(); // no need, maybe
    
    current = dc->current;
    while (1) {
	if (current != dc->current) {	    
	    current = dc->current;
	    if (current == -1)
		break;
	    req = dc->requests+current;
	    resp = dc->responses+current;
	    
	    resp->errno = 0;
	    if (req->taskfunc == NOP_TASK)
		resp->state = NSK_TSTOPPED;
	    else
		resp->state = NSK_TRUNNING;
	    resp->request_id = req->request_id;
	    
	    __threadfence_system();
	}
	if (allslavesdone()) {
	    resp->state = NSK_TSTOPPED;
	    resetslavesdone();
	    __threadfence_system();
	}
    }
}

__device__ int nop(nsk_hd_request_t *req)
{
    return 0;
}

__device__ int sha1(nsk_hd_request_t *req)
{
    return 0;
}

__device__ int iplookup(nsk_hd_request_t *req)
{
    return 0;
}

__device__ int decrypt(nsk_hd_request_t *req)
{
    return 0;
}

__device__ int encrypt(nsk_hd_request_t *req)
{
    return 0;
}

static void _fill_tasks(nsk_device_context_t *dc)
{
    int i;
    csc(cudaMemcpyFromSymbol(&(dc->task_funcs[0]), nop, sizeof(nsk_task_func_t)));
    csc(cudaMemcpyFromSymbol(&(dc->task_funcs[1]), sha1, sizeof(nsk_task_func_t)));
    csc(cudaMemcpyFromSymbol(&(dc->task_funcs[2]), iplookup, sizeof(nsk_task_func_t)));
    csc(cudaMemcpyFromSymbol(&(dc->task_funcs[3]), decrypt, sizeof(nsk_task_func_t)));
    csc(cudaMemcpyFromSymbol(&(dc->task_funcs[4]), encrypt, sizeof(nsk_task_func_t)));
    for(i=5; i<NSK_MAX_TASK_FUNC_NR; i++)
	dc->task_funcs[i] = NULL;
}

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
    
    alloc_hdmem(&h_dc, &d_dc, sizeof(nsk_device_context_t), PAGEABLE);
    memset((void*)h_dc, 0, sizeof(nsk_device_context_t));

    alloc_hdmem(&h_requests, &h_dc->requests,
	      sizeof(nsk_request_t)*NSK_MAX_REQ_NR, PINNED);
    alloc_hdmem(&h_responses, &h_dc->responses,
	      sizeof(nsk_response_t)*NSK_MAX_REQ_NR, PINNED);
    k_requests = (nsk_request_t*)malloc(sizeof(nsk_request_t)*NSK_MAX_REQ_NR);
    k_responses = (nsk_response_t*)malloc(sizeof(nsk_response_t)*NSK_MAX_REQ_NR)
    assert(k_requests && k_responses);

    for(i=0; i<3; i++){
	alloc_hdmem(&(h_mems[i]), &(h_dc->mems[i]), NSK_MEM_SIZE, PINNED);
	d_mems[i] = h_dc->mems[i];
    }
    csc( cudaHostAlloc((void**)&(h_mems[3]), NSK_MEM_SIZE, PINNED));
    for(i=0; i<4; i++)
	memset((void*)h_mems[i], 0, NSK_MEM_SIZE);
}

static void _free_mem()
{
    int i;

    for(i=0; i<3; i++)
	free_hdmem(&(h_mems[i]), &(h_dc->mems[i]), PINNED);
    csc(cudaFreeHost((void*)h_mems[3]));

    free_hdmem(&h_responses, &h_dc->responses, PINNED);
    free_hdmem(&h_requests, &h_dc->requests, PINNED);
    free_hemem(&h_dc, &d_dc, PAGEABLE);
    free(k_requests);
    free(k_responses);
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

    _fill_tasks(h_dc);
    _init_streams();
    current = 0;
}

static void _cleanup_context()
{
    _free_mem();
    _destroy_streams();
}

static void _copy_context()
{
    int i;
    csc( cudaMemcpy((void*)d_dc, (void*)h_dc,
		    sizeof(nsk_device_context_t),
		    cudaMemcpyHostToDevice));
    for (i=0; i<3; i++)
	csc( cudaMemcpy((void*)d_mems[i], (void*)h_mems[i],
			NSK_MEM_SIZE, cudaMemcpyHostToDevice));
}

static void* _get_next_device_mem()
{
    return NULL;
}

static void _put_device_mem(void* devmem)
{
}

static int _prepare_task(int next)
{
    nsk_request_t *nreq = h_requests+next;
    nsk_response_t *nresp = h_responses+next;
    nsk_request_t *kreq = k_requests+next;

    volatile void* dmem = _get_next_device_mem();
    if (dmem == NULL)
	return 0;

    csc( cudaMemcpyAsync(d_mem, kreq->inputs, kreq->insize,
			 cudaMemcpyHostToDevice,
			 sch2d));
    memcpy(nreq, kreq, sizeof(nsk_request_t));
    nreq->inputs = d_mem;
    nreq->outputs = d_mem + (kreq->outputs - kreq->inputs);

    nresp->request_id = nreq->request_id;
    nresp->state = NSK_TREADY;
    nresp->errno = 0;
    csc( cudaMemcpyAsync((void*)(h_dc->responses+next), (void*)nresp,
			 sizeof(nsk_response_t), cudaMemcpyHostToDevice, sch2d));
    csc( cudaMemcpyAsync((void*)(h_dc->requests+next), (void*)nreq,
			 sizeof(nsk_request_t), cudaMemcpyHostToDevice, sch2d));
    return 1;
}

static void _start_task(int which)
{
    h_dc->current = which;
    csc( cudaMemcpyAsync((void*)(&d_dc->current), (void*)(&h_dc->current),
			 sizeof(int), cudaMemcpyHostToDevice, sch2d));
    //csc( cudaStreamSynchronize(sch2d));
}

static void _finish_task(int which)
{
    if (which == -1)
	return;
    if (h_requests[which].taskfunc == NOP_TASK)
	return;
    nsk_request_t *hreq = h_requests+which;
    nsk_request_t *kreq = k_requests+which;

    csc( cudaMemcpyAsync(kreq->outputs, hreq->outputs,
				 cudaMemcpyDeviceToHost, scd2h));
    //csc( cudaStreamSynchronize(scd2h));   
}


static int _current_task_done()
{
    if (current == -1)
	return 1;
    if (h_requests[current].taskfunc == NOP_TASK)
	return 1;
    if (cudaStreamQuery(sch2d) == cudaSuccess) {
	csc( cudaMemcpyAsync((void*)(h_responses+current),
			     (void*)(h_dc->responses+current),
			     sizeof(nsk_response_t),
			     cudaMemcpyDeviceToHost, sdd2h));
	csc( cudaStreamSynchronize(sdd2h));
	if (h_responses[current].state == NSK_TSTOPPED)
	    return 1;
    }
    return 0;
}

static int _task_finished(int which)
{
    if (which == -1)
	return 1;
    if (h_requests[which].taskfunc == NOP_TASK)
	return 1;
    
    cudaError_t e = cudaStreamQuery(scd2h);
    switch (e) {
    case cudaSuccess:
	_put_device_mem(h_requests[which].inputs);
	k_responses[which].state = NSK_TSTOPPED;
	return 1;
    case cudaErrorNotReady:
	break;
    default:
	csc(e);
	break;
    }

    return 0;
}

static int _poll_next_task()
{
    return current;
}

static void _nskhost()
{
    _init_context();
    _copy_context();

    nskmaster<<<1,1,0,smaster>>>(d_dc);
    nskslave<<<griddim, blockdim, 0, sslave>>>(d_dc);

    while(1){
	if (_current_task_done()) {
	    if (_task_finished(last)) {
		_finish_task(current);
		last = current;
		if (next != current) {
		    current = next;
		    _start_task(current);
		    if (current == -1)
			break;
		}

	    }
	}
	if (next == current)) {
	    next = _poll_next_task();
	    if (next != current && next != -1) {
		_prepare_task(next);
	    }
	}        
    }

    _cleanup_context();
}
