#include <cuda.h>
#include "nsk.h"

__device__ volatile int slavesdone = 0;

__device__ int myblockid()
{
    return blockIdx.y*gridDim.x+blockIdx.x;
}

__device__ int mythreadid()
{
    return myblockid()*(blockDim.x*blockDim.y)
	+ threadIdx.y*blockDim.x+threadIdx.x;
}

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

__device__ void testtask(nsk_hd_request_t *req, int outval)
{
    volatile int *odata = (volatile int*)(req->outputs);
    volatile int *idata = (volatile int*)(req->inputs);

    int mytid = mythreadid();
    odata[mytid] = outval+idata[mytid];
    __threadfence_system();
}

__device__ int nop(nsk_hd_request_t *req)
{
    return 0;
}

__device__ int sha1(nsk_hd_request_t *req)
{
    testtask(req, 1);
    
    return 0;
}

__device__ int iplookup(nsk_hd_request_t *req)
{
    testtask(req, 2);
    return 0;
}

__device__ int decrypt(nsk_hd_request_t *req)
{
    testtask(req, 3);
    return 0;
}

__device__ int encrypt(nsk_hd_request_t *req)
{
    testtask(req, 4);
    return 0;
}

void fill_tasks(nsk_device_context_t *dc)
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

void start_device_kernels(nsk_device_context_t *dc, )
{    
    nskmaster<<<1,1,0,smaster>>>(dc);
    nskslave<<<griddim, blockdim, 0, sslave>>>(dc);
}
