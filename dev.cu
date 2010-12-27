#include <cuda.h>
#include "devutils.h"
#include "nsk.h"
#include <stdio.h>

__device__ volatile int		d_slavesdone;
__device__ nsk_task_func_t	d_task_funcs[NSK_MAX_TASK_FUNC_NR];
__device__ volatile nsk_request_t*	d_requests;
__device__ volatile nsk_response_t*	d_responses;
__device__ volatile int		d_current;
__device__ volatile int		d_taskdone;

nsk_request_t* dh_requests;
nsk_response_t* dh_responses;

dim3 blockdim = dim3(BLOCKS_X,1);
dim3 griddim = dim3(GRIDS_X,1);

void init_hd_context()
{
    init_hd_streams();
    init_hd_buffers();
    // set d_slavesdone, d_taskdone and d_current, the int value data		
    csc( cudaHostAlloc( (void**)&h_current, sizeof(int), 0 ) );
    csc( cudaHostAlloc( (void**)&h_taskdone, sizeof(int), 0 ) );
    *h_current = 0;
    *h_taskdone = 0;
    csc( h2d_sbl_a( d_slavesdone, h_current, sizeof(int), ss[SKERNEL] ) );	
    csc( h2d_sbl_a( d_current, h_current, sizeof(int), ss[SKERNEL] ) );
    csc( h2d_sbl_a( d_taskdone, h_taskdone, sizeof(int), ss[SKERNEL] ) );
    csc( cudaStreamSynchronize(ss[SKERNEL]) );	
	
    // set task function addresses
    nsk_task_func_t *tfs;
    csc( cudaHostAlloc( (void**)&tfs, sizeof(nsk_task_func_t)*NSK_MAX_TASK_FUNC_NR, 0 ) );
    fill_tasks(tfs);
    csc( h2d_sbl_a( d_task_funcs, tfs, sizeof(nsk_task_func_t)*NSK_MAX_TASK_FUNC_NR, ss[SKERNEL] ) );
    csc( cudaStreamSynchronize(ss[SKERNEL]) );
    csc( cudaFreeHost(tfs) );
	
    // set pointer-type device vars
    {
	void **hdp;
	csc( cudaHostAlloc( (void**)&hdp, sizeof(void*), 0 ) );
	void *hp, *dp;
	ALLOC_HDMEM( &hp, &dp, sizeof(nsk_request_t)*NSK_MAX_REQ_NR, PINNED );
	h_requests = (nsk_request_t*)hp;
	dh_requests = (nsk_request_t*)dp;
	*hdp = dp;
	csc( h2d_sbl_a( d_requests, hdp, sizeof(void*), ss[SKERNEL] ) );
	csc( cudaStreamSynchronize(ss[SKERNEL]) );
    }
	
    {
	void **hdp;
	csc( cudaHostAlloc( (void**)&hdp, sizeof(void*), 0 ) );
	void *hp, *dp;
	ALLOC_HDMEM( &hp, &dp, sizeof(nsk_response_t)*NSK_MAX_REQ_NR, PINNED );
	h_responses = (nsk_response_t*)hp;
	dh_responses = (nsk_response_t*)dp;
	*hdp = dp;
	csc( h2d_sbl_a( d_responses, hdp, sizeof(void*), ss[SKERNEL] ) );
	csc( cudaStreamSynchronize(ss[SKERNEL]) );
    }
	
    init_hd_buffers();
}

void clean_hd_context()
{
    // ignore this now
    csc(cudaThreadSynchronize());
    cudaThreadExit();
}

int prepare_hd_task(nsk_request_t *kreq, int which)
{
    nsk_request_t *hreq = h_requests+which;
    nsk_response_t *hresp = h_responses+which;
	
    nsk_request_t *dreq = dh_requests+which;
    nsk_response_t *dresp = dh_responses+which;
	
    memcpy( (void*)hreq, (void*)kreq, sizeof(nsk_request_t) );
	
    hreq->inputs = get_next_device_mem(which);
    if (hreq->inputs == NULL)
	return 0;
	
    unsigned long offset = (unsigned long)((unsigned long)(kreq->outputs) - (unsigned long)(kreq->inputs));
    hreq->outputs = (volatile void *)((unsigned long)(hreq->inputs) + offset);
	
    memset( (void*)hresp, 0, sizeof(nsk_response_t) );
    csc( h2d_cpy_a( hreq->inputs, kreq->inputs, kreq->insize, ss[SCOM] ) );
    csc( h2d_cpy_a( dresp, hresp, sizeof(nsk_response_t), ss[SCOM] ) );
    csc( h2d_cpy_a( dreq, hreq, sizeof(nsk_request_t), ss[SCOM] ) );
    csc( cudaStreamSynchronize(ss[SCOM]) );
	
    return 1;
}

void start_hd_task(int which)
{
    if (which == -1)
	return;
    *h_current = which;
    csc( h2d_sbl_a( d_current, h_current, sizeof(int), ss[SCOM] ) );
    csc( cudaStreamSynchronize(ss[SCOM]) );
	
}

int is_current_task_done()
{
    int oldtaskdone = *h_taskdone;

    csc( d2h_sbl_a( h_taskdone, d_taskdone, sizeof(int), ss[SCOM] ) );
    csc( cudaStreamSynchronize(ss[SCOM]) );
    if (*h_taskdone != oldtaskdone){		
	return 1;
    }
    else
	return 0;
}

void finish_task(int which, nsk_request_t *kreq)
{
    if (which == -1)
	return;
    if (h_requests[which].taskfunc == NOP_TASK)
	return;
	
    nsk_request_t *hreq = h_requests+which;
	
    csc( d2h_cpy_a( kreq->outputs, hreq->outputs, kreq->outsize, ss[SCOM] ) );
    csc( cudaStreamSynchronize(ss[SCOM]) );
    put_device_mem( hreq->inputs );
}

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
    return d_slavesdone == gridDim.x;
}

__device__ void resetslavesdone()
{
    d_slavesdone = 0;
    __threadfence_system();
}

__device__ void slavedone()
{
    atomicAdd((int*)&d_slavesdone,1);
    __threadfence_system();
}

__device__ void testtask(volatile nsk_request_t *req, int outval)
{
    volatile int *odata = (volatile int*)(req->outputs);
    volatile int *idata = (volatile int*)(req->inputs);

    int mytid = mythreadid();
    odata[mytid] = req->request_id;//outval+idata[mytid];
    __threadfence_system();
}

__device__ int nop(volatile nsk_request_t *req)
{
    return 0;
}

__device__ int sha1(volatile nsk_request_t *req)
{
    testtask(req, 1);
		
    return 0;
}

__device__ int iplookup(volatile nsk_request_t *req)
{
    testtask(req, 2);
	
    return 0;
}

__device__ int decrypt(volatile nsk_request_t *req)
{
    testtask(req, 3);
	
    return 0;
}

__device__ int encrypt(volatile nsk_request_t *req)
{
    testtask(req, 4);
	
    return 0;
}

__device__ nsk_task_func_t p_nop = nop;
__device__ nsk_task_func_t p_sha1 = sha1;
__device__ nsk_task_func_t p_iplookup = iplookup;
__device__ nsk_task_func_t p_decrypt = decrypt;
__device__ nsk_task_func_t p_encrypt = encrypt;

#define dev_vrequest(which) ((volatile nsk_request_t*)(&(d_requests[which])))
#define dev_vresponse(which) ((volatile nsk_response_t*)(&(d_responses[which])))
#define dev_taskfunction(which) (d_task_funcs[which])

__global__ void nskdevkernel()
{
    volatile int current = d_current;
    volatile nsk_request_t *req;
	
    while (d_current != -1) {
	if (current != d_current) {
	    current = d_current;
			
	    if (current == -1)
		break;	// risk of synchronization within a block
			
	    req = dev_vrequest(current);
	    __syncthreads();
	    if (req->taskfunc == -1)
		return;
	    dev_taskfunction(req->taskfunc)(req);
	    __syncthreads();
			
	    if (threadIdx.x == 0) {
		slavedone();
	    }
	    if (threadIdx.x == 0 && blockIdx.x == 0/*mythreadid() == 0*/) {
		while (!allslavesdone())
		    ;
		resetslavesdone();
				
		dev_vresponse(current)->state = NSK_TSTOPPED;
		atomicAdd((int*)&d_taskdone, 1);				
		__threadfence_system();
	    }
	    __syncthreads();
	}
    }
}



void fill_tasks(nsk_task_func_t tfs[])
{
    int i;
    csc( cudaMemcpyFromSymbolAsync( (void*)&tfs[0], p_nop, sizeof(nsk_task_func_t), 0, cudaMemcpyDeviceToHost, ss[SKERNEL] ));
    csc( cudaMemcpyFromSymbolAsync( (void*)&tfs[1], p_sha1, sizeof(nsk_task_func_t), 0, cudaMemcpyDeviceToHost, ss[SKERNEL] ));
    csc( cudaMemcpyFromSymbolAsync( (void*)&tfs[2], p_iplookup, sizeof(nsk_task_func_t), 0, cudaMemcpyDeviceToHost, ss[SKERNEL] ));
    csc( cudaMemcpyFromSymbolAsync( (void*)&tfs[3], p_decrypt, sizeof(nsk_task_func_t), 0, cudaMemcpyDeviceToHost, ss[SKERNEL] ));
    csc( cudaMemcpyFromSymbolAsync( (void*)&tfs[4], p_encrypt, sizeof(nsk_task_func_t), 0, cudaMemcpyDeviceToHost, ss[SKERNEL] ));
    csc( cudaStreamSynchronize(ss[SKERNEL]) );
    for (i=5; i<NSK_MAX_TASK_FUNC_NR; i++)
	tfs[i] = NULL;
}

void start_device_kernel()
{
    nskdevkernel<<<griddim, blockdim, 0, ss[SKERNEL]>>>();
}
