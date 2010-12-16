#include <cuda.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include "nsk.h"

nsk_task_func_t h_task_funcs[NSK_MAX_TASK_FUNC_NR];

nsk_request_t *h_requests;
volatile nsk_response_t *h_responses;
nsk_request_t *k_requests;
nsk_response_t *K_responses;

nsk_device_context_t *h_dc, *d_dc;

volatile void *d_mems[3];
volatile void *h_mems[4];

int devmemuses[3];

cudaStream_t smaster, sslave, sch2d, scd2h, sdh2d, sdd2h;
int current = 0;
int last = 0;
int next = 0;

#define NOP_TASK 0

dim3 blockdim = dim3(BLOCKS_X,1);
dim3 griddim = dim3(GRIDS_X,1);


static void init_context()
{
    int i;
    nsk_buf_info_t bufs[4];
    int nskkfd;
    
    ALLOC_HDMEM(&h_dc, &d_dc, sizeof(nsk_device_context_t), PAGEABLE);
    memset((void*)h_dc, 0, sizeof(nsk_device_context_t));

    ALLOC_HDMEM(&h_requests, &h_dc->requests,
	      sizeof(nsk_request_t)*NSK_MAX_REQ_NR, PINNED);
    ALLOC_HDMEM(&h_responses, &h_dc->responses,
	      sizeof(nsk_response_t)*NSK_MAX_REQ_NR, PINNED);
    
    k_requests = (nsk_request_t*)malloc(sizeof(nsk_request_t)*NSK_MAX_REQ_NR);
    k_responses = (nsk_response_t*)malloc(sizeof(nsk_response_t)*NSK_MAX_REQ_NR);
    mlock(k_requests, sizeof(nsk_request_t)*NSK_MAX_REQ_NR);
    mlock(k_responses, sizeof(nsk_response_t)*NSK_MAX_REQ_NR);

    for(i=0; i<3; i++){
	ALLOC_HDMEM(&(h_mems[i]), &(h_dc->mems[i]), NSK_MEM_SIZE, PINNED);
	d_mems[i] = h_dc->mems[i];
    }
    csc( cudaHostAlloc((void**)&(h_mems[3]), NSK_MEM_SIZE, PINNED));
    for(i=0; i<4; i++)
	memset((void*)h_mems[i], 0, NSK_MEM_SIZE);

    // init device memory buffers uses:
    for(i=0; i<3; i++)
	devmemuses[i] = -1; // nobody use
    
    
    // tell nsk kernel-size code the buffers
    for(i=0; i<4; i++) {
	bufs[i].addr = (void*)h_mems[i];
	bufs[i].size = NSK_MEM_SIZE;
    }
    
    nskkfd = scce(open(NSK_PROCFS_FILE, O_RDWR));
    scce(write(nskkfd, (void*)bufs, sizeof(nsk_buf_info_t)*4));
    close(nskkfd);  

    fill_tasks(h_dc);

    // create streams:
    csc(cudaStreamCreate(&smaster));
    csc(cudaStreamCreate(&sslave));
    csc(cudaStreamCreate(&sch2d));
    csc(cudaStreamCreate(&scd2h));
    csc(cudaStreamCreate(&sdh2d));
    csc(cudaStreamCreate(&sdd2h));

    // copy configurations from host to device:
    csc( cudaMemcpy((void*)d_dc, (void*)h_dc,
		    sizeof(nsk_device_context_t),
		    cudaMemcpyHostToDevice));
    for (i=0; i<3; i++)
	csc( cudaMemcpy((void*)d_mems[i], (void*)h_mems[i],
			NSK_MEM_SIZE, cudaMemcpyHostToDevice));
    
    current = 0;
}

static void cleanup_context()
{
    int i;

    for(i=0; i<3; i++)
	FREE_HDMEM(&(h_mems[i]), &(h_dc->mems[i]), PINNED);
    csc(cudaFreeHost((void*)h_mems[3]));

    FREE_HDMEM(&h_responses, &h_dc->responses, PINNED);
    FREE_HDMEM(&h_requests, &h_dc->requests, PINNED);
    free_hemem(&h_dc, &d_dc, PAGEABLE);
    free(k_requests);
    free(k_responses);
    
    csc(cudaStreamDestroy(smaster));
    csc(cudaStreamDestroy(sslave));
    csc(cudaStreamDestroy(sch2d));
    csc(cudaStreamDestroy(scd2h));
    csc(cudaStreamDestroy(sdh2d));
    csc(cudaStreamDestroy(sdd2h));
}

static void* get_next_device_mem(int which)
{
    int i;

    for (i=0; i<3; i++) {
	if (devmemuses[i] == -1) {
	    devmemuses[i] = which;
	    return d_mems[i];
	}
    }
    
    return NULL;
}

static void put_device_mem(void* devmem)
{
    int i;

    for (i=0; i<3; i++) {
	if (devmem == d_mems[i]) {
	    devmemuses[i] = -1;
	    return;
	}
    }
}

static int prepare_task(int next)
{
    nsk_request_t *nreq = h_requests+next;
    nsk_response_t *nresp = h_responses+next;
    nsk_request_t *kreq = k_requests+next;

    volatile void* d_mem = get_next_device_mem(next);
    if (d_mem == NULL)
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

static void start_task(int which)
{
    if (which == -1)
	return;
    
    h_dc->current = which;
    csc( cudaMemcpyAsync((void*)(&d_dc->current), (void*)(&h_dc->current),
			 sizeof(int), cudaMemcpyHostToDevice, sch2d));
    //csc( cudaStreamSynchronize(sch2d));
}

static void finish_task(int which)
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


static int current_task_done_on_device()
{
    if (current == -1)
	return 1;
    if (h_requests[current].taskfunc == NOP_TASK)
	return 1;
    if (current == last)
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

static int task_finished(int which)
{
    if (which == -1)
	return 1;
    if (h_requests[which].taskfunc == NOP_TASK)
	return 1;
    
    cudaError_t e = cudaStreamQuery(scd2h);
    switch (e) {
    case cudaSuccess:
	put_device_mem(h_requests[which].inputs);
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

static int poll_next_task()
{
    return current;
}

static void nskhost()
{
    init_context();
    start_device_kernels(d_dc, smaster, sslave);

    while(1){
	if (current_task_done_on_device())
	{
	    if (task_finished(last))
	    {
		if (current != last) {
		    finish_task(current);
		    last = current;
		}
		
		if (next != current) {
		    current = next;		    
		    if (current == -1)
			break;
		    
		    start_task(current);
		}

	    }
	}
	if (next == current)) {
	    next = poll_next_task();
	    if (next != current && next != -1) {
		prepare_task(next);
	    }
	}        
    }

    cleanup_context();
}

int main(int argc, char *argv[])
{
    nskhost();
    return 0;
}
