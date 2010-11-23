#include <cuda.h>
#include "../common/nsk.h"

__device__ void nskmaster(nsk_device_context_t *devctxt)
{
}


__global__ void nskmaster_launcher(
    nsk_device_context_t *devctxt)
{
    
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
