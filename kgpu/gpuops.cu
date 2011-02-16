#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include "helper.h"
#include "service.h"


extern "C" int alloc_gpu_mem(struct service_request *sreq);
extern "C" void free_gpu_mem(struct service_request *sreq);
extern "C" int alloc_stream(struct service_request *sreq);
extern "C" void free_stream(struct service_request *sreq);
extern "C" struct service_request* alloc_service_request();
extern "C" void free_service_request(struct service_request *sreq);

extern "C" int alloc_gpu_buffers(struct gpu_buffer gbufs[], int n, unsigned long size);
extern "C" void free_gpu_buffers(struct gpu_buffer gbufs[], int n);

#define MAX_STREAM_NR 4
cudaStream_t streams[MAX_STREAM_NR];

const dim3 default_block_size(32,1);
const dim3 default_grid_size(512,1);


cudaStream_t get_stream(int stid)
{
    if (stid < 0 || stid >= MAX_STREAM_NR)
	return 0;
    else
	return streams[stid];
}

int default_compute_size(struct service_request *sreq)
{
    sreq->block_x = default_block_size.x;
    sreq->block_y = default_block_size.y;
    sreq->grid_x = default_grid_size.x;
    sreq->grid_y = default_grid_size.y;
    return 0;
}

int default_prepare(struct service_request *sreq)
{
    cudaStream_t s = get_stream(sreq->stream_id);
    csc( ah2dcpy( sreq->dinput, sreq->kureq.input, sreq->kureq.insize, s) );
    return 0;
}

int default_post(struct service_request *sreq)
{
    cudaStream_t s = get_stream(sreq->stream_id);
    csc( ad2hcpy( sreq->kureq.output, sreq->doutput, sreq->kureq.outsize, s) );
    return 0;
}
