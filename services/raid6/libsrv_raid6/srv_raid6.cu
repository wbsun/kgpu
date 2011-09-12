/* This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2010-2011 University of Utah and the Flux Group.
 * All rights reserved.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "../../../kgpu/kgpu.h"
#include "../../../kgpu/gputils.h"
#include "../gpq.h"
#include "../r62_recov.h"

#define SECTOR_SIZE 512
#ifndef PAGE_SIZE
#define PAGE_SIZE 4096
#endif
#define BYTES_PER_THREAD 8
#define BYTES_PER_BLOCK (SECTOR_SIZE*8)
#define THREADS_PER_BLOCK (BYTES_PER_BLOCK/BYTES_PER_THREAD)

struct kgpu_service raid6_pq_srv;
struct kgpu_service r62_recov_srv;

/*
 * Include device code
 */
#include "dev.cu"

int r62_recov_compute_size(struct kgpu_service_request *sr)
{
    struct r62_recov_data *data = (struct r62_recov_data*)sr->hdata;
    
    sr->block_x = SECTOR_SIZE;
    sr->block_y = 1;
    sr->grid_x  = data->n;
    sr->grid_y  = PAGE_SIZE/SECTOR_SIZE;

    return 0;
}

int r62_recov_prepare(struct kgpu_service_request *sr)
{
    cudaStream_t s = (cudaStream_t)(sr->stream);
  
    csc( ah2dcpy( sr->din, sr->hin, sr->insize, s) );

    return 0;
}

int r62_recov_launch(struct kgpu_service_request *sr)
{
    struct r62_recov_data *data = (struct r62_recov_data*)sr->hdata;
    struct r62_recov_data *dd = (struct r62_recov_data*)sr->ddata;
    cudaStream_t s = (cudaStream_t)(sr->stream);

    raid6_recov_2data<<<dim3(sr->grid_x, sr->grid_y),
	dim3(sr->block_x, sr->block_y), 0, s>>>(
	    (u8*)(sr->din),
	    ((u8*)(sr->din))+data->bytes,
	    (u8*)(sr->dout),
	    ((u8*)(sr->dout))+data->bytes,
	    dd);
    
    return 0;
}

int r62_recov_post(struct kgpu_service_request *sr)
{
    cudaStream_t s = (cudaStream_t)(sr->stream);
    
    csc( ad2hcpy( sr->hout, sr->dout, sr->outsize, s ) );

    return 0;
}


int raid6_pq_compute_size(struct kgpu_service_request *sr)
{
    struct raid6_pq_data* data = (struct raid6_pq_data*)(sr->hdata);
    
    sr->block_x = THREADS_PER_BLOCK;
    sr->block_y = 1;
    sr->grid_x  = data->dsize/BYTES_PER_BLOCK;
    sr->grid_y  = 1;

    return 0;
}

int raid6_pq_prepare(struct kgpu_service_request *sr)
{
    struct raid6_pq_data* data = (struct raid6_pq_data*)(sr->hdata);
    cudaStream_t s = (cudaStream_t)(sr->stream);
  
    csc( ah2dcpy( sr->din, sr->hin, data->dsize*(data->nr_d-2), s) );

    return 0;
}

int raid6_pq_launch(struct kgpu_service_request *sr)
{
    struct raid6_pq_data* data = (struct raid6_pq_data*)(sr->hdata);
    cudaStream_t s = (cudaStream_t)(sr->stream);

    raid6_pq<<<dim3(sr->grid_x, sr->grid_y),
	dim3(sr->block_x, sr->block_y), 0,
      s>>>(
	  (unsigned int)data->nr_d,
	  (unsigned long)data->dsize, (u8*)sr->din);
      
    return 0;
}

int raid6_pq_post(struct kgpu_service_request *sr)
{
    cudaStream_t s = (cudaStream_t)(sr->stream);

    csc( ad2hcpy( sr->hout, sr->dout, sr->outsize, s ) );

    return 0;
}

extern "C" int init_service(void *lh, int (*reg_srv)(struct kgpu_service*, void*))
{
    int err;
    printf("[libsrv_raid6] Info: init raid6 services\n");

    csc( cudaFuncSetCacheConfig(raid6_pq, cudaFuncCachePreferL1) );
    csc( cudaFuncSetCacheConfig(raid6_recov_2data, cudaFuncCachePreferL1) );

    sprintf(raid6_pq_srv.name, "raid6_pq");
    raid6_pq_srv.sid = 0;
    raid6_pq_srv.compute_size = raid6_pq_compute_size;
    raid6_pq_srv.launch = raid6_pq_launch;
    raid6_pq_srv.prepare = raid6_pq_prepare;
    raid6_pq_srv.post = raid6_pq_post;

    err = reg_srv(&raid6_pq_srv, lh);
    if (err) {
	fprintf(stderr, "[libsrv_raid6] Error: failed"
	    " to register raid6_pq service\n");
	return err;
    }

    sprintf(r62_recov_srv.name, "r62_recov");
    r62_recov_srv.sid = 1;
    r62_recov_srv.compute_size = r62_recov_compute_size;
    r62_recov_srv.launch = r62_recov_launch;
    r62_recov_srv.prepare = r62_recov_prepare;
    r62_recov_srv.post = r62_recov_post;

    err = reg_srv(&r62_recov_srv, lh);
    if (err) {
	fprintf(stderr, "[libsrv_raid6] Error: failed"
	    " to register r62_recov service\n");
    }
    return err;
}

extern "C" int finit_service(void *lh, int (*unreg_srv)(const char *))
{
    int err1, err2;
    printf("[libsrv_raid6] Info: finit raid6 services\n");
    
    err1 = unreg_srv(raid6_pq_srv.name);
    if (err1) {
	fprintf(stderr, "[libsrv_raid6] Error: failed"
	    " to unregister raid6_pq service\n");
    }

    err2 = unreg_srv(r62_recov_srv.name);
    if (err2) {
	fprintf(stderr, "[libsrv_raid6] Error: failed"
	    " to unregister r62_recov service\n");
    }
    return err1|err2;
}

