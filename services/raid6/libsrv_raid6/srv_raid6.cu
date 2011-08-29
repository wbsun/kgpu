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

#define SECTOR_SIZE 512
#define BYTES_PER_THREAD 8
#define BYTES_PER_BLOCK (SECTOR_SIZE*8)
#define THREADS_PER_BLOCK (BYTES_PER_BLOCK/BYTES_PER_THREAD)

struct kgpu_service raid6_pq_srv;

/*
 * Include device code
 */
#include "dev.cu"

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

    raid6_pq<<<dim3(sr->grid_x, sr->grid_y), dim3(sr->block_x, sr->block_y), 0,
      s>>>((unsigned int)data->nr_d, (unsigned long)data->dsize, (u8*)sr->din);
      
    return 0;
}

int raid6_pq_post(struct kgpu_service_request *sr)
{
    // struct raid6_pq_data* data = (struct raid6_pq_data*)sr->data;
    cudaStream_t s = (cudaStream_t)(sr->stream);

    /* outsize should be 2*data->dsize */
    csc( ad2hcpy( sr->hout, sr->dout, sr->outsize, s ) );

    return 0;
}

extern "C" int init_service(void *lh, int (*reg_srv)(struct kgpu_service*, void*))
{
    int err;
    printf("[libsrv_raid6_pa] Info: init raid6_pq service\n");

    sprintf(raid6_pq_srv.name, "raid6_pq");
    raid6_pq_srv.sid = 0;
    raid6_pq_srv.compute_size = raid6_pq_compute_size;
    raid6_pq_srv.launch = raid6_pq_launch;
    raid6_pq_srv.prepare = raid6_pq_prepare;
    raid6_pq_srv.post = raid6_pq_post;

    err = reg_srv(&raid6_pq_srv, lh);
    if (err) {
	fprintf(stderr, "[libsrv_raid6_pq] Error: failed"
	    " to register raid6_pq service\n");
    }
    return err;
}

extern "C" int finit_service(void *lh, int (*unreg_srv)(const char *))
{
    int err;
    printf("[libsrv_raid6_pa] Info: finit raid6_pq service\n");
    
    err = unreg_srv(raid6_pq_srv.name);
    if (err) {
	fprintf(stderr, "[libsrv_raid6_pq] Error: failed"
	    " to unregister raid6_pq service\n");
    }
    return err;
}

