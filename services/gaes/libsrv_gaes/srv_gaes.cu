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
#include "../gaesu.h"

#define BYTES_PER_BLOCK  1024
#define BYTES_PER_THREAD 4
#define BYTES_PER_GROUP  16
#define THREAD_PER_BLOCK (BYTES_PER_BLOCK/BYTES_PER_THREAD)
#define WORDS_PER_BLOCK (BYTES_PER_BLOCK/4)

#define BPT_BYTES_PER_BLOCK 4096

struct service gaes_ecb_enc_srv;
struct service gaes_ecb_dec_srv;

struct service gaes_ctr_srv;
struct service gaes_lctr_srv;

struct service bp4t_gaes_ecb_enc_srv;
struct service bp4t_gaes_ecb_dec_srv;

struct gaes_ecb_data {
    u32 *d_key;
    u32 *h_key;
    int nrounds;
};

struct gaes_ctr_data {
    u32 *d_key;
    u32 *h_key;
    u8 *d_ctr;
    u8 *h_ctr;
    int nrounds;
};

static void dump_hex(u8* p, int rs, int cs)
{
    int r,c;
    printf("\n");
    for (r=0; r<rs; r++) {
	for (c=0; c<cs; c++) {
	    printf("%02x ", p[r*cs+c]);
	}
	printf("\n");
    }
}

/*
 * Include device code
 */
#include "dev.cu"

int gaes_ecb_compute_size_bpt(struct service_request *sr)
{
    sr->block_x = sr->kureq.outsize>=BPT_BYTES_PER_BLOCK? BPT_BYTES_PER_BLOCK/16: sr->kureq.outsize/16;
    sr->grid_x = sr->kureq.outsize/BPT_BYTES_PER_BLOCK? sr->kureq.outsize/BPT_BYTES_PER_BLOCK:1;
    sr->block_y = 1;
    sr->grid_y = 1;

    return 0;
}

int gaes_ecb_compute_size_bp4t(struct service_request *sr)
{
    sr->block_y = sr->kureq.outsize>=BYTES_PER_BLOCK? BYTES_PER_BLOCK/BYTES_PER_GROUP: (sr->kureq.outsize/BYTES_PER_GROUP);
    sr->grid_x = sr->kureq.outsize/BYTES_PER_BLOCK? sr->kureq.outsize/BYTES_PER_BLOCK:1;
    sr->block_x = BYTES_PER_GROUP/BYTES_PER_THREAD;
    sr->grid_y = 1;

    return 0;
}

int gaes_ecb_launch_bpt(struct service_request *sr)
{
    struct gaes_ecb_data *data = (struct gaes_ecb_data*)sr->data;
    
    if (sr->s == &gaes_ecb_dec_srv)        
	aes_decrypt_bpt<<<dim3(sr->grid_x, sr->grid_y), dim3(sr->block_x, sr->block_y), 0, (cudaStream_t)(sr->stream)>>>
	    (data->d_key, data->nrounds, (u8*)sr->doutput);
    else
	aes_encrypt_bpt<<<dim3(sr->grid_x, sr->grid_y), dim3(sr->block_x, sr->block_y), 0, (cudaStream_t)(sr->stream)>>>
	    (data->d_key, data->nrounds, (u8*)sr->doutput);
    return 0;
}

int gaes_ecb_launch_bp4t(struct service_request *sr)
{
    struct gaes_ecb_data *data = (struct gaes_ecb_data*)sr->data;
    
    if (sr->s == &gaes_ecb_dec_srv)        
	aes_decrypt_bp4t<<<dim3(sr->grid_x, sr->grid_y), dim3(sr->block_x, sr->block_y), 0, (cudaStream_t)(sr->stream)>>>
	    (data->d_key, data->nrounds, (u8*)sr->doutput);
    else
	aes_encrypt_bp4t<<<dim3(sr->grid_x, sr->grid_y), dim3(sr->block_x, sr->block_y), 0, (cudaStream_t)(sr->stream)>>>
	    (data->d_key, data->nrounds, (u8*)sr->doutput);
    return 0;
}

int gaes_ecb_prepare(struct service_request *sr)
{
    cudaStream_t s = (cudaStream_t)(sr->stream);//get_stream(sr->stream_id);
    struct gaes_ecb_data *data = (struct gaes_ecb_data *)malloc(sizeof(struct gaes_ecb_data));
    struct crypto_aes_ctx *ctx = (struct crypto_aes_ctx*)((u8*)(sr->kureq.input) + (sr->kureq.outsize));
    
    data->nrounds = ctx->key_length/4+6;
    data->h_key = (sr->s == &gaes_ecb_dec_srv)?ctx->key_dec: ctx->key_enc;                         
    data->d_key = (u32*)((u8*)(sr->dinput) + ((u8*)data->h_key - (u8*)sr->kureq.input));
    sr->data = data;
    
    csc( ah2dcpy( sr->dinput, sr->kureq.input, sr->kureq.insize, s) );
    return 0;
}

int gaes_ecb_post(struct service_request *sr)
{
    cudaStream_t s = (cudaStream_t)(sr->stream);//get_stream(sr->stream_id);
    csc( ad2hcpy( sr->kureq.output, sr->doutput, sr->kureq.outsize, s) );
    
    free(sr->data);
    sr->data = NULL;
    return 0;
}

#define gaes_ctr_compute_size gaes_ecb_compute_size_bpt
#define gaes_ctr_post gaes_ecb_post

int gaes_ctr_launch(struct service_request *sr)
{
    struct gaes_ctr_data *data = (struct gaes_ctr_data*)sr->data;

    aes_ctr_crypt<<<dim3(sr->grid_x, sr->grid_y), dim3(sr->block_x, sr->block_y), 0, (cudaStream_t)(sr->stream)>>>
	(data->d_key, data->nrounds, (u8*)sr->doutput, data->d_ctr);
    return 0;
}

int gaes_lctr_launch(struct service_request *sr)
{
    struct gaes_ctr_data *data = (struct gaes_ctr_data*)sr->data;

    aes_lctr_crypt<<<dim3(sr->grid_x, sr->grid_y), dim3(sr->block_x, sr->block_y), 0, (cudaStream_t)(sr->stream)>>>
	(data->d_key, data->nrounds, (u8*)sr->doutput, data->d_ctr);
    return 0;
}

int gaes_ctr_prepare(struct service_request *sr)
{
    cudaStream_t s = (cudaStream_t)(sr->stream);
    struct gaes_ctr_data *data = (struct gaes_ctr_data *)malloc(sizeof(struct gaes_ctr_data));
    struct crypto_gaes_ctr_info *info = (struct crypto_gaes_ctr_info*)
	((u8*)(sr->kureq.input) + (sr->kureq.outsize));
    
    data->nrounds = info->key_length/4+6;
    data->h_key = info->key_enc;
    data->d_key = (u32*)((u8*)(sr->dinput) + ((u8*)data->h_key - (u8*)sr->kureq.input));
    data->h_ctr = info->ctrblk;
    data->d_ctr = (u8*)((u8*)(sr->dinput) + ((u8*)data->h_ctr - (u8*)sr->kureq.input));
    sr->data = data;
    
    csc( ah2dcpy( sr->dinput, sr->kureq.input, sr->kureq.insize, s) );
    return 0;
}

/*
 * Naming convention of ciphers:
 * g{algorithm}_{mode}[-({enc}|{dev})]
 *
 * {}  : var value
 * []  : optional
 * (|) : or
 */
extern "C" int init_service(void *lh, int (*reg_srv)(struct service*, void*))
{
    int err;
    printf("[libsrv_gaes] Info: init gaes services\n");
    
    sprintf(gaes_ecb_enc_srv.name, "gaes_ecb-enc");
    gaes_ecb_enc_srv.sid = 0;
    gaes_ecb_enc_srv.compute_size = gaes_ecb_compute_size_bpt;
    gaes_ecb_enc_srv.launch = gaes_ecb_launch_bpt;
    gaes_ecb_enc_srv.prepare = gaes_ecb_prepare;
    gaes_ecb_enc_srv.post = gaes_ecb_post;
    
    sprintf(gaes_ecb_dec_srv.name, "gaes_ecb-dec");
    gaes_ecb_dec_srv.sid = 0;
    gaes_ecb_dec_srv.compute_size = gaes_ecb_compute_size_bpt;
    gaes_ecb_dec_srv.launch = gaes_ecb_launch_bpt;
    gaes_ecb_dec_srv.prepare = gaes_ecb_prepare;
    gaes_ecb_dec_srv.post = gaes_ecb_post;

    sprintf(gaes_ctr_srv.name, "gaes_ctr");
    gaes_ctr_srv.sid = 0;
    gaes_ctr_srv.compute_size = gaes_ctr_compute_size;
    gaes_ctr_srv.launch = gaes_ctr_launch;
    gaes_ctr_srv.prepare = gaes_ctr_prepare;
    gaes_ctr_srv.post = gaes_ctr_post;

    sprintf(gaes_lctr_srv.name, "gaes_lctr");
    gaes_lctr_srv.sid = 0;
    gaes_lctr_srv.compute_size = gaes_ctr_compute_size;
    gaes_lctr_srv.launch = gaes_lctr_launch;
    gaes_lctr_srv.prepare = gaes_ctr_prepare;
    gaes_lctr_srv.post = gaes_ctr_post;

    err = reg_srv(&gaes_ecb_enc_srv, lh);
    err |= reg_srv(&gaes_ecb_dec_srv, lh);
    err |= reg_srv(&gaes_ctr_srv, lh);
    err |= reg_srv(&gaes_lctr_srv, lh);
    if (err) {
    	fprintf(stderr, "[libsrv_gaes] Error: failed to register gaes services\n");
    } 
    
    return err;
}

extern "C" int finit_service(void *lh, int (*unreg_srv)(const char*))
{
    int err;
    printf("[libsrv_gaes] Info: finit gaes services\n");
    
    err = unreg_srv(gaes_ecb_enc_srv.name);
    err |= unreg_srv(gaes_ecb_dec_srv.name);
    err |= unreg_srv(gaes_ctr_srv.name);
    err |= unreg_srv(gaes_lctr_srv.name);
    if (err) {
    	fprintf(stderr, "[libsrv_gaes] Error: failed to unregister gaes services\n");
    }
    
    return err;
}


