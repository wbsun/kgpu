#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "../../../kgpu/helper.h"
#include "../../../kgpu/gputils.h"
#include "gecbaes.h"


u32 key_enc[AES_MAX_KEYLENGTH_U32];
u32 key_dec[AES_MAX_KEYLENGTH_U32];


__global__ void inc_kernel(int *din, int *dout)
{
    int id = threadIdx.x +  blockIdx.x*blockDim.x;

    dout[id] = din[id]+1;    
}

int gecb_compute_size(struct service_request *sr)
{
    sr->block_x = 32;
    sr->grid_x = sr->kureq.insize/128;
    sr->block_y = 1;
    sr->grid_y = 1;

    return 0;
}

int gecb_launch(struct service_request *sr)
{
    inc_kernel<<<dim3(sr->grid_x, sr->grid_y), dim3(sr->block_x, sr->block_y), 0, (cudaStream_t)(sr->stream)>>>
	((int*)sr->dinput, (int*)sr->doutput);
    return 0;
}

int gecb_prepare(struct service_request *sr)
{
    cudaStream_t s = (cudaStream_t)(sr->stream);//get_stream(sr->stream_id);
    csc( ah2dcpy( sr->dinput, sr->kureq.input, sr->kureq.insize, s) );
    return 0;
}

int gecb_post(struct service_request *sr)
{
    cudaStream_t s = (cudaStream_t)(sr->stream);//get_stream(sr->stream_id);
    csc( ad2hcpy( sr->kureq.output, sr->doutput, sr->kureq.outsize, s) );
    return 0;
}

struct service gecb_enc_srv;
struct service gecb_dec_srv;

extern "C" int init_service(void *lh, int (*reg_srv)(struct service*, void*))
{
    int err;
    printf("[libsrv_gecb] Info: init gecb service\n");
    
    sprintf(gecb_enc_srv.name, "gecb-enc");
    gecb_enc_srv.sid = 0;
    gecb_enc_srv.compute_size = gecb_compute_size;
    gecb_enc_srv.launch = gecb_launch;
    gecb_enc_srv.prepare = gecb_prepare;
    gecb_enc_srv.post = gecb_post;
    
    sprintf(gecb_enc_srv.name, "gecb-dec");
    gecb_dec_srv.sid = 0;
    gecb_dec_srv.compute_size = gecb_compute_size;
    gecb_dec_srv.launch = gecb_launch;
    gecb_dec_srv.prepare = gecb_prepare;
    gecb_dec_srv.post = gecb_post;
    
    err = reg_srv(&gecb_enc_srv, lh);
    if (err) {
    	fprintf(stderr, "[libsrv_gecb] Error: failed to register enc service\n");
    } else {
        err = reg_srv(&gecb_dec_srv, lh);
        if (err) {
    	    fprintf(stderr, "[libsrv_gecb] Error: failed to register dec service\n");
    	    unreg_srv(gecb_enc_srv.name);
        }
    }
    
    return err;
}

extern "C" int finit_service(void *lh, int (*unreg_srv)(const char*))
{
    int err1, err2;
    printf("[libsrv_gecb] Info: finit gecb service\n");
    
    err1 = unreg_srv(gecb_enc_srv.name);
    if (err1) {
    	fprintf(stderr, "[libsrv_gecb] Error: failed to unregister enc service\n");
    }
    err2 = unreg_srv(gecb_dec_srv.name);
    if (err2) {
    	fprintf(stderr, "[libsrv_gecb] Error: failed to unregister dec service\n");
    }
    
    return err1 | err2;
}

