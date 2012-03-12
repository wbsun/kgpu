/* This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2010-2011 University of Utah and the Flux Group.
 * All rights reserved.
 *
 *
 * P and Q disk computing function, mostly derived from the kernel:
 * /lib/raid6/int.uc
 * Support x86_64 only.
 *
 * To be included by others.
 */

typedef unsigned long u64;
typedef unsigned char u8;

#include "table.h"

#define NBYTES(x) ((x) * 0x0101010101010101UL)
#define NSIZE  8
#define NSHIFT 3

#define SHLBYTE(v) (((v)<<1)&(0xfefefefefefefefe))
//(((v)<<1)&NBYTES(0xfe))
#define MASK(v) ({ u64 vv = (v)&(0x8080808080808080); (vv<<1)-(vv>>7); })

__global__ void raid6_recov_2data_nc(
    u8 *p, u8 *q, u8 *dp, u8 *dq,
    const u8 *pbmul, const u8 *qmul)
{
    int tid = threadIdx.x+blockDim.x*blockIdx.x;
    u8 px = p[tid]^dp[tid];
    u8 qx = qmul[q[tid]^dq[tid]];
    dq[tid] = pbmul[px]^qx;
    dp[tid] = dq[tid]^px;
}

__global__ void raid6_recov_2data(u8 *p, u8 *q, u8 *dp, u8 *dq,
				  struct r62_recov_data* data)
{
    int tid = threadIdx.x+blockDim.x*(blockIdx.x*gridDim.y+blockIdx.y);
    const u8 *pbmul = draid6_gfmul[data->idx[blockIdx.x].pbidx];
    const u8 *qmul = draid6_gfmul[data->idx[blockIdx.x].qidx];
    
    u8 px = p[tid] ^ dp[tid];
    u8 qx = qmul[q[tid] ^ dq[tid]];
    dq[tid] = pbmul[px] ^ qx;
    dp[tid] = dq[tid] ^ px;
}



/*
 * @disks: number of disks, p and q included
 * @dsize: unit size, or a stripe?
 * @data: disk data 
 */
__global__ void raid6_pq(unsigned int disks, unsigned long dsize, u8 *data)
{
    u64 *d = (u64*)data;
    int z0, offset64, step64, tid;

    u64 wd0, wq0, wp0;
    
    tid = blockDim.x*blockIdx.x+threadIdx.x;
    step64 = dsize/sizeof(u64);
    z0 = disks-3;
    offset64 = step64*z0+tid;
    
    wq0 = wp0 = d[offset64];
    #pragma unroll 16
    for (offset64 -= step64; offset64>=0; offset64 -=step64) {
	wd0 = d[offset64];
	wp0 ^= wd0;
	wq0 = SHLBYTE(wq0) ^ (MASK(wq0)&(0x1d1d1d1d1d1d1d1d)) ^ wd0;
    }
    d[step64*(z0+1)+tid] = wp0;
    d[step64*(z0+2)+tid] = wq0;    
}

/*
 * PQ with stride
 * @disks: number of disks, p and q included
 * @dsize: unit size, or a stripe?
 * @data: disk data 
 */
__global__ void raid6_pq_str(unsigned int disks, unsigned long dsize, u8 *data, unsigned int stride)
{
    u64 *d = (u64*)data;
    int z0, offset64, step64, tid, i;

    u64 wd0, wq0, wp0;
    
    tid = blockDim.x*blockIdx.x+threadIdx.x;
    step64 = dsize/(sizeof(u64));
    z0 = disks-3;
    
    #pragma unroll 4
    for (i=0; i<stride; i++) 
    {
        offset64 = step64*z0+tid*stride+i;
    
        wq0 = wp0 = d[offset64];
        
        #pragma unroll 16
        for (offset64 -= step64; offset64>=0; offset64 -=step64) {
	    wd0 = d[offset64];
	    wp0 ^= wd0;
	    wq0 = SHLBYTE(wq0) ^ (MASK(wq0)&NBYTES(0x1d)) ^ wd0;
        }
        d[step64*(z0+1)+tid*stride+i] = wp0;
        d[step64*(z0+2)+tid*stride+i] = wq0;
    }
        
}
