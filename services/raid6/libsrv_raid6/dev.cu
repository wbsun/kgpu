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

#define NBYTES(x) ((x) * 0x0101010101010101UL)
#define NSIZE  8
#define NSHIFT 3

#define SHLBYTE(v) (((v)<<1)&NBYTES(0xfe))
#define MASK(v) ({ u64 vv = (v)&NBYTES(0x80); (vv<<1)-(vv>>7);})

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
    for (offset64 -= step64; offset64>=0; offset64 -=step64) {
	wd0 = d[offset64];
	wp0 ^= wd0;
	wq0 = SHLBYTE(wq0) ^ (MASK(wq0)&NBYTES(0x1d)) ^ wd0;
    }
    d[step64*(z0+1)+tid] = wp0;
    d[step64*(z0+2)+tid] = wq0;    
}

/*
 * Fixed number of disks version
 * Naming: _fdx, where x is the number of disks, including p and q.
 *
 */
__global__ void raid6_pq_fd6(unsigned int disks, unsigned long dsize, u8 *data)
{
    u64 *d;;
    int step64, tid;

    u64 wq0, wp0;

    tid = blockDim.x*blockIdx.x+threadIdx.x;
    step64 = dsize/sizeof(u64);
    d = ((u64*)data)+tid+3*step64;
    
    wq0 = wp0 = *d;
    d -= step64;
    
    wp0 ^= *d;
    wq0 =
	SHLBYTE(wq0) ^ (MASK(wq0)&NBYTES(0x1d)) ^ *d;
    d-= step64;
    
    wp0 ^= *d;
    wq0 =
	SHLBYTE(wq0) ^ (MASK(wq0)&NBYTES(0x1d)) ^ *d;
    d -= step64;
    
    wp0 ^= *d;
    wq0 =
	SHLBYTE(wq0) ^ (MASK(wq0)&NBYTES(0x1d)) ^ *d;
    d += 4*step64;
    
    *d = wp0;
    *(d+step64) = wq0;
}
