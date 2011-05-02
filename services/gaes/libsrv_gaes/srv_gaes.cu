/* This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2010-2011 University of Utah and the Flux Group.
 * All rights reserved.
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "../../../kgpu/helper.h"
#include "../../../kgpu/gputils.h"
#include "../gaesu.h"

#define BYTES_PER_BLOCK  1024
#define BYTES_PER_THREAD 4
#define BYTES_PER_GROUP  16
#define THREAD_PER_BLOCK (BYTES_PER_BLOCK/BYTES_PER_THREAD)
#define WORDS_PER_BLOCK (BYTES_PER_BLOCK/4)

#define BPT_BYTES_PER_BLOCK 4096

struct service gecb_enc_srv;
struct service gecb_dec_srv;

struct service bp4t_gecb_enc_srv;
struct service bp4t_gecb_dec_srv;

struct gecb_data {
        u32 *d_key;
        u32 *h_key;
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

__device__ int block_id()
{
    return blockIdx.y*gridDim.x + blockIdx.x;
}

__device__ int thread_id()
{
    return block_id()*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
}


__device__ void big_u128_add(u8 *ctr, u64 offset, u8 *res)
{
    u64 c;
    
    c = GETU32(ctr+12);
    *((u32*)(&c)+1) = GETU32(ctr+8);
    c+=offset;
    *(u64*)(res) = 0; //*(u64*)(ctr);
    *(u32*)(res+8) = GETU32((u32*)(&c)+1);
    *(u32*)(res+12) = GETU32((u32*)(&c));
}

__global__ void aes_ctr_crypt_bpt(u32 *rk, int nrounds, u8 *text, u8 *ctr)
{
    u32 s0, s1, s2, s3, t0, t1, t2, t3;
    u8 *txt = text+(16*(blockIdx.x*blockDim.x+threadIdx.x));

    s0 = GETU32(txt     ) ^ rk[0];
    s1 = GETU32(txt +  4) ^ rk[1];
    s2 = GETU32(txt +  8) ^ rk[2];
    s3 = GETU32(txt + 12) ^ rk[3];
    
    /* round 1: */
    t0 = Te0[s0 >> 24] ^ Te1[(s1 >> 16) & 0xff] ^ Te2[(s2 >>  8) & 0xff] ^ Te3[s3 & 0xff] ^ rk[ 4];
    t1 = Te0[s1 >> 24] ^ Te1[(s2 >> 16) & 0xff] ^ Te2[(s3 >>  8) & 0xff] ^ Te3[s0 & 0xff] ^ rk[ 5];
    t2 = Te0[s2 >> 24] ^ Te1[(s3 >> 16) & 0xff] ^ Te2[(s0 >>  8) & 0xff] ^ Te3[s1 & 0xff] ^ rk[ 6];
    t3 = Te0[s3 >> 24] ^ Te1[(s0 >> 16) & 0xff] ^ Te2[(s1 >>  8) & 0xff] ^ Te3[s2 & 0xff] ^ rk[ 7];
    /* round 2: */
    s0 = Te0[t0 >> 24] ^ Te1[(t1 >> 16) & 0xff] ^ Te2[(t2 >>  8) & 0xff] ^ Te3[t3 & 0xff] ^ rk[ 8];
    s1 = Te0[t1 >> 24] ^ Te1[(t2 >> 16) & 0xff] ^ Te2[(t3 >>  8) & 0xff] ^ Te3[t0 & 0xff] ^ rk[ 9];
    s2 = Te0[t2 >> 24] ^ Te1[(t3 >> 16) & 0xff] ^ Te2[(t0 >>  8) & 0xff] ^ Te3[t1 & 0xff] ^ rk[10];
    s3 = Te0[t3 >> 24] ^ Te1[(t0 >> 16) & 0xff] ^ Te2[(t1 >>  8) & 0xff] ^ Te3[t2 & 0xff] ^ rk[11];
    /* round 3: */
    t0 = Te0[s0 >> 24] ^ Te1[(s1 >> 16) & 0xff] ^ Te2[(s2 >>  8) & 0xff] ^ Te3[s3 & 0xff] ^ rk[12];
    t1 = Te0[s1 >> 24] ^ Te1[(s2 >> 16) & 0xff] ^ Te2[(s3 >>  8) & 0xff] ^ Te3[s0 & 0xff] ^ rk[13];
    t2 = Te0[s2 >> 24] ^ Te1[(s3 >> 16) & 0xff] ^ Te2[(s0 >>  8) & 0xff] ^ Te3[s1 & 0xff] ^ rk[14];
    t3 = Te0[s3 >> 24] ^ Te1[(s0 >> 16) & 0xff] ^ Te2[(s1 >>  8) & 0xff] ^ Te3[s2 & 0xff] ^ rk[15];
    /* round 4: */
    s0 = Te0[t0 >> 24] ^ Te1[(t1 >> 16) & 0xff] ^ Te2[(t2 >>  8) & 0xff] ^ Te3[t3 & 0xff] ^ rk[16];
    s1 = Te0[t1 >> 24] ^ Te1[(t2 >> 16) & 0xff] ^ Te2[(t3 >>  8) & 0xff] ^ Te3[t0 & 0xff] ^ rk[17];
    s2 = Te0[t2 >> 24] ^ Te1[(t3 >> 16) & 0xff] ^ Te2[(t0 >>  8) & 0xff] ^ Te3[t1 & 0xff] ^ rk[18];
    s3 = Te0[t3 >> 24] ^ Te1[(t0 >> 16) & 0xff] ^ Te2[(t1 >>  8) & 0xff] ^ Te3[t2 & 0xff] ^ rk[19];
    /* round 5: */
    t0 = Te0[s0 >> 24] ^ Te1[(s1 >> 16) & 0xff] ^ Te2[(s2 >>  8) & 0xff] ^ Te3[s3 & 0xff] ^ rk[20];
    t1 = Te0[s1 >> 24] ^ Te1[(s2 >> 16) & 0xff] ^ Te2[(s3 >>  8) & 0xff] ^ Te3[s0 & 0xff] ^ rk[21];
    t2 = Te0[s2 >> 24] ^ Te1[(s3 >> 16) & 0xff] ^ Te2[(s0 >>  8) & 0xff] ^ Te3[s1 & 0xff] ^ rk[22];
    t3 = Te0[s3 >> 24] ^ Te1[(s0 >> 16) & 0xff] ^ Te2[(s1 >>  8) & 0xff] ^ Te3[s2 & 0xff] ^ rk[23];
    /* round 6: */
    s0 = Te0[t0 >> 24] ^ Te1[(t1 >> 16) & 0xff] ^ Te2[(t2 >>  8) & 0xff] ^ Te3[t3 & 0xff] ^ rk[24];
    s1 = Te0[t1 >> 24] ^ Te1[(t2 >> 16) & 0xff] ^ Te2[(t3 >>  8) & 0xff] ^ Te3[t0 & 0xff] ^ rk[25];
    s2 = Te0[t2 >> 24] ^ Te1[(t3 >> 16) & 0xff] ^ Te2[(t0 >>  8) & 0xff] ^ Te3[t1 & 0xff] ^ rk[26];
    s3 = Te0[t3 >> 24] ^ Te1[(t0 >> 16) & 0xff] ^ Te2[(t1 >>  8) & 0xff] ^ Te3[t2 & 0xff] ^ rk[27];
    /* round 7: */
    t0 = Te0[s0 >> 24] ^ Te1[(s1 >> 16) & 0xff] ^ Te2[(s2 >>  8) & 0xff] ^ Te3[s3 & 0xff] ^ rk[28];
    t1 = Te0[s1 >> 24] ^ Te1[(s2 >> 16) & 0xff] ^ Te2[(s3 >>  8) & 0xff] ^ Te3[s0 & 0xff] ^ rk[29];
    t2 = Te0[s2 >> 24] ^ Te1[(s3 >> 16) & 0xff] ^ Te2[(s0 >>  8) & 0xff] ^ Te3[s1 & 0xff] ^ rk[30];
    t3 = Te0[s3 >> 24] ^ Te1[(s0 >> 16) & 0xff] ^ Te2[(s1 >>  8) & 0xff] ^ Te3[s2 & 0xff] ^ rk[31];
    /* round 8: */
    s0 = Te0[t0 >> 24] ^ Te1[(t1 >> 16) & 0xff] ^ Te2[(t2 >>  8) & 0xff] ^ Te3[t3 & 0xff] ^ rk[32];
    s1 = Te0[t1 >> 24] ^ Te1[(t2 >> 16) & 0xff] ^ Te2[(t3 >>  8) & 0xff] ^ Te3[t0 & 0xff] ^ rk[33];
    s2 = Te0[t2 >> 24] ^ Te1[(t3 >> 16) & 0xff] ^ Te2[(t0 >>  8) & 0xff] ^ Te3[t1 & 0xff] ^ rk[34];
    s3 = Te0[t3 >> 24] ^ Te1[(t0 >> 16) & 0xff] ^ Te2[(t1 >>  8) & 0xff] ^ Te3[t2 & 0xff] ^ rk[35];
    /* round 9: */
    t0 = Te0[s0 >> 24] ^ Te1[(s1 >> 16) & 0xff] ^ Te2[(s2 >>  8) & 0xff] ^ Te3[s3 & 0xff] ^ rk[36];
    t1 = Te0[s1 >> 24] ^ Te1[(s2 >> 16) & 0xff] ^ Te2[(s3 >>  8) & 0xff] ^ Te3[s0 & 0xff] ^ rk[37];
    t2 = Te0[s2 >> 24] ^ Te1[(s3 >> 16) & 0xff] ^ Te2[(s0 >>  8) & 0xff] ^ Te3[s1 & 0xff] ^ rk[38];
    t3 = Te0[s3 >> 24] ^ Te1[(s0 >> 16) & 0xff] ^ Te2[(s1 >>  8) & 0xff] ^ Te3[s2 & 0xff] ^ rk[39];
    if (nrounds > 10)
    {
      /* round 10: */
      s0 = Te0[t0 >> 24] ^ Te1[(t1 >> 16) & 0xff] ^ Te2[(t2 >>  8) & 0xff] ^ Te3[t3 & 0xff] ^ rk[40];
      s1 = Te0[t1 >> 24] ^ Te1[(t2 >> 16) & 0xff] ^ Te2[(t3 >>  8) & 0xff] ^ Te3[t0 & 0xff] ^ rk[41];
      s2 = Te0[t2 >> 24] ^ Te1[(t3 >> 16) & 0xff] ^ Te2[(t0 >>  8) & 0xff] ^ Te3[t1 & 0xff] ^ rk[42];
      s3 = Te0[t3 >> 24] ^ Te1[(t0 >> 16) & 0xff] ^ Te2[(t1 >>  8) & 0xff] ^ Te3[t2 & 0xff] ^ rk[43];
      /* round 11: */
      t0 = Te0[s0 >> 24] ^ Te1[(s1 >> 16) & 0xff] ^ Te2[(s2 >>  8) & 0xff] ^ Te3[s3 & 0xff] ^ rk[44];
      t1 = Te0[s1 >> 24] ^ Te1[(s2 >> 16) & 0xff] ^ Te2[(s3 >>  8) & 0xff] ^ Te3[s0 & 0xff] ^ rk[45];
      t2 = Te0[s2 >> 24] ^ Te1[(s3 >> 16) & 0xff] ^ Te2[(s0 >>  8) & 0xff] ^ Te3[s1 & 0xff] ^ rk[46];
      t3 = Te0[s3 >> 24] ^ Te1[(s0 >> 16) & 0xff] ^ Te2[(s1 >>  8) & 0xff] ^ Te3[s2 & 0xff] ^ rk[47];
      if (nrounds > 12)
      {
        /* round 12: */
        s0 = Te0[t0 >> 24] ^ Te1[(t1 >> 16) & 0xff] ^ Te2[(t2 >>  8) & 0xff] ^ Te3[t3 & 0xff] ^ rk[48];
        s1 = Te0[t1 >> 24] ^ Te1[(t2 >> 16) & 0xff] ^ Te2[(t3 >>  8) & 0xff] ^ Te3[t0 & 0xff] ^ rk[49];
        s2 = Te0[t2 >> 24] ^ Te1[(t3 >> 16) & 0xff] ^ Te2[(t0 >>  8) & 0xff] ^ Te3[t1 & 0xff] ^ rk[50];
        s3 = Te0[t3 >> 24] ^ Te1[(t0 >> 16) & 0xff] ^ Te2[(t1 >>  8) & 0xff] ^ Te3[t2 & 0xff] ^ rk[51];
        /* round 13: */
        t0 = Te0[s0 >> 24] ^ Te1[(s1 >> 16) & 0xff] ^ Te2[(s2 >>  8) & 0xff] ^ Te3[s3 & 0xff] ^ rk[52];
        t1 = Te0[s1 >> 24] ^ Te1[(s2 >> 16) & 0xff] ^ Te2[(s3 >>  8) & 0xff] ^ Te3[s0 & 0xff] ^ rk[53];
        t2 = Te0[s2 >> 24] ^ Te1[(s3 >> 16) & 0xff] ^ Te2[(s0 >>  8) & 0xff] ^ Te3[s1 & 0xff] ^ rk[54];
        t3 = Te0[s3 >> 24] ^ Te1[(s0 >> 16) & 0xff] ^ Te2[(s1 >>  8) & 0xff] ^ Te3[s2 & 0xff] ^ rk[55];
      }
    }
    rk += nrounds << 2;
    
    s0 =
    (Te4[(t0 >> 24)       ] & 0xff000000) ^
    (Te4[(t1 >> 16) & 0xff] & 0x00ff0000) ^
    (Te4[(t2 >>  8) & 0xff] & 0x0000ff00) ^
    (Te4[(t3      ) & 0xff] & 0x000000ff) ^
    rk[0];
    PUTU32(txt     , s0);
    s1 =
    (Te4[(t1 >> 24)       ] & 0xff000000) ^
    (Te4[(t2 >> 16) & 0xff] & 0x00ff0000) ^
    (Te4[(t3 >>  8) & 0xff] & 0x0000ff00) ^
    (Te4[(t0      ) & 0xff] & 0x000000ff) ^
    rk[1];
    PUTU32(txt +  4, s1);
    s2 =
    (Te4[(t2 >> 24)       ] & 0xff000000) ^
    (Te4[(t3 >> 16) & 0xff] & 0x00ff0000) ^
    (Te4[(t0 >>  8) & 0xff] & 0x0000ff00) ^
    (Te4[(t1      ) & 0xff] & 0x000000ff) ^
    rk[2];
    PUTU32(txt +  8, s2);
    s3 =
    (Te4[(t3 >> 24)       ] & 0xff000000) ^
    (Te4[(t0 >> 16) & 0xff] & 0x00ff0000) ^
    (Te4[(t1 >>  8) & 0xff] & 0x0000ff00) ^
    (Te4[(t2      ) & 0xff] & 0x000000ff) ^
    rk[3];
    PUTU32(txt + 12, s3);
}

__global__ void aes_encrypt_bpt(u32 *rk, int nrounds, u8* text)
{
    u32 s0, s1, s2, s3, t0, t1, t2, t3;
    u8 *txt = text+(16*(blockIdx.x*blockDim.x+threadIdx.x));

    s0 = GETU32(txt     ) ^ rk[0];
    s1 = GETU32(txt +  4) ^ rk[1];
    s2 = GETU32(txt +  8) ^ rk[2];
    s3 = GETU32(txt + 12) ^ rk[3];

    /* round 1: */
    t0 = Te0[s0 >> 24] ^ Te1[(s1 >> 16) & 0xff] ^ Te2[(s2 >>  8) & 0xff] ^ Te3[s3 & 0xff] ^ rk[ 4];
    t1 = Te0[s1 >> 24] ^ Te1[(s2 >> 16) & 0xff] ^ Te2[(s3 >>  8) & 0xff] ^ Te3[s0 & 0xff] ^ rk[ 5];
    t2 = Te0[s2 >> 24] ^ Te1[(s3 >> 16) & 0xff] ^ Te2[(s0 >>  8) & 0xff] ^ Te3[s1 & 0xff] ^ rk[ 6];
    t3 = Te0[s3 >> 24] ^ Te1[(s0 >> 16) & 0xff] ^ Te2[(s1 >>  8) & 0xff] ^ Te3[s2 & 0xff] ^ rk[ 7];
    /* round 2: */
    s0 = Te0[t0 >> 24] ^ Te1[(t1 >> 16) & 0xff] ^ Te2[(t2 >>  8) & 0xff] ^ Te3[t3 & 0xff] ^ rk[ 8];
    s1 = Te0[t1 >> 24] ^ Te1[(t2 >> 16) & 0xff] ^ Te2[(t3 >>  8) & 0xff] ^ Te3[t0 & 0xff] ^ rk[ 9];
    s2 = Te0[t2 >> 24] ^ Te1[(t3 >> 16) & 0xff] ^ Te2[(t0 >>  8) & 0xff] ^ Te3[t1 & 0xff] ^ rk[10];
    s3 = Te0[t3 >> 24] ^ Te1[(t0 >> 16) & 0xff] ^ Te2[(t1 >>  8) & 0xff] ^ Te3[t2 & 0xff] ^ rk[11];
    /* round 3: */
    t0 = Te0[s0 >> 24] ^ Te1[(s1 >> 16) & 0xff] ^ Te2[(s2 >>  8) & 0xff] ^ Te3[s3 & 0xff] ^ rk[12];
    t1 = Te0[s1 >> 24] ^ Te1[(s2 >> 16) & 0xff] ^ Te2[(s3 >>  8) & 0xff] ^ Te3[s0 & 0xff] ^ rk[13];
    t2 = Te0[s2 >> 24] ^ Te1[(s3 >> 16) & 0xff] ^ Te2[(s0 >>  8) & 0xff] ^ Te3[s1 & 0xff] ^ rk[14];
    t3 = Te0[s3 >> 24] ^ Te1[(s0 >> 16) & 0xff] ^ Te2[(s1 >>  8) & 0xff] ^ Te3[s2 & 0xff] ^ rk[15];
    /* round 4: */
    s0 = Te0[t0 >> 24] ^ Te1[(t1 >> 16) & 0xff] ^ Te2[(t2 >>  8) & 0xff] ^ Te3[t3 & 0xff] ^ rk[16];
    s1 = Te0[t1 >> 24] ^ Te1[(t2 >> 16) & 0xff] ^ Te2[(t3 >>  8) & 0xff] ^ Te3[t0 & 0xff] ^ rk[17];
    s2 = Te0[t2 >> 24] ^ Te1[(t3 >> 16) & 0xff] ^ Te2[(t0 >>  8) & 0xff] ^ Te3[t1 & 0xff] ^ rk[18];
    s3 = Te0[t3 >> 24] ^ Te1[(t0 >> 16) & 0xff] ^ Te2[(t1 >>  8) & 0xff] ^ Te3[t2 & 0xff] ^ rk[19];
    /* round 5: */
    t0 = Te0[s0 >> 24] ^ Te1[(s1 >> 16) & 0xff] ^ Te2[(s2 >>  8) & 0xff] ^ Te3[s3 & 0xff] ^ rk[20];
    t1 = Te0[s1 >> 24] ^ Te1[(s2 >> 16) & 0xff] ^ Te2[(s3 >>  8) & 0xff] ^ Te3[s0 & 0xff] ^ rk[21];
    t2 = Te0[s2 >> 24] ^ Te1[(s3 >> 16) & 0xff] ^ Te2[(s0 >>  8) & 0xff] ^ Te3[s1 & 0xff] ^ rk[22];
    t3 = Te0[s3 >> 24] ^ Te1[(s0 >> 16) & 0xff] ^ Te2[(s1 >>  8) & 0xff] ^ Te3[s2 & 0xff] ^ rk[23];
    /* round 6: */
    s0 = Te0[t0 >> 24] ^ Te1[(t1 >> 16) & 0xff] ^ Te2[(t2 >>  8) & 0xff] ^ Te3[t3 & 0xff] ^ rk[24];
    s1 = Te0[t1 >> 24] ^ Te1[(t2 >> 16) & 0xff] ^ Te2[(t3 >>  8) & 0xff] ^ Te3[t0 & 0xff] ^ rk[25];
    s2 = Te0[t2 >> 24] ^ Te1[(t3 >> 16) & 0xff] ^ Te2[(t0 >>  8) & 0xff] ^ Te3[t1 & 0xff] ^ rk[26];
    s3 = Te0[t3 >> 24] ^ Te1[(t0 >> 16) & 0xff] ^ Te2[(t1 >>  8) & 0xff] ^ Te3[t2 & 0xff] ^ rk[27];
    /* round 7: */
    t0 = Te0[s0 >> 24] ^ Te1[(s1 >> 16) & 0xff] ^ Te2[(s2 >>  8) & 0xff] ^ Te3[s3 & 0xff] ^ rk[28];
    t1 = Te0[s1 >> 24] ^ Te1[(s2 >> 16) & 0xff] ^ Te2[(s3 >>  8) & 0xff] ^ Te3[s0 & 0xff] ^ rk[29];
    t2 = Te0[s2 >> 24] ^ Te1[(s3 >> 16) & 0xff] ^ Te2[(s0 >>  8) & 0xff] ^ Te3[s1 & 0xff] ^ rk[30];
    t3 = Te0[s3 >> 24] ^ Te1[(s0 >> 16) & 0xff] ^ Te2[(s1 >>  8) & 0xff] ^ Te3[s2 & 0xff] ^ rk[31];
    /* round 8: */
    s0 = Te0[t0 >> 24] ^ Te1[(t1 >> 16) & 0xff] ^ Te2[(t2 >>  8) & 0xff] ^ Te3[t3 & 0xff] ^ rk[32];
    s1 = Te0[t1 >> 24] ^ Te1[(t2 >> 16) & 0xff] ^ Te2[(t3 >>  8) & 0xff] ^ Te3[t0 & 0xff] ^ rk[33];
    s2 = Te0[t2 >> 24] ^ Te1[(t3 >> 16) & 0xff] ^ Te2[(t0 >>  8) & 0xff] ^ Te3[t1 & 0xff] ^ rk[34];
    s3 = Te0[t3 >> 24] ^ Te1[(t0 >> 16) & 0xff] ^ Te2[(t1 >>  8) & 0xff] ^ Te3[t2 & 0xff] ^ rk[35];
    /* round 9: */
    t0 = Te0[s0 >> 24] ^ Te1[(s1 >> 16) & 0xff] ^ Te2[(s2 >>  8) & 0xff] ^ Te3[s3 & 0xff] ^ rk[36];
    t1 = Te0[s1 >> 24] ^ Te1[(s2 >> 16) & 0xff] ^ Te2[(s3 >>  8) & 0xff] ^ Te3[s0 & 0xff] ^ rk[37];
    t2 = Te0[s2 >> 24] ^ Te1[(s3 >> 16) & 0xff] ^ Te2[(s0 >>  8) & 0xff] ^ Te3[s1 & 0xff] ^ rk[38];
    t3 = Te0[s3 >> 24] ^ Te1[(s0 >> 16) & 0xff] ^ Te2[(s1 >>  8) & 0xff] ^ Te3[s2 & 0xff] ^ rk[39];
    if (nrounds > 10)
    {
      /* round 10: */
      s0 = Te0[t0 >> 24] ^ Te1[(t1 >> 16) & 0xff] ^ Te2[(t2 >>  8) & 0xff] ^ Te3[t3 & 0xff] ^ rk[40];
      s1 = Te0[t1 >> 24] ^ Te1[(t2 >> 16) & 0xff] ^ Te2[(t3 >>  8) & 0xff] ^ Te3[t0 & 0xff] ^ rk[41];
      s2 = Te0[t2 >> 24] ^ Te1[(t3 >> 16) & 0xff] ^ Te2[(t0 >>  8) & 0xff] ^ Te3[t1 & 0xff] ^ rk[42];
      s3 = Te0[t3 >> 24] ^ Te1[(t0 >> 16) & 0xff] ^ Te2[(t1 >>  8) & 0xff] ^ Te3[t2 & 0xff] ^ rk[43];
      /* round 11: */
      t0 = Te0[s0 >> 24] ^ Te1[(s1 >> 16) & 0xff] ^ Te2[(s2 >>  8) & 0xff] ^ Te3[s3 & 0xff] ^ rk[44];
      t1 = Te0[s1 >> 24] ^ Te1[(s2 >> 16) & 0xff] ^ Te2[(s3 >>  8) & 0xff] ^ Te3[s0 & 0xff] ^ rk[45];
      t2 = Te0[s2 >> 24] ^ Te1[(s3 >> 16) & 0xff] ^ Te2[(s0 >>  8) & 0xff] ^ Te3[s1 & 0xff] ^ rk[46];
      t3 = Te0[s3 >> 24] ^ Te1[(s0 >> 16) & 0xff] ^ Te2[(s1 >>  8) & 0xff] ^ Te3[s2 & 0xff] ^ rk[47];
      if (nrounds > 12)
      {
        /* round 12: */
        s0 = Te0[t0 >> 24] ^ Te1[(t1 >> 16) & 0xff] ^ Te2[(t2 >>  8) & 0xff] ^ Te3[t3 & 0xff] ^ rk[48];
        s1 = Te0[t1 >> 24] ^ Te1[(t2 >> 16) & 0xff] ^ Te2[(t3 >>  8) & 0xff] ^ Te3[t0 & 0xff] ^ rk[49];
        s2 = Te0[t2 >> 24] ^ Te1[(t3 >> 16) & 0xff] ^ Te2[(t0 >>  8) & 0xff] ^ Te3[t1 & 0xff] ^ rk[50];
        s3 = Te0[t3 >> 24] ^ Te1[(t0 >> 16) & 0xff] ^ Te2[(t1 >>  8) & 0xff] ^ Te3[t2 & 0xff] ^ rk[51];
        /* round 13: */
        t0 = Te0[s0 >> 24] ^ Te1[(s1 >> 16) & 0xff] ^ Te2[(s2 >>  8) & 0xff] ^ Te3[s3 & 0xff] ^ rk[52];
        t1 = Te0[s1 >> 24] ^ Te1[(s2 >> 16) & 0xff] ^ Te2[(s3 >>  8) & 0xff] ^ Te3[s0 & 0xff] ^ rk[53];
        t2 = Te0[s2 >> 24] ^ Te1[(s3 >> 16) & 0xff] ^ Te2[(s0 >>  8) & 0xff] ^ Te3[s1 & 0xff] ^ rk[54];
        t3 = Te0[s3 >> 24] ^ Te1[(s0 >> 16) & 0xff] ^ Te2[(s1 >>  8) & 0xff] ^ Te3[s2 & 0xff] ^ rk[55];
      }
    }
    rk += nrounds << 2;
    
    s0 =
    (Te4[(t0 >> 24)       ] & 0xff000000) ^
    (Te4[(t1 >> 16) & 0xff] & 0x00ff0000) ^
    (Te4[(t2 >>  8) & 0xff] & 0x0000ff00) ^
    (Te4[(t3      ) & 0xff] & 0x000000ff) ^
    rk[0];
    PUTU32(txt     , s0);
    s1 =
    (Te4[(t1 >> 24)       ] & 0xff000000) ^
    (Te4[(t2 >> 16) & 0xff] & 0x00ff0000) ^
    (Te4[(t3 >>  8) & 0xff] & 0x0000ff00) ^
    (Te4[(t0      ) & 0xff] & 0x000000ff) ^
    rk[1];
    PUTU32(txt +  4, s1);
    s2 =
    (Te4[(t2 >> 24)       ] & 0xff000000) ^
    (Te4[(t3 >> 16) & 0xff] & 0x00ff0000) ^
    (Te4[(t0 >>  8) & 0xff] & 0x0000ff00) ^
    (Te4[(t1      ) & 0xff] & 0x000000ff) ^
    rk[2];
    PUTU32(txt +  8, s2);
    s3 =
    (Te4[(t3 >> 24)       ] & 0xff000000) ^
    (Te4[(t0 >> 16) & 0xff] & 0x00ff0000) ^
    (Te4[(t1 >>  8) & 0xff] & 0x0000ff00) ^
    (Te4[(t2      ) & 0xff] & 0x000000ff) ^
    rk[3];
    PUTU32(txt + 12, s3);
}

__global__ void aes_decrypt_bpt(u32 *rk, int nrounds, u8* text)
{
  u32 s0, s1, s2, s3, t0, t1, t2, t3;
  u8 *txt = text+(16*(blockIdx.x*blockDim.x+threadIdx.x));

  /*
  * map byte array block to cipher state
  * and add initial round key:
  */
    s0 = GETU32(txt     ) ^ rk[0];
    s1 = GETU32(txt +  4) ^ rk[1];
    s2 = GETU32(txt +  8) ^ rk[2];
    s3 = GETU32(txt + 12) ^ rk[3];

    /* round 1: */
    t0 = Td0[s0 >> 24] ^ Td1[(s3 >> 16) & 0xff] ^ Td2[(s2 >>  8) & 0xff] ^ Td3[s1 & 0xff] ^ rk[ 4];
    t1 = Td0[s1 >> 24] ^ Td1[(s0 >> 16) & 0xff] ^ Td2[(s3 >>  8) & 0xff] ^ Td3[s2 & 0xff] ^ rk[ 5];
    t2 = Td0[s2 >> 24] ^ Td1[(s1 >> 16) & 0xff] ^ Td2[(s0 >>  8) & 0xff] ^ Td3[s3 & 0xff] ^ rk[ 6];
    t3 = Td0[s3 >> 24] ^ Td1[(s2 >> 16) & 0xff] ^ Td2[(s1 >>  8) & 0xff] ^ Td3[s0 & 0xff] ^ rk[ 7];
    /* round 2: */
    s0 = Td0[t0 >> 24] ^ Td1[(t3 >> 16) & 0xff] ^ Td2[(t2 >>  8) & 0xff] ^ Td3[t1 & 0xff] ^ rk[ 8];
    s1 = Td0[t1 >> 24] ^ Td1[(t0 >> 16) & 0xff] ^ Td2[(t3 >>  8) & 0xff] ^ Td3[t2 & 0xff] ^ rk[ 9];
    s2 = Td0[t2 >> 24] ^ Td1[(t1 >> 16) & 0xff] ^ Td2[(t0 >>  8) & 0xff] ^ Td3[t3 & 0xff] ^ rk[10];
    s3 = Td0[t3 >> 24] ^ Td1[(t2 >> 16) & 0xff] ^ Td2[(t1 >>  8) & 0xff] ^ Td3[t0 & 0xff] ^ rk[11];
    /* round 3: */
    t0 = Td0[s0 >> 24] ^ Td1[(s3 >> 16) & 0xff] ^ Td2[(s2 >>  8) & 0xff] ^ Td3[s1 & 0xff] ^ rk[12];
    t1 = Td0[s1 >> 24] ^ Td1[(s0 >> 16) & 0xff] ^ Td2[(s3 >>  8) & 0xff] ^ Td3[s2 & 0xff] ^ rk[13];
    t2 = Td0[s2 >> 24] ^ Td1[(s1 >> 16) & 0xff] ^ Td2[(s0 >>  8) & 0xff] ^ Td3[s3 & 0xff] ^ rk[14];
    t3 = Td0[s3 >> 24] ^ Td1[(s2 >> 16) & 0xff] ^ Td2[(s1 >>  8) & 0xff] ^ Td3[s0 & 0xff] ^ rk[15];
    /* round 4: */
    s0 = Td0[t0 >> 24] ^ Td1[(t3 >> 16) & 0xff] ^ Td2[(t2 >>  8) & 0xff] ^ Td3[t1 & 0xff] ^ rk[16];
    s1 = Td0[t1 >> 24] ^ Td1[(t0 >> 16) & 0xff] ^ Td2[(t3 >>  8) & 0xff] ^ Td3[t2 & 0xff] ^ rk[17];
    s2 = Td0[t2 >> 24] ^ Td1[(t1 >> 16) & 0xff] ^ Td2[(t0 >>  8) & 0xff] ^ Td3[t3 & 0xff] ^ rk[18];
    s3 = Td0[t3 >> 24] ^ Td1[(t2 >> 16) & 0xff] ^ Td2[(t1 >>  8) & 0xff] ^ Td3[t0 & 0xff] ^ rk[19];
    /* round 5: */
    t0 = Td0[s0 >> 24] ^ Td1[(s3 >> 16) & 0xff] ^ Td2[(s2 >>  8) & 0xff] ^ Td3[s1 & 0xff] ^ rk[20];
    t1 = Td0[s1 >> 24] ^ Td1[(s0 >> 16) & 0xff] ^ Td2[(s3 >>  8) & 0xff] ^ Td3[s2 & 0xff] ^ rk[21];
    t2 = Td0[s2 >> 24] ^ Td1[(s1 >> 16) & 0xff] ^ Td2[(s0 >>  8) & 0xff] ^ Td3[s3 & 0xff] ^ rk[22];
    t3 = Td0[s3 >> 24] ^ Td1[(s2 >> 16) & 0xff] ^ Td2[(s1 >>  8) & 0xff] ^ Td3[s0 & 0xff] ^ rk[23];
    /* round 6: */
    s0 = Td0[t0 >> 24] ^ Td1[(t3 >> 16) & 0xff] ^ Td2[(t2 >>  8) & 0xff] ^ Td3[t1 & 0xff] ^ rk[24];
    s1 = Td0[t1 >> 24] ^ Td1[(t0 >> 16) & 0xff] ^ Td2[(t3 >>  8) & 0xff] ^ Td3[t2 & 0xff] ^ rk[25];
    s2 = Td0[t2 >> 24] ^ Td1[(t1 >> 16) & 0xff] ^ Td2[(t0 >>  8) & 0xff] ^ Td3[t3 & 0xff] ^ rk[26];
    s3 = Td0[t3 >> 24] ^ Td1[(t2 >> 16) & 0xff] ^ Td2[(t1 >>  8) & 0xff] ^ Td3[t0 & 0xff] ^ rk[27];
    /* round 7: */
    t0 = Td0[s0 >> 24] ^ Td1[(s3 >> 16) & 0xff] ^ Td2[(s2 >>  8) & 0xff] ^ Td3[s1 & 0xff] ^ rk[28];
    t1 = Td0[s1 >> 24] ^ Td1[(s0 >> 16) & 0xff] ^ Td2[(s3 >>  8) & 0xff] ^ Td3[s2 & 0xff] ^ rk[29];
    t2 = Td0[s2 >> 24] ^ Td1[(s1 >> 16) & 0xff] ^ Td2[(s0 >>  8) & 0xff] ^ Td3[s3 & 0xff] ^ rk[30];
    t3 = Td0[s3 >> 24] ^ Td1[(s2 >> 16) & 0xff] ^ Td2[(s1 >>  8) & 0xff] ^ Td3[s0 & 0xff] ^ rk[31];
    /* round 8: */
    s0 = Td0[t0 >> 24] ^ Td1[(t3 >> 16) & 0xff] ^ Td2[(t2 >>  8) & 0xff] ^ Td3[t1 & 0xff] ^ rk[32];
    s1 = Td0[t1 >> 24] ^ Td1[(t0 >> 16) & 0xff] ^ Td2[(t3 >>  8) & 0xff] ^ Td3[t2 & 0xff] ^ rk[33];
    s2 = Td0[t2 >> 24] ^ Td1[(t1 >> 16) & 0xff] ^ Td2[(t0 >>  8) & 0xff] ^ Td3[t3 & 0xff] ^ rk[34];
    s3 = Td0[t3 >> 24] ^ Td1[(t2 >> 16) & 0xff] ^ Td2[(t1 >>  8) & 0xff] ^ Td3[t0 & 0xff] ^ rk[35];
    /* round 9: */
    t0 = Td0[s0 >> 24] ^ Td1[(s3 >> 16) & 0xff] ^ Td2[(s2 >>  8) & 0xff] ^ Td3[s1 & 0xff] ^ rk[36];
    t1 = Td0[s1 >> 24] ^ Td1[(s0 >> 16) & 0xff] ^ Td2[(s3 >>  8) & 0xff] ^ Td3[s2 & 0xff] ^ rk[37];
    t2 = Td0[s2 >> 24] ^ Td1[(s1 >> 16) & 0xff] ^ Td2[(s0 >>  8) & 0xff] ^ Td3[s3 & 0xff] ^ rk[38];
    t3 = Td0[s3 >> 24] ^ Td1[(s2 >> 16) & 0xff] ^ Td2[(s1 >>  8) & 0xff] ^ Td3[s0 & 0xff] ^ rk[39];
    if (nrounds > 10)
    {
      /* round 10: */
      s0 = Td0[t0 >> 24] ^ Td1[(t3 >> 16) & 0xff] ^ Td2[(t2 >>  8) & 0xff] ^ Td3[t1 & 0xff] ^ rk[40];
      s1 = Td0[t1 >> 24] ^ Td1[(t0 >> 16) & 0xff] ^ Td2[(t3 >>  8) & 0xff] ^ Td3[t2 & 0xff] ^ rk[41];
      s2 = Td0[t2 >> 24] ^ Td1[(t1 >> 16) & 0xff] ^ Td2[(t0 >>  8) & 0xff] ^ Td3[t3 & 0xff] ^ rk[42];
      s3 = Td0[t3 >> 24] ^ Td1[(t2 >> 16) & 0xff] ^ Td2[(t1 >>  8) & 0xff] ^ Td3[t0 & 0xff] ^ rk[43];
      /* round 11: */
      t0 = Td0[s0 >> 24] ^ Td1[(s3 >> 16) & 0xff] ^ Td2[(s2 >>  8) & 0xff] ^ Td3[s1 & 0xff] ^ rk[44];
      t1 = Td0[s1 >> 24] ^ Td1[(s0 >> 16) & 0xff] ^ Td2[(s3 >>  8) & 0xff] ^ Td3[s2 & 0xff] ^ rk[45];
      t2 = Td0[s2 >> 24] ^ Td1[(s1 >> 16) & 0xff] ^ Td2[(s0 >>  8) & 0xff] ^ Td3[s3 & 0xff] ^ rk[46];
      t3 = Td0[s3 >> 24] ^ Td1[(s2 >> 16) & 0xff] ^ Td2[(s1 >>  8) & 0xff] ^ Td3[s0 & 0xff] ^ rk[47];
      if (nrounds > 12)
      {
        /* round 12: */
        s0 = Td0[t0 >> 24] ^ Td1[(t3 >> 16) & 0xff] ^ Td2[(t2 >>  8) & 0xff] ^ Td3[t1 & 0xff] ^ rk[48];
        s1 = Td0[t1 >> 24] ^ Td1[(t0 >> 16) & 0xff] ^ Td2[(t3 >>  8) & 0xff] ^ Td3[t2 & 0xff] ^ rk[49];
        s2 = Td0[t2 >> 24] ^ Td1[(t1 >> 16) & 0xff] ^ Td2[(t0 >>  8) & 0xff] ^ Td3[t3 & 0xff] ^ rk[50];
        s3 = Td0[t3 >> 24] ^ Td1[(t2 >> 16) & 0xff] ^ Td2[(t1 >>  8) & 0xff] ^ Td3[t0 & 0xff] ^ rk[51];
        /* round 13: */
        t0 = Td0[s0 >> 24] ^ Td1[(s3 >> 16) & 0xff] ^ Td2[(s2 >>  8) & 0xff] ^ Td3[s1 & 0xff] ^ rk[52];
        t1 = Td0[s1 >> 24] ^ Td1[(s0 >> 16) & 0xff] ^ Td2[(s3 >>  8) & 0xff] ^ Td3[s2 & 0xff] ^ rk[53];
        t2 = Td0[s2 >> 24] ^ Td1[(s1 >> 16) & 0xff] ^ Td2[(s0 >>  8) & 0xff] ^ Td3[s3 & 0xff] ^ rk[54];
        t3 = Td0[s3 >> 24] ^ Td1[(s2 >> 16) & 0xff] ^ Td2[(s1 >>  8) & 0xff] ^ Td3[s0 & 0xff] ^ rk[55];
      }
    }
    rk += nrounds << 2;
    
  s0 =
    (Td4[(t0 >> 24)       ] & 0xff000000) ^
    (Td4[(t3 >> 16) & 0xff] & 0x00ff0000) ^
    (Td4[(t2 >>  8) & 0xff] & 0x0000ff00) ^
    (Td4[(t1      ) & 0xff] & 0x000000ff) ^
    rk[0];
  PUTU32(txt     , s0);
  s1 =
    (Td4[(t1 >> 24)       ] & 0xff000000) ^
    (Td4[(t0 >> 16) & 0xff] & 0x00ff0000) ^
    (Td4[(t3 >>  8) & 0xff] & 0x0000ff00) ^
    (Td4[(t2      ) & 0xff] & 0x000000ff) ^
    rk[1];
  PUTU32(txt +  4, s1);
  s2 =
    (Td4[(t2 >> 24)       ] & 0xff000000) ^
    (Td4[(t1 >> 16) & 0xff] & 0x00ff0000) ^
    (Td4[(t0 >>  8) & 0xff] & 0x0000ff00) ^
    (Td4[(t3      ) & 0xff] & 0x000000ff) ^
    rk[2];
  PUTU32(txt +  8, s2);
  s3 =
    (Td4[(t3 >> 24)       ] & 0xff000000) ^
    (Td4[(t2 >> 16) & 0xff] & 0x00ff0000) ^
    (Td4[(t1 >>  8) & 0xff] & 0x0000ff00) ^
    (Td4[(t0      ) & 0xff] & 0x000000ff) ^
    rk[3];
  PUTU32(txt + 12, s3);
}

__global__ void aes_encrypt_bp4t(u32 *rk, int nrounds, u8* text)
{
    __shared__ u32 s[WORDS_PER_BLOCK];
    __shared__ u32 t[WORDS_PER_BLOCK];
    /*int lid = threadIdx.y*4 + threadIdx.x;
    int bid = blockIdx.x;
    int n1 = threadIdx.y*4 + (threadIdx.x+1)%4;
    int n2 = threadIdx.y*4 + (threadIdx.x+2)%4;
    int n3 = threadIdx.y*4 + (threadIdx.x+3)%4;  */
    
    #define lid threadIdx.y*4 + threadIdx.x
    #define bid blockIdx.x
    #define n1  threadIdx.y*4 + (threadIdx.x+1)%4
    #define n2  threadIdx.y*4 + (threadIdx.x+2)%4
    #define n3  threadIdx.y*4 + (threadIdx.x+3)%4   
    
    s[lid] = GETU32((u32*)((u32*)text+bid*WORDS_PER_BLOCK+lid)) ^ rk[threadIdx.x];

    t[lid] = Te0[s[lid] >> 24] ^ Te1[(s[n1] >> 16) & 0xff] ^ Te2[(s[n2] >>  8) & 0xff] ^ Te3[s[n3] & 0xff] ^ rk[4+threadIdx.x];
    
    s[lid] = Te0[t[lid] >> 24] ^ Te1[(t[n1] >> 16) & 0xff] ^ Te2[(t[n2] >>  8) & 0xff] ^ Te3[t[n3] & 0xff] ^ rk[8+threadIdx.x];
    
    t[lid] = Te0[s[lid] >> 24] ^ Te1[(s[n1] >> 16) & 0xff] ^ Te2[(s[n2] >>  8) & 0xff] ^ Te3[s[n3] & 0xff] ^ rk[12+threadIdx.x];
    
    s[lid] = Te0[t[lid] >> 24] ^ Te1[(t[n1] >> 16) & 0xff] ^ Te2[(t[n2] >>  8) & 0xff] ^ Te3[t[n3] & 0xff] ^ rk[16+threadIdx.x];
    
    t[lid] = Te0[s[lid] >> 24] ^ Te1[(s[n1] >> 16) & 0xff] ^ Te2[(s[n2] >>  8) & 0xff] ^ Te3[s[n3] & 0xff] ^ rk[20+threadIdx.x];
    
    s[lid] = Te0[t[lid] >> 24] ^ Te1[(t[n1] >> 16) & 0xff] ^ Te2[(t[n2] >>  8) & 0xff] ^ Te3[t[n3] & 0xff] ^ rk[24+threadIdx.x];
    
    t[lid] = Te0[s[lid] >> 24] ^ Te1[(s[n1] >> 16) & 0xff] ^ Te2[(s[n2] >>  8) & 0xff] ^ Te3[s[n3] & 0xff] ^ rk[28+threadIdx.x];
    
    s[lid] = Te0[t[lid] >> 24] ^ Te1[(t[n1] >> 16) & 0xff] ^ Te2[(t[n2] >>  8) & 0xff] ^ Te3[t[n3] & 0xff] ^ rk[32+threadIdx.x];
    
    t[lid] = Te0[s[lid] >> 24] ^ Te1[(s[n1] >> 16) & 0xff] ^ Te2[(s[n2] >>  8) & 0xff] ^ Te3[s[n3] & 0xff] ^ rk[36+threadIdx.x];

    if (nrounds > 10)
    {    
      s[lid] = Te0[t[lid] >> 24] ^ Te1[(t[n1] >> 16) & 0xff] ^ Te2[(t[n2] >>  8) & 0xff] ^ Te3[t[n3] & 0xff] ^ rk[40+threadIdx.x];
    
      t[lid] = Te0[s[lid] >> 24] ^ Te1[(s[n1] >> 16) & 0xff] ^ Te2[(s[n2] >>  8) & 0xff] ^ Te3[s[n3] & 0xff] ^ rk[44+threadIdx.x];

      if (nrounds > 12)
      {      
        s[lid] = Te0[t[lid] >> 24] ^ Te1[(t[n1] >> 16) & 0xff] ^ Te2[(t[n2] >>  8) & 0xff] ^ Te3[t[n3] & 0xff] ^ rk[48+threadIdx.x];
    
      	t[lid] = Te0[s[lid] >> 24] ^ Te1[(s[n1] >> 16) & 0xff] ^ Te2[(s[n2] >>  8) & 0xff] ^ Te3[s[n3] & 0xff] ^ rk[52+threadIdx.x];
      }
    }
    rk += nrounds << 2;
    
    s[lid] =
    (Te4[(t[lid] >> 24)       ] & 0xff000000) ^
    (Te4[(t[n1] >> 16) & 0xff] & 0x00ff0000) ^
    (Te4[(t[n2] >>  8) & 0xff] & 0x0000ff00) ^
    (Te4[(t[n3]      ) & 0xff] & 0x000000ff) ^
    rk[threadIdx.x];
    PUTU32((u32*)((u32*)text+bid*WORDS_PER_BLOCK+lid), s[lid]);
}

__global__ void aes_decrypt_bp4t(u32 *rk, int nrounds, u8* text)
{    
    __shared__ u32 s[WORDS_PER_BLOCK];
    __shared__ u32 t[WORDS_PER_BLOCK];
    /*int lid = threadIdx.y*4 + threadIdx.x;
    int bid = blockIdx.x;
    int n1 = threadIdx.y*4 + (threadIdx.x+1)%4;
    int n2 = threadIdx.y*4 + (threadIdx.x+2)%4;
    int n3 = threadIdx.y*4 + (threadIdx.x+3)%4;  */
    
    s[lid] = GETU32((u32*)((u32*)text+bid*WORDS_PER_BLOCK+lid)) ^ rk[threadIdx.x];
    
    t[lid] = Td0[s[lid] >> 24] ^ Td1[(s[n3] >> 16) & 0xff] ^ Td2[(s[n2] >>  8) & 0xff] ^ Td3[s[n1] & 0xff] ^ rk[4+threadIdx.x];
    s[lid] = Td0[t[lid] >> 24] ^ Td1[(t[n3] >> 16) & 0xff] ^ Td2[(t[n2] >>  8) & 0xff] ^ Td3[t[n1] & 0xff] ^ rk[8+threadIdx.x];
    
    t[lid] = Td0[s[lid] >> 24] ^ Td1[(s[n3] >> 16) & 0xff] ^ Td2[(s[n2] >>  8) & 0xff] ^ Td3[s[n1] & 0xff] ^ rk[4+threadIdx.x];
    s[lid] = Td0[t[lid] >> 24] ^ Td1[(t[n3] >> 16) & 0xff] ^ Td2[(t[n2] >>  8) & 0xff] ^ Td3[t[n1] & 0xff] ^ rk[8+threadIdx.x];
    
    t[lid] = Td0[s[lid] >> 24] ^ Td1[(s[n3] >> 16) & 0xff] ^ Td2[(s[n2] >>  8) & 0xff] ^ Td3[s[n1] & 0xff] ^ rk[4+threadIdx.x];
    s[lid] = Td0[t[lid] >> 24] ^ Td1[(t[n3] >> 16) & 0xff] ^ Td2[(t[n2] >>  8) & 0xff] ^ Td3[t[n1] & 0xff] ^ rk[8+threadIdx.x];
    
    t[lid] = Td0[s[lid] >> 24] ^ Td1[(s[n3] >> 16) & 0xff] ^ Td2[(s[n2] >>  8) & 0xff] ^ Td3[s[n1] & 0xff] ^ rk[4+threadIdx.x];
    s[lid] = Td0[t[lid] >> 24] ^ Td1[(t[n3] >> 16) & 0xff] ^ Td2[(t[n2] >>  8) & 0xff] ^ Td3[t[n1] & 0xff] ^ rk[8+threadIdx.x];
    
    t[lid] = Td0[s[lid] >> 24] ^ Td1[(s[n3] >> 16) & 0xff] ^ Td2[(s[n2] >>  8) & 0xff] ^ Td3[s[n1] & 0xff] ^ rk[4+threadIdx.x];
    
    if (nrounds > 10)
    {
        s[lid] = Td0[t[lid] >> 24] ^ Td1[(t[n3] >> 16) & 0xff] ^ Td2[(t[n2] >>  8) & 0xff] ^ Td3[t[n1] & 0xff] ^ rk[8+threadIdx.x];
        t[lid] = Td0[s[lid] >> 24] ^ Td1[(s[n3] >> 16) & 0xff] ^ Td2[(s[n2] >>  8) & 0xff] ^ Td3[s[n1] & 0xff] ^ rk[4+threadIdx.x];
        
        if (nrounds > 12)
        {
            s[lid] = Td0[t[lid] >> 24] ^ Td1[(t[n3] >> 16) & 0xff] ^ Td2[(t[n2] >>  8) & 0xff] ^ Td3[t[n1] & 0xff] ^ rk[8+threadIdx.x];
            t[lid] = Td0[s[lid] >> 24] ^ Td1[(s[n3] >> 16) & 0xff] ^ Td2[(s[n2] >>  8) & 0xff] ^ Td3[s[n1] & 0xff] ^ rk[4+threadIdx.x];
        }
    }
    rk += nrounds << 2;
    
  s[lid] =
    (Td4[(t[lid] >> 24)       ] & 0xff000000) ^
    (Td4[(t[n3] >> 16) & 0xff] & 0x00ff0000) ^
    (Td4[(t[n2] >>  8) & 0xff] & 0x0000ff00) ^
    (Td4[(t[n1]      ) & 0xff] & 0x000000ff) ^
    rk[0];
  PUTU32((u32*)((u32*)text+bid*WORDS_PER_BLOCK+lid), s[lid]);
}


int gecb_compute_size_bpt(struct service_request *sr)
{
    sr->block_x = sr->kureq.outsize>=BPT_BYTES_PER_BLOCK? BPT_BYTES_PER_BLOCK/16: sr->kureq.outsize/16;
    sr->grid_x = sr->kureq.outsize/BPT_BYTES_PER_BLOCK? sr->kureq.outsize/BPT_BYTES_PER_BLOCK:1;
    sr->block_y = 1;
    sr->grid_y = 1;

    return 0;
}

int gecb_compute_size_bp4t(struct service_request *sr)
{
    sr->block_y = sr->kureq.outsize>=BYTES_PER_BLOCK? BYTES_PER_BLOCK/BYTES_PER_GROUP: (sr->kureq.outsize/BYTES_PER_GROUP);
    sr->grid_x = sr->kureq.outsize/BYTES_PER_BLOCK? sr->kureq.outsize/BYTES_PER_BLOCK:1;
    sr->block_x = BYTES_PER_GROUP/BYTES_PER_THREAD;
    sr->grid_y = 1;

    return 0;
}

int gecb_launch_bpt(struct service_request *sr)
{
        struct gecb_data *data = (struct gecb_data*)sr->data;
        
        if (sr->s == &gecb_dec_srv)        
        	aes_decrypt_bpt<<<dim3(sr->grid_x, sr->grid_y), dim3(sr->block_x, sr->block_y), 0, (cudaStream_t)(sr->stream)>>>
                (data->d_key, data->nrounds, (u8*)sr->doutput);
        else
	    aes_encrypt_bpt/*aes_encrypt_withpointer*/<<<dim3(sr->grid_x, sr->grid_y), dim3(sr->block_x, sr->block_y), 0, (cudaStream_t)(sr->stream)>>>
                (data->d_key, data->nrounds, (u8*)sr->doutput);
        return 0;
}

int gecb_launch_bp4t(struct service_request *sr)
{
        struct gecb_data *data = (struct gecb_data*)sr->data;
        
        if (sr->s == &gecb_dec_srv)        
        	aes_decrypt_bp4t<<<dim3(sr->grid_x, sr->grid_y), dim3(sr->block_x, sr->block_y), 0, (cudaStream_t)(sr->stream)>>>
                (data->d_key, data->nrounds, (u8*)sr->doutput);
        else
        	aes_encrypt_bp4t<<<dim3(sr->grid_x, sr->grid_y), dim3(sr->block_x, sr->block_y), 0, (cudaStream_t)(sr->stream)>>>
                (data->d_key, data->nrounds, (u8*)sr->doutput);
        return 0;
}

int gecb_prepare(struct service_request *sr)
{
    cudaStream_t s = (cudaStream_t)(sr->stream);//get_stream(sr->stream_id);
    struct gecb_data *data = (struct gecb_data *)malloc(sizeof(struct gecb_data));
    struct crypto_aes_ctx *ctx = (struct crypto_aes_ctx*)((u8*)(sr->kureq.input) + (sr->kureq.outsize));
    
    data->nrounds = ctx->key_length/4+6;
    data->h_key = (sr->s == &gecb_dec_srv)?ctx->key_dec: ctx->key_enc;                         
    data->d_key = (u32*)((u8*)(sr->dinput) + ((u8*)data->h_key - (u8*)sr->kureq.input));
    sr->data = data;
    
    csc( ah2dcpy( sr->dinput, sr->kureq.input, sr->kureq.insize, s) );
    return 0;
}

int gecb_post(struct service_request *sr)
{
    cudaStream_t s = (cudaStream_t)(sr->stream);//get_stream(sr->stream_id);
    csc( ad2hcpy( sr->kureq.output, sr->doutput, sr->kureq.outsize, s) );
    
    free(sr->data);
    sr->data = NULL;
    return 0;
}


extern "C" int init_service(void *lh, int (*reg_srv)(struct service*, void*))
{
    int err;
    printf("[libsrv_gecb] Info: init gecb service\n");
    
    sprintf(gecb_enc_srv.name, "gecb-enc");
    gecb_enc_srv.sid = 0;
    gecb_enc_srv.compute_size = gecb_compute_size_bpt;
    gecb_enc_srv.launch = gecb_launch_bpt;
    gecb_enc_srv.prepare = gecb_prepare;
    gecb_enc_srv.post = gecb_post;
    
    sprintf(gecb_dec_srv.name, "gecb-dec");
    gecb_dec_srv.sid = 0;
    gecb_dec_srv.compute_size = gecb_compute_size_bpt;
    gecb_dec_srv.launch = gecb_launch_bpt;
    gecb_dec_srv.prepare = gecb_prepare;
    gecb_dec_srv.post = gecb_post;
    
    err = reg_srv(&gecb_enc_srv, lh);
    if (err) {
    	fprintf(stderr, "[libsrv_gecb] Error: failed to register enc service\n");
    } else {
        err = reg_srv(&gecb_dec_srv, lh);
        if (err) {
    	    fprintf(stderr, "[libsrv_gecb] Error: failed to register dec service\n");
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


