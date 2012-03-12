/* This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2010-2011 University of Utah and the Flux Group.
 * All rights reserved.
 */

/*
 * Device code file for AES
 */

__device__ int block_id()
{
    return blockIdx.y*gridDim.x + blockIdx.x;
}

__device__ int thread_id()
{
    return block_id()*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
}


/*
 * Not used yet, just in case.
 * Code borrowed from:
 *   http://stackoverflow.com/questions/6162140/128-bit-integer-on-cuda/6220499#6220499
 */
__device__ uint4 add_uint128 (uint4 addend, uint4 augend)
{
    uint4 res;
    asm ("add.cc.u32      %0, %4, %8;\n\t"
	 "addc.cc.u32     %1, %5, %9;\n\t"
	 "addc.cc.u32     %2, %6, %10;\n\t"
	 "addc.u32        %3, %7, %11;\n\t"
	 : "=r"(res.x), "=r"(res.y), "=r"(res.z), "=r"(res.w)
	 : "r"(addend.x), "r"(addend.y), "r"(addend.z), "r"(addend.w),
	   "r"(augend.x), "r"(augend.y), "r"(augend.z), "r"(augend.w));
    return res;
}


/*
 * deal with lower 64bit only
 */
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


__device__ u64 thread_ctr_offset()
{
    return (u64)(blockIdx.x*blockDim.x+threadIdx.x);
}

/*
 * local counter mode: counter starts from 0, within a page-scope.
 */
__global__ void aes_lctr_crypt(u32 *rk, int nrounds, u8 *text, u8 *ctr)
{
    u32 s[4];
    u32 t[4];
    u32 *txt = (u32*)(text+(16*(blockIdx.x*blockDim.x+threadIdx.x)));

    big_u128_add(ctr, (u64)(threadIdx.x), (u8*)s);

    for (int i=0; i<4; i++)
	s[i] = GETU32(s+i)^rk[i];
    
    /* round 1: */
    t[0] = Te0[s[0] >> 24] ^ Te1[(s[1] >> 16) & 0xff] ^ Te2[(s[2] >>  8) & 0xff] ^ Te3[s[3] & 0xff] ^ rk[ 4];
    t[1] = Te0[s[1] >> 24] ^ Te1[(s[2] >> 16) & 0xff] ^ Te2[(s[3] >>  8) & 0xff] ^ Te3[s[0] & 0xff] ^ rk[ 5];
    t[2] = Te0[s[2] >> 24] ^ Te1[(s[3] >> 16) & 0xff] ^ Te2[(s[0] >>  8) & 0xff] ^ Te3[s[1] & 0xff] ^ rk[ 6];
    t[3] = Te0[s[3] >> 24] ^ Te1[(s[0] >> 16) & 0xff] ^ Te2[(s[1] >>  8) & 0xff] ^ Te3[s[2] & 0xff] ^ rk[ 7];
    /* round 2: */
    s[0] = Te0[t[0] >> 24] ^ Te1[(t[1] >> 16) & 0xff] ^ Te2[(t[2] >>  8) & 0xff] ^ Te3[t[3] & 0xff] ^ rk[ 8];
    s[1] = Te0[t[1] >> 24] ^ Te1[(t[2] >> 16) & 0xff] ^ Te2[(t[3] >>  8) & 0xff] ^ Te3[t[0] & 0xff] ^ rk[ 9];
    s[2] = Te0[t[2] >> 24] ^ Te1[(t[3] >> 16) & 0xff] ^ Te2[(t[0] >>  8) & 0xff] ^ Te3[t[1] & 0xff] ^ rk[10];
    s[3] = Te0[t[3] >> 24] ^ Te1[(t[0] >> 16) & 0xff] ^ Te2[(t[1] >>  8) & 0xff] ^ Te3[t[2] & 0xff] ^ rk[11];
    /* round 3: */
    t[0] = Te0[s[0] >> 24] ^ Te1[(s[1] >> 16) & 0xff] ^ Te2[(s[2] >>  8) & 0xff] ^ Te3[s[3] & 0xff] ^ rk[12];
    t[1] = Te0[s[1] >> 24] ^ Te1[(s[2] >> 16) & 0xff] ^ Te2[(s[3] >>  8) & 0xff] ^ Te3[s[0] & 0xff] ^ rk[13];
    t[2] = Te0[s[2] >> 24] ^ Te1[(s[3] >> 16) & 0xff] ^ Te2[(s[0] >>  8) & 0xff] ^ Te3[s[1] & 0xff] ^ rk[14];
    t[3] = Te0[s[3] >> 24] ^ Te1[(s[0] >> 16) & 0xff] ^ Te2[(s[1] >>  8) & 0xff] ^ Te3[s[2] & 0xff] ^ rk[15];
    /* round 4: */
    s[0] = Te0[t[0] >> 24] ^ Te1[(t[1] >> 16) & 0xff] ^ Te2[(t[2] >>  8) & 0xff] ^ Te3[t[3] & 0xff] ^ rk[16];
    s[1] = Te0[t[1] >> 24] ^ Te1[(t[2] >> 16) & 0xff] ^ Te2[(t[3] >>  8) & 0xff] ^ Te3[t[0] & 0xff] ^ rk[17];
    s[2] = Te0[t[2] >> 24] ^ Te1[(t[3] >> 16) & 0xff] ^ Te2[(t[0] >>  8) & 0xff] ^ Te3[t[1] & 0xff] ^ rk[18];
    s[3] = Te0[t[3] >> 24] ^ Te1[(t[0] >> 16) & 0xff] ^ Te2[(t[1] >>  8) & 0xff] ^ Te3[t[2] & 0xff] ^ rk[19];
    /* round 5: */
    t[0] = Te0[s[0] >> 24] ^ Te1[(s[1] >> 16) & 0xff] ^ Te2[(s[2] >>  8) & 0xff] ^ Te3[s[3] & 0xff] ^ rk[20];
    t[1] = Te0[s[1] >> 24] ^ Te1[(s[2] >> 16) & 0xff] ^ Te2[(s[3] >>  8) & 0xff] ^ Te3[s[0] & 0xff] ^ rk[21];
    t[2] = Te0[s[2] >> 24] ^ Te1[(s[3] >> 16) & 0xff] ^ Te2[(s[0] >>  8) & 0xff] ^ Te3[s[1] & 0xff] ^ rk[22];
    t[3] = Te0[s[3] >> 24] ^ Te1[(s[0] >> 16) & 0xff] ^ Te2[(s[1] >>  8) & 0xff] ^ Te3[s[2] & 0xff] ^ rk[23];
    /* round 6: */
    s[0] = Te0[t[0] >> 24] ^ Te1[(t[1] >> 16) & 0xff] ^ Te2[(t[2] >>  8) & 0xff] ^ Te3[t[3] & 0xff] ^ rk[24];
    s[1] = Te0[t[1] >> 24] ^ Te1[(t[2] >> 16) & 0xff] ^ Te2[(t[3] >>  8) & 0xff] ^ Te3[t[0] & 0xff] ^ rk[25];
    s[2] = Te0[t[2] >> 24] ^ Te1[(t[3] >> 16) & 0xff] ^ Te2[(t[0] >>  8) & 0xff] ^ Te3[t[1] & 0xff] ^ rk[26];
    s[3] = Te0[t[3] >> 24] ^ Te1[(t[0] >> 16) & 0xff] ^ Te2[(t[1] >>  8) & 0xff] ^ Te3[t[2] & 0xff] ^ rk[27];
    /* round 7: */
    t[0] = Te0[s[0] >> 24] ^ Te1[(s[1] >> 16) & 0xff] ^ Te2[(s[2] >>  8) & 0xff] ^ Te3[s[3] & 0xff] ^ rk[28];
    t[1] = Te0[s[1] >> 24] ^ Te1[(s[2] >> 16) & 0xff] ^ Te2[(s[3] >>  8) & 0xff] ^ Te3[s[0] & 0xff] ^ rk[29];
    t[2] = Te0[s[2] >> 24] ^ Te1[(s[3] >> 16) & 0xff] ^ Te2[(s[0] >>  8) & 0xff] ^ Te3[s[1] & 0xff] ^ rk[30];
    t[3] = Te0[s[3] >> 24] ^ Te1[(s[0] >> 16) & 0xff] ^ Te2[(s[1] >>  8) & 0xff] ^ Te3[s[2] & 0xff] ^ rk[31];
    /* round 8: */
    s[0] = Te0[t[0] >> 24] ^ Te1[(t[1] >> 16) & 0xff] ^ Te2[(t[2] >>  8) & 0xff] ^ Te3[t[3] & 0xff] ^ rk[32];
    s[1] = Te0[t[1] >> 24] ^ Te1[(t[2] >> 16) & 0xff] ^ Te2[(t[3] >>  8) & 0xff] ^ Te3[t[0] & 0xff] ^ rk[33];
    s[2] = Te0[t[2] >> 24] ^ Te1[(t[3] >> 16) & 0xff] ^ Te2[(t[0] >>  8) & 0xff] ^ Te3[t[1] & 0xff] ^ rk[34];
    s[3] = Te0[t[3] >> 24] ^ Te1[(t[0] >> 16) & 0xff] ^ Te2[(t[1] >>  8) & 0xff] ^ Te3[t[2] & 0xff] ^ rk[35];
    /* round 9: */
    t[0] = Te0[s[0] >> 24] ^ Te1[(s[1] >> 16) & 0xff] ^ Te2[(s[2] >>  8) & 0xff] ^ Te3[s[3] & 0xff] ^ rk[36];
    t[1] = Te0[s[1] >> 24] ^ Te1[(s[2] >> 16) & 0xff] ^ Te2[(s[3] >>  8) & 0xff] ^ Te3[s[0] & 0xff] ^ rk[37];
    t[2] = Te0[s[2] >> 24] ^ Te1[(s[3] >> 16) & 0xff] ^ Te2[(s[0] >>  8) & 0xff] ^ Te3[s[1] & 0xff] ^ rk[38];
    t[3] = Te0[s[3] >> 24] ^ Te1[(s[0] >> 16) & 0xff] ^ Te2[(s[1] >>  8) & 0xff] ^ Te3[s[2] & 0xff] ^ rk[39];
    if (nrounds > 10)
    {
	/* round 10: */
	s[0] = Te0[t[0] >> 24] ^ Te1[(t[1] >> 16) & 0xff] ^ Te2[(t[2] >>  8) & 0xff] ^ Te3[t[3] & 0xff] ^ rk[40];
	s[1] = Te0[t[1] >> 24] ^ Te1[(t[2] >> 16) & 0xff] ^ Te2[(t[3] >>  8) & 0xff] ^ Te3[t[0] & 0xff] ^ rk[41];
	s[2] = Te0[t[2] >> 24] ^ Te1[(t[3] >> 16) & 0xff] ^ Te2[(t[0] >>  8) & 0xff] ^ Te3[t[1] & 0xff] ^ rk[42];
	s[3] = Te0[t[3] >> 24] ^ Te1[(t[0] >> 16) & 0xff] ^ Te2[(t[1] >>  8) & 0xff] ^ Te3[t[2] & 0xff] ^ rk[43];
	/* round 11: */
	t[0] = Te0[s[0] >> 24] ^ Te1[(s[1] >> 16) & 0xff] ^ Te2[(s[2] >>  8) & 0xff] ^ Te3[s[3] & 0xff] ^ rk[44];
	t[1] = Te0[s[1] >> 24] ^ Te1[(s[2] >> 16) & 0xff] ^ Te2[(s[3] >>  8) & 0xff] ^ Te3[s[0] & 0xff] ^ rk[45];
	t[2] = Te0[s[2] >> 24] ^ Te1[(s[3] >> 16) & 0xff] ^ Te2[(s[0] >>  8) & 0xff] ^ Te3[s[1] & 0xff] ^ rk[46];
	t[3] = Te0[s[3] >> 24] ^ Te1[(s[0] >> 16) & 0xff] ^ Te2[(s[1] >>  8) & 0xff] ^ Te3[s[2] & 0xff] ^ rk[47];
	if (nrounds > 12)
	{
	    /* round 12: */
	    s[0] = Te0[t[0] >> 24] ^ Te1[(t[1] >> 16) & 0xff] ^ Te2[(t[2] >>  8) & 0xff] ^ Te3[t[3] & 0xff] ^ rk[48];
	    s[1] = Te0[t[1] >> 24] ^ Te1[(t[2] >> 16) & 0xff] ^ Te2[(t[3] >>  8) & 0xff] ^ Te3[t[0] & 0xff] ^ rk[49];
	    s[2] = Te0[t[2] >> 24] ^ Te1[(t[3] >> 16) & 0xff] ^ Te2[(t[0] >>  8) & 0xff] ^ Te3[t[1] & 0xff] ^ rk[50];
	    s[3] = Te0[t[3] >> 24] ^ Te1[(t[0] >> 16) & 0xff] ^ Te2[(t[1] >>  8) & 0xff] ^ Te3[t[2] & 0xff] ^ rk[51];
	    /* round 13: */
	    t[0] = Te0[s[0] >> 24] ^ Te1[(s[1] >> 16) & 0xff] ^ Te2[(s[2] >>  8) & 0xff] ^ Te3[s[3] & 0xff] ^ rk[52];
	    t[1] = Te0[s[1] >> 24] ^ Te1[(s[2] >> 16) & 0xff] ^ Te2[(s[3] >>  8) & 0xff] ^ Te3[s[0] & 0xff] ^ rk[53];
	    t[2] = Te0[s[2] >> 24] ^ Te1[(s[3] >> 16) & 0xff] ^ Te2[(s[0] >>  8) & 0xff] ^ Te3[s[1] & 0xff] ^ rk[54];
	    t[3] = Te0[s[3] >> 24] ^ Te1[(s[0] >> 16) & 0xff] ^ Te2[(s[1] >>  8) & 0xff] ^ Te3[s[2] & 0xff] ^ rk[55];
	}
    }
    rk += nrounds << 2;
    
    s[0] =
	(Te4[(t[0] >> 24)       ] & 0xff000000) ^
	(Te4[(t[1] >> 16) & 0xff] & 0x00ff0000) ^
	(Te4[(t[2] >>  8) & 0xff] & 0x0000ff00) ^
	(Te4[(t[3]      ) & 0xff] & 0x000000ff) ^
	rk[0];
    txt[0] ^= GETU32(s+0);
 
    s[1] =
	(Te4[(t[1] >> 24)       ] & 0xff000000) ^
	(Te4[(t[2] >> 16) & 0xff] & 0x00ff0000) ^
	(Te4[(t[3] >>  8) & 0xff] & 0x0000ff00) ^
	(Te4[(t[0]      ) & 0xff] & 0x000000ff) ^
	rk[1];
    txt[1] ^= GETU32(s+1);
    
    s[2] =
	(Te4[(t[2] >> 24)       ] & 0xff000000) ^
	(Te4[(t[3] >> 16) & 0xff] & 0x00ff0000) ^
	(Te4[(t[0] >>  8) & 0xff] & 0x0000ff00) ^
	(Te4[(t[1]      ) & 0xff] & 0x000000ff) ^
	rk[2];
    txt[2] ^= GETU32(s+2);
    
    s[3] =
	(Te4[(t[3] >> 24)       ] & 0xff000000) ^
	(Te4[(t[0] >> 16) & 0xff] & 0x00ff0000) ^
	(Te4[(t[1] >>  8) & 0xff] & 0x0000ff00) ^
	(Te4[(t[2]      ) & 0xff] & 0x000000ff) ^
	rk[3];
    txt[3] ^= GETU32(s+3);
    
}

__global__ void aes_ctr_crypt(u32 *rk, int nrounds, u8 *text, u8 *ctr)
{
    u32 s[4];
    u32 t[4];
    u32 *txt = (u32*)(text+(16*(blockIdx.x*blockDim.x+threadIdx.x)));

    big_u128_add(ctr, thread_ctr_offset(), (u8*)s);

    for (int i=0; i<4; i++)
	s[i] = GETU32(s+i)^rk[i];
    
    /* round 1: */
    t[0] = Te0[s[0] >> 24] ^ Te1[(s[1] >> 16) & 0xff] ^ Te2[(s[2] >>  8) & 0xff] ^ Te3[s[3] & 0xff] ^ rk[ 4];
    t[1] = Te0[s[1] >> 24] ^ Te1[(s[2] >> 16) & 0xff] ^ Te2[(s[3] >>  8) & 0xff] ^ Te3[s[0] & 0xff] ^ rk[ 5];
    t[2] = Te0[s[2] >> 24] ^ Te1[(s[3] >> 16) & 0xff] ^ Te2[(s[0] >>  8) & 0xff] ^ Te3[s[1] & 0xff] ^ rk[ 6];
    t[3] = Te0[s[3] >> 24] ^ Te1[(s[0] >> 16) & 0xff] ^ Te2[(s[1] >>  8) & 0xff] ^ Te3[s[2] & 0xff] ^ rk[ 7];
    /* round 2: */
    s[0] = Te0[t[0] >> 24] ^ Te1[(t[1] >> 16) & 0xff] ^ Te2[(t[2] >>  8) & 0xff] ^ Te3[t[3] & 0xff] ^ rk[ 8];
    s[1] = Te0[t[1] >> 24] ^ Te1[(t[2] >> 16) & 0xff] ^ Te2[(t[3] >>  8) & 0xff] ^ Te3[t[0] & 0xff] ^ rk[ 9];
    s[2] = Te0[t[2] >> 24] ^ Te1[(t[3] >> 16) & 0xff] ^ Te2[(t[0] >>  8) & 0xff] ^ Te3[t[1] & 0xff] ^ rk[10];
    s[3] = Te0[t[3] >> 24] ^ Te1[(t[0] >> 16) & 0xff] ^ Te2[(t[1] >>  8) & 0xff] ^ Te3[t[2] & 0xff] ^ rk[11];
    /* round 3: */
    t[0] = Te0[s[0] >> 24] ^ Te1[(s[1] >> 16) & 0xff] ^ Te2[(s[2] >>  8) & 0xff] ^ Te3[s[3] & 0xff] ^ rk[12];
    t[1] = Te0[s[1] >> 24] ^ Te1[(s[2] >> 16) & 0xff] ^ Te2[(s[3] >>  8) & 0xff] ^ Te3[s[0] & 0xff] ^ rk[13];
    t[2] = Te0[s[2] >> 24] ^ Te1[(s[3] >> 16) & 0xff] ^ Te2[(s[0] >>  8) & 0xff] ^ Te3[s[1] & 0xff] ^ rk[14];
    t[3] = Te0[s[3] >> 24] ^ Te1[(s[0] >> 16) & 0xff] ^ Te2[(s[1] >>  8) & 0xff] ^ Te3[s[2] & 0xff] ^ rk[15];
    /* round 4: */
    s[0] = Te0[t[0] >> 24] ^ Te1[(t[1] >> 16) & 0xff] ^ Te2[(t[2] >>  8) & 0xff] ^ Te3[t[3] & 0xff] ^ rk[16];
    s[1] = Te0[t[1] >> 24] ^ Te1[(t[2] >> 16) & 0xff] ^ Te2[(t[3] >>  8) & 0xff] ^ Te3[t[0] & 0xff] ^ rk[17];
    s[2] = Te0[t[2] >> 24] ^ Te1[(t[3] >> 16) & 0xff] ^ Te2[(t[0] >>  8) & 0xff] ^ Te3[t[1] & 0xff] ^ rk[18];
    s[3] = Te0[t[3] >> 24] ^ Te1[(t[0] >> 16) & 0xff] ^ Te2[(t[1] >>  8) & 0xff] ^ Te3[t[2] & 0xff] ^ rk[19];
    /* round 5: */
    t[0] = Te0[s[0] >> 24] ^ Te1[(s[1] >> 16) & 0xff] ^ Te2[(s[2] >>  8) & 0xff] ^ Te3[s[3] & 0xff] ^ rk[20];
    t[1] = Te0[s[1] >> 24] ^ Te1[(s[2] >> 16) & 0xff] ^ Te2[(s[3] >>  8) & 0xff] ^ Te3[s[0] & 0xff] ^ rk[21];
    t[2] = Te0[s[2] >> 24] ^ Te1[(s[3] >> 16) & 0xff] ^ Te2[(s[0] >>  8) & 0xff] ^ Te3[s[1] & 0xff] ^ rk[22];
    t[3] = Te0[s[3] >> 24] ^ Te1[(s[0] >> 16) & 0xff] ^ Te2[(s[1] >>  8) & 0xff] ^ Te3[s[2] & 0xff] ^ rk[23];
    /* round 6: */
    s[0] = Te0[t[0] >> 24] ^ Te1[(t[1] >> 16) & 0xff] ^ Te2[(t[2] >>  8) & 0xff] ^ Te3[t[3] & 0xff] ^ rk[24];
    s[1] = Te0[t[1] >> 24] ^ Te1[(t[2] >> 16) & 0xff] ^ Te2[(t[3] >>  8) & 0xff] ^ Te3[t[0] & 0xff] ^ rk[25];
    s[2] = Te0[t[2] >> 24] ^ Te1[(t[3] >> 16) & 0xff] ^ Te2[(t[0] >>  8) & 0xff] ^ Te3[t[1] & 0xff] ^ rk[26];
    s[3] = Te0[t[3] >> 24] ^ Te1[(t[0] >> 16) & 0xff] ^ Te2[(t[1] >>  8) & 0xff] ^ Te3[t[2] & 0xff] ^ rk[27];
    /* round 7: */
    t[0] = Te0[s[0] >> 24] ^ Te1[(s[1] >> 16) & 0xff] ^ Te2[(s[2] >>  8) & 0xff] ^ Te3[s[3] & 0xff] ^ rk[28];
    t[1] = Te0[s[1] >> 24] ^ Te1[(s[2] >> 16) & 0xff] ^ Te2[(s[3] >>  8) & 0xff] ^ Te3[s[0] & 0xff] ^ rk[29];
    t[2] = Te0[s[2] >> 24] ^ Te1[(s[3] >> 16) & 0xff] ^ Te2[(s[0] >>  8) & 0xff] ^ Te3[s[1] & 0xff] ^ rk[30];
    t[3] = Te0[s[3] >> 24] ^ Te1[(s[0] >> 16) & 0xff] ^ Te2[(s[1] >>  8) & 0xff] ^ Te3[s[2] & 0xff] ^ rk[31];
    /* round 8: */
    s[0] = Te0[t[0] >> 24] ^ Te1[(t[1] >> 16) & 0xff] ^ Te2[(t[2] >>  8) & 0xff] ^ Te3[t[3] & 0xff] ^ rk[32];
    s[1] = Te0[t[1] >> 24] ^ Te1[(t[2] >> 16) & 0xff] ^ Te2[(t[3] >>  8) & 0xff] ^ Te3[t[0] & 0xff] ^ rk[33];
    s[2] = Te0[t[2] >> 24] ^ Te1[(t[3] >> 16) & 0xff] ^ Te2[(t[0] >>  8) & 0xff] ^ Te3[t[1] & 0xff] ^ rk[34];
    s[3] = Te0[t[3] >> 24] ^ Te1[(t[0] >> 16) & 0xff] ^ Te2[(t[1] >>  8) & 0xff] ^ Te3[t[2] & 0xff] ^ rk[35];
    /* round 9: */
    t[0] = Te0[s[0] >> 24] ^ Te1[(s[1] >> 16) & 0xff] ^ Te2[(s[2] >>  8) & 0xff] ^ Te3[s[3] & 0xff] ^ rk[36];
    t[1] = Te0[s[1] >> 24] ^ Te1[(s[2] >> 16) & 0xff] ^ Te2[(s[3] >>  8) & 0xff] ^ Te3[s[0] & 0xff] ^ rk[37];
    t[2] = Te0[s[2] >> 24] ^ Te1[(s[3] >> 16) & 0xff] ^ Te2[(s[0] >>  8) & 0xff] ^ Te3[s[1] & 0xff] ^ rk[38];
    t[3] = Te0[s[3] >> 24] ^ Te1[(s[0] >> 16) & 0xff] ^ Te2[(s[1] >>  8) & 0xff] ^ Te3[s[2] & 0xff] ^ rk[39];
    if (nrounds > 10)
    {
	/* round 10: */
	s[0] = Te0[t[0] >> 24] ^ Te1[(t[1] >> 16) & 0xff] ^ Te2[(t[2] >>  8) & 0xff] ^ Te3[t[3] & 0xff] ^ rk[40];
	s[1] = Te0[t[1] >> 24] ^ Te1[(t[2] >> 16) & 0xff] ^ Te2[(t[3] >>  8) & 0xff] ^ Te3[t[0] & 0xff] ^ rk[41];
	s[2] = Te0[t[2] >> 24] ^ Te1[(t[3] >> 16) & 0xff] ^ Te2[(t[0] >>  8) & 0xff] ^ Te3[t[1] & 0xff] ^ rk[42];
	s[3] = Te0[t[3] >> 24] ^ Te1[(t[0] >> 16) & 0xff] ^ Te2[(t[1] >>  8) & 0xff] ^ Te3[t[2] & 0xff] ^ rk[43];
	/* round 11: */
	t[0] = Te0[s[0] >> 24] ^ Te1[(s[1] >> 16) & 0xff] ^ Te2[(s[2] >>  8) & 0xff] ^ Te3[s[3] & 0xff] ^ rk[44];
	t[1] = Te0[s[1] >> 24] ^ Te1[(s[2] >> 16) & 0xff] ^ Te2[(s[3] >>  8) & 0xff] ^ Te3[s[0] & 0xff] ^ rk[45];
	t[2] = Te0[s[2] >> 24] ^ Te1[(s[3] >> 16) & 0xff] ^ Te2[(s[0] >>  8) & 0xff] ^ Te3[s[1] & 0xff] ^ rk[46];
	t[3] = Te0[s[3] >> 24] ^ Te1[(s[0] >> 16) & 0xff] ^ Te2[(s[1] >>  8) & 0xff] ^ Te3[s[2] & 0xff] ^ rk[47];
	if (nrounds > 12)
	{
	    /* round 12: */
	    s[0] = Te0[t[0] >> 24] ^ Te1[(t[1] >> 16) & 0xff] ^ Te2[(t[2] >>  8) & 0xff] ^ Te3[t[3] & 0xff] ^ rk[48];
	    s[1] = Te0[t[1] >> 24] ^ Te1[(t[2] >> 16) & 0xff] ^ Te2[(t[3] >>  8) & 0xff] ^ Te3[t[0] & 0xff] ^ rk[49];
	    s[2] = Te0[t[2] >> 24] ^ Te1[(t[3] >> 16) & 0xff] ^ Te2[(t[0] >>  8) & 0xff] ^ Te3[t[1] & 0xff] ^ rk[50];
	    s[3] = Te0[t[3] >> 24] ^ Te1[(t[0] >> 16) & 0xff] ^ Te2[(t[1] >>  8) & 0xff] ^ Te3[t[2] & 0xff] ^ rk[51];
	    /* round 13: */
	    t[0] = Te0[s[0] >> 24] ^ Te1[(s[1] >> 16) & 0xff] ^ Te2[(s[2] >>  8) & 0xff] ^ Te3[s[3] & 0xff] ^ rk[52];
	    t[1] = Te0[s[1] >> 24] ^ Te1[(s[2] >> 16) & 0xff] ^ Te2[(s[3] >>  8) & 0xff] ^ Te3[s[0] & 0xff] ^ rk[53];
	    t[2] = Te0[s[2] >> 24] ^ Te1[(s[3] >> 16) & 0xff] ^ Te2[(s[0] >>  8) & 0xff] ^ Te3[s[1] & 0xff] ^ rk[54];
	    t[3] = Te0[s[3] >> 24] ^ Te1[(s[0] >> 16) & 0xff] ^ Te2[(s[1] >>  8) & 0xff] ^ Te3[s[2] & 0xff] ^ rk[55];
	}
    }
    rk += nrounds << 2;
    
    s[0] =
	(Te4[(t[0] >> 24)       ] & 0xff000000) ^
	(Te4[(t[1] >> 16) & 0xff] & 0x00ff0000) ^
	(Te4[(t[2] >>  8) & 0xff] & 0x0000ff00) ^
	(Te4[(t[3]      ) & 0xff] & 0x000000ff) ^
	rk[0];
    txt[0] ^= GETU32(s+0);
 
    s[1] =
	(Te4[(t[1] >> 24)       ] & 0xff000000) ^
	(Te4[(t[2] >> 16) & 0xff] & 0x00ff0000) ^
	(Te4[(t[3] >>  8) & 0xff] & 0x0000ff00) ^
	(Te4[(t[0]      ) & 0xff] & 0x000000ff) ^
	rk[1];
    txt[1] ^= GETU32(s+1);
    
    s[2] =
	(Te4[(t[2] >> 24)       ] & 0xff000000) ^
	(Te4[(t[3] >> 16) & 0xff] & 0x00ff0000) ^
	(Te4[(t[0] >>  8) & 0xff] & 0x0000ff00) ^
	(Te4[(t[1]      ) & 0xff] & 0x000000ff) ^
	rk[2];
    txt[2] ^= GETU32(s+2);
    
    s[3] =
	(Te4[(t[3] >> 24)       ] & 0xff000000) ^
	(Te4[(t[0] >> 16) & 0xff] & 0x00ff0000) ^
	(Te4[(t[1] >>  8) & 0xff] & 0x0000ff00) ^
	(Te4[(t[2]      ) & 0xff] & 0x000000ff) ^
	rk[3];
    txt[3] ^= GETU32(s+3);
    
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

#define lid threadIdx.y*4 + threadIdx.x
#define bid blockIdx.x
#define n1  threadIdx.y*4 + (threadIdx.x+1)%4
#define n2  threadIdx.y*4 + (threadIdx.x+2)%4
#define n3  threadIdx.y*4 + (threadIdx.x+3)%4   

__global__ void aes_encrypt_bp4t(u32 *rk, int nrounds, u8* text)
{
    __shared__ u32 s[WORDS_PER_BLOCK];
    __shared__ u32 t[WORDS_PER_BLOCK];
        
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


typedef struct {
    u32 a,b,c,d;
} cube128;


__global__ void xts_encrypt(u32 *rk, int nrounds, u8* text, be128 *ivs)
{
    u32 s0, s1, s2, s3, t0, t1, t2, t3;
    u8 *txt = text+(16*(blockIdx.x*blockDim.x+threadIdx.x));
    
    cube128 *iv = (cube128*)(ivs+threadIdx.x);
    
    s0= iv->a ^ (*(u32*)(txt));
    s1= iv->b ^ (*(u32*)(txt+4));
    s2= iv->c ^ (*(u32*)(txt+8));
    s3= iv->d ^ (*(u32*)(txt+12));

    /*
     * map byte array block to cipher state
     * and add initial round key:
     */
    s0 = __byte_perm(s0, 0, ENDIAN_SELECTOR) ^ rk[0];
    s1 = __byte_perm(s1, 0, ENDIAN_SELECTOR) ^ rk[1];
    s2 = __byte_perm(s2, 0, ENDIAN_SELECTOR) ^ rk[2];
    s3 = __byte_perm(s3, 0, ENDIAN_SELECTOR) ^ rk[3];

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
    //PUTU32(&s0, s0);
    s0 = __byte_perm(s0, 0, ENDIAN_SELECTOR);
    s1 =
	(Te4[(t1 >> 24)       ] & 0xff000000) ^
	(Te4[(t2 >> 16) & 0xff] & 0x00ff0000) ^
	(Te4[(t3 >>  8) & 0xff] & 0x0000ff00) ^
	(Te4[(t0      ) & 0xff] & 0x000000ff) ^
	rk[1];
    //PUTU32(&s1, s1);
    s1 = __byte_perm(s1, 0, ENDIAN_SELECTOR);
    s2 =
	(Te4[(t2 >> 24)       ] & 0xff000000) ^
	(Te4[(t3 >> 16) & 0xff] & 0x00ff0000) ^
	(Te4[(t0 >>  8) & 0xff] & 0x0000ff00) ^
	(Te4[(t1      ) & 0xff] & 0x000000ff) ^
	rk[2];
    //PUTU32(&s2, s2);
    s2 = __byte_perm(s2, 0, ENDIAN_SELECTOR);
    s3 =
	(Te4[(t3 >> 24)       ] & 0xff000000) ^
	(Te4[(t0 >> 16) & 0xff] & 0x00ff0000) ^
	(Te4[(t1 >>  8) & 0xff] & 0x0000ff00) ^
	(Te4[(t2      ) & 0xff] & 0x000000ff) ^
	rk[3];
    //PUTU32(&s3, s3);
    s3 = __byte_perm(s3, 0, ENDIAN_SELECTOR);
    
    *(u32*)(txt)    = iv->a ^ s0;
    *(u32*)(txt+4)  = iv->b ^ s1;
    *(u32*)(txt+8)  = iv->c ^ s2;
    *(u32*)(txt+12) = iv->d ^ s3;
}

__global__ void xts_decrypt(u32 *rk, int nrounds, u8* text, be128 *ivs)
{
    u32 s0, s1, s2, s3, t0, t1, t2, t3;
    u8 *txt = text+(16*(blockIdx.x*blockDim.x+threadIdx.x));
    cube128 *iv = (cube128*)(ivs+threadIdx.x);
    
    s0= iv->a ^ (*(u32*)(txt));
    s1= iv->b ^ (*(u32*)(txt+4));
    s2= iv->c ^ (*(u32*)(txt+8));
    s3= iv->d ^ (*(u32*)(txt+12));

    /*
     * map byte array block to cipher state
     * and add initial round key:
     */
    s0 = __byte_perm(s0, 0, ENDIAN_SELECTOR) ^ rk[0];
    s1 = __byte_perm(s1, 0, ENDIAN_SELECTOR) ^ rk[1];
    s2 = __byte_perm(s2, 0, ENDIAN_SELECTOR) ^ rk[2];
    s3 = __byte_perm(s3, 0, ENDIAN_SELECTOR) ^ rk[3];
  
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
    s0 = __byte_perm(s0, 0, ENDIAN_SELECTOR);
    //PUTU32(&s0, s0);
    s1 =
	(Td4[(t1 >> 24)       ] & 0xff000000) ^
	(Td4[(t0 >> 16) & 0xff] & 0x00ff0000) ^
	(Td4[(t3 >>  8) & 0xff] & 0x0000ff00) ^
	(Td4[(t2      ) & 0xff] & 0x000000ff) ^
	rk[1];
    s1 = __byte_perm(s1, 0, ENDIAN_SELECTOR);
    //PUTU32(&s1, s1);
    s2 =
	(Td4[(t2 >> 24)       ] & 0xff000000) ^
	(Td4[(t1 >> 16) & 0xff] & 0x00ff0000) ^
	(Td4[(t0 >>  8) & 0xff] & 0x0000ff00) ^
	(Td4[(t3      ) & 0xff] & 0x000000ff) ^
	rk[2];
    s2 = __byte_perm(s2, 0, ENDIAN_SELECTOR);
    //PUTU32(&s2, s2);
    s3 =
	(Td4[(t3 >> 24)       ] & 0xff000000) ^
	(Td4[(t2 >> 16) & 0xff] & 0x00ff0000) ^
	(Td4[(t1 >>  8) & 0xff] & 0x0000ff00) ^
	(Td4[(t0      ) & 0xff] & 0x000000ff) ^
	rk[3];
    s3 = __byte_perm(s3, 0, ENDIAN_SELECTOR);
    //PUTU32(&s3, s3);
    
    *(u32*)(txt)    = iv->a ^ s0;
    *(u32*)(txt+4)  = iv->b ^ s1;
    *(u32*)(txt+8)  = iv->c ^ s2;
    *(u32*)(txt+12) = iv->d ^ s3;
    
}

