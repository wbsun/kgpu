/* This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2010-2011 University of Utah and the Flux Group.
 * All rights reserved.
 *
 * KGPU GAES common header
 */

#ifndef __GAES_COMMON_H__
#define __GAES_COMMON_H__

struct crypto_gaes_ctr_info {
    u32 key_enc[AES_MAX_KEYLENGTH_U32];
    u32 key_dec[AES_MAX_KEYLENGTH_U32];
    u32 key_length;
    u8  padding[28];
    u8  ctrblk[AES_BLOCK_SIZE];	
};


#endif
