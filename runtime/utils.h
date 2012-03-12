/*
 * This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2012 University of Utah and the Flux Group.
 * All rights reserved.
 */

/*
 * utilities used by KGPU runtime.
 */
#ifndef __UTILS_H__
#define __UTILS_H__


#define TO_UL(v) ((unsigned long)(v))

#define ADDR_WITHIN(pointer, base, size)		\
    (TO_UL(pointer) >= TO_UL(base) &&			\
     (TO_UL(pointer) < TO_UL(base)+TO_UL(size)))

#define ADDR_REBASE(dst_base, src_base, pointer)	\
    (TO_UL(dst_base) + (				\
	TO_UL(pointer)-TO_UL(src_base)))


/* u/s types */
#if !defined __KERNEL__ && !defined __KGPU_NO_USTYPES__

typedef unsigned char  u8;
typedef unsigned short u16;
typedef unsigned int   u32;
typedef unsigned long  u64;

typedef char  s8;
typedef short s16;
typedef int   s32;
typedef long  s64;

#endif

/*
 * Request id generated in kernel space is positive,
 * and negtive in user space.
 */
#define IS_KRID(id) ((id) > 0)
#define IS_URID(id) ((id) < 0)


#endif
