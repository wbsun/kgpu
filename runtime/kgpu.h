/*
 * This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2012 University of Utah and the Flux Group.
 * All rights reserved.
 *
 * Common header for userspace helper, kernel mode KGPU and KGPU clients
 *
 */
#ifndef __KGPU_H__
#define __KGPU_H__

#include "utils.h"

#define KGPU_SERVICE_NAME_SIZE 32

/* KGPU's errno */
#define KGPU_OK 0
#define KGPU_NO_RESPONSE 1
#define KGPU_NO_SERVICE 2
#define KGPU_TERMINATED 3

#define KGPU_DEV_NAME "kgpu"

/* Level of dependence  */
#define KGPU_DEP_ORDER 0   /* Just in order */
#define KGPU_DEP_DEVICE 1  /* Must be on same device + level ORDER */
#define KGPU_DEP_DATA 2    /* Shared data + level DEVICE */


#endif
