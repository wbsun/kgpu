/*
 * This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2012 University of Utah and the Flux Group.
 * All rights reserved.
 */
/*
 * KGPU userspace runtime header
 */
#ifndef __URT_H__
#define __URT_H__

#include "service.h"

struct kg_service * lookup_service(const char *name);
int register_service(struct kg_service *s, void *libhandle);
int unregister_service(const char *name);
int load_service(const char *libpath);
int load_all_services(const char *libdir);
int unload_service(const char *name);
int unload_all_services();

#endif
