/*
 * This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2010-2011 University of Utah and the Flux Group.
 * All rights reserved.
 *
 */

#ifndef __SERVICE_H__
#define __SERVICE_H__

struct service {
    char name[SERVICE_NAME_SIZE];
    int sid;
    int (*compute_size)(struct service_request *sreq);
    int (*launch)(struct service_request *sreq);
    int (*prepare)(struct service_request *sreq);
    int (*post)(struct service_request *sreq);
};

#define SERVICE_INIT "init_service"
#define SERVICE_FINIT "finit_service"
#define SERVICE_LIB_PREFIX "libsrv_"

typedef int (*fn_init_service)(
    void* libhandle, int (*reg_srv)(struct service *, void*));
typedef int (*fn_finit_service)(
    void* libhandle, int (*unreg_srv)(const char*));

struct service * lookup_service(const char *name);
int register_service(struct service *s, void *libhandle);
int unregister_service(const char *name);
int load_service(const char *libpath);
int load_all_services(const char *libdir);
int unload_service(const char *name);
int unload_all_services();

#endif
