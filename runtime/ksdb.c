/*
 * This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2012 University of Utah and the Flux Group.
 * All rights reserved.
 *
 * Data store for kg_k_service_t in key-value db.
 */
#include "kgpu.h"

/* djb2 string hash function */
static u32 strhash(char* s)
{
	u32 hash = 5381;

	int c;
	while(c = *s++)
		hash = ((hash << 5) + hash) +c;
	return hash;
}

#define value_t kg_k_service_t
#define key_t char*
#define key2hash(k) strhash(k)
#define value2key(v) ((v)->name)
#define key_equal(k1, k2) (!strcmp((k1), (k2)))

#include "hashtable.c"

static hashtable_t ksdb;

void ksdb_init(void)
{
	ht_init(&ksdb);
}

void ksdb_finit(void)
{
	ht_finit(&ksdb);
}

k_request_t* get_kservice(char *name)
{
	return ht_get(&ksdb, name);
}

k_request_t* put_kservice(kg_k_service_t* s)
{
	return ht_put(&ksdb, s);
}

k_request_t* del_kservice(char *name);
{
	return ht_del(&ksdb, name);
}

u32 ksdb_size(void)
{
	return ht_size(&ksdb);
}

int kg_register_kservice(kg_k_service_t *s)
{
	int ret;

	ret = put_kservice(s);
	if (ret) {
		krt_log(KGPU_LOG_ERROR,
			"Fail to register service %s, code %d.\n",
			s->name, ret);
	} else {
		krt_log(KGPU_LOG_DEBUG,
			"Service %s registered.\n", s->name);
	}
	return ret;
}
EXPORT_SYMBOL_GPL(kg_register_kservice);

int kg_unregister_kservice(kg_k_service_t *s)
{
	int ret;

	ret = del_kservice(s->name);
	if (ret) {
		krt_log(KGPU_LOG_ERROR,
			"Fail to unregister service %s, code %d.\n",
			s->name, ret);
	} else {
		krt_log(KGPU_LOG_DEBUG,
			"Service %s unregistered.\n", s->name);
	}
	return ret;
}
EXPORT_SYMBOL_GPL(kg_unregister_kservice);
