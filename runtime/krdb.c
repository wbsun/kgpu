/*
 * This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2012 University of Utah and the Flux Group.
 * All rights reserved.
 *
 * Data store for k_request_t in key-value db.
 */
#include "kgpu.h"

#define value_t k_request_t
#define key_t int
#define key2hash(k) ((u32)(k))
#define value2key(v) ((int)(v->id))
#define key_equal(k1, k2) ((k1) == (k2))

#include "hashtable.c"

static hashtable_t krdb;

void krdb_init(void)
{
	ht_init(&krdb);
}

void krdb_finit(void)
{
	ht_finit(&krdb);
}

k_request_t* get_krequest(int id)
{
	return ht_get(&krdb, id);
}

k_request_t* put_krequest(k_request_t* r)
{
	return ht_put(&krdb, r);
}

k_request_t* del_krequest(int id);
{
	return ht_del(&krdb, id);
}

u32 krdb_size(void)
{
	return ht_size(&krdb);
}
