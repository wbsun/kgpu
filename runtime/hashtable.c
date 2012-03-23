/*
 * This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2012 University of Utah and the Flux Group.
 * All rights reserved.
 *
 * Common hashtable implementation for both kernel and user spaces.
 */

/* Required defs:  k_request_t as example */
/*
#define value_t k_request_t
#define key_t int
#define key2hash(k) ((u32)(k))
#define value2key(v) ((key_t)(v->id))
#define key_equal(k1, k2) ((k1) == (k2))
*/

/* Rules:
 * - A value has a hlist_node field for hash list, named 'hashnode';
 * - Keys mapped to u32 hash value;
 * - Include this c file in customized implementation file;
 * - Define required defs before including, define customized functions after;
 */

#ifdef __KERNEL__

#include <linux/list.h>
#include <linux/hash.h>

/* There is a chance that kmalloc may fail to alloc enough memory when 
 * we alloc for buckets. Better method is to define an alloc_big_mem
 * that uses vmalloc instead of kmalloc.
 */
#define alloc_mem(size) kmalloc(size, GFP_KERNEL)
#define free_mem(p) kfree(p)

#else

#include "list.h"
#define alloc_mem(size) malloc(size)
#define free_mem(p) free(p)
#define EXPORT_SYMBOL_GPL(s)

#endif

typedef struct hashtable_t {
	u32 prime_idx;
	struct hlist_head* buckets;
	u32 size;
} hashtable_t;

static const int max_load_factor = 5;

static const u32 primes[] = {
	53, 97, 193, 389,
	769, 1543, 3079, 6151,
	12289, 24593, 49157, 98317,
	196613, 393241, 786433, 1572869,
	3145739, 6291469, 12582917, 25165843,
	50331653, 100663319, 201326611, 402653189,
	805306457, 1610612741
};
static const u32 nr_primes = sizeof(primes)/sizeof(u32);

/* Prototype definitions: */
static hashtable_t* mk_ht(hashtable_t* ht, int pidx);
static int rehash_ht(hashtable_t* ht);

static hashtable_t* ht_init(hashtable_t* ht);
static void ht_finit(hashtable_t* ht);
static u32 ht_size(hashtable_t* ht);
static value_t* ht_get(hashtable_t* ht, key_t k);
static value_t* ht_del(hashtable_t* ht, key_t k);
static value_t* ht_put(hashtable_t* ht, value_t* v);


hashtable_t* ht_init(hashtable_t* ht)
{
	return mk_ht(ht, 0);
}

void ht_finit(hashtable_t* ht)
{
	free_mem(ht->buckets);
	ht->buckets = NULL;
	ht->size = 0;
	ht->prime_idx = 0;
}

u32 ht_size(hashtable_t* ht)
{
	return ht->size;
}

static hashtable_t* mk_ht(hashtable_t* ht, int pidx)
{
	int i;
	ht->prime_idx = pidx;
	ht->buckets = (struct hlist_head*)alloc_mem(
		sizeof(struct hlist_head)*primes[ht->prime_idx]);
	if (!ht->buckets) { 
                /* kg_log(KGPU_LOG_ERROR, "Out of memory for ht!\n"); */
		return NULL;
	}

	for (i=0; i<primes[pidx]; i++)
		INIT_HLIST_HEAD(ht->buckets + i);
	ht->size = 0;
	return ht;
}

value_t* ht_get(hashtable_t* ht, key_t k)
{
	int ib;
	struct hlist_head *buck;
	struct hlist_node *pos;
	value_t *v;

	ib = key2hash(k) % primes[ht->prime_idx];
	buck = ht->buckets + ib;
	
	hlist_for_each(pos, buck) {
		v = hlist_entry(pos, value_t, hashnode);
		if (key_equal(value2key(v), k)
			return v;
	}
	return NULL;
}

value_t* ht_del(hashtable_t* ht, key_t k)
{
	value_t* v = ht_get(ht, k);
	if (v) {
		hlist_del(&(v->hashnode));
		INIT_HLIST_NODE(&(v->hashnode));
		ht->size--;
	}
	return v;
}

static int rehash_ht(hashtable_t* ht)
{
	hashtable_t tht;
	struct hlist_head *h, *nh;
	struct hlist_node *pos, *n;
	int i, ib;

        if (ht->prime_idx == nr_primes-1) {
		return -ENOMEM;
	}

	if (!mk_ht(&tht, ht->prime_idx+1)) {
		return -ENOMEM;
	}

	for (i=0; i<primes[ht->prime_idx]; i++) {
		h = ht->buckets+i;
		hlist_for_each_safe(pos, n, h) {
			value_t *v = hlist_entry(pos, value_t, hashnode);
			ib = key2hash(value2key(v)) % primes[tht.prime_idx];
			nh = tht.buckets + ib;
			hlist_del(pos);
			hlist_add_head(&(v->hashnode), nh);
		}
	}

	free_mem(ht->buckets);
	ht->buckets = tht.buckets;
	ht->prime_idx++;
}

static inline void hlist_replace(struct hlist_node *old, struct hlist_node *new)
{
	new->next = old->next;
	new->pprev = old->pprev;
	if (new->next)
		new->next->pprev = &new->next;
	if (new->pprev)
		*(new->pprev) = new;
}

value_t* ht_put(hashtable_t* ht, value_t* v)
{
	int ib;
	struct hlist_head *buck;

	value_t* vt = ht_get(ht, value2key(v));
	if (vt) {
		if (vt != v)
			hlist_replace(&(vt->hashnode), &(v->hashnode));
		return v;
	}
	
	if ((ht->size+1)/primes[ht->prime_idx] > max_load_factor) {
		if (rehash_ht(ht))
			return NULL;
	}

	ib = key2hash(value2key(v)) % primes[ht->prime_idx];
	buck = ht->buckets + ib;

	hlist_add_head(&(v->hashnode), buck);
	ht->size++;

	return v;	
}





