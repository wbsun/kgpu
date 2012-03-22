#include "kgpu.h"

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

typedef struct rhashtable {
	u32 prime_idx;
	struct hlist_head* buckets;
	u32 size;
} rhashtable;

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
static rhashtable* mk_rbase(rhashtable* rb, int pidx);

rhashtable* rbase_init(rhashtable* rb)
{
	return mk_rbase(rb, 0);
}
EXPORT_SYMBOL_GPL(rbase_init);

static rhashtable* mk_rbase(rhashtable* rb, int pidx)
{
	int i;
	rb->prime_idx = pidx;
	rb->buckets = (struct hlist_head*)alloc_mem(
		sizeof(struct hlist_head)*primes[rb->prime_idx]);
	if (!rb->buckets) {
		kg_log(KGPU_LOG_ERROR, "Out of memory for rbase!\n");
		return NULL;
	}

	for (i=0; i<primes[pidx]; i++)
		INIT_HLIST_HEAD(rb->buckets + i);
	rb->size = 0;
	return rb;
}



