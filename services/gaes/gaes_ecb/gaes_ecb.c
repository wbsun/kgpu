/*
 * GPU accelerated AES-ECB cipher
 * The cipher and the algorithm are binded closely.
 *
 * This cipher is mostly derived from the crypto/ecb.c in Linux kernel tree.
 *
 * The cipher can only handle data with the size of multiples of PAGE_SIZE
 */
#include <crypto/algapi.h>
#include <linux/err.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/scatterlist.h>
#include <linux/slab.h>
#include <crypto/aes.h>
#include <linux/string.h>
#include "../../../kgpu/kkgpu.h"
#include "../gaesk.h"


struct crypto_gecb_ctx {
    struct crypto_cipher *child;
    struct crypto_aes_ctx aes_ctx;    
    u8 key[32];
};

static int crypto_gecb_setkey(struct crypto_tfm *parent, const u8 *key,
			     unsigned int keylen)
{
    struct crypto_gecb_ctx *ctx = crypto_tfm_ctx(parent);
    struct crypto_cipher *child = ctx->child;
    int err;

    crypto_cipher_clear_flags(child, CRYPTO_TFM_REQ_MASK);
    crypto_cipher_set_flags(child, crypto_tfm_get_flags(parent) &
			    CRYPTO_TFM_REQ_MASK);

    err = crypto_aes_expand_key(&ctx->aes_ctx,
				key, keylen);
    err = crypto_cipher_setkey(child, key, keylen);

    
    cvt_endian_u32(ctx->aes_ctx.key_enc, AES_MAX_KEYLENGTH_U32);
    cvt_endian_u32(ctx->aes_ctx.key_dec, AES_MAX_KEYLENGTH_U32);
    
    memcpy(ctx->key, key, keylen);
    
    crypto_tfm_set_flags(parent, crypto_cipher_get_flags(child) &
			 CRYPTO_TFM_RES_MASK);
    return err;
}

static int crypto_gecb_crypt(struct blkcipher_desc *desc,
                            struct scatterlist *dst, struct scatterlist *src,
			    unsigned int sz,
			    int enc)
{
    int err=0;
    unsigned int nbytes;
    u8* gpos;
    int i = 0;    
    
    struct kgpu_req *req;
    struct kgpu_resp *resp;
    struct kgpu_buffer *buf;

    struct crypto_blkcipher *tfm = desc->tfm;
    struct crypto_gecb_ctx *ctx = crypto_blkcipher_ctx(tfm);
    struct blkcipher_walk walk;


    blkcipher_walk_init(&walk, dst, src, sz);
    
    buf = alloc_gpu_buffer();
    if (!buf) {
	printk("[gecb] Error: GPU buffer is null.\n");
	return -EFAULT;
    }

    req = alloc_kgpu_request();
    resp = alloc_kgpu_response();
    if (!req || !resp) {
	return -EFAULT;
    }

    err = blkcipher_walk_virt(desc, &walk);

    while ((nbytes = walk.nbytes)) {
	u8 *wsrc = walk.src.virt.addr;
	if (nbytes > KGPU_BUF_FRAME_SIZE) {
	    return -EFAULT;
	}

#ifndef _NDEBUG
	if (nbytes != PAGE_SIZE)
	    printk("[gecb] WARNING: %u is not PAGE_SIZE\n", nbytes);
	    
#endif

	gpos = buf->paddrs[i++];
	memcpy(__va(gpos), wsrc, nbytes);

	err = blkcipher_walk_done(desc, &walk, 0);
    }

    gpos = buf->paddrs[i];
    memcpy(__va(gpos), &(ctx->aes_ctx), sizeof(struct crypto_aes_ctx));   

    strcpy(req->kureq.sname, enc?"gecb-enc":"gecb-dec");
    req->kureq.input = buf->gb.addr;
    req->kureq.output = buf->gb.addr;
    req->kureq.insize = sz+PAGE_SIZE;
    req->kureq.outsize = sz;

    if (call_gpu_sync(req, resp)) {
	err = -EFAULT;
	printk("[gecb] Error: callgpu error\n");
    } else {
	i=0;
	blkcipher_walk_init(&walk, dst, src, sz);
	err = blkcipher_walk_virt(desc, &walk);
	
	while ((nbytes = walk.nbytes)) {
	    u8 *wdst = walk.dst.virt.addr;
	    if (nbytes > KGPU_BUF_FRAME_SIZE) {
		return -EFAULT;
	    }

#ifndef _NDEBUG
	    if (nbytes != PAGE_SIZE)
		printk("[gecb] WARNING: %u is not PAGE_SIZE\n", nbytes);
#endif

	    gpos = buf->paddrs[i++];	
	    memcpy(wdst, __va(gpos), nbytes);       

	    err = blkcipher_walk_done(desc, &walk, 0);
	}
    }
    
    free_kgpu_request(req);
    free_kgpu_response(resp);
    free_gpu_buffer(buf);

    return err;
}

static int crypto_ecb_crypt(struct blkcipher_desc *desc,
			    struct blkcipher_walk *walk,
			    struct crypto_cipher *tfm,
			    void (*fn)(struct crypto_tfm *, u8 *, const u8 *))
{
	int bsize = crypto_cipher_blocksize(tfm);
	unsigned int nbytes;
	int err;

	err = blkcipher_walk_virt(desc, walk);

	while ((nbytes = walk->nbytes)) {
		u8 *wsrc = walk->src.virt.addr;
		u8 *wdst = walk->dst.virt.addr;

		do {
			fn(crypto_cipher_tfm(tfm), wdst, wsrc);

			wsrc += bsize;
			wdst += bsize;
		} while ((nbytes -= bsize) >= bsize);

		err = blkcipher_walk_done(desc, walk, nbytes);
	}

	return err;
}

static int crypto_ecb_encrypt(struct blkcipher_desc *desc,
			      struct scatterlist *dst, struct scatterlist *src,
			      unsigned int nbytes)
{
	struct blkcipher_walk walk;
	struct crypto_blkcipher *tfm = desc->tfm;
	struct crypto_gecb_ctx *ctx = crypto_blkcipher_ctx(tfm);
	struct crypto_cipher *child = ctx->child;

	blkcipher_walk_init(&walk, dst, src, nbytes);
	return crypto_ecb_crypt(desc, &walk, child,
				crypto_cipher_alg(child)->cia_encrypt);
}

static int crypto_ecb_decrypt(struct blkcipher_desc *desc,
			      struct scatterlist *dst, struct scatterlist *src,
			      unsigned int nbytes)
{
	struct blkcipher_walk walk;
	struct crypto_blkcipher *tfm = desc->tfm;
	struct crypto_gecb_ctx *ctx = crypto_blkcipher_ctx(tfm);
	struct crypto_cipher *child = ctx->child;

	blkcipher_walk_init(&walk, dst, src, nbytes);
	return crypto_ecb_crypt(desc, &walk, child,
				crypto_cipher_alg(child)->cia_decrypt);
}

static int crypto_gecb_encrypt(struct blkcipher_desc *desc,
			      struct scatterlist *dst, struct scatterlist *src,
			      unsigned int nbytes)
{    
    if (nbytes % PAGE_SIZE != 0)
    	return crypto_ecb_encrypt(desc, dst, src, nbytes);
    return crypto_gecb_crypt(desc, dst, src, nbytes, 1);
}

static int crypto_gecb_decrypt(struct blkcipher_desc *desc,
			      struct scatterlist *dst, struct scatterlist *src,
			      unsigned int nbytes)
{
    if (nbytes % PAGE_SIZE != 0)
    	return crypto_ecb_decrypt(desc, dst, src, nbytes);
    return crypto_gecb_crypt(desc, dst, src, nbytes, 0);
}

static int crypto_gecb_init_tfm(struct crypto_tfm *tfm)
{
    struct crypto_instance *inst = (void *)tfm->__crt_alg;
    struct crypto_spawn *spawn = crypto_instance_ctx(inst);
    struct crypto_gecb_ctx *ctx = crypto_tfm_ctx(tfm);
    struct crypto_cipher *cipher;

    cipher = crypto_spawn_cipher(spawn);
    if (IS_ERR(cipher))
	return PTR_ERR(cipher);

    ctx->child = cipher;
    return 0;
}

static void crypto_gecb_exit_tfm(struct crypto_tfm *tfm)
{
    struct crypto_gecb_ctx *ctx = crypto_tfm_ctx(tfm);
    crypto_free_cipher(ctx->child);
}

static struct crypto_instance *crypto_gecb_alloc(struct rtattr **tb)
{
    struct crypto_instance *inst;
    struct crypto_alg *alg;
    int err;

    err = crypto_check_attr_type(tb, CRYPTO_ALG_TYPE_BLKCIPHER);
    if (err)
	return ERR_PTR(err);

    alg = crypto_get_attr_alg(tb, CRYPTO_ALG_TYPE_CIPHER,
			      CRYPTO_ALG_TYPE_MASK);
    if (IS_ERR(alg))
	return ERR_CAST(alg);

    inst = crypto_alloc_instance("gaes_ecb", alg);
    if (IS_ERR(inst)) {
	printk("[gecb] Error: cannot alloc crypto instance\n");
	goto out_put_alg;
    }

    inst->alg.cra_flags = CRYPTO_ALG_TYPE_BLKCIPHER;
    inst->alg.cra_priority = alg->cra_priority;
    inst->alg.cra_blocksize = alg->cra_blocksize;
    inst->alg.cra_alignmask = alg->cra_alignmask;
    inst->alg.cra_type = &crypto_blkcipher_type;

    inst->alg.cra_blkcipher.min_keysize = alg->cra_cipher.cia_min_keysize;
    inst->alg.cra_blkcipher.max_keysize = alg->cra_cipher.cia_max_keysize;

    inst->alg.cra_ctxsize = sizeof(struct crypto_gecb_ctx);

    inst->alg.cra_init = crypto_gecb_init_tfm;
    inst->alg.cra_exit = crypto_gecb_exit_tfm;

    inst->alg.cra_blkcipher.setkey = crypto_gecb_setkey;
    inst->alg.cra_blkcipher.encrypt = crypto_gecb_encrypt;
    inst->alg.cra_blkcipher.decrypt = crypto_gecb_decrypt;

out_put_alg:
    crypto_mod_put(alg);
    return inst;
}

static void crypto_gecb_free(struct crypto_instance *inst)
{
    crypto_drop_spawn(crypto_instance_ctx(inst));
    kfree(inst);
}

static struct crypto_template crypto_gecb_tmpl = {
    .name = "gaes_ecb",
    .alloc = crypto_gecb_alloc,
    .free = crypto_gecb_free,
    .module = THIS_MODULE,
};

static int __init crypto_gecb_module_init(void)
{
    return crypto_register_template(&crypto_gecb_tmpl);
}

static void __exit crypto_gecb_module_exit(void)
{
    crypto_unregister_template(&crypto_gecb_tmpl);
}

module_init(crypto_gecb_module_init);
module_exit(crypto_gecb_module_exit);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("gECB block cipher algorithm");
