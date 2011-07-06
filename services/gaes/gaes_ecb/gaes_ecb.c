/*
 * This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2010-2011 University of Utah and the Flux Group.
 * All rights reserved.
 * 
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
#include "../../../kgpu/kgpu.h"
#include "../gaesk.h"


struct crypto_gaes_ecb_ctx {
    struct crypto_cipher *child;
    struct crypto_aes_ctx aes_ctx;    
    u8 key[32];
};

static int
crypto_gaes_ecb_setkey(
    struct crypto_tfm *parent, const u8 *key,
    unsigned int keylen)
{
    struct crypto_gaes_ecb_ctx *ctx = crypto_tfm_ctx(parent);
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

static int
crypto_gaes_ecb_crypt(
    struct blkcipher_desc *desc,
    struct scatterlist *dst, struct scatterlist *src,
    unsigned int sz,
    int enc)
{
    int err=0;
    unsigned int rsz = roundup(sz, PAGE_SIZE);
    unsigned int nbytes;
    u8* gpos;
    unsigned long cpdbytes=0;  
    
    struct kgpu_req *req;
    struct kgpu_resp *resp;
    struct kgpu_buffer *buf;

    struct crypto_blkcipher *tfm    = desc->tfm;
    struct crypto_gaes_ecb_ctx *ctx = crypto_blkcipher_ctx(tfm);
    struct blkcipher_walk walk;


    blkcipher_walk_init(&walk, dst, src, sz);
    
    buf = alloc_gpu_buffer(rsz+2*PAGE_SIZE);
    if (!buf) {
	printk("[gaes_ecb] Error: GPU buffer is null.\n");
	return -EFAULT;
    }

    req  = alloc_kgpu_request();
    resp = alloc_kgpu_response();
    if (!req || !resp) {
	return -EFAULT;
    }

    err = blkcipher_walk_virt(desc, &walk);

    while ((nbytes = walk.nbytes)) {
	u8 *wsrc             = walk.src.virt.addr;
	unsigned long offset = cpdbytes&(PAGE_SIZE-1);
	unsigned long idx    = cpdbytes>>PAGE_SHIFT;
	
	if (nbytes > PAGE_SIZE) {
	    return -EFAULT;
	}

	gpos = (u8*)(buf->pas[idx])+offset;
	while (nbytes > PAGE_SIZE-offset) { /* 'if' should be fine */
	    unsigned long realsz = PAGE_SIZE-offset;
	    memcpy(__va(gpos), wsrc, realsz);
	    cpdbytes += realsz;
	    nbytes   -= realsz;
	    idx       = cpdbytes>>PAGE_SHIFT;
	    offset    = cpdbytes&(PAGE_SIZE-1);
	    wsrc     += realsz;
	    gpos      = (u8*)(buf->pas[idx])+offset;
	}
	memcpy(__va(gpos), wsrc, nbytes);
	cpdbytes += nbytes;

	err = blkcipher_walk_done(desc, &walk, 0);
    }

    gpos = (u8*)(buf->pas[rsz>>PAGE_SHIFT])+(rsz&(PAGE_SIZE-1));
    memcpy(__va(gpos), &(ctx->aes_ctx), sizeof(struct crypto_aes_ctx));   

    strcpy(req->kureq.sname, enc?"gaes_ecb-enc":"gaes_ecb-dec");
    req->kureq.input    = buf->va;
    req->kureq.output   = buf->va;
    req->kureq.insize   = rsz+PAGE_SIZE;
    req->kureq.outsize  = rsz;
    req->kureq.data     = (u8*)(buf->va) + rsz;
    req->kureq.datasize = sizeof(struct crypto_aes_ctx);

    if (call_gpu_sync(req, resp)) {
	err = -EFAULT;
	printk("[gaes_ecb] Error: callgpu error\n");
    } else {
	cpdbytes = 0;
	blkcipher_walk_init(&walk, dst, src, sz);
	err = blkcipher_walk_virt(desc, &walk);
 	
	while ((nbytes = walk.nbytes)) {
	    u8 *wdst             = walk.dst.virt.addr;
	    unsigned long offset = cpdbytes&(PAGE_SIZE-1);
	    unsigned long idx    = cpdbytes>>PAGE_SHIFT;
	
	    if (nbytes > PAGE_SIZE) {
		return -EFAULT;
	    }

	    gpos = (u8*)(buf->pas[idx])+offset;
	    while (nbytes > PAGE_SIZE-offset) { /* 'if' should be fine */
		unsigned long realsz = PAGE_SIZE-offset;
		memcpy(wdst, __va(gpos), realsz);
		cpdbytes += realsz;
		nbytes   -= realsz;
		idx       = cpdbytes>>PAGE_SHIFT;
		offset    = cpdbytes&(PAGE_SIZE-1);
		wdst     += realsz;
		gpos      = (u8*)(buf->pas[idx])+offset;
	    }
	    memcpy(wdst, __va(gpos), nbytes);
	    cpdbytes += nbytes;
	
	    err = blkcipher_walk_done(desc, &walk, 0);
	}
    }
    
    free_kgpu_request(req);
    free_kgpu_response(resp);
    free_gpu_buffer(buf);

    return err;
}

static int
crypto_ecb_crypt(
    struct blkcipher_desc *desc,
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

static int
crypto_ecb_encrypt(
    struct blkcipher_desc *desc,
    struct scatterlist *dst, struct scatterlist *src,
    unsigned int nbytes)
{
    struct blkcipher_walk walk;
    struct crypto_blkcipher *tfm = desc->tfm;
    struct crypto_gaes_ecb_ctx *ctx = crypto_blkcipher_ctx(tfm);
    struct crypto_cipher *child = ctx->child;

    blkcipher_walk_init(&walk, dst, src, nbytes);
    return crypto_ecb_crypt(desc, &walk, child,
			    crypto_cipher_alg(child)->cia_encrypt);
}

static int
crypto_ecb_decrypt(
    struct blkcipher_desc *desc,
    struct scatterlist *dst, struct scatterlist *src,
    unsigned int nbytes)
{
    struct blkcipher_walk walk;
    struct crypto_blkcipher *tfm = desc->tfm;
    struct crypto_gaes_ecb_ctx *ctx = crypto_blkcipher_ctx(tfm);
    struct crypto_cipher *child = ctx->child;

    blkcipher_walk_init(&walk, dst, src, nbytes);
    return crypto_ecb_crypt(desc, &walk, child,
			    crypto_cipher_alg(child)->cia_decrypt);
}

static int
crypto_gaes_ecb_encrypt(
    struct blkcipher_desc *desc,
    struct scatterlist *dst, struct scatterlist *src,
    unsigned int nbytes)
{    
    if (/*nbytes%PAGE_SIZE != 0 ||*/ nbytes <= GAES_ECB_SIZE_THRESHOLD)
    	return crypto_ecb_encrypt(desc, dst, src, nbytes);
    return crypto_gaes_ecb_crypt(desc, dst, src, nbytes, 1);
}

static int
crypto_gaes_ecb_decrypt(
    struct blkcipher_desc *desc,
    struct scatterlist *dst, struct scatterlist *src,
    unsigned int nbytes)
{
    if (/*nbytes%PAGE_SIZE != 0 ||*/ nbytes <= GAES_ECB_SIZE_THRESHOLD)
    	return crypto_ecb_decrypt(desc, dst, src, nbytes);
    return crypto_gaes_ecb_crypt(desc, dst, src, nbytes, 0);
}

static int crypto_gaes_ecb_init_tfm(struct crypto_tfm *tfm)
{
    struct crypto_instance *inst = (void *)tfm->__crt_alg;
    struct crypto_spawn *spawn = crypto_instance_ctx(inst);
    struct crypto_gaes_ecb_ctx *ctx = crypto_tfm_ctx(tfm);
    struct crypto_cipher *cipher;

    cipher = crypto_spawn_cipher(spawn);
    if (IS_ERR(cipher))
	return PTR_ERR(cipher);

    ctx->child = cipher;
    return 0;
}

static void crypto_gaes_ecb_exit_tfm(struct crypto_tfm *tfm)
{
    struct crypto_gaes_ecb_ctx *ctx = crypto_tfm_ctx(tfm);
    crypto_free_cipher(ctx->child);
}

static struct crypto_instance *crypto_gaes_ecb_alloc(struct rtattr **tb)
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
	printk("[gaes_ecb] Error: cannot alloc crypto instance\n");
	goto out_put_alg;
    }

    inst->alg.cra_flags = CRYPTO_ALG_TYPE_BLKCIPHER;
    inst->alg.cra_priority = alg->cra_priority;
    inst->alg.cra_blocksize = alg->cra_blocksize;
    inst->alg.cra_alignmask = alg->cra_alignmask;
    inst->alg.cra_type = &crypto_blkcipher_type;

    inst->alg.cra_blkcipher.min_keysize = alg->cra_cipher.cia_min_keysize;
    inst->alg.cra_blkcipher.max_keysize = alg->cra_cipher.cia_max_keysize;

    inst->alg.cra_ctxsize = sizeof(struct crypto_gaes_ecb_ctx);

    inst->alg.cra_init = crypto_gaes_ecb_init_tfm;
    inst->alg.cra_exit = crypto_gaes_ecb_exit_tfm;

    inst->alg.cra_blkcipher.setkey = crypto_gaes_ecb_setkey;
    inst->alg.cra_blkcipher.encrypt = crypto_gaes_ecb_encrypt;
    inst->alg.cra_blkcipher.decrypt = crypto_gaes_ecb_decrypt;

out_put_alg:
    crypto_mod_put(alg);
    return inst;
}

static void crypto_gaes_ecb_free(struct crypto_instance *inst)
{
    crypto_drop_spawn(crypto_instance_ctx(inst));
    kfree(inst);
}

static struct crypto_template crypto_gaes_ecb_tmpl = {
    .name = "gaes_ecb",
    .alloc = crypto_gaes_ecb_alloc,
    .free = crypto_gaes_ecb_free,
    .module = THIS_MODULE,
};

static int __init crypto_gaes_ecb_module_init(void)
{
    return crypto_register_template(&crypto_gaes_ecb_tmpl);
}

static void __exit crypto_gaes_ecb_module_exit(void)
{
    crypto_unregister_template(&crypto_gaes_ecb_tmpl);
}

module_init(crypto_gaes_ecb_module_init);
module_exit(crypto_gaes_ecb_module_exit);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("gaes_ecb block cipher algorithm");
