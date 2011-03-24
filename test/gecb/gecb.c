
#include <crypto/algapi.h>
#include <linux/err.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/scatterlist.h>
#include <linux/slab.h>
#include <crypto/aes.h>
#include <linux/string.h>
#include "../../kgpu/_kgpu.h"

#define aes_enc_key aes_ctx.key_enc
#define aes_dec_key aes_ctx.key_dec
#define aes_key_len aes_ctx.key_length

struct crypto_gecb_ctx {
    struct crypto_cipher *child;
    struct crypto_aes_ctx aes_ctx;    
};

static int crypto_ecb_setkey(struct crypto_tfm *parent, const u8 *key,
			     unsigned int keylen)
{
    struct crypto_gecb_ctx *ctx = crypto_tfm_ctx(parent);
    struct crypto_cipher *child = ctx->child;
    int err;

    crypto_cipher_clear_flags(child, CRYPTO_TFM_REQ_MASK);
    crypto_cipher_set_flags(child, crypto_tfm_get_flags(parent) &
			    CRYPTO_TFM_REQ_MASK);

    err = crypto_aes_expand_key(&ctx.aes_ctx,
				key, keylen);
    
    crypto_tfm_set_flags(parent, crypto_cipher_get_flags(child) &
			 CRYPTO_TFM_RES_MASK);
    return err;
}

static int crypto_ecb_crypt(struct blkcipher_desc *desc,
			    struct blkcipher_walk *walk,
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
    
    buf = alloc_kgpu_buffer();
    if (!buf) {
	printk("[gecb] Error: GPU buffer is null.\n");
	return -EFAULT;
    }

    req = alloc_kgpu_request();
    resp = alloc_kgpu_response();
    if (!req || !resp) {
	return -EFAULT;
    }

    err = blkcipher_walk_virt(desc, walk);

    while ((nbytes = walk->nbytes)) {
	u8 *wsrc = walk->src.virt.addr;
	if (nbytes > KGPU_BUF_FRAME_SIZE) {
	    return -EFAULT;
	}

#ifndef _NDEBUG
	if (nbytes != PAGE_SIZE)
	    printk("[gecb] WARNING: %u is not PAGE_SIZE\n", nbytes);
#endif

	gpos = buf->paddrs[i++];
	memcpy(__va(gpos), wsrc, nbytes);        

	err = blkcipher_walk_done(desc, walk, nbytes);
    }

    gpos = buf->paddrs[i];
    memcpy(__va(gpos), &(ctx->aes_ctx), sizeof(struct crypto_aes_ctx));    

    strcpy(req->kureq.sname, enc?"aes-enc":"aes-dec");
    req->kureq.input = buf->gb.addr;
    req->kureq.output = buf->gb.addr;
    req->kureq.insize = PAGE_SIZE*((sz-1)/PAGE_SIZE+1)
	+ sizeof(struct crypto_aes_ctx);
    req->kureq.outsize = sz;

    if (call_gpu_sync(req, resp)) {
	err = -EFAULT;
    } else {
	i=0;
	blkcipher_walk_init(&walk, dst, src, nbytes);
	err = blkcipher_walk_virt(desc, walk);
	
	while ((nbytes = walk->nbytes)) {
	    u8 *wdst = walk->dst.virt.addr;
	    if (nbytes > KGPU_BUF_FRAME_SIZE) {
		return -EFAULT;
	    }

#ifndef _NDEBUG
	    if (nbytes != PAGE_SIZE)
		printk("[gecb] WARNING: %u is not PAGE_SIZE\n", nbytes);
#endif

	    gpos = buf->paddrs[i++];
	    memcpy(wdst, __va(gpos), nbytes);        

	    err = blkcipher_walk_done(desc, walk, nbytes);
	}
    }
    
    free_kgpu_request(req);
    free_kgpu_response(resp);
    free_kgpu_buffer(buf);

    return err;
}

static int crypto_ecb_encrypt(struct blkcipher_desc *desc,
			      struct scatterlist *dst, struct scatterlist *src,
			      unsigned int nbytes)
{
    struct blkcipher_walk walk;

    blkcipher_walk_init(&walk, dst, src, nbytes);
    return crypto_ecb_crypt(desc, &walk, nbytes, 1);
}

static int crypto_ecb_decrypt(struct blkcipher_desc *desc,
			      struct scatterlist *dst, struct scatterlist *src,
			      unsigned int nbytes)
{
    struct blkcipher_walk walk;

    blkcipher_walk_init(&walk, dst, src, nbytes);
    return crypto_ecb_crypt(desc, &walk, nbytes, 0);
}

static int crypto_ecb_init_tfm(struct crypto_tfm *tfm)
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

static void crypto_ecb_exit_tfm(struct crypto_tfm *tfm)
{
    struct crypto_ecb_ctx *ctx = crypto_tfm_ctx(tfm);
    crypto_free_cipher(ctx->child);
}

static struct crypto_instance *crypto_ecb_alloc(struct rtattr **tb)
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

    inst = crypto_alloc_instance("gecb", alg);
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

    inst->alg.cra_init = crypto_ecb_init_tfm;
    inst->alg.cra_exit = crypto_ecb_exit_tfm;

    inst->alg.cra_blkcipher.setkey = crypto_ecb_setkey;
    inst->alg.cra_blkcipher.encrypt = crypto_ecb_encrypt;
    inst->alg.cra_blkcipher.decrypt = crypto_ecb_decrypt;

out_put_alg:
    crypto_mod_put(alg);
    return inst;
}

static void crypto_ecb_free(struct crypto_instance *inst)
{
    crypto_drop_spawn(crypto_instance_ctx(inst));
    kfree(inst);
}

static struct crypto_template crypto_gecb_tmpl = {
    .name = "gecb",
    .alloc = crypto_ecb_alloc,
    .free = crypto_ecb_free,
    .module = THIS_MODULE,
};

static int __init crypto_ecb_module_init(void)
{
    return crypto_register_template(&crypto_gecb_tmpl);
}

static void __exit crypto_ecb_module_exit(void)
{
    crypto_unregister_template(&crypto_gecb_tmpl);
}

module_init(crypto_ecb_module_init);
module_exit(crypto_ecb_module_exit);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("gECB block cipher algorithm");
