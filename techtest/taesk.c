#include <linux/module.h>
#include <linux/init.h>
#include <linux/types.h>
#include <linux/crypto.h>
//#include <crypto/aes.h>
#include <linux/scatterlist.h>
#include <linux/string.h>
#include <linux/gfp.h>
#include <linux/err.h>
#include <linux/jiffies.h>
#include <linux/timex.h>

#define AES_GENERIC "ecb(aes-generic)"
#define AES_ASM "ecb(aes-asm)"

#define MAX_BLK_SIZE (64*1024*1024)
#define MIN_BLK_SIZE (16)

#define TEST_TIMES 10

void test_aes(void)
{
	struct crypto_blkcipher *tfm;
	struct blkcipher_desc desc;
	u32 bs;
	int i,j;
	u32 npages;
	
	struct scatterlist *src;
	struct scatterlist *dst;
	char *buf;
	char **ins, **outs;
	
	unsigned int ret;
	
	u8 key[] = {0x00, 0x01, 0x02, 0x03, 0x05, 0x06, 0x07, 0x08, 0x0A, 0x0B, 0x0C, 0x0D, 0x0F, 0x10, 0x11, 0x12};
	
	npages = MAX_BLK_SIZE/PAGE_SIZE;
	
	src = kmalloc(npages*sizeof(struct scatterlist), __GFP_ZERO|GFP_KERNEL);
	if (!src) {
		printk("taes ERROR: failed to alloc src\n");		
		return;
	}
	dst = kmalloc(npages*sizeof(struct scatterlist), __GFP_ZERO|GFP_KERNEL);
	if (!dst) {
		printk("taes ERROR: failed to alloc dst\n");
		kfree(src);		
		return;
	}
	ins = kmalloc(npages*sizeof(char*), __GFP_ZERO|GFP_KERNEL);
	if (!ins) {
		printk("taes ERROR: failed to alloc ins\n");
		kfree(src);
		kfree(dst);		
		return;
	}
	outs = kmalloc(npages*sizeof(char*), __GFP_ZERO|GFP_KERNEL);
	if (!outs) {
		printk("taes ERROR: failed to alloc outs\n");
		kfree(src);
		kfree(dst);
		kfree(ins);		
		return;
	}
	
	tfm = crypto_alloc_blkcipher(AES_GENERIC, 0, 0);
	
	if (IS_ERR(tfm)) {
		printk("failed to load transform for %s: %ld\n", AES_GENERIC,
			PTR_ERR(tfm));
		goto out;
	}
	desc.tfm = tfm;
	desc.flags = 0;
	
	ret = crypto_blkcipher_setkey(tfm, key, sizeof(key));
	if (ret) {
		printk("setkey() failed flags=%x\n",
				crypto_blkcipher_get_flags(tfm));
	 	goto out;
	}
	
	sg_init_table(src, npages);
	for (i=0; i<npages; i++) {
		buf = (void *)__get_free_page(GFP_KERNEL);
		if (!buf) {
			printk("taes ERROR: alloc free page error\n");
			goto free_err_pages;
		}
		ins[i] = buf;
		sg_set_buf(src+i, buf, PAGE_SIZE);
		buf = (void *)__get_free_page(GFP_KERNEL);
		if (!buf) {
			printk("taes ERROR: alloc free page error\n");
			goto free_err_pages;
		}
		outs[i] = buf;
		sg_set_buf(dst+i, buf, PAGE_SIZE);
	}
	
	for (bs = MIN_BLK_SIZE; bs <= MAX_BLK_SIZE; bs*=2) {
		struct timeval t0, t1;
		long int enc, dec;

		do_gettimeofday(&t0);
		for (j=0; j<TEST_TIMES; j++) {
			for (i=0; i<npages && i*PAGE_SIZE < bs; i++) {
				memcpy((void*)(ins[i]), (void*)outs[i], PAGE_SIZE);
			}
			
			if (j%2==0)
				ret = crypto_blkcipher_encrypt(&desc, dst, src, bs);
			else
				ret = crypto_blkcipher_encrypt(&desc, src, dst, bs);
			if (ret) {
				printk("taes ERROR: enc error\n");
				goto free_err_pages;
			}
			for (i=0; i<npages && i*PAGE_SIZE < bs; i++) {
				memcpy((void*)(outs[i]), (void*)ins[i], PAGE_SIZE);
			}
		}
		do_gettimeofday(&t1);
		enc = 1000000*(t1.tv_sec-t0.tv_sec) + 
			((int)(t1.tv_usec) - (int)(t0.tv_usec));
			
		do_gettimeofday(&t0);
		for (j=0; j<TEST_TIMES; j++) {
			for (i=0; i<npages && i*PAGE_SIZE < bs; i++) {
				memcpy((void*)(ins[i]), (void*)outs[i], PAGE_SIZE);
			}
			
			if (j%2==0)
				ret = crypto_blkcipher_decrypt(&desc, src, dst, bs);
			else
				ret = crypto_blkcipher_decrypt(&desc, dst, src, bs);
			if (ret) {
				printk("taes ERROR: dec error\n");
				goto free_err_pages;
			}
			for (i=0; i<npages && i*PAGE_SIZE < bs; i++) {
				memcpy((void*)(outs[i]), (void*)ins[i], PAGE_SIZE);
			}
		}
		do_gettimeofday(&t1);
		dec = 1000000*(t1.tv_sec-t0.tv_sec) + 
			((int)(t1.tv_usec) - (int)(t0.tv_usec));
		
		printk("Size %u, enc %ld, dec %ld\n",
			bs, enc, dec);
	}
	
	
free_err_pages:
	for (i=0; i<npages && ins[i]; i++){		
		free_page((unsigned long)ins[i]);
	}
	for (i=0; i<npages && outs[i]; i++){
		free_page((unsigned long)outs[i]);
	}
out:
	kfree(src);
	kfree(dst);
	kfree(ins);
	kfree(outs);
	crypto_free_blkcipher(tfm);	
}

static int __init taes_init(void)
{
	printk("taes loaded\n");
	test_aes();
	return 0;
}

static void __exit taes_exit(void)
{
	printk("taes unloaded\n");
}

module_init(taes_init);
module_exit(taes_exit);

MODULE_DESCRIPTION("Test AES");
MODULE_LICENSE("GPL");

