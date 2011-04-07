/*
 * KGPU GAES header
 */

#ifndef __GAESK_H__
#define __GAESK_H__

#define GAES_ECB_SIZE_THRESHOLD (2*PAGE_SIZE)

static void cvt_endian_u32(u32* buf, int n)
{
  u8* b = (u8*)buf;
  int nb = n*4;
  
  u8 t;
  int i;
  
  for (i=0; i<nb; i+=4, b+=4) {
    t = b[0];
    b[0] = b[3];
    b[3] = t;
    
    t = b[1];
    b[1] = b[2];
    b[2] = t;
  }
}


static void dump_page_content(u8 *p)
{
    int r,c;
    printk("dump page content:\n");
    for (r=0; r<16; r++) {
	for (c=0; c<32; c++)
	    printk("%02x ", p[r*32+c]);
	printk("\n");
    }
}

#endif
