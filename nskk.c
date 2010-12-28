/*
 * NSK kernel module
 */

/*
 * all headers are included from a 'build' symbol link to the kernel's
 * build directory so that we can link with any kernel versions.
 */
#include "build/include/linux/module.h"
#include "build/include/linux/kernel.h"
#include "build/include/linux/init.h"

#include "build/include/linux/kthread.h"
#include "build/include/linux/netfilter.h"
#include "build/include/linux/ip.h"
#include "build/include/linux/netfilter_ipv4.h"
#include "build/include/linux/netdevice.h"

#include "build/include/linux/proc_fs.h"
#include "build/include/linux/uaccess.h"
#include "build/arch/x86/include/asm/page.h"
#include "build/include/linux/mm.h"
#include "build/include/linux/mm_types.h"
#include "build/include/linux/types.h"
#include "build/include/linux/string.h"
#include "build/arch/x86/include/asm/system.h"
#include "build/arch/x86/include/asm/pgtable.h"

#include "nskku.h"

static int bad_address(void *p)
{
    unsigned long dummy;
    return probe_kernel_address((unsigned long*)p, dummy);
}

/*
 * map any virtual address of the current process to its
 * physical one.
 */
static unsigned long any_v2p(unsigned long vaddr)
{
    pgd_t *pgd = pgd_offset(current->mm, vaddr);
    pud_t *pud;
    pmd_t *pmd;
    pte_t *pte;

    /* to lock the page */
    struct page *pg;
    unsigned long paddr;

    if (bad_address(pgd)) {
	printk(KERN_ALERT "[nskk] Alert: bad address of pgd %p\n", pgd);
	goto bad;
    }
    if (!pgd_present(*pgd)) {
	printk(KERN_ALERT "[nskk] Alert: pgd not present %lu\n", *pgd);
	goto out;
    }

    pud = pud_offset(pgd, vaddr);
    if (bad_address(pud)) {
	printk(KERN_ALERT "[nskk] Alert: bad address of pud %p\n", pud);
	goto bad;
    }
    if (!pud_present(*pud) || pud_large(*pud)) {
	printk(KERN_ALERT "[nskk] Alert: pud not present %lu\n", *pud);
	goto out;
    }

    pmd = pmd_offset(pud, vaddr);
    if (bad_address(pmd)) {
	printk(KERN_ALERT "[nskk] Alert: bad address of pmd %p\n", pmd);
	goto bad;
    }
    if (!pmd_present(*pmd) || pmd_large(*pmd)) {
	printk(KERN_ALERT "[nskk] Alert: pmd not present %lu\n", *md);
	goto out;
    }

    pte = pte_offset_kernel(pmd, vaddr);
    if (bad_address(pte)) {
	printk(KERN_ALERT "[nskk] Alert: bad address of pte %p\n", pte);
	goto bad;
    }
    if (!pte_present(*pte)) {
	printk(KERN_ALERT "[nskk] Alert: pte not present %lu\n", *pte);
	goto out;
    }

    pg = pte_page(*pte);
    paddr = (pte_val(*pte) & PHYSICAL_PAGE_MASK) | (vaddr&(PAGE_SIZE-1));

out:
    return paddr;
bad:
    printk(KERN_ALERT "[nskk] Alert: Bad address\n");
    return 0;
}

static struct proc_dir_entry *reqfile, *respfile;

static int reqf_read(char *buf, char **bufloc, off_t offset, int buflen,
		     int *eof, void *data)
{
    
}

static int reqf_write(struct file *file, const char *buf, unsigned long count,
		      void *data)
{
}

static int respf_read(char *buf, char **bufloc, off_t offset, int buflen,
		      int *eof, void *data)
{
    
}

static int respf_write(struct file *file, const char *buf, unsigned long count,
		       void *data)
{
}

static void register_kucommfs(void)
{
    reqfile = create_proc_entry(NSK_PROCFS_REQ_FILE, 0777, NULL);
    respfile = create_proc_entry(NSK_PROCFS_RESP_FILE, 0777, NULL);

    if (reqfile == NULL || respfile == NULL) {
	remove_proc_entry(NSK_PROCFS_REQ_FILE, NULL);
	remove_proc_entry(NSK_PROCFS_RESP_FILE, NULL);

	printk(KERN_ALERT "[nskk] Error: Could not initialize proc fs.\n");
	return;
    }

    reqfile->read_proc = reqf_read;
    reqfile->write_proc = reqf_write;
    reqfile->mode = S_IFREG|S_IRUGO;
    reqfile->uid = 0;
    reqfile->gid = 0;
    reqfile->size = 0;

    respfile->read_proc = respf_read;
    respfile->write_proc = respf_write;
    respfile->mode = S_IFREG|S_IRUGO;
    respfile->uid = 0;
    respfile->gid = 0;
    respfile->size = 0;
}

static int __init nskk_init(void)
{
    register_kucommfs();
    return 0;
}

static void _exit nskk_exit(void)
{
    remove_proc_entry(NSK_PROCFS_REQ_FILE, NULL);
    remove_proc_entry(NSK_PROCFS_RESP_FILE, NULL);
}

module_init(nskk_init);
module_exit(nskk_exit);
