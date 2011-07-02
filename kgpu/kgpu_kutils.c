/*
 * This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2010-2011 University of Utah and the Flux Group.
 * All rights reserved.
 *
 * KGPU kernel module utilities
 *
 */
#include "kkgpu.h"
#include <linux/kernel.h>
#include <linux/sched.h>
#include <linux/mm.h>
#include <linux/mm_types.h>
#include <linux/uaccess.h>
#include <linux/slab.h>
#include <asm/current.h>

static int bad_address(void *p)
{
    unsigned long dummy;
    return probe_kernel_address((unsigned long*)p, dummy);
}

/*
 * map any virtual address of the current process to its
 * physical one.
 */
unsigned long kgpu_virt2phy(unsigned long vaddr)
{
    pgd_t *pgd = pgd_offset(current->mm, vaddr);
    pud_t *pud;
    pmd_t *pmd;
    pte_t *pte;

    /* to lock the page */
    struct page *pg;
    unsigned long paddr = 0;

    if (bad_address(pgd)) {
	kgpu_log(KGPU_LOG_ALERT, "bad address of pgd %p\n", pgd);
	goto bad;
    }
    if (pgd_none(*pgd) || pgd_bad(*pgd)) {
	kgpu_log(KGPU_LOG_ALERT, "pgd not present %lu\n", pgd_val(*pgd));
	goto out;
    }

    pud = pud_offset(pgd, vaddr);
    if (bad_address(pud)) {
	kgpu_log(KGPU_LOG_ALERT, "bad address of pud %p\n", pud);
	goto bad;
    }
    if (pud_none(*pud) || pud_bad(*pud)) {
	kgpu_log(KGPU_LOG_ALERT, "pud not present %lu\n", pud_val(*pud));
	goto out;
    }

    pmd = pmd_offset(pud, vaddr);
    if (bad_address(pmd)) {
	kgpu_log(KGPU_LOG_ALERT, "bad address of pmd %p\n", pmd);
	goto bad;
    }
    if (pmd_none(*pmd) || pmd_bad(*pmd)) {
	kgpu_log(KGPU_LOG_ALERT, "pmd not present %lu\n", pmd_val(*pmd));
	goto out;
    }

    pte = pte_offset_map/*kernel*/(pmd, vaddr);
    if (bad_address(pte)) {
	kgpu_log(KGPU_LOG_ALERT, "bad address of pte %p\n", pte);
	goto bad;
    }    
    if (!pte_present(*pte)) {
	kgpu_log(KGPU_LOG_ALERT, "pte not present %lu\n", pte_val(*pte));
	goto out;
    }

    pg = pte_page(*pte);
    paddr = (pte_val(*pte) & PHYSICAL_PAGE_MASK) | (vaddr&(PAGE_SIZE-1));

out:
    return paddr;
bad:
    kgpu_log(KGPU_LOG_ALERT, "Bad address\n");
    return 0;
}


int
kgpu_check_phy_consecutive(unsigned long vaddr, size_t sz, size_t framesz)
{
    unsigned long paddr, lpa;
    size_t offset = 0;

    if (framesz == PAGE_SIZE)
	return 1;

    do {
	paddr = kgpu_virt2phy(vaddr+offset);
	lpa = kgpu_virt2phy(vaddr+offset-PAGE_SIZE+framesz);
	if (!lpa || !paddr) {
	    kgpu_log(KGPU_LOG_ERROR, "PA for 0x%lx or 0x%lx not found\n",
		   (vaddr+offset), (vaddr+offset-PAGE_SIZE+framesz));
	    return 0;
	}
	
	if (lpa != paddr+framesz-PAGE_SIZE) {
	    kgpu_log(KGPU_LOG_ERROR, "VA from 0x%lx to 0x%lx not consecutive\n",
		   (vaddr+offset), (vaddr+offset-PAGE_SIZE+framesz));
	    return 0;
	}
	
	offset += framesz;
    } while (offset < sz);

    return 1;    
}

void
kgpu_dump_pages(unsigned long vaddr, unsigned long sz)
{
    void* page;
    unsigned long offset=0;
    sz -= PAGE_SIZE;

    do {
	page = virt_to_page(vaddr+offset);
	dbg("%s %s %p @ page %p (%p)\n",
	    (virt_addr_valid(vaddr+offset)?"valid va":"invalid va"),
	    (virt_addr_valid(page)?"valid page":"invalid page"),
	    (void*)(vaddr+offset), page, (void*)__pa(page));
	offset += PAGE_SIZE;
    } while (offset < sz-1);
}


void
kgpu_test_memory_pages(unsigned long vaddr, unsigned long sz)
{
    int npages = sz/PAGE_SIZE;

    struct page **pages = kmalloc(npages*sizeof(struct page*), GFP_KERNEL);

    int rt;
    struct vm_area_struct *vma;

    down_read(&current->mm->mmap_sem);
    rt = get_user_pages(current, current->mm, vaddr, npages,
			    0, 0, pages, NULL);
    up_read(&current->mm->mmap_sem);

    vma = find_vma(current->mm, vaddr);
    if (!vma) {
	dbg("no VMA for %p\n", (void*)vaddr);
    } else {
	dbg("VMA(0x%lx ~ 0x%lx) flags for %p is 0x%lx\n",
	    vma->vm_start, vma->vm_end,
	    (void*)vaddr, vma->vm_flags);
    }
    
    if (rt<=0) {
	dbg("no page pinned %d\n", rt);
    } else {
	dbg("get pages\n");
	for (npages=0; npages<rt; npages++) {
	    put_page(pages[npages]);
	}
    }

    kfree(pages);    
}
