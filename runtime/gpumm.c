/*
 * This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2012 University of Utah and the Flux Group.
 * All rights reserved.
 */

/* Function prototypes: */
static void mm_clear_host_alloc_buf(void);
static void mm_clear_host_map_buf(void);
static void mm_clear_dev_buf(void);

void gpumm_init(void);
void gpumm_finit(void);

static struct page* uv2page(unsigned long v);

int mm_set_host_alloc_buf(ku_meminfo_t* info);
int mm_set_host_map_buf(ku_meminfo_t* info);
int mm_set_host_map_type(int maptype);
int mm_set_dev_buf(ku_devmeminfo_t* info);

void* kg_vmalloc(u64 nbytes);
void  kg_vfree(void* p);

u64 kg_alloc_mmap_area(u64 size);
void kg_free_mmap_area(u64 start);

void kg_unmap_area(u64 start);
int kg_map_page(struct page *p, u64 addr);

void* kg_map_pages(struct page **ps, int nr); 


/* Globals: */
int kgpu_mem_map_type = KGPU_MMAP_MAP_SELF_DEVICE_VMA;

static k_hostmem_pool_t hmpool;
static k_hostmem_vma_t hmvma;

static void mm_clear_host_alloc_buf(void)
{
	if (hmpool.pages)
		vfree(hmpool.pages);
	if (hmpool.bitmap)
		vfree(hmpool.bitmap);
	if (hmpool.alloc_sz)
		vfree(hmpool.alloc_sz);
	vunmap((void*)hmpool.kva);
	return 0;
}

static void mm_clear_host_map_buf(void)
{
	// TODO write this
}

static void mm_clear_dev_buf(void)
{
	// TODO write this
}

void gpumm_init(void)
{
	memset(&hmpool, 0, sizeof(k_hostmem_pool_t));
	memset(&hmvma, 0, sizeof(k_hostmem_vma_t));2

	krt_log(KGPU_LOG_INFO,
		"KRT memory manager initialized.\n");
}

void gpumm_finit(void)
{
	mm_clear_host_alloc_buf();
	mm_clear_host_map_buf();
	mm_clear_dev_buf();
	krt_log(KGPU_LOG_INFO,
		"KRT memory manager finished.\n");
}

static struct page* uv2page(unsigned long v)
{
	struct page *p = NULL;
	pgd_t *pgd = pgd_offset(current->mm, v);
	
	if (!pgd_none(*pgd)) {
		pud_t *pud = pud_offset(pgd, v);
		if (!pud_none(*pud)) {
			pmd_t *pmd = pmd_offset(pud, v);
			if (!pmd_none(*pmd)) {
				pte_t *pte;
				
				pte = pte_offset_map(pmd, v);
				if (pte_present(*pte))
					p = pte_page(*pte);
				
				pte_unmap(pte);
			}
		}
	}
	if (!p)
		krt_log(KGPU_LOG_ALERT, "bad address 0x%lX\n", v);
	return p;
}

/*
 * Initialize host memory buffer that is for normal allocation,
 * which maps CUDA pages into kernel address space.
 */
int mm_set_host_alloc_buf(ku_meminfo_t* info)
{
	int i;
	u64 uva;
	int ret = 0;
	
	if (hmpool.uva) {
		krt_log(KGPU_LOG_ERROR,
			"Re-set host mem buffer is not allowed!\n");
		return -EINVAL;
	}
	spin_lock_init(&hmpool.lock);
	hmpool.uva = info->uva;
	hmpool.npages = info->size>>PAGE_SHIFT;
	hmpool.nunits = info->size>>KGPU_BUF_UNIT_SHIFT;
	
	hmpool.pages = vmalloc(sizeof(struct page*) * hmpool.npages);
	if (!hmpool.pages) {
		krt_log(KGPU_LOG_ERROR,
			"Out of memory for hmpool pages.\n");
		ret = -ENOMEM;
		goto err_get_out;
	}	
	uva = info->uva;
	for (i=0; i<hmpool->npages; i++) {
		hmpool->pages[i] = uv2page(uva);
		uva += PAGE_SIZE;
	}
	
	hmpool.bitmap = vmalloc(sizeof(u64) * BITS_TO_LONGS(hmpool.nunits));
	if (!hmpool.bitmap) {
		krt_log(KGPU_LOG_ERROR,
			"Out of memory for hmpool bitmap.\n");
		ret = -ENOMEM;
		goto err_get_out;
	}
	bitmap_zero(hmpool.bitmap, hmpool.nunits);
	
	hmpool.alloc_sz = vmalloc(sizeof(u32)*hmpool.nunits);
	if (!hmpool.alloc_sz) {
		krt_log(KGPU_LOG_ERROR,
			"Out of memory for hmpool allocation sizes.\n");
		ret = -ENOMEM;
		goto err_get_out;
	}
	memset(hmpool.alloc_sz, 0, hmpool.nunits*sizeof(u32));
	
	hmpool.kva = (u64)vmap(hmpool.pages, 
			       hmpool.npages, 
			       GFP_KERNEL, 
			       PAGE_KERNEL);
	if (!hmpool.kva) {
		krt_log(KGPU_LOG_ERROR, "Failed to map pages into kernel.\n");
		ret = -EFAULT;
		goto err_get_out;
	}
	
	goto get_out;
	
err_get_out:
	if (hmpool.pages)
		vfree(hmpool.pages);
	if (hmpool.bitmap)
		vfree(hmpool.bitmap);
	if (hmpool.alloc_sz)
		vfree(hmpool.alloc_sz);
get_out:
	return ret;	
}

int mm_set_host_map_type(int maptype)
{
	kgpu_mem_map_type = maptype;
}

int mm_set_host_map_buf(ku_meminfo_t* info)
{
	int ret = 0;
	int i;
	
	/* This may fail, a proper way is to get VMA according to map type. */
	hmvma.vma = find_vma(current->mm, info->uva);
	if (!hmvma.vma) {
		krt_log(KGPU_LOG_ERROR,
			"Can't find VMA for 0x%lX.\n",
			info->uva);
		ret = -EINVAL;
		goto err_get_out;
	}
	
	hmvma.start = info->uva;
	hmvma.end = info->uva + info->size;
	hmvma.npages = info->size >> PAGE_SHIFT;
	
	spin_lock_init(&hmvma.lock);
	
	hmvma.alloc_sz = vmalloc(sizeof(u32)*hmvma.npages);
	if (!hmvma.alloc_sz) {
		krt_log(KGPU_LOG_ERROR,
			"Out of memory for hmvma allocation sizes.\n");
		ret = -ENOMEM;
		goto err_get_out;
	}
	memset(hmvma.alloc_sz, 0, sizeof(u32)*hmvma.npages);
	
	hmvma.bitmap = vmalloc(sizeof(u64)*BITS_TO_LONGS(hmvma.npages));
	if (!hmvma.bitmap) {
		krt_log(KGPU_LOG_ERROR,
			"Out of memory for hmvma bitmap.\n");
		ret = -ENOMEM;
		goto err_get_out;
	}
	bitmap_zero(hmvma.bitmap, hmvma.npages);
	
	goto get_out;
	
err_get_out:
	if (hmvma.alloc_sz)
		vfree(hmvma.alloc_sz);
	if (hmvma.bitmap)
		vfree(hmvma.bitmap);
get_out:
	return ret;
}

/*
 * Pre-Condition: GPU device array must be initialized.
 */
int mm_set_dev_buf(ku_devmeminfo_t *info)
{
	k_gpu_t* gpu;
	k_devmem_pool_t* dmp;
	int ret;
	
	gpu = get_k_gpu(info->device);
	if (!gpu) {
		krt_log(KGPU_LOG_ERROR,
			"Can't find GPU device %d.\n",
			info->gpu);
		ret = -EINVAL;
		goto err_get_out;
	}
	
	dmp = &gpu->devmem;
	dmp->start = info->uva;
	dmp->end = info->uva + info->size;
	dmp->size = info->size;
	dmp->nunits = info->size >> KGPU_BUF_UNIT_SHIFT;
	
	spin_lock_init(&hmvma.lock);
	
	dmp->alloc_sz = vmalloc(sizeof(u32)*dmp->nunits);
	if (!dmp->alloc_sz) {
		krt_log(KGPU_LOG_ERROR,
			"Out of memory for GPU %d allocation sizes.\n",
			info->device);
		ret = -ENOMEM;
		goto err_get_out;
	}
	memset(dmp->alloc_sz, 0, sizeof(u32)*dmp->nunits);
	
	dmp->bitmap = vmalloc(sizeof(u64)*BITS_TO_LONGS(dmp->nunits));
	if (!dmp->bitmap) {
		krt_log(KGPU_LOG_ERROR,
			"Out of memory for GPU %d bitmap.\n",
			info->device);
		ret = -ENOMEM;
		goto err_get_out;
	}
	bitmap_zero(dmp->bitmap, dmp->nunits);
	
	goto get_out;
	
err_get_out:
	if (dmp->alloc_sz)
		vfree(dmp->alloc_sz);
	if (dmp->bitmap)
		vfree(dmp->bitmap);
get_out:
	return ret;		
}

void* kg_vmalloc(u64 nbytes)
{
	u32 nunits = DIV_ROUND_UP(nbytes, KGPU_BUF_UNIT_SIZE);
	void *p = NULL;
	u64 idx;
	
	spin_lock(&hmpool.lock);
	
	idx = bitmap_find_next_zero_area(hmpool.bitmap, 
					 hmpool.nunits, 
					 0, 
					 nunits, 
					 0);
	
	if (idx < hmpool.nunits) {
		bitmap_set(hmpool.bitmap, idx, nunits);
		p = (void*)(hmpool.kva + (idx<<KGPU_BUF_UNIT_SHIFT));
		hmpool.alloc_sz[idx] = nunits;
	} else {
		krt_log(KGPU_LOG_ERROR,
			"Out of host memory for kg_vmalloc %lu.\n",
			nbytes);
	}
	spin_unlock(&hmpool.lock);
	
	return p;
}
EXPORT_SYMBOL_GPL(kg_vmalloc);

void kg_vfree(void* p)
{
	u64 idx = (TO_UL(p) - hmpool.kva) >> KGPU_BUF_UNIT_SHIFT;
	u32 nunits;
	
	if (!p) return;
	
	if (idx < 0 || idx >= hmpool.nunits) {
		krt_log(KGPU_LOG_ERROR,
			"Invalid host mem pointer %p to free.\n",
			p);
		return;
	}
	
	nunits = hmpool.alloc_sz[idx];
	if (nunits == 0) return;
	
	if (nunits > hmpool.nunits - idx) {
		krt_log(KGPU_LOG_ERROR,
			"Invalid host mem allocation info: "
			"allocated %u units at index %u.\n",
			nunits, idx);
		return;
	}
	
	spin_lock(&hmpool.lock);
	bitmap_clear(hmpool.bitmap, idx, nunits);
	hmpool.alloc_sz[idx] = 0;
	spin_unlock(&hmpool.lock);
}
EXPORT_SYMBOL_GPL(kg_vfree);

u64 kg_alloc_mmap_area(u64 size)
{
	u32 npages = DIV_ROUND_UP(size, PAGE_SIZE);
	u64 p = 0;
	u64 idx;
	
	spin_lock(&hmvma.lock);
	
	idx = bitmap_find_next_zero_area(hmvma.bitmap,
					 hmvma.npages,
					 0, npages, 0);
	if (idx < hmvma.npages) {
		bitmap_set(hmvma.bitmap, idx, npages);
		p = hmvma.start + (idx<<PAGE_SHIFT);
		hmvma.alloc_sz[idx] = n;
	} else {
		krt_log(KGPU_LOG_ERROR, 
			"Out of mmap area for mapping "
			"ask for %u page %lu size, idx %lu.\n",
			npages, size, idx);
	}
	
	spin_unlock(&hmvma.lock);
	
	return p;
}
EXPORT_SYMBOL_GPL(kg_alloc_mmap_area);

void kg_free_mmap_area(u64 start)
{
	u64 idx = (start - hmvma.start)>>PAGE_SHIFT;
	u32 npages;
	
	if (start == 0) return;
	
	if (idx < 0 || idx >= hmvma.npages) {
		krt_log(KGPU_LOG_ERROR,
			 "Invalid GPU mmap pointer 0x%8lX to free.\n", start);
		return;
	}
	
	npages = hmvma.alloc_sz[idx];
	if (npages > (hmvma.npages - idx)) {
		krt_log(KGPU_LOG_ERROR,
			 "Invalid GPU mmap allocation info: "
			 "allocated %u pages at index %u.\n", npages, idx);
		return;
	}
	if (npages > 0) {
		spin_lock(&hmvma.lock);
		bitmap_clear(hmvma.bitmap, idx, npages);
		spin_unlock(&hmvma.lock);
	}
	hmvma.alloc_sz[idx] = 0;
}
EXPORT_SYMBOL_GPL(kg_free_mmap_area);

void kg_unmap_area(u64 start)
{
	
}
EXPORT_SYMBOL_GPL(kg_unmap_area);

int kg_map_page(struct page *p, u64 addr)
{
	
}
EXPORT_SYMBOL_GPL(kg_map_page);

void* kg_map_pages(struct page **ps, int nr)
{
	
}
EXPORT_SYMBOL_GPL(kg_map_pages);





