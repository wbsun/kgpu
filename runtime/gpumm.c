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

static int bitmap_mm_alloc(u64 size,
			   u64 *p,
			   u32 unitsize,
			   spinlock_t *lock,
			   u64 *bitmap,
			   u32 *alloc_sz,
			   u32 nunits,
			   u64 start);
static int bitmap_mm_free(u64 p,
			  u32 unitsize,
			  spinlock_t *lock,
			  u64 *bitmap,
			  u32 *alloc_sz,
			  u32 nunits,
			  u64 start);

int mm_alloc_krequest_devmem(k_request_t *r);
int mm_free_krequest_devmem(k_request_t *r); 


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

/*
 * Memory allocation in KGPU adopts a simple mechansim: a bitmap is used to
 * track all allocation units, which can be defined by changing KGPU_BUF_UNIT_SIZE
 * macro. When allocating a trunk of memory, consecutive zero bits are searched
 * linearly from the beginning of the bitmap to fit the requested size of memory,
 * then those bits are set for allocation. To track the allocated size, a silly
 * sizes array is created, which is the 'alloc_sz' filed in those k_meminfo_*
 * structures, each allocation unit has its own allocation size record in that
 * array. The unit at the beginning of the allocated memory trunk has its record
 * filled with the allocation size, other records are ignored. So when freeing
 * the memory trunk, kg_vfree can simply get the size record of the beginning
 * unit and free certain number of bits according to the size.
 */
void* kg_vmalloc(u64 nbytes)
{
	int ret;
	u64 addr = 0;

	ret = bitmap_mm_alloc(nbytes, &addr, KGPU_BUF_UNIT_SIZE,
			      &hmpool.lock,
			      hmpool.bitmap,
			      hmpool.alloc_sz,
			      hmpool.nunits,
			      hmpool.kva);
	if (unlikely(ret)) {
		if (ret == KGPU_ENO_MEM) {
			krt_log(KGPU_LOG_ERROR,
				"Out of host memory for %lu.\n",
				nbytes);
		} else {
			krt_log(KGPU_LOG_ERROR,
				"Error when allocating host mem"
				" for %lu bytes, code %d.\n",
				nbytes, ret);
		}			
	}
	
	return (void*)addr;
}
EXPORT_SYMBOL_GPL(kg_vmalloc);

void kg_vfree(void* p)
{
	int ret;

	ret = bitmap_mm_free(TO_UL(p), KGPU_BUF_UNIT_SIZE,
			     &hmpool.lock,
			     hmpool.bitmap,
			     hmpool.alloc_sz,
			     hmpool.nunits,
			     hmpool.kva);

	if (unlikely(ret)) {
		switch (ret) {
		case KGPU_EINVALID_POINTER:
			krt_log(KGPU_LOG_ERROR,
				"Invalid host mem pointer %p to free.\n",
				p);
			break;
		case KGPU_EINVALID_MALLOC_INFO:
			krt_log(KGPU_LOG_ERROR,
				"Invalid host mem allocation info: "
				"pointer %p.\n",
				(void*)p);
			break;
		default:
			krt_log(KGPU_LOG_ERROR,
				"Error when free host memory %p, code: %d.\n",
				p, ret);
			break;			
		}
	}
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
	u64 idx = (start-hmvma.start)>>PAGE_SHIFT;
	
	if (idx < 0 || idx >= hmvma.npages) {
		krt_log(KGPU_LOG_ERROR,
			"Invalid GPU mmap pointer 0x%lX to unmap\n",
			start);
	} else {
		u32 n = hmvma.alloc_sz[idx];
		if (n > (hmvma.npages - idx)) {
			krt_log(KGPU_LOG_ERROR,
				"Invalid GPU mmap allocation info: "
				"allocated %u pages at index %u\n", n, idx);
			return;
		}
		if (n > 0) {
			int ret;
			spin_lock(&hmvma.lock);
			bitmap_clear(hmvma.bitmap, idx, n);
			hmvma.alloc_sz[idx] = 0;
			
			/*unmap_mapping_range(hmvma.vma->vm_file->f_mapping,
			 start, n<<PAGE_SHIFT, 1);*/
			
			hmvma.vma->vm_flags |= VM_PFNMAP;
			ret = zap_vma_ptes(hmvma.vma, start, n<<PAGE_SHIFT);
			if (ret)
				krt_log(KGPU_LOG_ALERT,
					"zap_vma_ptes returns %d\n", ret);
			hmvma.vma->vm_flags &= ~VM_PFNMAP;
			spin_unlock(&hmvma.lock);
		}
	}
}
EXPORT_SYMBOL_GPL(kg_unmap_area);

int kg_map_page(struct page *p, u64 addr)
{
	int ret = 0;
	
	down_write(&hmvma.vma->vm_mm->mmap_sem);
	ret = vm_insert_page(hmvma.vma, addr, p);
	if (unlikely(ret < 0)) {
		krt_log(KGPU_LOG_ERROR,
			"Can't remap pfn %lu, error %d, count %d.\n",
			page_to_pfn(p), ret, page_count(p));
	}
	up_write(&hmvma.vma->vm_mm->mmap_sem);
	
	return ret;
}
EXPORT_SYMBOL_GPL(kg_map_page);

void* kg_map_pages(struct page **ps, int nr)
{
	u64 addr, a;
	int i;
	int ret;
	
	addr = kg_alloc_mmap_area(nr<<PAGE_SHIFT);
	if (!addr) {
		return NULL;
	}
	
	down_write(&hmvma.vma->vm_mm->mmap_sem);
	
	a = addr;
	for (i=0; i<n; i++) {
		ret = vm_insert_page(hmvma.vma, a, ps[i]);		
		if (unlikely(ret < 0)) {
			up_write(&hmvma.vma->vm_mm->mmap_sem);
			
			/* Not sure whether to free the mmap area or not. */
			
			krt_log(KGPU_LOG_ERROR,
				"Can't remap %d pfn %lu, error code %d\n",
				i, page_to_pfn(ps[i]), ret);
			return NULL;
		}
	}
	
	up_write(&hmvma.vma->vm_mm->mmap_sem);
	
	return (void*)addr;
}
EXPORT_SYMBOL_GPL(kg_map_pages);

static int bitmap_mm_alloc(u64 size,
			   u64 *p,
			   u32 unitsize,
			   spinlock_t *lock,
			   u64 *bitmap,
			   u32 *alloc_sz,
			   u32 nunits,
			   u64 start)
{
	u32 req_nunits = DIV_ROUND_UP(size, unitsize);
	u64 idx;
	int ret = 0;
	
	*p = 0;	
	spin_lock(lock);
	
	idx = bitmap_find_next_zero_area(bitmap, 
					 nunits, 
					 0, 
					 req_nunits, 
					 0);
	if (idx < nunits) {
		bitmap_set(bitmap, idx, req_nunits);
		*p = start  + (idx*unitsize);
		alloc_sz[idx] = req_nunits;
	} else
		ret = KGPU_ENO_MEM;
	
	spin_unlock(lock);

	return ret;
}

static u64 alloc_devmem(u64 size, k_gpu_t *gpu)
{
	int ret;
	u64 addr = 0;

	ret = bitmap_mm_alloc(size, &addr, KGPU_BUF_UNIT_SIZE,
			      &gpu->devmem.lock,
			      gpu->devmem.bitmap,
			      gpu->devmem.alloc_sz,
			      gpu->devmem.nunits,
			      gpu->devmem.start);
	if (unlikely(ret)) {
		if (ret == KGPU_ENO_MEM) {
			krt_log(KGPU_LOG_ERROR,
				"Out of device memory on GPU %d for %lu.\n",
				gpu->id, size);
		} else {
			krt_log(KGPU_LOG_ERROR,
				"Error when allocating device mem on GPU %d"
				" for %lu, code %d.\n",
				gpu->id, size, ret);
		}			
	}
	
	return addr;	
}

static int bitmap_mm_free(u64 p,
			  u32 unitsize,
			  spinlock_t *lock,
			  u64 *bitmap,
			  u32 *alloc_sz,
			  u32 nunits,
			  u64 start)
{
	u64 idx = (p - start) / unitsize;
	u32 alloc_nunits;
	int ret = 0;
	
	if (!p) return ret;
	
	if (idx < 0 || idx >= nunits) {
		ret = KGPU_EINVALID_POINTER;
	} else {	
		alloc_nunits = alloc_sz[idx];
		if (alloc_nunits == 0) return ret;
		
		if (alloc_nunits > nunits - idx) 
			ret = KGPU_EINVALID_MALLOC_INFO;
		else {
			spin_lock(lock);
			bitmap_clear(bitmap, idx, alloc_nunits);
			alloc_sz[idx] = 0;
			spin_unlock(lock);
		}
	}

	return ret;	
}

static void free_devmem(u64 p, k_gpu_t *gpu)
{
	int ret;

	ret = bitmap_mm_free(p, KGPU_BUF_UNIT_SIZE,
			     &gpu->devmem.lock,
			     gpu->devmem.bitmap,
			     gpu->devmem.alloc_sz,
			     gpu->devmem.nunits,
			     gpu->devmem.start);

	if (unlikely(ret)) {
		switch (ret) {
		case KGPU_EINVALID_POINTER:
			krt_log(KGPU_LOG_ERROR,
				"Invalid device mem pointer %p on GPU %d to free.\n",
				(void*)p, gpu->id);
			break;
		case KGPU_EINVALID_MALLOC_INFO:
			krt_log(KGPU_LOG_ERROR,
				"Invalid device mem allocation info: "
				"pointer %p on GPU %d.\n",
				(void*)p, gpu->id);
			break;
		default:
			krt_log(KGPU_LOG_ERROR,
				"Error when free device memory %p, code: %d.\n",
				(void*)p, ret);
			break;			
		}
	}
}

/*
 * Allocation logic:
 * IF r has depon:
 *     build_depon_outs() to fill 'out' and 'outsize' in kg_depon_t array
 *     IF memflags includes NO_DEV_MEM_ALLOC:
 *         fill din/dout/ddata with NULL, their sizes to be 0, DONE
 *     ELSE-IF
 */
int mm_alloc_krequest_devmem(k_request_t *r)
{
	int i;
	kg_depon_t *depby;
	k_gpu_t *gpu;

	gpu = get_k_gpu(t->device);
	if (!gpu) {
		return KGPU_EINVALID_DEVICE;
	}
	
}

int mm_free_krequest_devmem(k_request_t *r)
{
	
}



