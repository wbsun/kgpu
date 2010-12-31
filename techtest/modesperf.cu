
#include <cuda.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

enum memoryMode { PINNED=1, PAGEABLE, MAPPED, WC };

memoryMode mm = MAPPED;

dim3 grids = dim3(32,1);

//#define csc(err) cutilSafeCall(err)

#define csc(err) __cusafecall(err)

#define cutilSafeCall(err) __cusafecall(err)

#define NSAMPLES 100

static void __cusafecall(cudaError_t e)
{
    if (e != cudaSuccess){
	printf("Error: %s\n", cudaGetErrorString(e));
	cudaThreadExit();
	exit(0);
    }
}


static void initCUDA()
{
    cudaSetDeviceFlags(cudaDeviceMapHost);
}

void __allocMem(void **pph, void **ppd, unsigned int size, memoryMode memMode)
{
    switch(memMode) {
    case PINNED:
	cutilSafeCall( cudaHostAlloc(pph, size, 0) );
	cutilSafeCall( cudaMalloc(ppd, size) );
	break;
    case PAGEABLE:
        *pph = malloc(size);
	cutilSafeCall( cudaMalloc(ppd, size) );
	break;
    case MAPPED:
	cutilSafeCall( cudaHostAlloc(pph, size, cudaHostAllocMapped) );
	cutilSafeCall( cudaHostGetDevicePointer(ppd, *pph, 0) );
	break;
    case WC:
	cutilSafeCall( cudaHostAlloc(pph, size, cudaHostAllocWriteCombined|cudaHostAllocMapped) );
	//cutilSafeCall( cudaHostGetDevicePointer(ppd, *pph, 0) );
	cutilSafeCall( cudaMalloc(ppd, size) );
    default:
	break;
    }
}

void __freeMem(void **pph, void **ppd, memoryMode memMode)
{
    switch(memMode) {
    case PINNED:
    case MAPPED:
    case WC:
	cutilSafeCall(cudaFreeHost(*pph));
	break;
    case PAGEABLE:
	free(*pph);
    default:
	break;
    }

    *pph = NULL;
    cutilSafeCall(cudaFree(*ppd));
}

#define allocMem(pph, ppd, sz, memMode) __allocMem((void **)(pph), (void **)(ppd), (size_t)(sz), memMode)

#define freeMem(pph, ppd, memMode) __freeMem((void **)(pph), (void **)(ppd), memMode)

__global__ void simple_comm(volatile int *inputs, volatile int *results, volatile int *current, volatile int *tdone, int *count) 
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    int mycur = *current;
    int oc = mycur;
    
    int c = *count;
    
    while (1) {
    	oc = *current;
    	if (mycur != oc) {
    		mycur = oc;
    		if (mycur == -1)
    			break;
    		for (int i=0; i< c; i++) {
    			results[c*idx+i] = inputs[c*idx+i]+idx;
    		} 
    		__syncthreads();
    		if (threadIdx.x == 0){
    			atomicAdd((int*)tdone, 1);
    		}
    		//__threadfence_system();
    	}
    	//__syncthreads();
    }
}

__global__ void old_comm(int *inputs, int *results, int *count)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    
    int c = *count;
    
    for (int i=0; i< c; i++) {
    	results[c*idx+i] = inputs[c*idx+i]+idx;
    }
}

#include <time.h>

static void nsleep(long ns)
{
    struct timespec tv;
    
    tv.tv_sec = ns/1000000000;
    tv.tv_nsec = ns%1000000000;

    nanosleep(&tv, NULL);
}

typedef struct {
    timespec start, stop;
} timer;

double ts2d(timespec *ts)
{
    double d = ts->tv_sec;
    d += ts->tv_nsec/1000000000.0;
    return d;
}

long double ts2ld_us(timespec *ts) 
{
	long double ld = ts->tv_sec*1000000;
	ld += ((long double)(ts->tv_nsec))/1000.0;
	return ld;
}

timespec get_timer_val(timer *tm)
{
    timespec temp;
    if ((tm->stop.tv_nsec - tm->start.tv_nsec)<0) {
	temp.tv_sec = tm->stop.tv_sec - tm->start.tv_sec-1;
	temp.tv_nsec = 1000000000+tm->stop.tv_nsec - tm->start.tv_nsec;
    } else {
	temp.tv_sec = tm->stop.tv_sec - tm->start.tv_sec;
	temp.tv_nsec = tm->stop.tv_nsec - tm->start.tv_nsec;
    }
    return temp;
}

void start_timer(timer *tm)
{
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tm->start);
}

timespec stop_timer(timer *tm)
{
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tm->stop);
    return get_timer_val(tm);
}

#define fillarray(arry, sz, val)		\
    do {					\
	for (int i=0; i < (sz); i++) {		\
	    arry[i] = (val);			\
	}					\
    } while(0)


#define printarray(arry, cols, rows)			\
    do {						\
	for (int row = 0; row < (rows); row++) {	\
	    for (int col = 0; col < (cols); col++) {	\
		int idx = col + row*(cols);		\
		printf("%4d", arry[idx]);		\
	    }						\
	    printf("\n");				\
	}						\
    } while(0)


#define checkarry(arry, sz, val, ok)		\
    do {					\
	ok = 1;					\
	for (int i=0; i<(sz); i++) {		\
	    if (arry[i] != (val)) {		\
		ok = 0;				\
		break;				\
	    }					\
	}					\
    } while(0)


#define MAX_TSIZE (16*1024)


void test_old_kernel()
{
    int *h_inputs;
    int *d_inputs;
    int *h_results;
    int *d_results;
    int *d_count;
    int *h_count;

    dim3 blocks = dim3(32, 1);
    
    int nthreads = grids.x*grids.y*blocks.x*blocks.y;
    
    int count = 1;
    
    size_t dsize = MAX_TSIZE*nthreads;
    
    //memoryMode mm = PAGEABLE;
    
    int nt = 0;
    int t = MAX_TSIZE/sizeof(int);
    while(t>0) {
    	t>>=1;
    	nt++;
    }
    long double *ress = (long double*)malloc(sizeof(long double)*(nt+1));
    memset(ress, 0, sizeof(long double)*(nt+1));
    
    //cudaStream_t s1;
    //cudaStreamCreate(&s1);
    
    allocMem(&h_inputs, &d_inputs, dsize, mm);
    allocMem(&h_results, &d_results, dsize, mm);
    allocMem(&h_count, &d_count, sizeof(int), mm);
    
    int k=0;
    
    for (; count<=MAX_TSIZE/sizeof(int); count*=2, k++) { 
    
    	timer tm;
    	//long long int nsec = 0;
    	//int sec = 0;

    	long double gt=0;
    	
    for ( int j=1; j < NSAMPLES+6; j++) {

    	memset(h_inputs, 0, dsize);
    	memset(h_results, 0, dsize);
    	*h_count = count;
    	
    	start_timer(&tm);   	

    	if (mm == PINNED || mm == PAGEABLE) {
		cudaMemcpy((void*)d_inputs,(void*)h_inputs, count*nthreads*sizeof(int), cudaMemcpyHostToDevice);		
		cudaMemcpy((void*)d_count, (void*)h_count, sizeof(int), cudaMemcpyHostToDevice);
    	}
    	
    	old_comm<<<grids, blocks>>>(d_inputs, d_results, d_count);
    	
    	cudaThreadSynchronize();

	if (mm == PINNED || mm == PAGEABLE) {
	    	cudaMemcpyAsync((void*)h_results, (void*)d_results, count*nthreads*sizeof(int), cudaMemcpyDeviceToHost);
	}
	
	cudaThreadSynchronize();
	
	timespec tt = stop_timer(&tm);
	//int ok = 0;  
	//checkarry(h_results, nblks, j, ok);

	if (j>=6)
	gt += ts2ld_us(&tt);
    }
    	//printf("Size: %lu - CPU time: %Lfus\n", count*sizeof(int), gt/NSAMPLES);
    	printf("%10luB", count*sizeof(int));
    	ress[k] = gt/NSAMPLES;
    }
    
    printf("\n");
    
    for (count=1,k=0; count<=MAX_TSIZE/sizeof(int); count*=2, k++) {
    	printf("%11.3Lf", ress[k]);
    }
    printf("\n");

    freeMem(&h_inputs, &d_inputs, mm);
    freeMem(&h_results, &d_results, mm);
    freeMem(&h_count, &d_count, mm);
    
    //cudaStreamDestroy(s1);
   
    //cudaThreadExit();
}

void test_comm_kernel()
{
    int *h_inputs;
    int *d_inputs;
    int *h_results;
    int *d_results;
    int *d_count;
    int *h_count;
    int *h_current, *d_current;
    int *h_tdone, *d_tdone;   

    dim3 blocks = dim3(32, 1);
    
    int nthreads = grids.x*grids.y*blocks.x*blocks.y;
    
    int count = 1;
    
    size_t dsize = MAX_TSIZE*nthreads;
    
    int nt = 0;
    int t = MAX_TSIZE/sizeof(int);
    while(t>0) {
    	t>>=1;
    	nt++;
    }
    long double *ress = (long double*)malloc(sizeof(long double)*(nt+1));
    memset(ress, 0, sizeof(long double)*(nt+1));
    
    allocMem(&h_inputs, &d_inputs, dsize, mm);
    allocMem(&h_results, &d_results, dsize, mm);
    allocMem(&h_count, &d_count, sizeof(int), mm);
    allocMem(&h_current, &d_current, sizeof(int), WC);
    allocMem(&h_tdone, &d_tdone, sizeof(int), mm);
    
    cudaStream_t s1, s2, s3;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);
    //cudaStreamCreate(&s3);
    
    int k=0;
    for (; count<=MAX_TSIZE/sizeof(int); count*=2, k++) { 
    
    	timer tm;
    	//long long int nsec = 0;
    	//int sec = 0;

    	long double gt=0;   	
    	
    	*h_tdone = 0;
    	*h_current = 0;
    	*h_count = count;
    	
    	if (mm == PINNED || mm == PAGEABLE) {
    		cudaMemcpyAsync((void*)d_current, (void*)h_current, sizeof(int), cudaMemcpyHostToDevice, s1); 
    		cudaMemcpyAsync((void *)d_tdone, (void *)h_tdone, sizeof(int), cudaMemcpyHostToDevice, s1);	
    		cudaMemcpyAsync((void*)d_count, (void*)h_count, sizeof(int), cudaMemcpyHostToDevice, s1);
    		cudaStreamSynchronize(s1); 
    	}    	   	
    	
    	simple_comm<<<grids, blocks, 0, s1>>>(d_inputs, d_results, d_current, d_tdone, d_count);
    	
    for ( int j=1; j < NSAMPLES+6; j++) {

    	memset(h_inputs, 0, dsize);
    	memset(h_results, 0, dsize);
    	
    	start_timer(&tm);   	

    	if (mm == PINNED || mm == PAGEABLE) {		
		cudaMemcpyAsync((void *)d_inputs, (void *)h_inputs, count*nthreads*sizeof(int), cudaMemcpyHostToDevice, s2);
		//cudaStreamSynchronize(s2);	
		//cudaMemcpyAsync((void *)d_current, (void *)h_current, sizeof(int), cudaMemcpyHostToDevice, s2);
    	}    	
    	*h_current = j;
    	cudaMemcpyAsync((void *)d_current, (void *)h_current, sizeof(int), cudaMemcpyHostToDevice, s2);
    	
    	while(1) {
    		cudaMemcpyAsync((void*)h_tdone, (void*)d_tdone, sizeof(int), cudaMemcpyDeviceToHost, s2);
    		cudaStreamSynchronize(s2);
    		
    		if (*h_tdone == j*grids.x*grids.y)
    			break;
    		//printf("%d, ", *h_tdone);
    	}

	if (mm == PINNED || mm == PAGEABLE) {
	    	cudaMemcpyAsync((void*)h_results, (void*)d_results, count*nthreads*sizeof(int), cudaMemcpyDeviceToHost, s2);
	}
	
	timespec tt = stop_timer(&tm);
	
	//int ok = 0;  
	//checkarry(h_results, nblks, j, ok);

	if (j>=6)
	gt += ts2ld_us(&tt);
    }
    	//printf("Size: %lu - CPU time: %Lfus\n", count*sizeof(int), gt/NSAMPLES);
    	printf("%10luB", count*sizeof(int));
    	ress[k] = gt/NSAMPLES;
    
    	*h_current = -1;
    	cudaMemcpyAsync((void *)d_current, (void *)h_current, sizeof(int), cudaMemcpyHostToDevice, s2);
    	cudaStreamSynchronize(s2);
    	cudaStreamSynchronize(s1);
    	cudaThreadSynchronize();
    }    	
    
    printf("\n");
    
    for (count=1,k=0; count<=MAX_TSIZE/sizeof(int); count*=2, k++) {
    	printf("%11.3Lf", ress[k]);
    }
    printf("\n");

    freeMem(&h_inputs, &d_inputs, mm);
    freeMem(&h_results, &d_results, mm);
    freeMem(&h_count, &d_count, mm);
    freeMem(&h_tdone, &d_tdone, mm);
    freeMem(&h_current, &d_current, WC);
    
    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);
    //cudaStreamDestroy(s3);
}


int main(int argc, char *argv[])
{
    if (argc > 1) {
	switch(argv[1][1]) {
	case 'p':
	    mm = PINNED;
	    break;
	case 'g': // not supported by our model
	    mm = PAGEABLE;
	    break;
	case 'm':
	    mm = MAPPED;
	    break;
	case 'w':
	    mm = WC;
	    nsleep(10);
	    break;
	default:
	    break;
	}
    }

    if (argc > 3) {
	grids = dim3(atoi(argv[2]), atoi(argv[3]));
    }
    
    
    initCUDA();    
    printf("\n-----------------------\nMany kernel launch:\n");
    test_old_kernel();
    cudaThreadExit();
    
    initCUDA();
    printf("\n-----------------------\nSingle kernel:\n");
    test_comm_kernel();
    cudaThreadExit();
    return 0;
}
