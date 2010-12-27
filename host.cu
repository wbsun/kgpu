#include <time.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "nsk.h"
#include "hostutils.h"

nsk_request_t *h_requests;
nsk_response_t *h_responses;
nsk_request_t *k_requests;
nsk_response_t *k_responses;

int hostmemuses[4];

volatile int *h_current;
volatile int *h_taskdone;

volatile int current = 0;
volatile int last = 0;
volatile int next = 0;

void init_ku_context()
{
    k_requests = (nsk_request_t*)malloc( sizeof(nsk_request_t)*NSK_MAX_REQ_NR );
    k_responses = (nsk_response_t*)malloc( sizeof(nsk_response_t)*NSK_MAX_REQ_NR );
    mlock( k_requests, sizeof(nsk_request_t)*NSK_MAX_REQ_NR );
    mlock( k_responses, sizeof(nsk_response_t)*NSK_MAX_REQ_NR );
	
    memset((void*)k_requests, 0, sizeof(nsk_request_t)*NSK_MAX_REQ_NR);
	
    // init host memory buffers uses:
    for (int i=0; i<4; i++)
	hostmemuses[i] = -1;
		
    nsk_buf_info_t bufs[4];	
    // tell nsk kernel-size code the buffers
    for (int i=0; i<4; i++) {
	bufs[i].addr = (void*)h_mems[i];
	bufs[i].size = NSK_MEM_SIZE;
    }
    
    int nskkfd = ssce(open(NSK_PROCFS_FILE, O_RDWR));
    ssce(write(nskkfd, (void*)bufs, sizeof(nsk_buf_info_t)*4));
    close(nskkfd);
}

int fake_task(int cur)
{
    static int idseq = 1;
    int which = (cur+1)%NSK_MAX_REQ_NR;

    int taskfunc = rand()%4 + 1;

    if (idseq > 10000)
	taskfunc = -1;

    nsk_request_t *kreq = k_requests+which;
    nsk_response_t *kresp = k_responses+which;

    kreq->request_id = idseq;
    kresp->request_id = idseq++;

    kreq->taskfunc = taskfunc;
    kreq->insize = sizeof(int)*GRIDS_X*BLOCKS_X;
    kreq->outsize = kreq->insize;
    kreq->inputs = get_next_host_mem(which);
    kreq->outputs = (char*)(kreq->inputs)+kreq->insize;
    memset((void*)(kreq->inputs), 0, 2*(kreq->insize));

    printf("fake %d at %d\n", taskfunc, which);
    
    return which;
}

#define checkarray(arry, sz, val, ok)		\
    do {					\
	ok = 1;					\
	for (int i=0; i<(sz); i++) {		\
	    if (arry[i] != (val)) {		\
		ok = 0;				\
		break;				\
	    }					\
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


void fake_post_req(int which, nsk_request_t *kreq)
{
    int ok;
    int *outputs = (int*)(kreq->outputs);
    // just for test of fake task generator:
    checkarray(outputs,
	       GRIDS_X*BLOCKS_X,
	       kreq->request_id,
	       ok);
    if (ok) {
	printf("%d OK\n", kreq->request_id);
    } else {
	printarray(outputs,
		   GRIDS_X, BLOCKS_X);
    }
    put_host_mem(kreq->inputs);
}

int poll_next_task(int cur)
{    
    return fake_task(cur);
}

void host()
{
    volatile int i=0;
    init_hd_context();
    init_ku_context();
	
    start_device_kernel();
	
    while (1) {
	if ( is_current_task_done() || i==0)
	{
	    if (i!= 0) {
		finish_task(current, k_requests+current);
		fake_post_req(current, k_requests+current);
		printf("%d finished\n", current);
	    }
			
	    if (current != next)
	    {
		start_hd_task(next);
		printf("start %d\n", next);
		current = next;
		i++;
		if ((k_requests+next)->taskfunc == -1)
		    break;
	    }
	}
		
	if (current == next)
	{
	    next = poll_next_task(current);
	    if (next != current) {
		if (!prepare_hd_task(k_requests+next, next)) {
		    next = current;
		}
	    }
	}
    }
    printf("Exit\n");
    clean_hd_context();
}

int main(int argc, char* argv[])
{
    host();
    return 0;
}
