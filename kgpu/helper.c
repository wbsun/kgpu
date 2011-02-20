#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include "helper.h"

static int reqfd, respfd;

static struct gpu_buffer gbufs[KGPU_BUF_NR];

/* lists of requests of different states */
LIST_HEAD(all_reqs);
LIST_HEAD(init_reqs);
LIST_HEAD(memdone_reqs);
LIST_HEAD(prepared_reqs);
LIST_HEAD(running_reqs);
LIST_HEAD(post_exec_reqs);
LIST_HEAD(done_reqs);

#define ssc(...) _safe_syscall(__VA_ARGS__, __FILE__, __LINE__)

int _safe_syscall(int r, const char *file, int line)
{
    if (r<0) {
	printf("Error in %s:%d, ", file, line);
	perror("");
	abort();
    }
    return r;
}

int init_kgpu(void)
{
    char fname[128];
    int  i, len, r;
    
    snprintf(fname, 128, "/proc/%s", REQ_PROC_FILE);
    reqfd = ssc(open(fname, O_RDWR));

    snprintf(fname, 128, "/proc/%s", RESP_PROC_FILE);
    respfd = ssc(open(fname, O_WRONLY));

    init_gpu();

    /* alloc GPU Pinned memory buffers */
    for (i=0; i<KGPU_BUF_NR; i++) {
	gbufs[i].addr = alloc_pinned_mem(KGPU_BUF_SIZE);
	gbufs[i].size = KGPU_BUF_SIZE;
    }

    len = KGPU_BUF_NR*sizeof(struct gpu_buffer);

    /* tell kernel the buffers */
    r = write(reqfd, gbufs, len);
    if (r < 0) {
	perror("Write req file for buffers.");
	abort();
    }
    else if (r != len) {
	printf("Write req file for buffers failed!\n");
	abort();
    }

    return 0;
}


int send_ku_response(struct ku_response *resp)
{
    ssc(write(respfd, resp, sizeof(struct ku_response)));
    return 0;
}


int get_next_service_request(struct service_request **psreq)
{
    int err;

    struct service_request *sreq = alloc_service_request();
    *psreq = NULL;
    
    if (!sreq)
	return 1;

    err = read(reqfd, &sreq->kureq, sizeof(struct ku_request));
    if (err < 0) {
	if (errno == EAGAIN) {
	    return 1;
	} else {
	    perror("Read request.");
	    abort();
	}
    } else {
	list_add_tail(&allreqs, &sreq->glist);
	*psreq = sreq;
	sreq->stream_id = -1;
	
	sreq->s = lookup_service(sreq->kureq.sname);
	if (!sreq->s) {
	    sreq->errno = KGPU_NO_SERVICE;
	    sreq->state = REQ_DONE;	    
	    list_add_tail(&donereqs, &sreq->list);
	}
	else {	
	    sreq->s->compute_size(sreq);
	    sreq->state = REQ_INIT;
	    sreq->errno = 0;
	    list_add_tail(&initreqs, &sreq->list);
	}
	return 0;
    }
}

int service_request_alloc_mem(struct service_request *sreq)
{
    int r = alloc_gpu_mem(sreq);
    if (r) {
	return 1;
    } else {
	sreq->state = REQ_MEM_DONE;
	list_del(&sreq->list);
	list_add_tail(&memdone_reqs, &sreq->list);
	return 0;
    }
}

int prepare_exec(struct service_request *sreq)
{
    if (alloc_stream(sreq)) {
	return 1;
    } else {
	sreq->s->prepare(sreq);
	sreq->state = REQ_PREPARED;
	list_del(&sreq->list);
	list_add_tail(&prepared_reqs, &sreq->list);
	return 0;
    }
}
	
int launch_exec(struct service_request *sreq)
{
    int r = sreq->s->launch(sreq);
    if (r) {
	sreq->state = REQ_DONE;
	sreq->errno = r;
	list_del(&sreq->list);
	list_add_tail(&done_reqs, &sreq->list);
    } else {
	sreq->state = REQ_RUNNING;
	list_del(&sreq->list);
	list_add_tail(&running_reqs, &sreq->list);
    }
    return 0;
}

int post_exec(struct service_request *sreq)
{
    if (execution_finished(sreq)) {
	sreq->state = REQ_POST_EXEC;
	list_del(&sreq->list);
	list_add_tail(&post_exec_reqs, &sreq->list);
	sreq->s->post(sreq);
	return 0;
    }

    return 1;
}

int finish_post(struct service_request *sreq)
{
    if (post_finished(sreq)) {
	sreq->state = REQ_DONE;
	list_del(&sreq->list);
	list_add_tail(&done_reqs, &sreq->list);
	return 0;
    }

    return 1;
}

int service_done(struct service_request *sreq)
{
    struct ku_response resp;

    resp.id = sreq->kureq.id;
    resp.errno = sreq->errno;

    send_ku_response(&resp);
    
    list_del(&sreq->list);
    list_del(&sreq->glist);
    free_gpu_mem(sreq);
    free_stream(sreq);   
    free_service_request(sreq);
    return 0;
}

int main_loop()
{
}
