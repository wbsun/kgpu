#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include "list.h"
#include "helper.h"

struct sritem {
    struct service_request sr;
    struct list_head glist;
    struct list_head list;
};

static int reqfd, respfd;

static struct gpu_buffer gbufs[KGPU_BUF_NR];

volatile int loop_continue = 1;

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
	fprintf(stderr, "Error in %s:%d, ", file, line);
	perror("");
	abort();
    }
    return r;
}

#ifndef _NDEBUG
#define dbg(...) fprintf(stderr, __VA_ARGS__)
#else
#define dbg(...)
#endif

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
	gbufs[i].addr = (void*)alloc_pinned_mem(KGPU_BUF_SIZE);
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


int finit_kgpu(void)
{
    int i;
    
    close(reqfd);
    close(respfd);
    finit_gpu();

    for (i=0; i<KGPU_BUF_NR; i++) {
	free_pinned_mem(gbufs[i].addr);
    }
    return 0;
}

int send_ku_response(struct ku_response *resp)
{
    ssc(write(respfd, resp, sizeof(struct ku_response)));
    return 0;
}

void fail_request(struct sritem *sreq, int serr)
{
    sreq->sr.state = REQ_DONE;
    sreq->sr.errcode = serr;
    list_del(&sreq->list);
    list_add_tail(&done_reqs, &sreq->list);
}

struct sritem *alloc_service_request()
{
    struct sritem *s = (struct sritem *)
	malloc(sizeof(struct sritem));
    return s;
}

void free_service_request(struct sritem *s)
{
    free(s);
}

int get_next_service_request()
{
    int err;

    struct sritem *sreq = alloc_service_request();
    
    if (!sreq)
	return -1;

    /* strange trick to tell the read to be blocking IO */
    if (list_empty(&all_reqs))
	sreq->sr.kureq.id = -1;

    err = read(reqfd, &(sreq->sr.kureq), sizeof(struct ku_request));
    if (err < 0) {
	if (errno == EAGAIN) {
	    return -1;
	} else {
	    perror("Read request.");
	    abort();
	}
    } else {
	list_add_tail(&all_reqs, &sreq->glist);
	sreq->sr.stream_id = -1;
	
	sreq->sr.s = lookup_service(sreq->sr.kureq.sname);
	if (!sreq->sr.s) {
	    fail_request(sreq, KGPU_NO_SERVICE);
	}
	else {	
	    sreq->sr.s->compute_size(&sreq->sr);
	    sreq->sr.state = REQ_INIT;
	    sreq->sr.errcode = 0;
	    list_add_tail(&init_reqs, &sreq->list);
	}
	return 0;
    }
}

int service_request_alloc_mem(struct sritem *sreq)
{
    int r = alloc_gpu_mem(&sreq->sr);
    if (r) {
	return -1;
    } else {
	sreq->sr.state = REQ_MEM_DONE;
	list_del(&sreq->list);
	list_add_tail(&memdone_reqs, &sreq->list);
	return 0;
    }
}

int prepare_exec(struct sritem *sreq)
{
    int r;
    if (alloc_stream(&sreq->sr)) {
	r = -1;
    } else {
	r = sreq->sr.s->prepare(&sreq->sr);
	if (r) {
	    fail_request(sreq, r);
	} else {
	    sreq->sr.state = REQ_PREPARED;
	    list_del(&sreq->list);
	    list_add_tail(&prepared_reqs, &sreq->list);
	}
    }

    return r;
}
	
int launch_exec(struct sritem *sreq)
{
    int r = sreq->sr.s->launch(&sreq->sr);
    if (r) {
	fail_request(sreq, r);	
    } else {
	sreq->sr.state = REQ_RUNNING;
	list_del(&sreq->list);
	list_add_tail(&running_reqs, &sreq->list);
    }
    return 0;
}

int post_exec(struct sritem *sreq)
{
    int r = 1;
    if (execution_finished(&sreq->sr)) {
	if (!(r = sreq->sr.s->post(&sreq->sr))) {
	    sreq->sr.state = REQ_POST_EXEC;
	    list_del(&sreq->list);
	    list_add_tail(&post_exec_reqs, &sreq->list);
	}
	else
	    fail_request(sreq, r);
    }

    return r;
}

int finish_post(struct sritem *sreq)
{
    if (post_finished(&sreq->sr)) {
	sreq->sr.state = REQ_DONE;
	list_del(&sreq->list);
	list_add_tail(&done_reqs, &sreq->list);
	return 0;
    }

    return 1;
}

int service_done(struct sritem *sreq)
{
    struct ku_response resp;

    resp.id = sreq->sr.kureq.id;
    resp.errcode = sreq->sr.errcode;

    send_ku_response(&resp);
    
    list_del(&sreq->list);
    list_del(&sreq->glist);
    free_gpu_mem(&sreq->sr);
    free_stream(&sreq->sr);   
    free_service_request(sreq);
    return 0;
}

static int __process_request(int (*op)(struct sritem *),
			      struct list_head *lst, int once)
{
    struct list_head *pos, *n;
    int r = 0;
    
    list_for_each_safe(pos, n, lst) {
	r = op(list_entry(pos, struct sritem, list));
	if (!r && once)
	    break;
    }

    return r;	
}

int main_loop()
{    
    while (loop_continue)
    {
	__process_request(service_done, &done_reqs, 0);
	__process_request(finish_post, &post_exec_reqs, 0);
	__process_request(post_exec, &running_reqs, 1);
	__process_request(launch_exec, &prepared_reqs, 1);
	__process_request(prepare_exec, &memdone_reqs, 1);
	__process_request(service_request_alloc_mem, &init_reqs, 0);
	get_next_service_request();
	dbg("one time loop\n");
    }

    return 0;
}

int main()
{
    init_kgpu();
    main_loop();
    finit_kgpu();
    return 0;
}
