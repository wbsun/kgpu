/*
 * TODO:
 *     - In get_next_service_request, read directly, no need of poll because:
 *         when a service is done, the service_done function will check if
 *         the global all_reqs list will be empty, if YES: do fcntl to set
 *         the devfd to be blocking, otherwise keep non-blocking.
 *         AND in get_next_service_request, when a request is comming, check
 *         if the global all_reqs list was empty previously, if YES: do
 *         fcntl to set devfd to be non-blocking, otherwise no change.
 *
 *     - Cleanup all headers and sources for well-organized code/defs...
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <poll.h>
#include "list.h"
#include "helper.h"

struct sritem {
    struct service_request sr;
    struct list_head glist;
    struct list_head list;
};

static int devfd;

static struct gpu_buffer gbufs[KGPU_BUF_NR];

volatile int loop_continue = 1;

static char *service_lib_dir;
static char *kgpudev;

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

typedef unsigned char u8;

static void dump_hex(u8* p, int rs, int cs)
{
        int r,c;
        printf("\n");
        for (r=0; r<rs; r++) {
                for (c=0; c<cs; c++) {
                        printf("%02x ", p[r*cs+c]);
                }
        	printf("\n");
        }
}

int init_kgpu(void)
{
    /*char fname[128];*/
    int  i, len, r;
    
    /*(snprintf(fname, 128, "/dev/%s", KGPU_DEV_NAME);*/
    devfd = ssc(open(kgpudev, O_RDWR));

    init_gpu();

    /* alloc GPU Pinned memory buffers */
    for (i=0; i<KGPU_BUF_NR; i++) {
	gbufs[i].addr = (void*)alloc_pinned_mem(KGPU_BUF_SIZE);
	/*dbg("%p \n", gbufs[i].addr);*/
	memset(gbufs[i].addr, 0, KGPU_BUF_SIZE);
	ssc( mlock(gbufs[i].addr, KGPU_BUF_SIZE));
    }
    
    len = KGPU_BUF_NR*sizeof(struct gpu_buffer);

    /* tell kernel the buffers */
    r = ioctl(devfd, KGPU_IOC_SET_GPU_BUFS, (unsigned long)gbufs);
    if (r < 0) {
	perror("Write req file for buffers.");
	abort();
    }

    return 0;
}


int finit_kgpu(void)
{
    int i;

    ioctl(devfd, KGPU_IOC_SET_STOP);
    close(devfd);
    finit_gpu();

    for (i=0; i<KGPU_BUF_NR; i++) {
	free_pinned_mem(gbufs[i].addr);
    }
    return 0;
}

int send_ku_response(struct ku_response *resp)
{
    ssc(write(devfd, resp, sizeof(struct ku_response)));
    return 0;
}

void fail_request(struct sritem *sreq, int serr)
{
    sreq->sr.state = REQ_DONE;
    sreq->sr.errcode = serr;
    list_del(&sreq->list);
    list_add_tail(&sreq->list, &done_reqs);
}

struct sritem *alloc_service_request()
{
    struct sritem *s = (struct sritem *)
	malloc(sizeof(struct sritem));
    if (s) {
    	memset(s, 0, sizeof(struct sritem));
	INIT_LIST_HEAD(&s->list);
	INIT_LIST_HEAD(&s->glist);
    }
    return s;
}

void free_service_request(struct sritem *s)
{
    free(s);
}

int get_next_service_request()
{
    int err;
    struct pollfd pfd;

    struct sritem *sreq;

    /*dbg("read is %s\n", list_empty(&all_reqs)?"blocking":"non-blocking");*/

    pfd.fd = devfd;
    pfd.events = POLLIN;
    pfd.revents = 0;

    err = poll(&pfd, 1, list_empty(&all_reqs)? -1:0);
    if (err == 0 || (err && !(pfd.revents & POLLIN)) ) {
	return -1;
    } else if (err == 1 && pfd.revents & POLLIN)
    {
	sreq = alloc_service_request();
    
	if (!sreq)
	    return -1;

	err = read(devfd, (char*)(&(sreq->sr.kureq)), sizeof(struct ku_request));
	if (err <= 0) {
	    if (errno == EAGAIN || err == 0) {
		free_service_request(sreq);
		return -1;
	    } else {
		perror("Read request.");
		abort();
	    }
	} else {
	
	    /*dbg("request %d %s %p %p %d\n", sreq->sr.kureq.id, sreq->sr.kureq.sname,
		sreq->sr.kureq.input, sreq->sr.kureq.output, *(int*)(sreq->sr.kureq.input));*/
	
	    list_add_tail(&sreq->glist, &all_reqs);
	    sreq->sr.stream_id = -1;
	
	    sreq->sr.s = lookup_service(sreq->sr.kureq.sname);
	    if (!sreq->sr.s) {
		fail_request(sreq, KGPU_NO_SERVICE);
	    }
	    else {	
		sreq->sr.s->compute_size(&sreq->sr);
		sreq->sr.state = REQ_INIT;
		sreq->sr.errcode = 0;
		list_add_tail(&sreq->list, &init_reqs);
	    }
	
	    return 0;
	}
    } else {
	if (err < 0) {
	    perror("Poll request");
	    abort();
	} else {
	    fprintf(stderr, "Poll returns multiple fd's results\n");
	    abort();
	}
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
	list_add_tail(&sreq->list, &memdone_reqs);
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
	    list_add_tail(&sreq->list, &prepared_reqs);
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
	list_add_tail(&sreq->list, &running_reqs);
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
	    list_add_tail(&sreq->list, &post_exec_reqs);
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
	list_add_tail(&sreq->list, &done_reqs);
	
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
	
	/*dbg("one loop\n");*/
	
    }

    return 0;
}

int main(int argc, char *argv[])
{
    int c;
    kgpudev = "/dev/kgpu";
    service_lib_dir = "./";

    while ((c = getopt(argc, argv, "d:l:")) != -1)
    {
	switch (c)
	{
	case 'd':
	    kgpudev = optarg;
	    break;
	case 'l':
	    service_lib_dir = optarg;
	    break;
	default:
	    fprintf(stderr, "Usage %s [-d device] [-l service_lib_dir]\n",
		argv[0]);
	    return 0;
	}
    }
    
    init_kgpu();
    load_all_services(service_lib_dir);
    main_loop();
    finit_kgpu();
    return 0;
}
