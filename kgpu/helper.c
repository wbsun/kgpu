/*
 * This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2010-2011 University of Utah and the Flux Group.
 * All rights reserved.
 *
 * Userspace helper program.
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

struct _kgpu_sritem {
    struct kgpu_service_request sr;
    struct list_head glist;
    struct list_head list;
};

static int devfd;

struct kgpu_gpu_mem_info hostbuf;

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

static int init_kgpu(void)
{
    int  i, len, r;
    
    devfd = ssc(open(kgpudev, O_RDWR));

    init_gpu();

    /* alloc GPU Pinned memory buffers */
    /* for (i=0; i<KGPU_BUF_NR; i++) { */
	hostbuf.uva = (void*)alloc_pinned_mem(KGPU_BUF_SIZE);
	hostbuf.size = KGPU_BUF_SIZE;
	dbg("%p \n", hostbuf.uva);
	memset(hostbuf.uva, 0, KGPU_BUF_SIZE);
	ssc( mlock(hostbuf.uva, KGPU_BUF_SIZE));
    /* } */
    
    len = sizeof(struct kgpu_gpu_mem_info);

    /* tell kernel the buffers */
    r = ioctl(devfd, KGPU_IOC_SET_GPU_BUFS, (unsigned long)&hostbuf);
    if (r < 0) {
	perror("Write req file for buffers.");
	abort();
    }

    return 0;
}


static int finit_kgpu(void)
{
    int i;

    ioctl(devfd, KGPU_IOC_SET_STOP);
    close(devfd);
    finit_gpu();

    /* for (i=0; i<KGPU_BUF_NR; i++) { */
	free_pinned_mem(hostbuf.uva);
    /* } */
    return 0;
}

static int send_kgpu_ku_response(struct kgpu_ku_response *resp)
{
    ssc(write(devfd, resp, sizeof(struct kgpu_ku_response)));
    return 0;
}

static void fail_request(struct _kgpu_sritem *sreq, int serr)
{
    sreq->sr.state = KGPU_REQ_DONE;
    sreq->sr.errcode = serr;
    list_del(&sreq->list);
    list_add_tail(&sreq->list, &done_reqs);
}

static struct _kgpu_sritem *alloc_kgpu_service_request()
{
    struct _kgpu_sritem *s = (struct _kgpu_sritem *)
	malloc(sizeof(struct _kgpu_sritem));
    if (s) {
    	memset(s, 0, sizeof(struct _kgpu_sritem));
	INIT_LIST_HEAD(&s->list);
	INIT_LIST_HEAD(&s->glist);
    }
    return s;
}

static void free_kgpu_service_request(struct _kgpu_sritem *s)
{
    free(s);
}

static void init_kgpu_service_request(struct _kgpu_sritem *item,
			       struct kgpu_ku_request *kureq)
{
    dbg("get request %d %s\n", kureq->id, kureq->service_name);
    
    list_add_tail(&item->glist, &all_reqs);

    memset(&item->sr, 0, sizeof(struct kgpu_service_request));
    item->sr.id = kureq->id;
    item->sr.hin = kureq->in;
    item->sr.hout = kureq->out;
    item->sr.hdata = kureq->data;
    item->sr.insize = kureq->insize;
    item->sr.outsize = kureq->outsize;
    item->sr.datasize = kureq->datasize;
    item->sr.stream_id = -1;
    item->sr.s = kgpu_lookup_service(kureq->service_name);
    if (!item->sr.s) {
	fail_request(item, KGPU_NO_SERVICE);
    } else {
	item->sr.s->compute_size(&item->sr);
	item->sr.state = KGPU_REQ_INIT;
	item->sr.errcode = 0;
	list_add_tail(&item->list, &init_reqs);
    }
}

static int get_next_kgpu_service_request(void)
{
    int err;
    struct pollfd pfd;

    struct _kgpu_sritem *sreq;
    struct kgpu_ku_request kureq;

    dbg("read is %s\n", list_empty(&all_reqs)?"blocking":"non-blocking");

    pfd.fd = devfd;
    pfd.events = POLLIN;
    pfd.revents = 0;

    err = poll(&pfd, 1, list_empty(&all_reqs)? -1:0);
    if (err == 0 || (err && !(pfd.revents & POLLIN)) ) {
	return -1;
    } else if (err == 1 && pfd.revents & POLLIN)
    {
	sreq = alloc_kgpu_service_request();
    
	if (!sreq)
	    return -1;

	err = read(devfd, (char*)(&kureq), sizeof(struct kgpu_ku_request));
	if (err <= 0) {
	    if (errno == EAGAIN || err == 0) {
		free_kgpu_service_request(sreq);
		return -1;
	    } else {
		perror("Read request.");
		abort();
	    }
	} else {
	    init_kgpu_service_request(sreq, &kureq);	
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

static int kgpu_service_request_alloc_mem(struct _kgpu_sritem *sreq)
{
    int r = alloc_gpu_mem(&sreq->sr);
    if (r) {
	return -1;
    } else {
	sreq->sr.state = KGPU_REQ_MEM_DONE;
	list_del(&sreq->list);
	list_add_tail(&sreq->list, &memdone_reqs);
	return 0;
    }
}

static int prepare_exec(struct _kgpu_sritem *sreq)
{
    int r;
    if (alloc_stream(&sreq->sr)) {
	r = -1;
    } else {
	r = sreq->sr.s->prepare(&sreq->sr);
	if (r) {
	    fail_request(sreq, r);
	} else {
	    sreq->sr.state = KGPU_REQ_PREPARED;
	    list_del(&sreq->list);
	    list_add_tail(&sreq->list, &prepared_reqs);
	}
    }

    return r;
}
	
static int launch_exec(struct _kgpu_sritem *sreq)
{
    int r = sreq->sr.s->launch(&sreq->sr);
    if (r) {
	fail_request(sreq, r);	
    } else {
	sreq->sr.state = KGPU_REQ_RUNNING;
	list_del(&sreq->list);
	list_add_tail(&sreq->list, &running_reqs);
    }
    return 0;
}

static int post_exec(struct _kgpu_sritem *sreq)
{
    int r = 1;
    if (execution_finished(&sreq->sr)) {
	if (!(r = sreq->sr.s->post(&sreq->sr))) {
	    sreq->sr.state = KGPU_REQ_POST_EXEC;
	    list_del(&sreq->list);
	    list_add_tail(&sreq->list, &post_exec_reqs);
	}
	else
	    fail_request(sreq, r);
    }

    return r;
}

static int finish_post(struct _kgpu_sritem *sreq)
{
    if (post_finished(&sreq->sr)) {
	sreq->sr.state = KGPU_REQ_DONE;
	list_del(&sreq->list);
	list_add_tail(&sreq->list, &done_reqs);
	
	return 0;
    }

    return 1;
}

static int service_done(struct _kgpu_sritem *sreq)
{
    struct kgpu_ku_response resp;

    resp.id = sreq->sr.id;
    resp.errcode = sreq->sr.errcode;

    send_kgpu_ku_response(&resp);
    
    list_del(&sreq->list);
    list_del(&sreq->glist);
    free_gpu_mem(&sreq->sr);
    free_stream(&sreq->sr);   
    free_kgpu_service_request(sreq);
    return 0;
}

static int __process_request(int (*op)(struct _kgpu_sritem *),
			      struct list_head *lst, int once)
{
    struct list_head *pos, *n;
    int r = 0;
    
    list_for_each_safe(pos, n, lst) {
	r = op(list_entry(pos, struct _kgpu_sritem, list));
	if (!r && once)
	    break;
    }

    return r;	
}

static int main_loop()
{    
    while (loop_continue)
    {
	__process_request(service_done, &done_reqs, 0);
	__process_request(finish_post, &post_exec_reqs, 0);
	__process_request(post_exec, &running_reqs, 1);
	__process_request(launch_exec, &prepared_reqs, 1);
	__process_request(prepare_exec, &memdone_reqs, 1);
	__process_request(kgpu_service_request_alloc_mem, &init_reqs, 0);
	get_next_kgpu_service_request();	
    }

    return 0;
}

int main(int argc, char *argv[])
{
    int c;
    kgpudev = "/dev/kgpu";
    service_lib_dir = "./";

    while ((c = getopt(argc, argv, "d:l:v:")) != -1)
    {
	switch (c)
	{
	case 'd':
	    kgpudev = optarg;
	    break;
	case 'l':
	    service_lib_dir = optarg;
	    break;
	case 'v':
	    kgpu_log_level = atoi(optarg);
	    break;
	default:
	    fprintf(stderr,
		    "Usage %s"
		    " [-d device]"
		    " [-l service_lib_dir]"
		    " [-v log_level"
		    "\n",
		    argv[0]);
	    return 0;
	}
    }
    
    init_kgpu();
    kgpu_load_all_services(service_lib_dir);
    main_loop();
    finit_kgpu();
    return 0;
}
