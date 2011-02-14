#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include "kgpu.h"

int reqfd, respfd;

struct gpu_buffer gbufs[KGPU_BUF_NR];

int alloc_gpu_buffers(struct gpu_buffer gbufs[], int n, unsigned long size);


int set_kgpu_buffers(void)
{
    int len = KGPU_BUF_NR*sizeof(struct gpu_buffer);
    int r;

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

int init_kgpu(void)
{
    char fname[128];
    
    snprintf(fname, 128, "/proc/%s", REQ_PROC_FILE);
    reqfd = ssc(open(fname, O_RDWR));

    snprintf(fname, 128, "/proc/%s", RESP_PROC_FILE);
    respfd = ssc(open(fname, O_WRONLY));

    alloc_gpu_buffers(gbufs, KGPU_BUF_NR, KGPU_BUF_SIZE);
    set_kgpu_buffers();
}

int get_request(struct ku_request *req)
{
    int r;

    r = read(reqfd, req, sizeof(struct ku_request));
    if (r < 0) {
	if (errno == EAGAIN) {
	    return 0;
	} else {
	    perror("Read request.");
	    abort();
	}
    } else
	return 1;
}

int put_response(struct ku_response *resp)
{
    ssc(write(respfd, resp, sizeof(struct ku_response)));
    return 1;
}
