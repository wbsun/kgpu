#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

union{
    char *buf;
    struct {
	void *startaddr;
        unsigned long size;
    } info;
} capp_comm_mem[2];


int main(){
    int fd = open("/proc/capp", O_RDWR);
    if (fd == -1){
	perror("open failed");
	return 0;
    }
    volatile int *m = malloc(1024*1024*1024); // huge mem

    if (mlock((void*)m, 1024*32) == -1) {
	perror("mlock failed");
	close(fd);
	return 0;
    }

    capp_comm_mem[0].info.startaddr = (void*)m;
    capp_comm_mem[0].info.size = 1024*4;
    
    capp_comm_mem[1].info.startaddr = (void*)((char*)m+1024*16);
    capp_comm_mem[1].info.size = 1024*4;

    if (write(fd, (void*)capp_comm_mem, sizeof(capp_comm_mem)) == -1){
	perror("write failed");
	close(fd);
	return 0;
    }

    close(fd);
    
    while(1) {
	sleep(1);
	printf("%d\n", *m);
    }

    
    return 0;
}
