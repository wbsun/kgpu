#ifndef __HOST_H__
#define __HOST_H__

typedef struct {
    void *addr;
    unsigned int size; // no more than 4GB
} nsk_buf_info_t;

#define NSK_PROCFS_FILE "/proc/nsk"

#endif
