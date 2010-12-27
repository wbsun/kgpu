#ifndef __DEV_UTILS_H__
#define __DEV_UTILS_H__

#include <cuda.h>

void _csc(cudaError_t e, const char *file, int line);

#define csc(...) _csc(__VA_ARGS__, __FILE__, __LINE__)

enum mem_mode_t {
    PINNED,
    PAGEABLE,
    MAPPED,
    WC,
};

void alloc_hdmem(void **pph, void **ppd, unsigned int size, mem_mode_t memMode);
void free_hdmem(void **pph, void **ppd, mem_mode_t memMode);

#define ALLOC_HDMEM(pph, ppd, sz, mm)					\
    alloc_hdmem((void**)(pph), (void**)(ppd), (unsigned int)(sz), mm)

#define FREE_HDMEM(pph, ppd, mm) free_hdmem((void**)(pph), (void**)(ppd), mm)

extern cudaStream_t ss[3];

void init_hd_buffers();
void init_hd_streams();

volatile void* get_next_device_mem(int user);
void put_device_mem(volatile void* devmem);

#define h2d_cpy_a(dst, src, sz, stream) cudaMemcpyAsync((void*)(dst), (void*)(src), (sz), cudaMemcpyHostToDevice, (stream))
#define d2h_cpy_a(dst, src, sz, stream) cudaMemcpyAsync((void*)(dst), (void*)(src), (sz), cudaMemcpyDeviceToHost, (stream))
#define h2d_sbl_a(dst, src, sz, stream) cudaMemcpyToSymbolAsync(dst, (void*)(src), (sz), 0, cudaMemcpyHostToDevice, (stream))
#define d2h_sbl_a(dst, src, sz, stream) cudaMemcpyFromSymbolAsync((void*)(dst), src, (sz), 0, cudaMemcpyDeviceToHost, (stream))

#endif
