/**
 * AES CUDA test, modified from Engine_cudamrg
 *
 * @author Paolo Margara <paolo.margara@gmail.com>
 * @author Weibin Sun
 *
 * Copyright 2010 Paolo Margara
 *
 */

#include "aes_cuda.h"
#include <time.h>

#define MAX_FILE_SIZE (64*1024*1024)
#define TEST_TIMES 10

typedef struct {
    struct timespec start, stop;
} timer;

double ts2d_us(struct timespec *ts)
{
    double d = ts->tv_sec*1000000;
    d += ts->tv_nsec/1000.0;
    return d;
}

struct timespec get_timer_val(timer *tm)
{
    struct timespec temp;
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

struct timespec stop_timer(timer *tm)
{
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tm->stop);
    return get_timer_val(tm);
}


int main(int argc, char **argv) {
	uint8_t key128[AES_KEY_SIZE_128] = { 0x95, 0xA8, 0xEE, 0x8E, 0x89, 0x97,
			0x9B, 0x9E, 0xFD, 0xCB, 0xC6, 0xEB, 0x97, 0x97, 0x52, 0x8D };

	int i, bs;
	uint8_t *outs, *ins;
	AES_KEY *ak;
	timer tm;
	struct timespec ts;
	double tv;

	ak = (AES_KEY *) malloc(sizeof(AES_KEY));		
	AES_cuda_init(&ins, &outs, MAX_FILE_SIZE);

	AES_set_encrypt_key((unsigned char*) key128, 128, ak);
	AES_cuda_transfer_key(ak);
	
	printf("Encrypt:\n%12s %12s\n", "Size", "Time");	
		
	for (bs=16; bs < MAX_FILE_SIZE; bs*=2) {
		start_timer(&tm);
		for (i=0; i < TEST_TIMES; i++) {
			if (i%2==0)
				AES_cuda_encrypt(ins, outs, bs);
			else
				AES_cuda_encrypt(outs, ins, bs);
		}
		ts = stop_timer(&tm);
		tv = ts2d_us(&ts);
		printf("%12d %12.3f\n", bs, tv/TEST_TIMES);
	}
		
	AES_set_decrypt_key((unsigned char*) key128, 128, ak);
	AES_cuda_transfer_key(ak);
	
	printf("Decrypt:\n");
	
	for (bs=16; bs < MAX_FILE_SIZE; bs*=2) {
		start_timer(&tm);
		for (i=0; i < TEST_TIMES; i++) {
			if (i%2==0)
				AES_cuda_decrypt(ins, outs, bs);
			else
				AES_cuda_decrypt(outs, ins, bs);
		}
		ts = stop_timer(&tm);
		tv = ts2d_us(&ts);
		printf("%12d %12.3f\n", bs, tv/TEST_TIMES);
	}

	AES_cuda_finish();
	return 0;
}
