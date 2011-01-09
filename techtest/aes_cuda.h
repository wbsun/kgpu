/**
 * AES CUDA test, modified from Engine_cudamrg
 *
 * @author Paolo Margara <paolo.margara@gmail.com>
 * @author Weibin Sun
 *
 * Copyright 2010 Paolo Margara
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <memory.h>
#include <openssl/aes.h>
#include <openssl/engine.h>
#include <cuda.h>


#define MAX_THREAD		256
#define STATE_THREAD		4

#define AES_KEY_SIZE_128	16
#define AES_KEY_SIZE_192	24
#define AES_KEY_SIZE_256	32

#define OUTPUT_QUIET		0
#define OUTPUT_NORMAL		1
#define OUTPUT_VERBOSE		2

void ExpandKey (unsigned char *key, unsigned char *expkey);
void AES_cuda_transfer_key(AES_KEY *key);
void AES_cuda_encrypt(const unsigned char *in, unsigned char *out,size_t nbytes);
void AES_cuda_decrypt(const unsigned char *in, unsigned char *out,size_t nbytes);

void AES_cuda_init(unsigned char **ins, unsigned char **outs, int bufsize);
void AES_cuda_finish();
