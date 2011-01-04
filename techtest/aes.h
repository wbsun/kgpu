// advanced encryption standard
// author: karl malbrain, malbrain@yahoo.com

/*
This work, including the source code, documentation
and related data, is placed into the public domain.

The orginal author is Karl Malbrain.

THIS SOFTWARE IS PROVIDED AS-IS WITHOUT WARRANTY
OF ANY KIND, NOT EVEN THE IMPLIED WARRANTY OF
MERCHANTABILITY. THE AUTHOR OF THIS SOFTWARE,
ASSUMES _NO_ RESPONSIBILITY FOR ANY CONSEQUENCE
RESULTING FROM THE USE, MODIFICATION, OR
REDISTRIBUTION OF THIS SOFTWARE.
*/

#ifndef _AES_H_INC_
#define _AES_H_INC_


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>

typedef long long __int64;
typedef unsigned char uchar;


// AES only supports Nb=4
#define Nb 4			// number of columns in the state & expanded key
#define Nk 4			// number of columns in a key
#define Nr 10			// number of rounds in encryption


typedef struct {
	uchar Sbox[256];
	uchar InvSbox[256];
	uchar Xtime2Sbox[256];
	uchar Xtime3Sbox[256];
	uchar Xtime2[256];
	uchar Xtime9[256];
	uchar XtimeB[256];
	uchar XtimeD[256];
	uchar XtimeE[256];
} t_global;

extern uchar *devmem;
extern uchar *hostmem;

#define MAX_FILE_SIZE (64 * 1024 * 1024)

extern uchar Sbox[256];


#endif // _AES_H_INC_
