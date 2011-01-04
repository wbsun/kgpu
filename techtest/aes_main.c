// ECE 1724 -- GPU Programming
// Winter 2009
//
// Michael Kipper
// Joshua Slavkin
// Dmitry Denisenko
//
// AES implementation on CUDA and how it compares to CPU

// This is the main driver file. Both the CPU and GPU implementations of
// AES are taken through:
//		- correctness verification (on a single known string)
//		- reversibility verification (on randomly-generated strings)
//		- benchmarking (on fixed strings)



// GLOBAL CONSTANTS
// --------------------------------------
// Name of file containing plaintext for benchmarking
static const char *benchmarking_filename = "benchmarking_input.txt";
// How many times to run encryption and decryption during the benchmarking phase
static const int NUM_BENCHMARKING_ITERATIONS = 20;



// INCLUDES
// --------------------------------------
#include <time.h>
#include <math.h>
#include "aes.h"



// DEFINES
// --------------------------------------
#define my_assert(x) if(!(x)) { printf ("\nAssert " #x " failed at %s:%d\n\n", __FILE__, __LINE__); exit(1); }

// FORWARD DECLARATIONS
// --------------------------------------
static double benchmark (void);

extern void gpu_init(int argc, char** argv);
extern double gpu_encrypt_string(uchar* in, int in_len, uchar* key, uchar* out);
extern double gpu_decrypt_string(uchar* in, int in_len, uchar* key, uchar* out);


// FUNCTION DEFINITIONS
// --------------------------------------
int main (int argc, char *argv[])
{
	gpu_init(argc, argv);

	double elapsed;
	printf ("\tBenchmarking....");
	elapsed = benchmark ();
	printf ("Done testing.\n\n");

	return 0;
}


static double benchmark (void)
// Benchmark performance of the cipher functions.
// As plaintext, using 1.5 MB text version of Jame Joyce's Ulysses.
// Taken from Project Gutenberg (www.gutenberg.org). 
// The text is stored in local file called benchmarking_input.txt.
{
	FILE *f_in;
	const int START_VALUE = 16;
	
	uchar *in = hostmem;
	uchar *out = hostmem+MAX_FILE_SIZE;
	uchar *tmp = hostmem+2*MAX_FILE_SIZE;

	// Base key. Will be perturbed on every test
	unsigned char key[] = {0x00, 0x01, 0x02, 0x03, 0x05, 0x06, 0x07, 0x08, 0x0A, 0x0B, 0x0C, 0x0D, 0x0F, 0x10, 0x11, 0x12};
	
	int i, size, num_read;
	double base, elapsed;

	// Read the file into memory
	f_in = fopen (benchmarking_filename, "rb");
	my_assert (f_in != NULL);
	num_read = fread (in, sizeof(char), MAX_FILE_SIZE, f_in);
	fclose (f_in);

	// Remember how many bytes were read, and round it up to a multiple of 16
	num_read = (int)(ceil (num_read / 16.0) * 16);
	my_assert (num_read < MAX_FILE_SIZE);

	//printf("\n%16s%16s%16s\n", "Size", "Total (us)", "Unit (us)");
	printf("\n%16s%16s\n", "Size", "Unit (us)");
	for ( size = START_VALUE; size <= MAX_FILE_SIZE; size *= 2 )
	{
		elapsed = 0.0;

		// Encrypt it 100 times using slightly different keys
		for ( i = 0; i < NUM_BENCHMARKING_ITERATIONS; ++i )
		{
			// Change the key on every iteration so the loop is not optimized away
			key[3] = i;
			elapsed += gpu_encrypt_string (in,  size, key, tmp);
			elapsed += gpu_decrypt_string (tmp, size, key, out);

			//my_assert( memcmp (in, out, size) == 0 );
		}

		if ( size == START_VALUE ) base = elapsed;

		// Report the average time for a single operation
		//printf ("%16i%16.3f%16.3f\n", size, elapsed, elapsed / NUM_BENCHMARKING_ITERATIONS / 2.0 );
		printf ("%16i%16.3f\n", size, elapsed / NUM_BENCHMARKING_ITERATIONS / 2.0 );
	}

	return elapsed;
}
