
all: gaes_ecb.h srv_gaes_ecb.cu
	nvcc -arch=sm_20 --shared --compiler-options '-fPIC' -o libsrv_gaes_ecb.so srv_gaes_ecb.cu

clean:
	rm -f *.o *.so
