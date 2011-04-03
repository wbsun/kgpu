
all: gecb_aes.h srv_gecb.cu
	nvcc -arch=sm_20 --shared --compiler-options '-fPIC' -o libsrv_gecb.so srv_gecb.cu

clean:
	rm -f *.o *.so
