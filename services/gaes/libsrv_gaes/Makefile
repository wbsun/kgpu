
all: srv_gaes.cu
	nvcc -arch=sm_20 --shared --compiler-options '-fPIC' -o libsrv_gaes.so srv_gaes.cu

clean:
	rm -f *.o *.so
