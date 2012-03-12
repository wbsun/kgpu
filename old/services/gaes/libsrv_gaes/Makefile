
all: srv_gaes.cu
	nvcc -arch=sm_20 -O2 --shared --compiler-options '-fPIC' -o libsrv_gaes.so srv_gaes.cu
	$(if $(BUILD_DIR), cp libsrv_gaes.so $(BUILD_DIR)/ ) 

clean:
	rm -f *.o *.so
