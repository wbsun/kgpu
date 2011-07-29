
all: srv_raid6.cu
	nvcc -arch=sm_20 --shared --compiler-options '-fPIC' -o libsrv_raid6.so srv_raid6.cu
	$(if $(BUILD_DIR), cp libsrv_raid6.so $(BUILD_DIR)/ ) 

clean:
	rm -f *.o *.so
