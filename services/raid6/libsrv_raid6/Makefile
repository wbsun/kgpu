
all: service test

service: srv_raid6.cu
	nvcc -arch=sm_20 --shared -D__LIB_SERVICE__ --compiler-options '-fPIC' -o libsrv_raid6.so srv_raid6.cu
	$(if $(BUILD_DIR), cp libsrv_raid6.so $(BUILD_DIR)/ )

test: test.cu
	nvcc -arch=sm_20 -o testpq test.cu
	$(if $(BUILD_DIR), cp testpq $(BUILD_DIR)/ )

clean:
	rm -f *.o *.so
