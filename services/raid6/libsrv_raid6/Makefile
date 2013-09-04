
all: service

service: srv_raid6.cu
	nvcc -O3 -arch=sm_35 --shared -D__LIB_SERVICE__ --compiler-options '-fPIC' -o libsrv_raid6.so srv_raid6.cu
	$(if $(BUILD_DIR), cp libsrv_raid6.so $(BUILD_DIR)/ )

testlib: testlib.cu
	nvcc -arch=sm_35 -O3 --shared --compiler-options '-fPIC' -o libcudapq.so testlib.cu
	$(if $(BUILD_DIR), cp libcudapq.so $(BUILD_DIR)/ )

test: test.c testlib.cu
	nvcc -arch=sm_35 -O3 -c testlib.cu -o testlib.o
	nvcc -O3 testlib.o test.c -o test

test-old: test.c testlib.cu
	nvcc -arch=sm_20 -O3 -c testlib.cu -o testlib-old.o
	nvcc -O3 testlib-old.o test.c -o test-old

clean:
	rm -f test
	rm -f *.o *.so
