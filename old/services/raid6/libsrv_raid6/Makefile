
all: service

service: srv_raid6.cu
	nvcc -O2 -arch=sm_20 --shared -D__LIB_SERVICE__ --compiler-options '-fPIC' -o libsrv_raid6.so srv_raid6.cu
	$(if $(BUILD_DIR), cp libsrv_raid6.so $(BUILD_DIR)/ )

testlib: testlib.cu
	nvcc -arch=sm_20 -O3 --shared --compiler-options '-fPIC' -o libcudapq.so testlib.cu
	$(if $(BUILD_DIR), cp libcudapq.so $(BUILD_DIR)/ )

test: test.c testlib
	gcc -O2 -Wl,-rpath=./ -L. -lcudapq test.c -o test

clean:
	rm -f test
	rm -f *.o *.so
