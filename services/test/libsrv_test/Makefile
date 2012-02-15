
all:
	nvcc --shared --compiler-options '-fPIC' -o libsrv_test.so srv_test.cu
	$(if $(BUILD_DIR), cp libsrv_test.so $(BUILD_DIR)/ )

clean:
	rm -f *.o *.so
