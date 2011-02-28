obj-m += kgpu.o

all: service.o helper.o
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules
	nvcc -c -arch=sm_20 gpuops.cu
	nvcc -link -arch=sm_20 service.o helper.o gpuops.o -o helper -ldl

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
	rm -f helper
