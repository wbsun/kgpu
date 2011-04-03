obj-m += kgpu.o

km:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules

helper:
	gcc -c helper.c
	gcc -c service.c
	nvcc -c -arch=sm_20 gpuops.cu
	nvcc -link -arch=sm_20 service.o helper.o gpuops.o -o helper -ldl

all:	km helper
	

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
	rm -f helper
