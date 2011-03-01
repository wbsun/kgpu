obj-m += kgpu.o

km:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules

helper:
	gcc -c -g helper.c
	gcc -c -g service.c
	nvcc -c -g -arch=sm_20 gpuops.cu
	nvcc -link -g -arch=sm_20 service.o helper.o gpuops.o -o helper -ldl

all:	km helper
	

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
	rm -f helper
