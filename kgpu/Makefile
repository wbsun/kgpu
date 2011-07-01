obj-m += kgpu_km.o

kgpu_km-objs := kgpu.o kgpu_kutils.o kgpu_common.o

km:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules

helper:
	gcc -O2 -c helper.c
	gcc -O2 -c service.c
	gcc -O2 -c kgpu_common.c -o kgpu_common_user.o
	nvcc -O2 -c -arch=sm_20 gpuops.cu
	nvcc -link -O2 -arch=sm_20 service.o helper.o kgpu_common_user.o gpuops.o -o helper -ldl

all:	km helper


clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
	rm -f helper
