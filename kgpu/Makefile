obj-m += kgpu_km.o

kgpu_km-objs := kgpu.o kgpu_kutils.o kgpu_log.o

km:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules

kgpu_log:
	gcc -O2 -D__KGPU__ -c kgpu_log.c -o kgpu_log_user.o
	ar -rcs kgpu_log.a kgpu_log_user.o

helper: kgpu_log
	gcc -O2 -D__KGPU__ -c helper.c
	gcc -O2 -D__KGPU__ -c service.c
	nvcc -O2 -D__KGPU__ -c -arch=sm_20 gpuops.cu
	nvcc -link -D__KGPU__ -O2 -arch=sm_20 service.o helper.o kgpu_log_user.o gpuops.o -o helper -ldl

all:	km helper

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
	rm -f helper
	rm -f kgpu_log.a
	rm -f *.o
