obj-m += kgpu.o

all:	kgpu helper

kgpu-objs := main.o kgpu_kutils.o kgpu_log.o

kgpu:
	make -C /lib/modules/$(shell uname -r)/build M=`pwd` modules
	$(if $(BUILD_DIR), cp kgpu.ko $(BUILD_DIR)/ ) 

kgpu_log:
	gcc -O2 -D__KGPU__ -c kgpu_log.c -o kgpu_log_user.o
	ar -rcs kgpu_log.a kgpu_log_user.o
	$(if $(BUILD_DIR), cp kgpu_log.a $(BUILD_DIR)/ )

helper: kgpu_log
	gcc -O2 -D__KGPU__ -c helper.c
	gcc -O2 -D__KGPU__ -c service.c
	nvcc -O2 -D__KGPU__ -c -arch=sm_20 gpuops.cu
	nvcc -link -D__KGPU__ -O2 -arch=sm_20 service.o helper.o kgpu_log_user.o gpuops.o -o helper -ldl
	$(if $(BUILD_DIR), cp helper $(BUILD_DIR)/ )

clean:
	make -C /lib/modules/$(shell uname -r)/build M=`pwd` clean
	rm -f helper
	rm -f kgpu_log.a
	rm -f *.o
