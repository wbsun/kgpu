obj-m += callgpu.o

all:
	cp ../../kgpu/Module.symvers ./
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
