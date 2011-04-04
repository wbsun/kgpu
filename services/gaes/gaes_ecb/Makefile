obj-m += gaes_ecb.o

all:
	cp ../../../kgpu/Module.symvers ./
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
