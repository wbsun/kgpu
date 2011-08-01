obj-m += gpq.o

all:
	cp ../../../kgpu/Module.symvers ./
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) modules
	$(if $(BUILD_DIR), cp gpq.ko $(BUILD_DIR)/ ) 

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) clean
