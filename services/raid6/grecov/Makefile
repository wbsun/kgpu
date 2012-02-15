obj-m += async_raid6_recov.o
obj-m += raid6test.o

all:
	cp ../../../kgpu/Module.symvers ./
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) modules
	$(if $(BUILD_DIR), cp async_raid6_recov.ko $(BUILD_DIR)/ ) 
	$(if $(BUILD_DIR), cp raid6test.ko $(BUILD_DIR)/ ) 

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) clean
