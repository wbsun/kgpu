#
# Makefile for the kernel software RAID 456
#
raid456-y += raid5.o
obj-m += raid456.o

all:
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) modules
	$(if $(BUILD_DIR), cp raid456.ko $(BUILD_DIR)/ )

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) clean
