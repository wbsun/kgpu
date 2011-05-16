#
# Makefile for the Linux 2.6 dm-crypt
#

obj-m += dm-crypt.o

all:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
