obj-m += capp.o

utest:
	nvcc -arch=sm_20 -c -g nskdev.cu
	nvcc -arch=sm_20 -c -g nsk.cu
	nvcc -arch=sm_20 -c -g nskutils.cu
	nvcc -arch=sm_20 -link -g nskdev.o nskutils.o nsk.o

all:
	rm build
	ln -s /lib/modules/$(shell uname -r)/build build
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules
	gcc capp_user.c -o capp_user

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
	rm *.o
	rm capp_user
