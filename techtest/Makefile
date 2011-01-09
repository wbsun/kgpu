obj-m += taesk.o

taesk:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules

aescuda: aes_cuda_test.c aes_cuda.cu aes_cuda.h
	nvcc -lcrypto aes_cuda_test.c aes_cuda.cu -o aescuda

utest:
	nvcc -arch=sm_20 -c -g nskdev.cu
	nvcc -arch=sm_20 -c -g nsk.cu
	nvcc -arch=sm_20 -c -g nskutils.cu
	nvcc -arch=sm_20 -link -g nskdev.o nskutils.o nsk.o

capp:
	rm build
	ln -s /lib/modules/$(shell uname -r)/build build
	obj-m += capp.o
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules
	gcc capp_user.c -o capp_user

clean:
	obj-m += capp.o
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
	rm *.o
	rm capp_user
