obj-m += nskk.o

nsku:
	nvcc -arch=sm_20 dev.cu devutils.cu host.cu hostutils.cu -o nsku

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
	rm *.o
	rm nsku
