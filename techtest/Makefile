obj-m += capp.o

all:
	rm build
	ln -s /lib/modules/$(shell uname -r)/build build
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules
	gcc capp_user.c -o capp_user

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
	rm *.o
	rm capp_user
