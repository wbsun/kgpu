# fill this later
SUBDIRS = kgpu services

all: $(SUBDIRS)


.PHONY: $(SUBDIRS)

$(SUBDIRS): mkbuilddir
	$(MAKE) -C $@ $(TARGET) BUILD_DIR=`pwd`/build

mkbuilddir:
	mkdir -p build

services: kgpu

distclean:
	$(MAKE) all TARGET=clean

clean: distclean
	rm -rf build
