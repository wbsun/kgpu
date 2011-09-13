#
# Makefile for the kernel software RAID 456
#
all:
ifeq ($(kv), 38)
	$(MAKE) -C 38 all BUILD_DIR=$(BUILD_DIR)
else
	$(MAKE) -C 39 all BUILD_DIR=$(BUILD_DIR)
endif

clean:
ifeq ($(kv), 38)
	$(MAKE) -C 38 clean BUILD_DIR=$(BUILD_DIR)
else
	$(MAKE) -C 39 clean BUILD_DIR=$(BUILD_DIR)
endif
