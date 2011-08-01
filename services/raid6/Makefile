SUBDIRS = libsrv_raid6 gpq

all: $(SUBDIRS)

.PHONY: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ $(TARGET) BUILD_DIR=$(BUILD_DIR)

clean:
	$(MAKE) all TARGET=clean