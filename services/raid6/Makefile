SUBDIRS = libsrv_raid6 gpq grecov raid456

all: $(SUBDIRS)

.PHONY: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ $(TARGET) kv=$(kv) BUILD_DIR=$(BUILD_DIR)

clean:
	$(MAKE) all TARGET=clean