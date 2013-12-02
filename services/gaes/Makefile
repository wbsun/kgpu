# SUBDIRS = callaes ecryptfs gaes_ctr libsrv_gaes dm-crypt gaes_ecb gaes_xts
SUBDIRS = callaes gaes_ctr libsrv_gaes gaes_ecb

all: $(SUBDIRS)

.PHONY: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ $(TARGET) BUILD_DIR=$(BUILD_DIR)

clean:
	$(MAKE) all TARGET=clean