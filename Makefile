version := $(shell cat version)

run:
	@go run server/*.go

CFLAGS=$(shell pkg-config --cflags opencv4)
LDFLAGS=$(shell pkg-config --libs opencv4)

.PHONY: build
build: clean
	CGO_ENABLED=1 \
	CGO_CFLAGS="$(CFLAGS)" \
	CGO_LDFLAGS="$(LDFLAGS)" \
	go build -o build/face_cropper server/*.go

clean:
	@rm -rf build

image:
	@docker build --network host -t face_cropper:$(version) .