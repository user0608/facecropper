FROM golang:1.24.3-alpine AS builder
RUN apk add --no-cache build-base pkgconfig opencv-dev git make
WORKDIR /app
COPY go.mod .
COPY go.sum .
RUN go mod download
COPY . .
ENV CGO_ENABLED=1
RUN go build -o build/face_cropper server/*.go
RUN chmod +x build/face_cropper

FROM alpine:3.22
RUN apk add --no-cache opencv-dev libstdc++
WORKDIR /app
COPY ./opencv_models/ opencv_models/
COPY  --from=builder /app/build/face_cropper .
CMD ["./face_cropper"]