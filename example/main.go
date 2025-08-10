// cmd/main.go
package main

import (
	"context"
	"fmt"
	"os"
	"time"

	"log/slog"

	"github.com/user0608/facecropper"
)

func main() {

	model := "face_detection_yunet_2023mar.onnx"
	c, err := facecropper.New(model, &facecropper.Options{
		ScoreThreshold: 0.7,
		NMSThreshold:   0.3,
		TopK:           5000,
		OutputWidth:    480,
		OutputHeight:   600,
		MarginScaleW:   1.6,
		MarginScaleH:   2.0,
		AlignByEyes:    false,
	})
	if err != nil {
		slog.Error("init", "err", err)
		return
	}
	defer c.Close()

	in, err := os.ReadFile("input.jpg")
	if err != nil {
		slog.Error("leer input", "err", err)
		return
	}
	start := time.Now()
	out, err := c.Process(context.Background(), in)
	if err != nil {
		slog.Error("procesar", "err", err)
		return
	}
	fmt.Println("duration (s):", time.Since(start).Seconds())
	if err := os.WriteFile("output_face.jpg", out, 0o644); err != nil {
		slog.Error("guardar out", "err", err)
		return
	}
}
