// cmd/main.go
package main

import (
	"context"
	"log"
	"os"
	"path"
	"path/filepath"

	"log/slog"

	"github.com/user0608/facecropper"
)

const inputDir = "input"
const outputDir = "output"

func main() {
	model := "../opencv_models/haarcascade_frontalface_default.xml"
	cropper, err := facecropper.New(model, &facecropper.Options{
		ScoreThreshold: 0.5,
		NMSThreshold:   0.3,
		TopK:           5000,
		OutputWidth:    354,
		OutputHeight:   472,
		PaddingPct:     0.18,
	})
	if err != nil {
		slog.Error("init", "err", err)
		return
	}
	defer cropper.Close()
	entries, err := os.ReadDir(inputDir)
	if err != nil {
		log.Fatalln(err)
	}
	if len(entries) > 0 {
		if err := os.MkdirAll(outputDir, 0755); err != nil {
			log.Fatalln(err)
		}
	}
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		imageBytes, err := os.ReadFile(path.Join(inputDir, entry.Name()))
		if err != nil {
			slog.Error("reading input", "error", err)
			continue
		}
		output, err := cropper.Process(context.Background(), imageBytes)
		if err != nil {
			slog.Error("response", "error", err)
		}
		if len(output) == 0 {
			slog.Info("empty response", "input", entry.Name())
			continue
		}
		outpath := filepath.Join(outputDir, entry.Name())
		if err := os.WriteFile(outpath, output, 0755); err != nil {
			log.Fatalln(err)
		}

	}
}
