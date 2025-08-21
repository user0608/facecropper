package main

import (
	"errors"
	"io"
	"log/slog"
	"net/http"
	"slices"

	"github.com/gabriel-vasile/mimetype"
	"github.com/labstack/echo/v4"
	"github.com/user0608/facecropper"
)

var acceptedTypes = []string{"image/png", "image/jpeg"}

func NewFaceCropHandle() echo.HandlerFunc {
	return func(c echo.Context) error {
		content, err := io.ReadAll(c.Request().Body)
		if err != nil {
			if errors.Is(err, io.EOF) {
				return c.JSON(http.StatusBadRequest, echo.Map{"message": "el cuerpo de la solicitud está vacío"})
			}
			if errors.Is(err, io.ErrUnexpectedEOF) {
				return c.JSON(http.StatusBadRequest, echo.Map{"message": "el cuerpo de la solicitud está incompleto"})
			}
			return c.JSON(http.StatusInternalServerError, echo.Map{"message": "error al leer el cuerpo de la solicitud"})
		}
		mine := mimetype.Detect(content)
		if !slices.Contains(acceptedTypes, mine.String()) {
			return c.JSON(http.StatusInternalServerError, echo.Map{"message": "tipo invalido"})
		}
		fc, err := facecropper.New(
			"opencv_models/haarcascade_frontalface_default.xml",
			&facecropper.Options{
				ScoreThreshold: 0.5,
				NMSThreshold:   0.3,
				TopK:           5000,
				OutputWidth:    354,
				OutputHeight:   472,
				PaddingPct:     0.18,
			})
		if err != nil {
			slog.Error("creating facecopper", "error", err)
			return c.JSON(http.StatusInternalServerError, echo.Map{"message": "error creating facecopper"})
		}
		defer fc.Close()

		resultBytes, err := fc.Process(c.Request().Context(), content)
		if err != nil {
			slog.Error("processing image", "error", err)
			return c.JSON(http.StatusInternalServerError, echo.Map{"message": "error processing image"})
		}
		return c.Blob(http.StatusOK, mimetype.Detect(resultBytes).String(), resultBytes)
	}
}
