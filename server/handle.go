package main

import (
	"errors"
	"io"
	"net/http"
	"slices"

	"github.com/gabriel-vasile/mimetype"
	"github.com/labstack/echo/v4"
	"github.com/user0608/facecropper"
	"github.com/user0608/goones/answer"
	"github.com/user0608/goones/errs"
)

var acceptedTypes = []string{"image/png", "image/jpeg"}

const model = "opencv_models/haarcascade_frontalface_default.xml"

var defaultOptions = facecropper.Options{
	ScoreThreshold: 0.5,
	NMSThreshold:   0.3,
	TopK:           5000,
	OutputWidth:    354,
	OutputHeight:   472,
	PaddingPct:     0.18,
}

func NewFaceCropHandle() echo.HandlerFunc {
	return func(c echo.Context) error {
		content, err := io.ReadAll(c.Request().Body)
		if err != nil {
			if errors.Is(err, io.EOF) {
				return answer.Err(c, errs.BadRequestDirect("la foto enviada en la solicitud está vacía"))
			}
			if errors.Is(err, io.ErrUnexpectedEOF) {
				return answer.Err(c, errs.BadRequestDirect("la foto enviada está incompleta o dañada"))
			}
			return answer.Err(c, errs.InternalErrorDirect("no se pudo leer el cuerpo de la solicitud"))
		}
		mine := mimetype.Detect(content)
		if !slices.Contains(acceptedTypes, mine.String()) {
			return answer.Err(c, errs.BadRequestDirect("solo se aceptan imágenes en formato PNG o JPG"))
		}
		fc, err := facecropper.New(model, &defaultOptions)
		if err != nil {
			return answer.Err(c, err)

		}
		defer fc.Close()

		resultBytes, err := fc.Process(c.Request().Context(), content)
		if err != nil {
			return answer.Err(c, err)
		}
		return c.Blob(http.StatusOK, mimetype.Detect(resultBytes).String(), resultBytes)
	}
}
