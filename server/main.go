package main

import (
	"context"
	"net/http"
	"os"
	"os/signal"
	"time"

	"github.com/labstack/echo/v4"
	"github.com/labstack/gommon/log"
)

func main() {

	e := echo.New()
	e.Logger.SetLevel(log.INFO)
	e.HideBanner = true

	e.GET("/", func(c echo.Context) error { return c.JSON(http.StatusOK, "OK") })
	e.POST("/facecrop", NewFaceCropHandle())

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	go func() {
		addr := os.Getenv("LISTEN_ADDR")
		if addr == "" {
			addr = ":1323"
		}
		if err := e.Start(":1323"); err != nil && err != http.ErrServerClosed {
			e.Logger.Fatal("shutting down the server")
		}
	}()

	<-ctx.Done()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := e.Shutdown(ctx); err != nil {
		e.Logger.Fatal(err)
	}
}
