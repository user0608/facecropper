package facecropper

import (
	"context"
	"errors"
	"image"
	"log/slog"
	"math"
	"sync"

	"gocv.io/x/gocv"
)

type Options struct {
	ScoreThreshold float32
	NMSThreshold   float32
	TopK           int
	OutputWidth    int
	OutputHeight   int
	PaddingPct     float64
}

func DefaultOptions() Options {
	return Options{
		ScoreThreshold: 0.6,
		NMSThreshold:   0.3,
		TopK:           5000,
		OutputWidth:    354,
		OutputHeight:   472,
		PaddingPct:     0.15,
	}
}

type Cropper struct {
	opts Options
	mu   sync.Mutex
	cls  gocv.CascadeClassifier
}

func New(modelPath string, opts *Options) (*Cropper, error) {
	if modelPath == "" {
		slog.Error("ruta de modelo vacía")
		return nil, errors.New("modelo requerido")
	}
	if opts == nil {
		def := DefaultOptions()
		opts = &def
	}
	cls := gocv.NewCascadeClassifier()
	if !cls.Load(modelPath) {
		slog.Error("no se pudo cargar haarcascade", "path", modelPath)
		return nil, errors.New("carga de haarcascade falló")
	}
	return &Cropper{opts: *opts, cls: cls}, nil
}

func (c *Cropper) Close() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.cls.Close()
}

func clamp(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func (c *Cropper) Process(ctx context.Context, imgBytes []byte) ([]byte, error) {
	if len(imgBytes) == 0 {
		return nil, errors.New("imagen vacía")
	}
	img, err := gocv.IMDecode(imgBytes, gocv.IMReadColor)
	if err != nil {
		return nil, err
	}
	if img.Empty() {
		return nil, errors.New("decode vacío")
	}
	defer img.Close()

	W, H := img.Cols(), img.Rows()
	if W == 0 || H == 0 {
		return nil, errors.New("dimensiones inválidas")
	}

	gray := gocv.NewMat()
	defer gray.Close()
	gocv.CvtColor(img, &gray, gocv.ColorBGRToGray)

	c.mu.Lock()
	rects := c.cls.DetectMultiScale(gray)
	c.mu.Unlock()
	if len(rects) == 0 {
		return nil, errors.New("sin rostro")
	}

	best := rects[0]
	bestArea := best.Dx() * best.Dy()
	for i := 1; i < len(rects); i++ {
		a := rects[i].Dx() * rects[i].Dy()
		if a > bestArea {
			best = rects[i]
			bestArea = a
		}
	}

	p := c.opts.PaddingPct
	if p < 0 {
		p = 0
	}
	bw, bh := best.Dx(), best.Dy()
	padX := int(math.Round(float64(bw) * p))
	padY := int(math.Round(float64(bh) * p))

	x1 := clamp(best.Min.X-padX, 0, W)
	y1 := clamp(best.Min.Y-padY, 0, H)
	x2 := clamp(best.Max.X+padX, 0, W)
	y2 := clamp(best.Max.Y+padY, 0, H)

	targetAR := float64(c.opts.OutputWidth) / float64(c.opts.OutputHeight)
	cw := x2 - x1
	ch := y2 - y1
	cx := x1 + cw/2
	cy := y1 + ch/2
	curAR := float64(cw) / float64(ch)

	if curAR > targetAR {
		newH := int(math.Round(float64(cw) / targetAR))
		y1 = cy - newH/2
		y2 = y1 + newH
	} else if curAR < targetAR {
		newW := int(math.Round(float64(ch) * targetAR))
		x1 = cx - newW/2
		x2 = x1 + newW
	}

	if x1 < 0 {
		shift := -x1
		x1 += shift
		x2 += shift
	}
	if y1 < 0 {
		shift := -y1
		y1 += shift
		y2 += shift
	}
	if x2 > W {
		shift := x2 - W
		x1 -= shift
		x2 -= shift
	}
	if y2 > H {
		shift := y2 - H
		y1 -= shift
		y2 -= shift
	}

	x1 = clamp(x1, 0, W)
	y1 = clamp(y1, 0, H)
	x2 = clamp(x2, 0, W)
	y2 = clamp(y2, 0, H)
	if x2-x1 <= 1 || y2-y1 <= 1 {
		return nil, errors.New("recorte inválido")
	}

	roi := img.Region(image.Rect(x1, y1, x2, y2))
	defer roi.Close()

	out := gocv.NewMat()
	defer out.Close()
	gocv.Resize(roi, &out, image.Pt(c.opts.OutputWidth, c.opts.OutputHeight), 0, 0, gocv.InterpolationLanczos4)

	buf, err := gocv.IMEncode(gocv.JPEGFileExt, out)
	if err != nil {
		return nil, err
	}
	bytes := make([]byte, len(buf.GetBytes()))
	copy(bytes, buf.GetBytes())
	buf.Close()
	return bytes, nil
}
