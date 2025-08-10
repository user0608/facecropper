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

	OutputWidth  int
	OutputHeight int

	MarginScaleW float64
	MarginScaleH float64

	AlignByEyes bool
}

func DefaultOptions() Options {
	return Options{
		ScoreThreshold: 0.7,
		NMSThreshold:   0.3,
		TopK:           5000,
		OutputWidth:    480,
		OutputHeight:   600,
		MarginScaleW:   1.6,
		MarginScaleH:   2.0,
		AlignByEyes:    true,
	}
}

type Cropper struct {
	opts Options
	mu   sync.Mutex
	det  gocv.FaceDetectorYN
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
	det := gocv.NewFaceDetectorYN(modelPath, "", image.Pt(320, 320))
	det.SetScoreThreshold(opts.ScoreThreshold)
	det.SetNMSThreshold(opts.NMSThreshold)
	det.SetTopK(opts.TopK)

	return &Cropper{opts: *opts, det: det}, nil
}

func (c *Cropper) Close() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.det.Close()
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

// Process decodes an image (JPEG, PNG, BMP, WebP, TIFF, etc.), detects the largest face,
// optionally aligns it by eyes, and crops it to a 4:5 ratio.
// Returns:
//   - cropped JPEG face image
//   - debug JPEG image with detection and crop boxes drawn
//   - error if decoding, detection, or encoding fails
func (c *Cropper) Process(ctx context.Context, imgBytes []byte) ([]byte, error) {
	if len(imgBytes) == 0 {
		slog.Error("imagen vacía")
		return nil, errors.New("imagen vacía")
	}
	img, err := gocv.IMDecode(imgBytes, gocv.IMReadColor)
	if err != nil {
		slog.Error("IMDecode error", "err", err)
		return nil, err
	}
	if img.Empty() {
		slog.Error("IMDecode produjo Mat vacío")
		return nil, errors.New("decode vacío")
	}
	defer img.Close()

	w, h := img.Cols(), img.Rows()
	if w == 0 || h == 0 {
		slog.Error("dimensiones inválidas", "w", w, "h", h)
		return nil, errors.New("dimensiones inválidas")
	}

	aligned := img
	freeAligned := false

	c.mu.Lock()
	c.det.SetInputSize(image.Pt(w, h))
	faces := gocv.NewMat()
	c.det.Detect(img, &faces)
	c.mu.Unlock()
	defer faces.Close()

	if faces.Empty() || faces.Rows() == 0 {
		slog.Error("sin rostro")
		return nil, errors.New("sin rostro")
	}

	bestIdx := 0
	bestArea := float32(0)
	for i := 0; i < faces.Rows(); i++ {
		area := faces.GetFloatAt(i, 2) * faces.GetFloatAt(i, 3)
		if area > bestArea {
			bestArea = area
			bestIdx = i
		}
	}

	x := faces.GetFloatAt(bestIdx, 0)
	y := faces.GetFloatAt(bestIdx, 1)
	wb := faces.GetFloatAt(bestIdx, 2)
	hb := faces.GetFloatAt(bestIdx, 3)
	lx := faces.GetFloatAt(bestIdx, 4)
	ly := faces.GetFloatAt(bestIdx, 5)
	rx := faces.GetFloatAt(bestIdx, 6)
	ry := faces.GetFloatAt(bestIdx, 7)

	if c.opts.AlignByEyes {
		angle := math.Atan2(float64(ry-ly), float64(rx-lx)) * 180.0 / math.Pi
		center := image.Pt(int(x+wb/2), int(y+hb/2))
		M := gocv.GetRotationMatrix2D(center, -angle, 1.0)
		tmp := gocv.NewMat()
		gocv.WarpAffine(img, &tmp, M, image.Pt(w, h))
		aligned = tmp
		freeAligned = true

		faces2 := gocv.NewMat()
		defer faces2.Close()
		c.mu.Lock()
		c.det.SetInputSize(image.Pt(w, h))
		c.det.Detect(aligned, &faces2)
		c.mu.Unlock()

		if faces2.Empty() || faces2.Rows() == 0 {
			if freeAligned {
				aligned.Close()
			}
			slog.Error("sin rostro tras alineación")
			return nil, errors.New("sin rostro tras alineación")
		}

		bestIdx = 0
		bestArea = 0
		for i := 0; i < faces2.Rows(); i++ {
			area := faces2.GetFloatAt(i, 2) * faces2.GetFloatAt(i, 3)
			if area > bestArea {
				bestArea = area
				bestIdx = i
			}
		}
		x = faces2.GetFloatAt(bestIdx, 0)
		y = faces2.GetFloatAt(bestIdx, 1)
		wb = faces2.GetFloatAt(bestIdx, 2)
		hb = faces2.GetFloatAt(bestIdx, 3)
	}

	cx := int(x + wb/2)
	cy := int(y + hb*0.45)
	bw := float64(wb) * c.opts.MarginScaleW
	bh := float64(hb) * c.opts.MarginScaleH
	target := float64(c.opts.OutputWidth) / float64(c.opts.OutputHeight)
	if bw/bh > target {
		bh = bw / target
	} else {
		bw = bh * target
	}

	halfW := int(math.Round(bw / 2.0))
	halfH := int(math.Round(bh / 2.0))
	x1 := clamp(cx-halfW, 0, w)
	y1 := clamp(cy-halfH, 0, h)
	x2 := clamp(cx+halfW, 0, w)
	y2 := clamp(cy+halfH, 0, h)
	if x2-x1 < 2*halfW {
		if x1 == 0 {
			x2 = clamp(2*halfW, 0, w)
		} else if x2 == w {
			x1 = clamp(w-2*halfW, 0, w)
		}
	}
	if y2-y1 < 2*halfH {
		if y1 == 0 {
			y2 = clamp(2*halfH, 0, h)
		} else if y2 == h {
			y1 = clamp(h-2*halfH, 0, h)
		}
	}
	if x2 <= x1 || y2 <= y1 {
		if freeAligned {
			aligned.Close()
		}
		slog.Error("rect recorte inválido", "x1", x1, "y1", y1, "x2", x2, "y2", y2)
		return nil, errors.New("recorte inválido")
	}

	roi := aligned.Region(image.Rect(x1, y1, x2, y2))
	defer roi.Close()

	out := gocv.NewMat()
	defer out.Close()
	gocv.Resize(roi, &out, image.Pt(c.opts.OutputWidth, c.opts.OutputHeight), 0, 0, gocv.InterpolationLanczos4)

	// --- encode OUT ---
	outBuf, err := gocv.IMEncode(gocv.JPEGFileExt, out)
	if err != nil {
		slog.Error("IMEncode(out) error", "err", err)
		if freeAligned {
			aligned.Close()
		}
		return nil, err
	}
	tmpOut := outBuf.GetBytes()
	outBytes := make([]byte, len(tmpOut))
	copy(outBytes, tmpOut)
	outBuf.Close()
	return outBytes, nil
}
