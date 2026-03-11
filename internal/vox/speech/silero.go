// Package speech - Silero VAD implementation.
//
// Silero VAD is a lightweight neural network for voice activity detection.
// Unlike energy-based VAD, it detects actual speech, not just loudness.
//
// Model: silero_vad.onnx (~1.8MB)
// Input: 512 samples (32ms at 16kHz) per frame
// Output: Speech probability 0.0-1.0
//
// Requirements:
// - ONNX Runtime (brew install onnxruntime on macOS)
// - Silero model (make setup-silero)
package speech

import (
	"fmt"
	"os"
	"runtime"
	"sync"
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

func init() {
	// Set ONNX Runtime shared library path based on platform
	var libPath string
	switch runtime.GOOS {
	case "darwin":
		paths := []string{
			"/opt/homebrew/lib/libonnxruntime.dylib", // ARM64 Homebrew
			"/usr/local/lib/libonnxruntime.dylib",    // Intel Homebrew
		}
		for _, p := range paths {
			if _, err := os.Stat(p); err == nil {
				libPath = p
				break
			}
		}
	case "linux":
		libPath = "/usr/lib/libonnxruntime.so"
	}

	if libPath != "" {
		ort.SetSharedLibraryPath(libPath)
	}
}

const (
	// SileroWindowSamples is the number of samples per VAD frame.
	// Silero expects exactly 512 samples (32ms at 16kHz).
	SileroWindowSamples = 512

	// SileroSampleRate is the required sample rate.
	SileroSampleRate = 16000

	// SileroHiddenSize is the LSTM hidden state dimension.
	SileroHiddenSize = 64

	// SileroNumLayers is the number of LSTM layers (2 for h/c).
	SileroNumLayers = 2
)

// SileroDetector implements SpeechDetector using Silero VAD.
type SileroDetector struct {
	session *ort.DynamicAdvancedSession
	config  DetectorConfig

	// LSTM hidden states (persisted across calls)
	h []float32
	c []float32

	// Buffering state
	audioBuffer []byte     // Accumulates PCM16 audio
	speechBuf   []byte     // Accumulates speech audio
	speaking    bool       // Currently in speech segment
	speechStart time.Time  // When current speech started
	silenceMs   int        // Consecutive silence duration in ms
	streamStart time.Time  // When the stream started

	mu sync.Mutex
}

// Compile-time check that SileroDetector implements SpeechDetector.
var _ SpeechDetector = (*SileroDetector)(nil)

// NewSileroDetector creates a Silero VAD-based speech detector.
func NewSileroDetector(modelPath string, cfg DetectorConfig) (*SileroDetector, error) {
	// Check model exists
	if _, err := os.Stat(modelPath); err != nil {
		return nil, fmt.Errorf("silero model not found: %s (run 'make setup-silero')", modelPath)
	}

	// Initialize ONNX Runtime
	if err := initONNXRuntime(); err != nil {
		return nil, fmt.Errorf("init onnx runtime: %w", err)
	}

	// Create session options
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("create session options: %w", err)
	}
	defer opts.Destroy()

	// Create dynamic session for Silero VAD
	// Input names: input, sr, h, c
	// Output names: output, hn, cn
	session, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{"input", "sr", "h", "c"},
		[]string{"output", "hn", "cn"},
		opts,
	)
	if err != nil {
		return nil, fmt.Errorf("create silero session: %w", err)
	}

	// Initialize hidden states to zeros
	stateSize := SileroNumLayers * 1 * SileroHiddenSize // [2, 1, 64]
	h := make([]float32, stateSize)
	c := make([]float32, stateSize)

	return &SileroDetector{
		session:     session,
		config:      cfg,
		h:           h,
		c:           c,
		audioBuffer: make([]byte, 0, SileroWindowSamples*4), // Pre-alloc
		speechBuf:   make([]byte, 0, cfg.SampleRate*2*30),   // Up to 30s
		streamStart: time.Now(),
	}, nil
}

// Process feeds an audio chunk to the detector and returns complete speech segments.
func (d *SileroDetector) Process(chunk []byte) []SpeechSegment {
	d.mu.Lock()
	defer d.mu.Unlock()

	// Add chunk to buffer
	d.audioBuffer = append(d.audioBuffer, chunk...)

	var segments []SpeechSegment

	// Process complete windows (512 samples = 1024 bytes)
	windowBytes := SileroWindowSamples * 2
	for len(d.audioBuffer) >= windowBytes {
		window := d.audioBuffer[:windowBytes]
		d.audioBuffer = d.audioBuffer[windowBytes:]

		// Run VAD on this window
		prob, err := d.runVAD(window)
		if err != nil {
			// Log error but continue
			continue
		}

		isSpeech := prob >= d.config.SpeechThreshold

		// State machine: detect speech start/end
		if isSpeech {
			if !d.speaking {
				// Speech started
				d.speaking = true
				d.speechStart = time.Now()
				d.speechBuf = d.speechBuf[:0]
				d.silenceMs = 0
			}
			d.speechBuf = append(d.speechBuf, window...)
			d.silenceMs = 0

			// Check max duration
			speechMs := int(time.Since(d.speechStart).Milliseconds())
			if speechMs >= d.config.MaxSpeechMs {
				// Force segment end
				if seg := d.emitSegment(); seg != nil {
					segments = append(segments, *seg)
				}
			}
		} else {
			if d.speaking {
				// Still in speech segment, accumulate silence
				d.speechBuf = append(d.speechBuf, window...)
				d.silenceMs += 32 // 512 samples at 16kHz = 32ms

				if d.silenceMs >= d.config.MinSilenceMs {
					// Enough silence to end segment
					if seg := d.emitSegment(); seg != nil {
						segments = append(segments, *seg)
					}
				}
			}
			// If not speaking and silence, just discard
		}
	}

	return segments
}

// Flush forces emission of any buffered speech.
func (d *SileroDetector) Flush() []SpeechSegment {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.speaking {
		if seg := d.emitSegment(); seg != nil {
			return []SpeechSegment{*seg}
		}
	}
	return nil
}

// Reset clears all internal state.
func (d *SileroDetector) Reset() {
	d.mu.Lock()
	defer d.mu.Unlock()

	// Reset hidden states
	for i := range d.h {
		d.h[i] = 0
	}
	for i := range d.c {
		d.c[i] = 0
	}

	// Reset buffers
	d.audioBuffer = d.audioBuffer[:0]
	d.speechBuf = d.speechBuf[:0]
	d.speaking = false
	d.silenceMs = 0
	d.streamStart = time.Now()
}

// Close releases ONNX resources.
func (d *SileroDetector) Close() error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.session != nil {
		if err := d.session.Destroy(); err != nil {
			return fmt.Errorf("destroy silero session: %w", err)
		}
		d.session = nil
	}
	return nil
}

// emitSegment creates a segment from current speech buffer.
// Caller must hold mutex.
func (d *SileroDetector) emitSegment() *SpeechSegment {
	speechMs := int(time.Since(d.speechStart).Milliseconds())

	// Check minimum duration
	if speechMs < d.config.MinSpeechMs {
		// Too short, discard
		d.speaking = false
		d.speechBuf = d.speechBuf[:0]
		d.silenceMs = 0
		return nil
	}

	// Trim trailing silence from buffer
	silenceBytes := d.silenceMs * d.config.SampleRate * 2 / 1000
	audioLen := len(d.speechBuf) - silenceBytes
	if audioLen < 0 {
		audioLen = 0
	}

	// Make a copy of the audio
	audio := make([]byte, audioLen)
	copy(audio, d.speechBuf[:audioLen])

	seg := &SpeechSegment{
		Audio:     audio,
		StartTime: d.speechStart,
		Duration:  time.Duration(audioLen/2) * time.Second / time.Duration(d.config.SampleRate),
	}

	// Reset state
	d.speaking = false
	d.speechBuf = d.speechBuf[:0]
	d.silenceMs = 0

	return seg
}

// runVAD runs inference on a single window of audio.
// window must be exactly 1024 bytes (512 samples of PCM16).
func (d *SileroDetector) runVAD(window []byte) (float32, error) {
	// Convert PCM16 to float32
	samples := make([]float32, SileroWindowSamples)
	for i := 0; i < SileroWindowSamples; i++ {
		sample := int16(window[i*2]) | int16(window[i*2+1])<<8
		samples[i] = float32(sample) / 32768.0
	}

	// Create input tensors
	// input: [1, 512]
	inputShape := ort.NewShape(1, SileroWindowSamples)
	inputTensor, err := ort.NewTensor(inputShape, samples)
	if err != nil {
		return 0, fmt.Errorf("create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	// sr: scalar int64
	srShape := ort.NewShape(1)
	srData := []int64{SileroSampleRate}
	srTensor, err := ort.NewTensor(srShape, srData)
	if err != nil {
		return 0, fmt.Errorf("create sr tensor: %w", err)
	}
	defer srTensor.Destroy()

	// h: [2, 1, 64]
	hShape := ort.NewShape(SileroNumLayers, 1, SileroHiddenSize)
	hTensor, err := ort.NewTensor(hShape, d.h)
	if err != nil {
		return 0, fmt.Errorf("create h tensor: %w", err)
	}
	defer hTensor.Destroy()

	// c: [2, 1, 64]
	cShape := ort.NewShape(SileroNumLayers, 1, SileroHiddenSize)
	cTensor, err := ort.NewTensor(cShape, d.c)
	if err != nil {
		return 0, fmt.Errorf("create c tensor: %w", err)
	}
	defer cTensor.Destroy()

	// Create output tensors
	// output: [1, 1]
	outputShape := ort.NewShape(1, 1)
	outputData := make([]float32, 1)
	outputTensor, err := ort.NewTensor(outputShape, outputData)
	if err != nil {
		return 0, fmt.Errorf("create output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	// hn: [2, 1, 64]
	hnData := make([]float32, SileroNumLayers*1*SileroHiddenSize)
	hnTensor, err := ort.NewTensor(hShape, hnData)
	if err != nil {
		return 0, fmt.Errorf("create hn tensor: %w", err)
	}
	defer hnTensor.Destroy()

	// cn: [2, 1, 64]
	cnData := make([]float32, SileroNumLayers*1*SileroHiddenSize)
	cnTensor, err := ort.NewTensor(cShape, cnData)
	if err != nil {
		return 0, fmt.Errorf("create cn tensor: %w", err)
	}
	defer cnTensor.Destroy()

	// Run inference
	err = d.session.Run(
		[]ort.ArbitraryTensor{inputTensor, srTensor, hTensor, cTensor},
		[]ort.ArbitraryTensor{outputTensor, hnTensor, cnTensor},
	)
	if err != nil {
		return 0, fmt.Errorf("silero inference: %w", err)
	}

	// Update hidden states for next call
	copy(d.h, hnTensor.GetData())
	copy(d.c, cnTensor.GetData())

	// Return speech probability
	return outputTensor.GetData()[0], nil
}

// ONNX Runtime initialization (shared with speaker module)
var (
	onnxRuntimeInitOnce sync.Once
	onnxRuntimeInitErr  error
)

func initONNXRuntime() error {
	onnxRuntimeInitOnce.Do(func() {
		onnxRuntimeInitErr = ort.InitializeEnvironment()
	})
	return onnxRuntimeInitErr
}
