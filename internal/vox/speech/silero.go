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
	"log/slog"
	"os"
	"sync"
	"time"

	ort "github.com/yalue/onnxruntime_go"

	ortinit "github.com/ashinsabu/atlas/internal/vox/ort"
)

const (
	// SileroWindowSamples is the number of samples per VAD frame (32ms at 16kHz).
	SileroWindowSamples = 512

	// SileroContextSamples is the context window prepended to each frame (v5 model).
	SileroContextSamples = 64

	// SileroSampleRate is the required sample rate.
	SileroSampleRate = 16000

	// SileroHiddenSize is the LSTM hidden state dimension.
	// Model uses combined state tensor of shape [2, 1, 128].
	SileroHiddenSize = 128

	// SileroNumLayers is the number of LSTM layers.
	SileroNumLayers = 2
)

// SileroDetector implements SpeechDetector using Silero VAD.
type SileroDetector struct {
	session *ort.DynamicAdvancedSession
	config  DetectorConfig

	// LSTM state (combined h+c, persisted across calls).
	// Shape: [SileroNumLayers, 1, SileroHiddenSize] = [2, 1, 128]
	state []float32

	// context holds the last SileroContextSamples from the previous frame.
	// The v5 model requires these 64 samples to be prepended to each window.
	context []float32

	// Buffering state
	audioBuffer []byte    // Accumulates PCM16 audio
	speechBuf   []byte    // Accumulates speech audio
	speaking    bool      // Currently in speech segment
	speechStart time.Time // When current speech started
	silenceMs   int       // Consecutive silence duration in ms
	streamStart time.Time // When the stream started

	// OnFrame is called after each VAD inference (every 32ms).
	// prob is speech probability [0,1], speaking is the updated state.
	// Used for debug visualization; nil in normal operation.
	OnFrame func(prob float32, speaking bool)

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

	// Initialize ONNX Runtime (shared once across all packages)
	if err := ortinit.Initialize(); err != nil {
		return nil, fmt.Errorf("init onnx runtime: %w", err)
	}

	// Create session options
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("create session options: %w", err)
	}
	defer opts.Destroy()

	// Create dynamic session for Silero VAD.
	// Input names: input, state, sr
	// Output names: output, stateN
	session, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{"input", "state", "sr"},
		[]string{"output", "stateN"},
		opts,
	)
	if err != nil {
		return nil, fmt.Errorf("create silero session: %w", err)
	}

	return &SileroDetector{
		session:     session,
		config:      cfg,
		state:       make([]float32, SileroNumLayers*1*SileroHiddenSize),
		context:     make([]float32, SileroContextSamples),
		audioBuffer: make([]byte, 0, SileroWindowSamples*4),
		speechBuf:   make([]byte, 0, cfg.SampleRate*2*30),
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
			slog.Debug("vad inference error", "err", err)
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

			// Check max duration using buffer length (accurate for both live and offline).
			speechMs := len(d.speechBuf) * 1000 / (d.config.SampleRate * 2)
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

		if d.OnFrame != nil {
			d.OnFrame(prob, d.speaking)
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

	for i := range d.state {
		d.state[i] = 0
	}
	for i := range d.context {
		d.context[i] = 0
	}

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
	// Use buffer length for duration — accurate for both live mic and offline WAV processing.
	speechMs := len(d.speechBuf) * 1000 / (d.config.SampleRate * 2)

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
	// Convert PCM16 window to float32
	samples := make([]float32, SileroWindowSamples)
	for i := 0; i < SileroWindowSamples; i++ {
		s := int16(window[i*2]) | int16(window[i*2+1])<<8
		samples[i] = float32(s) / 32768.0
	}

	// Build input frame: prepend 64-sample context from previous call (v5 model requirement).
	frame := make([]float32, SileroContextSamples+SileroWindowSamples)
	copy(frame[:SileroContextSamples], d.context)
	copy(frame[SileroContextSamples:], samples)

	// input: [1, contextSize+windowSize]
	inputTensor, err := ort.NewTensor(ort.NewShape(1, SileroContextSamples+SileroWindowSamples), frame)
	if err != nil {
		return 0, fmt.Errorf("create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	// state: [2, 1, 128]
	stateTensor, err := ort.NewTensor(ort.NewShape(SileroNumLayers, 1, SileroHiddenSize), d.state)
	if err != nil {
		return 0, fmt.Errorf("create state tensor: %w", err)
	}
	defer stateTensor.Destroy()

	// sr: 0-dim scalar (model requires zero-dimensional, not shape [1])
	srScalar, err := ort.NewScalar[int64](SileroSampleRate)
	if err != nil {
		return 0, fmt.Errorf("create sr scalar: %w", err)
	}
	defer srScalar.Destroy()

	// Let ORT auto-allocate outputs (correct approach for dynamic-shape outputs)
	outputs := []ort.Value{nil, nil}
	err = d.session.Run(
		[]ort.Value{inputTensor, stateTensor, srScalar},
		outputs,
	)
	if err != nil {
		return 0, fmt.Errorf("silero inference: %w", err)
	}
	defer outputs[0].Destroy()
	defer outputs[1].Destroy()

	// Read probability from output[0]
	outputTensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return 0, fmt.Errorf("unexpected output type: %T", outputs[0])
	}
	prob := outputTensor.GetData()[0]

	// Read updated state from output[1] and persist for next call
	stateNTensor, ok := outputs[1].(*ort.Tensor[float32])
	if !ok {
		return 0, fmt.Errorf("unexpected stateN type: %T", outputs[1])
	}
	copy(d.state, stateNTensor.GetData())

	// Persist context: save last SileroContextSamples of the input frame
	copy(d.context, frame[SileroWindowSamples:])

	return prob, nil
}
