// Package stt provides speech-to-text using Whisper.cpp CGO bindings.
//
// Uses whisper.cpp via Go bindings (CGO). Requires libwhisper.a built from
// third_party/whisper.cpp. Run `make whisper-lib` to build.
//
// Usage:
//
//	w := stt.NewWhisper(stt.WhisperConfig{ModelPath: "models/ggml-small.bin"})
//	text, err := w.TranscribeBytes(audioData)
//	defer w.Close()
package stt

import (
	"fmt"
	"io"
	"path/filepath"
	"strings"
	"time"

	"github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
)

// TokenTiming represents a transcribed token with timing information.
// Used for word-level alignment in speaker diarization.
type TokenTiming struct {
	// Transcribed token text
	Text string

	// Start time relative to audio start
	Start time.Duration

	// End time relative to audio start
	End time.Duration

	// Probability/confidence score from Whisper (0.0-1.0)
	Probability float32
}

// WhisperConfig holds configuration for Whisper STT.
type WhisperConfig struct {
	// Path to model file (required)
	ModelPath string

	// Language code (default: "en")
	Language string

	// Temperature controls decoding randomness (default: 0.0 = deterministic)
	// Lower = more deterministic, higher = more creative/hallucinations
	Temperature float32

	// BeamSize for beam search (default: 5)
	// Higher = more thorough search, slower inference
	BeamSize int

	// InitialPrompt biases the model toward expected vocabulary
	// Include domain-specific words to improve recognition
	InitialPrompt string
}

// DefaultWhisperConfig returns sensible defaults for Atlas.
func DefaultWhisperConfig() WhisperConfig {
	return WhisperConfig{
		Language:      "en",
		Temperature:   0.0, // Deterministic decoding - no hallucinations
		BeamSize:      5,   // Search 5 token paths for accuracy
		InitialPrompt: "",  // No bias - let the model transcribe naturally
	}
}

// Whisper wraps whisper.cpp via CGO bindings for speech-to-text.
type Whisper struct {
	model       whisper.Model
	modelPath   string
	lang        string
	temperature float32
	beamSize    int
	prompt      string
}

// Compile-time check that Whisper implements STT.
var _ STT = (*Whisper)(nil)

// NewWhisper creates a Whisper instance and loads the model.
// The model is loaded once and reused for all transcriptions.
func NewWhisper(cfg WhisperConfig) (*Whisper, error) {
	if cfg.ModelPath == "" {
		return nil, fmt.Errorf("ModelPath is required")
	}

	model, err := whisper.New(cfg.ModelPath)
	if err != nil {
		return nil, fmt.Errorf("load whisper model: %w", err)
	}

	// Apply defaults
	lang := cfg.Language
	if lang == "" {
		lang = "en"
	}
	beamSize := cfg.BeamSize
	if beamSize <= 0 {
		beamSize = 5
	}

	return &Whisper{
		model:       model,
		modelPath:   cfg.ModelPath,
		lang:        lang,
		temperature: cfg.Temperature,
		beamSize:    beamSize,
		prompt:      cfg.InitialPrompt,
	}, nil
}

// Close releases the model resources. Must be called when done.
func (w *Whisper) Close() error {
	if w.model != nil {
		return w.model.Close()
	}
	return nil
}

// Transcribe converts raw PCM audio to text.
// Input: 16-bit little-endian PCM at 16kHz mono.
// Implements the STT interface.
func (w *Whisper) Transcribe(pcmData []byte) (string, error) {
	return w.TranscribeBytes(pcmData)
}

// TranscribeBytes converts raw PCM audio to text.
// Input: 16-bit little-endian PCM at 16kHz mono.
// Deprecated: Use Transcribe instead.
func (w *Whisper) TranscribeBytes(pcmData []byte) (string, error) {
	if len(pcmData) == 0 {
		return "", nil
	}

	// Convert 16-bit PCM to float32 samples normalized to [-1.0, 1.0]
	samples := pcmToFloat32(pcmData)

	// Normalize audio levels for consistent input
	samples = normalizeAudio(samples)

	// Create processing context
	ctx, err := w.model.NewContext()
	if err != nil {
		return "", fmt.Errorf("create whisper context: %w", err)
	}

	// Configure language
	if err := ctx.SetLanguage(w.lang); err != nil {
		return "", fmt.Errorf("set language: %w", err)
	}

	// Configure tuning parameters for accuracy
	ctx.SetTemperature(w.temperature)
	ctx.SetBeamSize(w.beamSize)
	if w.prompt != "" {
		ctx.SetInitialPrompt(w.prompt)
	}

	// Process audio samples
	// Args: samples, encoderBeginCallback, segmentCallback, progressCallback
	if err := ctx.Process(samples, nil, nil, nil); err != nil {
		return "", fmt.Errorf("whisper process: %w", err)
	}

	// Collect transcription from segments
	var sb strings.Builder
	for {
		segment, err := ctx.NextSegment()
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", fmt.Errorf("read segment: %w", err)
		}
		sb.WriteString(segment.Text)
	}

	return strings.TrimSpace(sb.String()), nil
}

// TranscribeWithTimestamps converts raw PCM audio to text with word-level timing.
// Input: 16-bit little-endian PCM at 16kHz mono.
// Returns token timings suitable for speaker diarization alignment.
func (w *Whisper) TranscribeWithTimestamps(pcmData []byte) ([]TokenTiming, error) {
	if len(pcmData) == 0 {
		return nil, nil
	}

	// Convert 16-bit PCM to float32 samples normalized to [-1.0, 1.0]
	samples := pcmToFloat32(pcmData)

	// Normalize audio levels for consistent input
	samples = normalizeAudio(samples)

	// Create processing context
	ctx, err := w.model.NewContext()
	if err != nil {
		return nil, fmt.Errorf("create whisper context: %w", err)
	}

	// Configure language
	if err := ctx.SetLanguage(w.lang); err != nil {
		return nil, fmt.Errorf("set language: %w", err)
	}

	// Configure tuning parameters
	ctx.SetTemperature(w.temperature)
	ctx.SetBeamSize(w.beamSize)
	if w.prompt != "" {
		ctx.SetInitialPrompt(w.prompt)
	}

	// Enable token timestamps for word-level alignment
	ctx.SetTokenTimestamps(true)

	// Process audio samples
	if err := ctx.Process(samples, nil, nil, nil); err != nil {
		return nil, fmt.Errorf("whisper process: %w", err)
	}

	// Collect tokens with timestamps from all segments
	var tokens []TokenTiming
	for {
		segment, err := ctx.NextSegment()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("read segment: %w", err)
		}

		// Extract text tokens (filter out special tokens)
		for _, tok := range segment.Tokens {
			if !ctx.IsText(tok) {
				continue
			}

			text := strings.TrimSpace(tok.Text)
			if text == "" {
				continue
			}

			tokens = append(tokens, TokenTiming{
				Text:        text,
				Start:       tok.Start,
				End:         tok.End,
				Probability: tok.P,
			})
		}
	}

	return tokens, nil
}

// TranscribeSamples converts float32 audio samples to text.
// Input: float32 audio samples at 16kHz mono, normalized to [-1.0, 1.0].
// For word-level timing, use TranscribeSamplesWithTimestamps instead.
func (w *Whisper) TranscribeSamples(samples []float32) (string, error) {
	if len(samples) == 0 {
		return "", nil
	}

	// Normalize audio levels for consistent input
	normalizedSamples := make([]float32, len(samples))
	copy(normalizedSamples, samples)
	normalizedSamples = normalizeAudio(normalizedSamples)

	// Create processing context
	ctx, err := w.model.NewContext()
	if err != nil {
		return "", fmt.Errorf("create whisper context: %w", err)
	}

	// Configure language
	if err := ctx.SetLanguage(w.lang); err != nil {
		return "", fmt.Errorf("set language: %w", err)
	}

	// Configure tuning parameters for accuracy
	ctx.SetTemperature(w.temperature)
	ctx.SetBeamSize(w.beamSize)
	if w.prompt != "" {
		ctx.SetInitialPrompt(w.prompt)
	}

	// Process audio samples
	if err := ctx.Process(normalizedSamples, nil, nil, nil); err != nil {
		return "", fmt.Errorf("whisper process: %w", err)
	}

	// Collect transcription from segments
	var sb strings.Builder
	for {
		segment, err := ctx.NextSegment()
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", fmt.Errorf("read segment: %w", err)
		}
		sb.WriteString(segment.Text)
	}

	return strings.TrimSpace(sb.String()), nil
}

// TranscribeSamplesWithTimestamps converts float32 audio samples to text with timing.
// Input: float32 audio samples at 16kHz mono, normalized to [-1.0, 1.0].
// This avoids PCM conversion when audio is already in float32 format.
// Note: This makes a copy of the samples buffer for normalization. If you don't
// need to preserve the original samples, use TranscribeSamplesWithTimestampsInPlace.
func (w *Whisper) TranscribeSamplesWithTimestamps(samples []float32) ([]TokenTiming, error) {
	if len(samples) == 0 {
		return nil, nil
	}

	// Normalize audio levels for consistent input
	normalizedSamples := make([]float32, len(samples))
	copy(normalizedSamples, samples)
	normalizedSamples = normalizeAudio(normalizedSamples)

	// Create processing context
	ctx, err := w.model.NewContext()
	if err != nil {
		return nil, fmt.Errorf("create whisper context: %w", err)
	}

	// Configure language
	if err := ctx.SetLanguage(w.lang); err != nil {
		return nil, fmt.Errorf("set language: %w", err)
	}

	// Configure tuning parameters
	ctx.SetTemperature(w.temperature)
	ctx.SetBeamSize(w.beamSize)
	if w.prompt != "" {
		ctx.SetInitialPrompt(w.prompt)
	}

	// Enable token timestamps
	ctx.SetTokenTimestamps(true)

	// Process audio samples
	if err := ctx.Process(normalizedSamples, nil, nil, nil); err != nil {
		return nil, fmt.Errorf("whisper process: %w", err)
	}

	// Collect tokens with timestamps
	var tokens []TokenTiming
	for {
		segment, err := ctx.NextSegment()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("read segment: %w", err)
		}

		for _, tok := range segment.Tokens {
			if !ctx.IsText(tok) {
				continue
			}

			text := strings.TrimSpace(tok.Text)
			if text == "" {
				continue
			}

			tokens = append(tokens, TokenTiming{
				Text:        text,
				Start:       tok.Start,
				End:         tok.End,
				Probability: tok.P,
			})
		}
	}

	return tokens, nil
}

// TranscribeSamplesWithTimestampsInPlace converts float32 audio samples to text with timing.
// Unlike TranscribeSamplesWithTimestamps, this normalizes the samples buffer in-place,
// avoiding a ~320KB allocation for 10s of audio. The caller's buffer will be modified.
// Use this when you don't need to preserve the original audio data.
func (w *Whisper) TranscribeSamplesWithTimestampsInPlace(samples []float32) ([]TokenTiming, error) {
	if len(samples) == 0 {
		return nil, nil
	}

	// Normalize audio levels in-place (modifies caller's buffer)
	normalizeAudio(samples)

	// Create processing context
	ctx, err := w.model.NewContext()
	if err != nil {
		return nil, fmt.Errorf("create whisper context: %w", err)
	}

	// Configure language
	if err := ctx.SetLanguage(w.lang); err != nil {
		return nil, fmt.Errorf("set language: %w", err)
	}

	// Configure tuning parameters
	ctx.SetTemperature(w.temperature)
	ctx.SetBeamSize(w.beamSize)
	if w.prompt != "" {
		ctx.SetInitialPrompt(w.prompt)
	}

	// Enable token timestamps
	ctx.SetTokenTimestamps(true)

	// Process audio samples
	if err := ctx.Process(samples, nil, nil, nil); err != nil {
		return nil, fmt.Errorf("whisper process: %w", err)
	}

	// Collect tokens with timestamps
	var tokens []TokenTiming
	for {
		segment, err := ctx.NextSegment()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("read segment: %w", err)
		}

		for _, tok := range segment.Tokens {
			if !ctx.IsText(tok) {
				continue
			}

			text := strings.TrimSpace(tok.Text)
			if text == "" {
				continue
			}

			tokens = append(tokens, TokenTiming{
				Text:        text,
				Start:       tok.Start,
				End:         tok.End,
				Probability: tok.P,
			})
		}
	}

	return tokens, nil
}

// ModelInfo returns info about the loaded model.
func (w *Whisper) ModelInfo() string {
	return fmt.Sprintf("Whisper CGO, Model: %s", filepath.Base(w.modelPath))
}

// pcmToFloat32 converts 16-bit little-endian PCM to float32 samples.
// Output is normalized to [-1.0, 1.0] as expected by Whisper.
func pcmToFloat32(pcm []byte) []float32 {
	numSamples := len(pcm) / 2
	samples := make([]float32, numSamples)

	for i := 0; i < numSamples; i++ {
		// Read 16-bit little-endian sample
		sample := int16(pcm[i*2]) | int16(pcm[i*2+1])<<8
		// Normalize to [-1.0, 1.0]
		samples[i] = float32(sample) / 32768.0
	}

	return samples
}

// normalizeAudio applies peak normalization to audio samples.
// This ensures consistent input levels regardless of recording volume.
func normalizeAudio(samples []float32) []float32 {
	if len(samples) == 0 {
		return samples
	}

	// Find peak amplitude
	var peak float32
	for _, s := range samples {
		abs := s
		if abs < 0 {
			abs = -abs
		}
		if abs > peak {
			peak = abs
		}
	}

	// Normalize to 95% of max range (leave headroom to avoid clipping)
	// Only normalize if signal is above noise floor
	if peak > 0.01 {
		target := float32(0.95)
		scale := target / peak
		for i := range samples {
			samples[i] *= scale
		}
	}

	return samples
}
