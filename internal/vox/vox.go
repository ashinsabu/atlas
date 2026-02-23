// Package vox is the voice I/O module for Atlas.
//
// Vox handles:
// - Continuous audio capture
// - Voice Activity Detection (speech start/end)
// - Local speech-to-text via Whisper.cpp
// - Wake word detection ("Hey Atlas")
// - Command extraction and delivery to Brain
//
// Flow:
//  1. Listen continuously for audio
//  2. VAD detects speech start → start buffering
//  3. VAD detects speech end → send to Whisper
//  4. Check transcript for "Hey Atlas" → extract command
//  5. Send command to Brain (via callback)
//
// Usage:
//
//	v, err := vox.New(vox.Config{ModelPath: "models/ggml-small.bin"})
//	v.OnCommand = func(cmd string) { /* send to brain */ }
//	v.Run(ctx)
package vox

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/ashinsabu/atlas/internal/vox/audio"
	"github.com/ashinsabu/atlas/internal/vox/config"
	"github.com/ashinsabu/atlas/internal/vox/stt"
	"github.com/ashinsabu/atlas/internal/vox/vad"
	"github.com/ashinsabu/atlas/internal/vox/wakeword"
)

// STTMode controls transcription behavior
type STTMode string

const (
	// ModeStreaming uses 2s chunks for faster feedback
	ModeStreaming STTMode = "streaming"
	// ModeAccurate uses 5s chunks for better accuracy
	ModeAccurate STTMode = "accurate"
)

// Config holds Vox configuration.
type Config struct {
	// Path to Whisper model file (required)
	ModelPath string

	// STT mode: "streaming" (2s, fast) or "accurate" (5s, better)
	Mode STTMode

	// Audio capture settings
	Audio audio.Config

	// VAD settings
	VAD vad.Config

	// Debug logging
	Debug bool
}

// DefaultConfig returns defaults (you must set ModelPath).
func DefaultConfig() Config {
	return Config{
		Mode:  ModeStreaming, // Default to faster feedback
		Audio: audio.DefaultConfig(),
		VAD:   vad.DefaultConfig(),
		Debug: false,
	}
}

// Vox is the main voice interface.
type Vox struct {
	config   Config
	capture  *audio.Capture
	whisper  *stt.Whisper
	vad      *vad.VAD
	detector *wakeword.Detector

	// Callbacks
	OnCommand          func(command string)                // Called when command is complete
	OnWake             func()                              // Called when wake word detected
	OnListening        func()                              // Called when speech detected
	OnTranscript       func(text string)                   // Called with final transcript
	OnPartialTranscript func(text string, isFinal bool)    // Called with streaming transcripts (isFinal=true on last)
	OnError            func(error)                         // Called on errors
	OnEnergy           func(energy float32, isSpeech bool) // Called every chunk with energy level (debug)

	// Processing queues (serializes Whisper access - not thread-safe)
	utteranceChan chan []byte
	partialChan   chan []byte

	mu     sync.RWMutex
	status string
}

// NewFromConfig creates a Vox instance from the unified config package.
// This is the preferred way to create Vox when using .env configuration.
func NewFromConfig(cfg *config.Config) (*Vox, error) {
	// Convert to internal Config
	internalCfg := Config{
		ModelPath: cfg.ModelPath,
		Mode:      STTMode(cfg.Mode),
		Audio: audio.Config{
			SampleRate:      cfg.SampleRate,
			Channels:        cfg.Channels,
			FramesPerBuffer: cfg.FramesPerBuffer,
		},
		VAD: vad.Config{
			EnergyThreshold:   cfg.EnergyThreshold,
			SilenceTimeout:    cfg.SilenceTimeout,
			MinSpeechDuration: cfg.MinSpeechDuration,
			MaxSpeechDuration: cfg.MaxSpeechDuration,
			SampleRate:        cfg.SampleRate,
			StreamingInterval: cfg.StreamingInterval,
		},
		Debug: cfg.Debug,
	}
	return newVox(internalCfg, cfg.Language, cfg.WakeWords, cfg.ListenTimeout)
}

// New creates a Vox instance.
func New(cfg Config) (*Vox, error) {
	return newVox(cfg, "en", "hey atlas,atlas", 10*time.Second)
}

// newVox is the internal constructor.
func newVox(cfg Config, language, wakeWords string, listenTimeout time.Duration) (*Vox, error) {
	if cfg.ModelPath == "" {
		return nil, fmt.Errorf("ModelPath is required")
	}

	capture, err := audio.NewCapture(cfg.Audio)
	if err != nil {
		return nil, fmt.Errorf("audio capture: %w", err)
	}

	whisperCfg := stt.WhisperConfig{
		ModelPath: cfg.ModelPath,
		Language:  language,
	}
	whisper, err := stt.NewWhisper(whisperCfg)
	if err != nil {
		capture.Close()
		return nil, fmt.Errorf("whisper: %w", err)
	}

	vadCfg := cfg.VAD
	// Apply mode settings if MaxSpeechDuration not explicitly set
	if vadCfg.MaxSpeechDuration == 0 {
		switch cfg.Mode {
		case ModeStreaming:
			vadCfg.MaxSpeechDuration = 2 * time.Second // Fast feedback
		case ModeAccurate:
			vadCfg.MaxSpeechDuration = 5 * time.Second // Better context
		}
	}
	vadInstance := vad.New(vadCfg)

	// Parse wake words from comma-separated string
	words := strings.Split(wakeWords, ",")
	for i := range words {
		words[i] = strings.TrimSpace(words[i])
	}
	detector := wakeword.New().WithWakeWords(words...)
	detector.ListenTimeout = listenTimeout

	v := &Vox{
		config:        cfg,
		capture:       capture,
		whisper:       whisper,
		vad:           vadInstance,
		detector:      detector,
		utteranceChan: make(chan []byte, 5), // Buffer up to 5 utterances
		partialChan:   make(chan []byte, 3), // Buffer partial audio for streaming
		status:        "initialized",
	}

	// Wire up VAD callbacks
	vadInstance.OnSpeechStart = func() {
		if v.OnListening != nil {
			v.OnListening()
		}
	}

	vadInstance.OnSpeechEnd = func(audioData []byte) {
		// Send to processing queue (non-blocking to prevent input overflow)
		select {
		case v.utteranceChan <- audioData:
		default:
			// Queue full, drop utterance
		}
	}

	vadInstance.OnPartialAudio = func(audioData []byte) {
		// Send partial audio for streaming transcription (non-blocking)
		select {
		case v.partialChan <- audioData:
		default:
			// Queue full, skip this partial
		}
	}

	vadInstance.OnEnergy = func(energy float32, isSpeech bool) {
		if v.OnEnergy != nil {
			v.OnEnergy(energy, isSpeech)
		}
	}

	return v, nil
}

// Close releases all resources.
func (v *Vox) Close() error {
	var errs []error
	if v.whisper != nil {
		if err := v.whisper.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	if v.capture != nil {
		if err := v.capture.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	if len(errs) > 0 {
		return fmt.Errorf("close errors: %v", errs)
	}
	return nil
}

// Status returns current status string.
func (v *Vox) Status() string {
	v.mu.RLock()
	defer v.mu.RUnlock()
	return v.status
}

func (v *Vox) setStatus(s string) {
	v.mu.Lock()
	v.status = s
	v.mu.Unlock()
}

// Run starts the voice capture loop.
// Blocks until context is cancelled.
func (v *Vox) Run(ctx context.Context) error {
	v.setStatus("running")

	// Start utterance processor (serializes Whisper access)
	go v.processLoop(ctx)

	return v.capture.Start(ctx, func(chunk []byte) {
		v.vad.Process(chunk)
	})
}

// processLoop handles utterances serially (Whisper is not thread-safe)
func (v *Vox) processLoop(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case audioData := <-v.partialChan:
			// Process partial audio for streaming display
			v.processPartial(audioData)
		case audioData := <-v.utteranceChan:
			// Drain any pending partials (now stale since we have the final)
			v.drainPartials()
			// Process final utterance
			v.processUtterance(audioData)
		}
	}
}

// drainPartials discards any pending partial audio (now stale).
func (v *Vox) drainPartials() {
	for {
		select {
		case <-v.partialChan:
			// Discard
		default:
			return
		}
	}
}

// processPartial handles partial audio for streaming transcription.
func (v *Vox) processPartial(audioData []byte) {
	// Skip very short audio
	const minAudioBytes = 9600 // 0.3s at 16kHz 16-bit mono
	if len(audioData) < minAudioBytes {
		return
	}

	// Transcribe partial audio
	text, err := v.whisper.TranscribeBytes(audioData)
	if err != nil {
		return // Silently ignore partial errors
	}

	text = strings.TrimSpace(text)
	if text == "" {
		return
	}

	// Emit partial transcript
	if v.OnPartialTranscript != nil {
		v.OnPartialTranscript(text, false)
	}
}

// processUtterance handles a complete speech segment.
func (v *Vox) processUtterance(audioData []byte) {
	// Skip very short audio (< 0.5s at 16kHz 16-bit mono = 16000 bytes)
	// Short clips cause Whisper to hallucinate
	const minAudioBytes = 16000
	if len(audioData) < minAudioBytes {
		return
	}

	v.setStatus("transcribing")

	// Transcribe with Whisper
	text, err := v.whisper.TranscribeBytes(audioData)
	if err != nil {
		if v.OnError != nil {
			v.OnError(fmt.Errorf("transcribe: %w", err))
		}
		v.setStatus("running")
		return
	}

	text = strings.TrimSpace(text)
	if text == "" {
		v.setStatus("running")
		return
	}

	// Emit final partial transcript (for streaming UI to finalize)
	if v.OnPartialTranscript != nil {
		v.OnPartialTranscript(text, true)
	}

	// Also emit to legacy OnTranscript callback
	if v.OnTranscript != nil {
		v.OnTranscript(text)
	}

	// Check for wake word and extract command
	command := v.detector.Process(text, true)

	if command != "" {
		// Wake word was present, we have a command
		if v.OnWake != nil {
			v.OnWake()
		}
		if v.OnCommand != nil {
			v.OnCommand(command)
		}
	}

	v.setStatus("running")
}

// ModelInfo returns information about the loaded Whisper model.
func (v *Vox) ModelInfo() string {
	return v.whisper.ModelInfo()
}
