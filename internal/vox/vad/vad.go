// Package vad provides Voice Activity Detection.
//
// Detects when speech starts and ends based on audio energy levels.
// Used to segment continuous audio into utterances for transcription.
//
// Usage:
//
//	v := vad.New(vad.DefaultConfig())
//	v.OnSpeechStart = func() { /* start recording */ }
//	v.OnSpeechEnd = func(audio []byte) { /* transcribe */ }
//	v.Process(audioChunk)
package vad

import (
	"math"
	"sync"
	"time"
)

// Config holds VAD parameters.
type Config struct {
	// Energy threshold for speech detection (0.0 to 1.0)
	// Lower = more sensitive, Higher = less sensitive
	EnergyThreshold float32

	// Duration of silence before considering speech ended
	SilenceTimeout time.Duration

	// Minimum speech duration to trigger (filters noise bursts)
	MinSpeechDuration time.Duration

	// Maximum speech duration before forcing end (for continuous audio)
	MaxSpeechDuration time.Duration

	// Sample rate of input audio
	SampleRate int

	// StreamingInterval: How often to emit partial audio for streaming transcription
	// Set to 0 to disable streaming (default behavior)
	StreamingInterval time.Duration
}

// DefaultConfig returns sensible defaults for voice commands.
func DefaultConfig() Config {
	return Config{
		EnergyThreshold:   0.05,                   // Less sensitive - reduce ambient noise triggers
		SilenceTimeout:    800 * time.Millisecond, // Gap detection
		MinSpeechDuration: 500 * time.Millisecond, // Filter short noise bursts
		MaxSpeechDuration: 5 * time.Second,        // Allow complete sentences
		SampleRate:        16000,
	}
}

// State represents the VAD state.
type State int

const (
	Silence State = iota
	Speaking
)

// VAD detects voice activity in audio streams.
type VAD struct {
	config Config
	state  State
	mu     sync.Mutex

	// Callbacks
	OnSpeechStart  func()
	OnSpeechEnd    func(audio []byte)
	OnPartialAudio func(audio []byte) // Called periodically during speech for streaming transcription
	OnEnergy       func(energy float32, isSpeech bool)

	// Internal state
	speechStart      time.Time
	silenceStart     time.Time
	lastPartialEmit  time.Time // Track when we last emitted partial audio
	audioBuffer      []byte
	samplesPerChunk  int
}

// New creates a VAD instance.
func New(cfg Config) *VAD {
	return &VAD{
		config:          cfg,
		state:           Silence,
		samplesPerChunk: cfg.SampleRate / 10, // 100ms chunks
	}
}

// Process analyzes an audio chunk and updates state.
// Input: 16-bit PCM audio at configured sample rate.
func (v *VAD) Process(chunk []byte) {
	v.mu.Lock()
	defer v.mu.Unlock()

	energy := calculateEnergy(chunk)
	isSpeech := energy > v.config.EnergyThreshold
	now := time.Now()

	// Notify about energy level for visualization
	if v.OnEnergy != nil {
		v.OnEnergy(energy, isSpeech)
	}

	switch v.state {
	case Silence:
		if isSpeech {
			v.state = Speaking
			v.speechStart = now
			v.silenceStart = time.Time{}
			v.audioBuffer = make([]byte, 0, 16000*10) // Pre-alloc for ~10s
			v.audioBuffer = append(v.audioBuffer, chunk...)

			if v.OnSpeechStart != nil {
				v.OnSpeechStart()
			}
		}

	case Speaking:
		v.audioBuffer = append(v.audioBuffer, chunk...)
		speechDuration := now.Sub(v.speechStart)

		// Emit partial audio for streaming transcription
		if v.config.StreamingInterval > 0 && v.OnPartialAudio != nil {
			if v.lastPartialEmit.IsZero() || now.Sub(v.lastPartialEmit) >= v.config.StreamingInterval {
				// Only emit if we have enough audio (at least 0.3s)
				minBytes := v.config.SampleRate * 2 * 3 / 10 // 0.3s at 16-bit
				if len(v.audioBuffer) >= minBytes {
					// Make a copy of current buffer for partial transcription
					partial := make([]byte, len(v.audioBuffer))
					copy(partial, v.audioBuffer)
					v.lastPartialEmit = now
					v.OnPartialAudio(partial)
				}
			}
		}

		// Force end if max duration reached (handles continuous audio like TV)
		if v.config.MaxSpeechDuration > 0 && speechDuration >= v.config.MaxSpeechDuration {
			audio := v.audioBuffer
			v.state = Silence
			v.audioBuffer = nil
			v.lastPartialEmit = time.Time{}
			if v.OnSpeechEnd != nil {
				v.OnSpeechEnd(audio)
			}
			return
		}

		if isSpeech {
			// Reset silence timer
			v.silenceStart = time.Time{}
		} else {
			// Track silence duration
			if v.silenceStart.IsZero() {
				v.silenceStart = now
			}

			silenceDuration := now.Sub(v.silenceStart)

			// Check if silence timeout reached
			if silenceDuration >= v.config.SilenceTimeout {
				// Check minimum speech duration
				if speechDuration >= v.config.MinSpeechDuration {
					// Valid speech segment complete
					audio := v.audioBuffer
					v.state = Silence
					v.audioBuffer = nil
					v.lastPartialEmit = time.Time{}

					if v.OnSpeechEnd != nil {
						v.OnSpeechEnd(audio)
					}
				} else {
					// Too short, discard (noise)
					v.state = Silence
					v.audioBuffer = nil
					v.lastPartialEmit = time.Time{}
				}
			}
		}
	}
}

// Reset clears the VAD state.
func (v *VAD) Reset() {
	v.mu.Lock()
	defer v.mu.Unlock()
	v.state = Silence
	v.audioBuffer = nil
	v.speechStart = time.Time{}
	v.silenceStart = time.Time{}
}

// State returns current VAD state.
func (v *VAD) State() State {
	v.mu.Lock()
	defer v.mu.Unlock()
	return v.state
}

// IsSpeaking returns true if currently detecting speech.
func (v *VAD) IsSpeaking() bool {
	return v.State() == Speaking
}

// calculateEnergy computes RMS energy of audio chunk.
// Returns value between 0.0 and 1.0.
func calculateEnergy(chunk []byte) float32 {
	if len(chunk) < 2 {
		return 0
	}

	var sum float64
	numSamples := len(chunk) / 2

	for i := 0; i < numSamples; i++ {
		// Little-endian 16-bit to int16
		sample := int16(chunk[i*2]) | int16(chunk[i*2+1])<<8
		// Normalize and square
		normalized := float64(sample) / 32768.0
		sum += normalized * normalized
	}

	// RMS = sqrt(mean of squares)
	rms := float32(math.Sqrt(sum / float64(numSamples)))
	if rms > 1.0 {
		rms = 1.0
	}
	return rms
}
