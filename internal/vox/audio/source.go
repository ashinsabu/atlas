// Package audio provides audio input/output abstractions for Vox.
package audio

import "context"

// Source is the interface for audio input devices.
// Implementations stream raw audio chunks for processing.
type Source interface {
	// Start begins streaming audio. Blocks until ctx is cancelled.
	// handler receives PCM16 audio chunks (16kHz mono, little-endian).
	// Chunk size is implementation-defined but typically 100ms (3200 bytes).
	Start(ctx context.Context, handler func(chunk []byte)) error

	// Close releases audio resources. Safe to call multiple times.
	Close() error
}

// SourceConfig defines audio capture parameters.
type SourceConfig struct {
	SampleRate      int  // Audio sample rate in Hz (typically 16000)
	Channels        int  // Number of channels (1 = mono)
	FramesPerBuffer int  // Samples per chunk (determines latency)
	Debug           bool // Print device/stream info on startup
}

// DefaultSourceConfig returns settings optimized for speech processing.
func DefaultSourceConfig() SourceConfig {
	return SourceConfig{
		SampleRate:      16000, // 16kHz required by Whisper and Silero
		Channels:        1,     // Mono
		FramesPerBuffer: 512,   // 32ms chunks - matches Silero VAD window
		Debug:           false,
	}
}
