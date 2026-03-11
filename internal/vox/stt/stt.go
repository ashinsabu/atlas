// Package stt provides speech-to-text interfaces and implementations.
package stt

// STT is the interface for speech-to-text engines.
type STT interface {
	// Transcribe converts PCM16 audio to text.
	// Input: 16-bit little-endian PCM at 16kHz mono.
	// Returns the transcribed text (may be empty for silence/noise).
	Transcribe(audio []byte) (string, error)

	// Close releases resources.
	Close() error
}

// TranscribeResult contains transcription output with metadata.
type TranscribeResult struct {
	// Text is the transcribed speech.
	Text string

	// Confidence is the average token probability (0.0-1.0).
	// Higher = more confident. Not all implementations provide this.
	Confidence float32

	// Language detected (if auto-detection enabled).
	Language string
}
