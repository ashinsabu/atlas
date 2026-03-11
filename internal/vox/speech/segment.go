// Package speech provides speech detection and segmentation.
package speech

import "time"

// SpeechSegment represents a detected segment of speech audio.
type SpeechSegment struct {
	// Audio contains PCM16 audio data (16kHz mono, little-endian).
	Audio []byte

	// StartTime is when this speech segment began (relative to stream start).
	StartTime time.Time

	// Duration is the length of this speech segment.
	Duration time.Duration
}

// Samples returns the number of audio samples in this segment.
func (s *SpeechSegment) Samples() int {
	return len(s.Audio) / 2 // 16-bit = 2 bytes per sample
}

// Seconds returns the duration of audio in seconds.
func (s *SpeechSegment) Seconds() float64 {
	// 16kHz * 2 bytes = 32000 bytes per second
	return float64(len(s.Audio)) / 32000.0
}

// ToFloat32 converts the PCM16 audio to float32 samples normalized to [-1.0, 1.0].
// This is the format expected by speaker verification and Whisper.
func (s *SpeechSegment) ToFloat32() []float32 {
	numSamples := len(s.Audio) / 2
	samples := make([]float32, numSamples)

	for i := 0; i < numSamples; i++ {
		// Read 16-bit little-endian sample
		sample := int16(s.Audio[i*2]) | int16(s.Audio[i*2+1])<<8
		// Normalize to [-1.0, 1.0]
		samples[i] = float32(sample) / 32768.0
	}

	return samples
}
