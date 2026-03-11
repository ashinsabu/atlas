// Package speech provides speech detection and segmentation.
//
// SpeechDetector uses neural VAD (Silero) to detect speech segments
// in continuous audio streams. This replaces energy-based VAD which
// only detects loudness, not actual speech.
package speech

// SpeechDetector detects speech segments in an audio stream.
// Implementations buffer audio internally and emit complete speech segments.
type SpeechDetector interface {
	// Process feeds an audio chunk to the detector.
	// Returns speech segments that are complete (ended by silence).
	// Chunk must be PCM16 audio at 16kHz mono.
	Process(chunk []byte) []SpeechSegment

	// Flush forces emission of any buffered speech.
	// Call this when the audio stream ends.
	Flush() []SpeechSegment

	// Reset clears all internal state.
	Reset()

	// Close releases resources.
	Close() error
}

// DetectorConfig holds configuration for speech detection.
type DetectorConfig struct {
	// SpeechThreshold is the probability threshold for speech (0.0-1.0).
	// Lower = more sensitive, higher = stricter.
	// Default: 0.5
	SpeechThreshold float32

	// MinSpeechMs is the minimum speech duration to emit (filters noise bursts).
	// Default: 250ms
	MinSpeechMs int

	// MinSilenceMs is the silence duration required to end a segment.
	// Default: 300ms
	MinSilenceMs int

	// MaxSpeechMs is the maximum segment duration before forced split.
	// Default: 30000ms (30s)
	MaxSpeechMs int

	// SampleRate of input audio. Must be 16000.
	SampleRate int
}

// DefaultDetectorConfig returns sensible defaults for voice commands.
func DefaultDetectorConfig() DetectorConfig {
	return DetectorConfig{
		SpeechThreshold: 0.5,
		MinSpeechMs:     250,
		MinSilenceMs:    1500, // Wait 1.5s of silence before emitting (Siri-like)
		MaxSpeechMs:     30000,
		SampleRate:      16000,
	}
}
