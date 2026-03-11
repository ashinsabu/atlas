// Package debug provides the debug TUI and Debugger interface for pipeline instrumentation.
package debug

// Tea message types — one per pipeline event.

// ChunkMsg is sent every audio chunk (every 32ms from the audio callback).
type ChunkMsg struct {
	N        int
	RMS      float32
	VADProb  float32
	Speaking bool
}

// SegmentStartMsg is sent when a speech segment is handed to the STT queue.
type SegmentStartMsg struct {
	DurationSecs float64
}

// TranscriptionMsg is sent when STT returns a non-empty result.
type TranscriptionMsg struct {
	Text      string
	ElapsedMs int64
}

// TranscriptionEmptyMsg is sent when STT returns an empty string.
type TranscriptionEmptyMsg struct {
	ElapsedMs int64
}

// TranscriptionErrorMsg is sent when STT returns an error.
type TranscriptionErrorMsg struct {
	Err error
}

// WakeWordMsg is sent when a wake word is detected in the transcript.
type WakeWordMsg struct {
	Command string
}

// QueueDepthMsg is sent whenever the relay queue length changes.
type QueueDepthMsg struct {
	Depth int
}

// TickMsg is sent by the 100ms ticker to drive live duration updates.
type TickMsg struct{}

// SpeakerMsg is sent after speaker verification completes for a segment.
type SpeakerMsg struct {
	Name     string  // Speaker name (profile name if accepted, "Unknown" if rejected)
	Score    float32 // Cosine similarity score
	Accepted bool    // Whether the speaker was accepted
}
