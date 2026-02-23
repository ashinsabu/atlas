// Package wakeword detects the "Hey Atlas" trigger phrase.
//
// Currently, uses transcript-based detection (simple approach).
// Can be upgraded to Porcupine/Snowboy for lower latency later.
package wakeword

import (
	"strings"
	"sync"
	"time"
)

// State represents the current listening state.
type State int

const (
	// Idle = waiting for wake word
	Idle State = iota
	// Listening = wake word detected, capturing command
	Listening
)

// Detector monitors transcripts for wake word and manages state.
type Detector struct {
	wakeWords []string
	state     State
	mu        sync.RWMutex

	// Timeout for listening state (go back to idle if no speech)
	ListenTimeout time.Duration

	// Callback when wake word detected
	OnWake func()

	// Callback when command completed (final transcript after wake)
	OnCommand func(command string)

	// Track when we started listening
	listenStart time.Time

	// Buffer for building command
	commandBuffer strings.Builder
}

// New creates a detector with default wake words.
func New() *Detector {
	return &Detector{
		wakeWords:     []string{"hey atlas", "atlas"},
		state:         Idle,
		ListenTimeout: 10 * time.Second,
	}
}

// WithWakeWords sets custom wake words (case-insensitive).
func (d *Detector) WithWakeWords(words ...string) *Detector {
	d.wakeWords = make([]string, len(words))
	for i, w := range words {
		d.wakeWords[i] = strings.ToLower(w)
	}
	return d
}

// State returns the current detection state.
func (d *Detector) State() State {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.state
}

// Process handles an incoming transcript.
// Returns the display text (command portion if listening).
func (d *Detector) Process(transcript string, isFinal bool) string {
	d.mu.Lock()
	defer d.mu.Unlock()

	lower := strings.ToLower(transcript)

	switch d.state {
	case Idle:
		// Look for wake word
		for _, wake := range d.wakeWords {
			if idx := strings.Index(lower, wake); idx != -1 {
				// Found wake word - extract command portion
				commandStart := idx + len(wake)
				command := strings.TrimSpace(transcript[commandStart:])
				// Strip leading punctuation (Whisper often adds "," or ":")
				command = strings.TrimLeft(command, ",.;:!? ")

				d.state = Listening
				d.listenStart = time.Now()
				d.commandBuffer.Reset()

				if d.OnWake != nil {
					d.OnWake()
				}

				if command != "" {
					d.commandBuffer.WriteString(command)
				}

				return command
			}
		}
		return "" // No wake word, nothing to display

	case Listening:
		// Check for timeout
		if time.Since(d.listenStart) > d.ListenTimeout {
			d.state = Idle
			d.commandBuffer.Reset()
			return ""
		}

		// Check if this transcript contains another wake word (interruption)
		for _, wake := range d.wakeWords {
			if idx := strings.Index(lower, wake); idx != -1 {
				// New wake word - reset and start fresh
				commandStart := idx + len(wake)
				command := strings.TrimSpace(transcript[commandStart:])
				command = strings.TrimLeft(command, ",.;:!? ")

				d.listenStart = time.Now()
				d.commandBuffer.Reset()

				if d.OnWake != nil {
					d.OnWake()
				}

				if command != "" {
					d.commandBuffer.WriteString(command)
				}

				return command
			}
		}

		// No wake word - this is command content
		if isFinal {
			// Final result - command is complete
			command := strings.TrimSpace(transcript)
			d.state = Idle

			if d.OnCommand != nil && command != "" {
				d.OnCommand(command)
			}

			return command
		}

		// Interim result - show progress
		return transcript
	}

	return ""
}

// Reset returns to idle state.
func (d *Detector) Reset() {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.state = Idle
	d.commandBuffer.Reset()
}

// IsListening returns true if actively capturing a command.
func (d *Detector) IsListening() bool {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.state == Listening
}
