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

// New creates a detector. Call WithWakeWords to configure wake phrases before use.
func New() *Detector {
	return &Detector{
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

	switch d.state {
	case Idle:
		command, found := findWakeWord(transcript, d.wakeWords)
		if !found {
			return ""
		}
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

	case Listening:
		if time.Since(d.listenStart) > d.ListenTimeout {
			d.state = Idle
			d.commandBuffer.Reset()
			return ""
		}

		// Re-trigger on another wake word (interruption).
		if command, found := findWakeWord(transcript, d.wakeWords); found {
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

		if isFinal {
			command := strings.TrimSpace(transcript)
			d.state = Idle
			if d.OnCommand != nil && command != "" {
				d.OnCommand(command)
			}
			return command
		}
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

// findWakeWord searches text for any wake phrase and returns the command portion.
//
// Two-pass strategy:
//  1. Exact substring match (fast path, handles the common case).
//  2. Word-level fuzzy match: each word in the transcript is compared against
//     the corresponding phrase word; edit distance ≤ 1 is accepted. This catches
//     common Whisper transcription errors like "Atlast" or "Atlass" for "Atlas".
func findWakeWord(text string, phrases []string) (command string, found bool) {
	lower := strings.ToLower(text)

	// Pass 1: exact substring match.
	for _, phrase := range phrases {
		if idx := strings.Index(lower, phrase); idx != -1 {
			after := strings.TrimSpace(text[idx+len(phrase):])
			return strings.TrimLeft(after, ",.;:!? "), true
		}
	}

	// Pass 2: word-level fuzzy match.
	lowerWords := strings.Fields(lower)
	origWords := strings.Fields(text)

	for _, phrase := range phrases {
		parts := strings.Fields(phrase)
		if len(parts) == 0 || len(lowerWords) < len(parts) {
			continue
		}
		for i := 0; i <= len(lowerWords)-len(parts); i++ {
			match := true
			for j, pw := range parts {
				tw := strings.TrimRight(lowerWords[i+j], ",.!?;:")
				if tw != pw && levenshtein(tw, pw) > 1 {
					match = false
					break
				}
			}
			if match {
				after := strings.Join(origWords[i+len(parts):], " ")
				return strings.TrimLeft(after, ",.;:!? "), true
			}
		}
	}

	return "", false
}

// levenshtein computes the edit distance between two strings.
// Uses the standard two-row DP algorithm; O(len(a)*len(b)) time, O(len(b)) space.
func levenshtein(a, b string) int {
	ra, rb := []rune(a), []rune(b)
	la, lb := len(ra), len(rb)
	if la == 0 {
		return lb
	}
	if lb == 0 {
		return la
	}
	prev := make([]int, lb+1)
	curr := make([]int, lb+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 0; i < la; i++ {
		curr[0] = i + 1
		for j := 0; j < lb; j++ {
			cost := 1
			if ra[i] == rb[j] {
				cost = 0
			}
			curr[j+1] = min(curr[j]+1, min(prev[j+1]+1, prev[j]+cost))
		}
		prev, curr = curr, prev
	}
	return prev[lb]
}
