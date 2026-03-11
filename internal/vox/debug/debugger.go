package debug

import tea "github.com/charmbracelet/bubbletea"

// Debugger is the interface through which the pipeline emits debug events.
// Implementations must be safe to call from any goroutine.
type Debugger interface {
	OnChunk(n int, rms, vadProb float32, speaking bool)
	OnSegmentStart(durationSecs float64)
	OnTranscription(text string, elapsedMs int64)
	OnTranscriptionEmpty(elapsedMs int64)
	OnTranscriptionError(err error)
	OnWakeWord(command string)
	OnQueueDepth(n int)
	OnSpeakerVerified(name string, score float32, accepted bool)
}

// NopDebugger is a no-op Debugger used when debug mode is disabled.
type NopDebugger struct{}

func (NopDebugger) OnChunk(_ int, _, _ float32, _ bool)           {}
func (NopDebugger) OnSegmentStart(_ float64)                       {}
func (NopDebugger) OnTranscription(_ string, _ int64)             {}
func (NopDebugger) OnTranscriptionEmpty(_ int64)                   {}
func (NopDebugger) OnTranscriptionError(_ error)                   {}
func (NopDebugger) OnWakeWord(_ string)                            {}
func (NopDebugger) OnQueueDepth(_ int)                             {}
func (NopDebugger) OnSpeakerVerified(_ string, _ float32, _ bool) {}

// UIDebugger forwards pipeline events to a bubbletea program as messages.
// tea.Program.Send is goroutine-safe — safe to call from the audio callback,
// relay goroutine, or STT goroutine.
type UIDebugger struct {
	prog *tea.Program
}

func (u *UIDebugger) OnChunk(n int, rms, vadProb float32, speaking bool) {
	u.prog.Send(ChunkMsg{N: n, RMS: rms, VADProb: vadProb, Speaking: speaking})
}

func (u *UIDebugger) OnSegmentStart(durationSecs float64) {
	u.prog.Send(SegmentStartMsg{DurationSecs: durationSecs})
}

func (u *UIDebugger) OnTranscription(text string, elapsedMs int64) {
	u.prog.Send(TranscriptionMsg{Text: text, ElapsedMs: elapsedMs})
}

func (u *UIDebugger) OnTranscriptionEmpty(elapsedMs int64) {
	u.prog.Send(TranscriptionEmptyMsg{ElapsedMs: elapsedMs})
}

func (u *UIDebugger) OnTranscriptionError(err error) {
	u.prog.Send(TranscriptionErrorMsg{Err: err})
}

func (u *UIDebugger) OnWakeWord(command string) {
	u.prog.Send(WakeWordMsg{Command: command})
}

func (u *UIDebugger) OnQueueDepth(n int) {
	u.prog.Send(QueueDepthMsg{Depth: n})
}

func (u *UIDebugger) OnSpeakerVerified(name string, score float32, accepted bool) {
	u.prog.Send(SpeakerMsg{Name: name, Score: score, Accepted: accepted})
}
