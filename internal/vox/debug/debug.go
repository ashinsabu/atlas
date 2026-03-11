package debug

import (
	"context"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/ashinsabu/atlas/internal/monitor"
)

// New creates the bubbletea program, the Debugger that feeds it, and a
// monitor.Tracker pre-wired to the TUI model for live STATS display.
//
// cancel is called when the user presses q or ctrl+c inside the TUI.
// The caller wires the returned tracker into PipelineConfig.Monitor so the
// pipeline records stage timings that the TUI reads on each tick.
func New(cancel context.CancelFunc) (*tea.Program, Debugger, *monitor.Tracker) {
	tracker := monitor.New()
	m := newModel(cancel, tracker)
	prog := tea.NewProgram(m, tea.WithAltScreen())
	dbg := &UIDebugger{prog: prog}
	return prog, dbg, tracker
}
