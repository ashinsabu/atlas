package debug

import (
	"context"

	tea "github.com/charmbracelet/bubbletea"
)

// New creates the bubbletea program and the Debugger that feeds it.
// cancel is called when the user presses q or ctrl+c inside the TUI.
// The caller is responsible for running the returned program (prog.Run()).
func New(cancel context.CancelFunc) (*tea.Program, Debugger) {
	m := newModel(cancel)
	prog := tea.NewProgram(m, tea.WithAltScreen())
	dbg := &UIDebugger{prog: prog}
	return prog, dbg
}
