package debug

import "github.com/charmbracelet/lipgloss"

var (
	colCyan   = lipgloss.Color("6")
	colGreen  = lipgloss.Color("2")
	colYellow = lipgloss.Color("3")
	colGray   = lipgloss.Color("8")
	colRed    = lipgloss.Color("1")

	paneStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(colGray).
			Padding(0, 1)

	headerStyle = lipgloss.NewStyle().
			Foreground(colCyan).
			Bold(true)

	stCyan   = lipgloss.NewStyle().Foreground(colCyan)
	stGreen  = lipgloss.NewStyle().Foreground(colGreen)
	stYellow = lipgloss.NewStyle().Foreground(colYellow)
	stGray   = lipgloss.NewStyle().Foreground(colGray)
	stRed    = lipgloss.NewStyle().Foreground(colRed)
)
