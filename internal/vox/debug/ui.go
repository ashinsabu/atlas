package debug

import (
	"context"
	"fmt"
	"math"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/ashinsabu/atlas/internal/monitor"
)

const maxTranscriptLines = 200

// Model is the bubbletea model for the 4-pane debug TUI.
type Model struct {
	// Pane 1 — AUDIO
	chunkN     int
	rms, vad   float32
	speaking   bool
	speakStart time.Time

	// Pane 2 — PIPELINE
	sttState   string // "idle" | "transcribing"
	queueDepth int

	// Speaker verification state (last result for the most recent segment)
	lastSpeakerName     string
	lastSpeakerScore    float32
	lastSpeakerAccepted bool
	hasSpeakerResult    bool // false until first segment is verified

	// Pane 3 — TRANSCRIPT
	lines        []string
	scrollOffset int // index of last visible line (single-model mode only)

	// Pane 4 — STATS
	tracker *monitor.Tracker
	snap    monitor.Snapshot

	// Multi-model state (populated when len(modelNames) > 1).
	modelNames       []string            // ["large-v3", "distil"] — ordered; [0] = primary
	compareLines     map[string][]string // per-compare-model transcript lines
	compareSTTStates map[string]string   // "idle" | "transcribing" per compare model

	width, height int

	cancel context.CancelFunc
}

func newModel(cancel context.CancelFunc, tracker *monitor.Tracker, modelNames []string) Model {
	m := Model{
		sttState:   "idle",
		cancel:     cancel,
		tracker:    tracker,
		modelNames: modelNames,
	}
	if len(modelNames) > 1 {
		m.compareLines = make(map[string][]string)
		m.compareSTTStates = make(map[string]string)
		for _, name := range modelNames[1:] {
			m.compareSTTStates[name] = "idle"
		}
	}
	return m
}

func (m Model) Init() tea.Cmd {
	return tickCmd()
}

func tickCmd() tea.Cmd {
	return tea.Tick(100*time.Millisecond, func(_ time.Time) tea.Msg {
		return TickMsg{}
	})
}

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height

	case TickMsg:
		// Re-arm the ticker and refresh the stats snapshot for the STATS pane.
		if m.tracker != nil {
			m.snap = m.tracker.Snapshot()
		}
		return m, tickCmd()

	case ChunkMsg:
		m.chunkN = msg.N
		m.rms = msg.RMS
		m.vad = msg.VADProb
		if msg.Speaking && !m.speaking {
			m.speakStart = time.Now()
		}
		m.speaking = msg.Speaking

	case SegmentStartMsg:
		m.sttState = "transcribing"
		for k := range m.compareSTTStates {
			m.compareSTTStates[k] = "transcribing"
		}

	case CompareTranscriptionMsg:
		m.compareSTTStates[msg.Model] = "idle"
		line := "> " + msg.Text
		if msg.Text == "" {
			line = stGray.Render("—")
		}
		m.compareLines[msg.Model] = append(m.compareLines[msg.Model], line)
		if len(m.compareLines[msg.Model]) > maxTranscriptLines {
			m.compareLines[msg.Model] = m.compareLines[msg.Model][1:]
		}

	case TranscriptionMsg:
		m.sttState = "idle"
		m = m.addLine(fmt.Sprintf("> %s", msg.Text))

	case TranscriptionEmptyMsg:
		m.sttState = "idle"

	case TranscriptionErrorMsg:
		m.sttState = "idle"
		m = m.addLine(fmt.Sprintf("✗ error: %v", msg.Err))

	case WakeWordMsg:
		m = m.addLine(fmt.Sprintf("◆ [Command] %s", msg.Command))

	case QueueDepthMsg:
		m.queueDepth = msg.Depth

	case SpeakerMsg:
		m.lastSpeakerName = msg.Name
		m.lastSpeakerScore = msg.Score
		m.lastSpeakerAccepted = msg.Accepted
		m.hasSpeakerResult = true
		// Also emit to transcript so speaker result appears inline with the text.
		if msg.Accepted {
			m = m.addLine(fmt.Sprintf("  └ [%s] %.2f ✓", msg.Name, msg.Score))
		} else {
			m = m.addLine(fmt.Sprintf("  └ [%s] %.2f ✗", msg.Name, msg.Score))
		}

	case tea.KeyMsg:
		switch msg.String() {
		case "q", "ctrl+c":
			m.cancel()
			return m, tea.Quit
		case "j", "down":
			if m.scrollOffset < len(m.lines)-1 {
				m.scrollOffset++
			}
		case "k", "up":
			if m.scrollOffset > 0 {
				m.scrollOffset--
			}
		}
	}
	return m, nil
}

// addLine appends a line and auto-scrolls to the bottom.
func (m Model) addLine(line string) Model {
	m.lines = append(m.lines, line)
	if len(m.lines) > maxTranscriptLines {
		m.lines = m.lines[len(m.lines)-maxTranscriptLines:]
	}
	m.scrollOffset = len(m.lines) - 1
	return m
}

func (m Model) View() string {
	if m.width == 0 {
		return "Initializing..."
	}
	if len(m.modelNames) > 1 {
		return m.viewMultiModel()
	}
	return m.viewSingleModel()
}

func (m Model) viewSingleModel() string {
	// Split width into two equal halves; each pane gets its share minus borders.
	halfW := m.width / 2
	leftInner := halfW - 4  // 2 border chars + 2 padding chars per side
	rightInner := m.width - halfW - 4

	// Height: reserve 1 line for status bar; each pane row is half the rest.
	usableH := m.height - 1
	topH := usableH / 2
	botH := usableH - topH
	paneInnerH := topH - 2 // minus top+bottom border lines

	audioPane := m.renderAudioPane(leftInner, paneInnerH)
	pipelinePane := m.renderPipelinePane(rightInner, paneInnerH)
	transcriptPane := m.renderTranscriptPane(leftInner, botH-2)
	statsPane := m.renderStatsPane(rightInner, botH-2)

	topRow := lipgloss.JoinHorizontal(lipgloss.Top, audioPane, pipelinePane)
	botRow := lipgloss.JoinHorizontal(lipgloss.Top, transcriptPane, statsPane)
	main := lipgloss.JoinVertical(lipgloss.Left, topRow, botRow)

	statusBar := stGray.Render("  ↑↓/jk scroll transcript · q quit")
	return lipgloss.JoinVertical(lipgloss.Left, main, statusBar)
}

func (m Model) viewMultiModel() string {
	n := len(m.modelNames)

	usableH := m.height - 1
	topH := usableH / 4
	statsH := usableH / 4
	transcriptH := usableH - topH - statsH

	// Top row: AUDIO (1/3 width) + PIPELINE (2/3 width).
	audioW := m.width / 3
	audioPane := m.renderAudioPane(audioW-4, topH-2)
	pipelinePane := m.renderPipelinePane(m.width-audioW-4, topH-2)
	topRow := lipgloss.JoinHorizontal(lipgloss.Top, audioPane, pipelinePane)

	// Transcript columns — side by side.
	colW := m.width / n
	var cols []string
	cols = append(cols,
		m.renderTranscriptColPane(colW-4, transcriptH-2, m.modelNames[0], m.lines, m.sttState))
	for i, name := range m.modelNames[1:] {
		innerW := colW - 4
		if i == len(m.modelNames)-2 {
			innerW = m.width - (n-1)*colW - 4
		}
		cols = append(cols,
			m.renderTranscriptColPane(innerW, transcriptH-2, name, m.compareLines[name], m.compareSTTStates[name]))
	}
	transcriptRow := lipgloss.JoinHorizontal(lipgloss.Top, cols...)

	// Single full-width stats pane.
	statsPane := m.renderCombinedStatsPane(m.width-4, statsH-2)

	main := lipgloss.JoinVertical(lipgloss.Left, topRow, transcriptRow, statsPane)
	statusBar := stGray.Render("  q quit")
	return lipgloss.JoinVertical(lipgloss.Left, main, statusBar)
}

func (m Model) renderAudioPane(w, h int) string {
	var sb strings.Builder
	sb.WriteString(stGray.Render(fmt.Sprintf("#%04d\n", m.chunkN)))
	sb.WriteString(fmt.Sprintf("MIC  %s %4.2f\n", barStr(m.rms*10.0, 10), m.rms))
	sb.WriteString(fmt.Sprintf("VAD  %s %4.2f\n", barStr(m.vad, 10), m.vad))

	return paneStyle.Width(w).Height(h).Render(
		headerStyle.Render("AUDIO") + "\n" + sb.String(),
	)
}

func (m Model) renderPipelinePane(w, h int) string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Mic    %s\n", stGreen.Render("● active")))

	if m.speaking {
		dur := time.Since(m.speakStart).Seconds()
		sb.WriteString(fmt.Sprintf("VAD    %s %.1fs\n", stGreen.Render("◉ speaking"), dur))
	} else {
		sb.WriteString(fmt.Sprintf("VAD    %s\n", stGray.Render("· idle")))
	}

	switch m.sttState {
	case "transcribing":
		sb.WriteString(fmt.Sprintf("STT    %s\n", stYellow.Render("⏳ transcribing...")))
	default:
		sb.WriteString(fmt.Sprintf("STT    %s\n", stGray.Render("· idle")))
	}

	sb.WriteString(fmt.Sprintf("Queue  %s\n", stGray.Render(
		fmt.Sprintf("%d segments pending", m.queueDepth),
	)))

	// SPK row: shows speaker verification result for the last segment.
	if m.hasSpeakerResult {
		if m.lastSpeakerAccepted {
			sb.WriteString(fmt.Sprintf("SPK    %s\n",
				stGreen.Render(fmt.Sprintf("[%s] %.2f ✓", m.lastSpeakerName, m.lastSpeakerScore))))
		} else {
			sb.WriteString(fmt.Sprintf("SPK    %s\n",
				stRed.Render(fmt.Sprintf("[%s] %.2f ✗", m.lastSpeakerName, m.lastSpeakerScore))))
		}
	} else {
		sb.WriteString(fmt.Sprintf("SPK    %s\n", stGray.Render("—")))
	}

	return paneStyle.Width(w).Height(h).Render(
		headerStyle.Render("PIPELINE") + "\n" + sb.String(),
	)
}

func (m Model) renderTranscriptPane(w, h int) string {
	innerH := h - 1 // reserve 1 line for the "TRANSCRIPT" header row
	if innerH < 1 {
		innerH = 1
	}

	// Compute visible slice: scrollOffset is the index of the last visible line.
	end := m.scrollOffset + 1
	if end > len(m.lines) {
		end = len(m.lines)
	}
	start := end - innerH
	if start < 0 {
		start = 0
	}

	var rows []string
	for _, line := range m.lines[start:end] {
		switch {
		case strings.HasPrefix(line, "◆"):
			rows = append(rows, stCyan.Render(line))
		case strings.HasPrefix(line, "✗"):
			rows = append(rows, stRed.Render(line))
		case strings.HasSuffix(line, "✗"):
			// Unknown speaker annotation (e.g. "  └ [Unknown] 0.42 ✗")
			rows = append(rows, stRed.Render(line))
		case strings.HasSuffix(line, "✓"):
			// Accepted speaker annotation (e.g. "  └ [Ashin] 0.81 ✓")
			rows = append(rows, stGreen.Render(line))
		default:
			rows = append(rows, stGreen.Render(line))
		}
	}
	// Pad to fill pane height so lipgloss border stays stable.
	for len(rows) < innerH {
		rows = append(rows, "")
	}

	return paneStyle.Width(w).Height(h).Render(
		headerStyle.Render("TRANSCRIPT") + "\n" + strings.Join(rows, "\n"),
	)
}

func (m Model) renderStatsPane(w, h int) string {
	var sb strings.Builder

	// Stage latencies — fixed display order: vad, stt, spk.
	sb.WriteString(stGray.Render("Stage Latencies") + "\n")
	for _, name := range []string{"vad", "stt", "spk"} {
		s, ok := m.snap.Stages[name]
		if !ok || s.Count == 0 {
			continue
		}
		sb.WriteString(fmt.Sprintf("  %-4s  last %-8s avg %s\n",
			strings.ToUpper(name), fmtDur(s.Last), fmtDur(s.Avg)))
	}

	sb.WriteString("\n")
	sb.WriteString(fmt.Sprintf("Queue   %s\n",
		stGray.Render(fmt.Sprintf("%d pending", m.queueDepth))))

	// Runtime section.
	sb.WriteString("\n")
	sb.WriteString(stGray.Render("Runtime") + "\n")
	sb.WriteString(fmt.Sprintf("  Goroutines   %-4d  GOMAXPROCS  %d\n",
		m.snap.Goroutines, m.snap.GOMAXPROCS))
	sb.WriteString(fmt.Sprintf("  Heap         %.1f MB / %.1f MB sys\n",
		m.snap.HeapAllocMB, m.snap.HeapSysMB))
	sb.WriteString(fmt.Sprintf("  Next GC      %.1f MB  (GC runs: %d)\n",
		m.snap.NextGCMB, m.snap.NumGC))

	return paneStyle.Width(w).Height(h).Render(
		headerStyle.Render("STATS") + "\n" + sb.String(),
	)
}

// renderTranscriptColPane renders a single transcript column for multi-model mode.
// Lines auto-scroll to the bottom; no manual scroll in this mode.
func (m Model) renderTranscriptColPane(w, h int, name string, lines []string, sttState string) string {
	innerH := h - 1 // reserve 1 line for the header row
	if innerH < 1 {
		innerH = 1
	}

	end := len(lines)
	start := end - innerH
	if start < 0 {
		start = 0
	}

	var rows []string
	for _, line := range lines[start:end] {
		switch {
		case strings.HasPrefix(line, "◆"):
			rows = append(rows, stCyan.Render(line))
		case strings.HasPrefix(line, "✗"):
			rows = append(rows, stRed.Render(line))
		case strings.HasSuffix(line, "✗"):
			rows = append(rows, stRed.Render(line))
		case strings.HasSuffix(line, "✓"):
			rows = append(rows, stGreen.Render(line))
		default:
			rows = append(rows, stGreen.Render(line))
		}
	}
	for len(rows) < innerH {
		rows = append(rows, "")
	}

	header := headerStyle.Render(name)
	if sttState == "transcribing" {
		header += " " + stYellow.Render("⏳")
	}
	return paneStyle.Width(w).Height(h).Render(header + "\n" + strings.Join(rows, "\n"))
}

// renderCombinedStatsPane renders a single full-width stats pane for multi-model mode.
// Rows: one per model (avg, max, runs), then process-wide runtime.
func (m Model) renderCombinedStatsPane(w, h int) string {
	var sb strings.Builder

	// Header row.
	sb.WriteString(fmt.Sprintf("  %-22s  %s  %s  %s\n",
		stGray.Render("model"),
		stGray.Render("avg      "),
		stGray.Render("max      "),
		stGray.Render("runs")))

	for i, name := range m.modelNames {
		stageName := "stt"
		if i > 0 {
			stageName = "stt:" + name
		}
		s, ok := m.snap.Stages[stageName]
		if !ok || s.Count == 0 {
			sb.WriteString(fmt.Sprintf("  %-22s  %s\n", name, stGray.Render("—")))
		} else {
			sb.WriteString(fmt.Sprintf("  %-22s  %-9s  %-9s  %d\n",
				name, fmtDur(s.Avg), fmtDur(s.Max), s.Count))
		}
	}

	sb.WriteString(fmt.Sprintf("\n  %s\n",
		stGray.Render(fmt.Sprintf("Goroutines %d   Heap %.1f MB   Queue %d pending",
			m.snap.Goroutines, m.snap.HeapAllocMB, m.queueDepth))))

	return paneStyle.Width(w).Height(h).Render(headerStyle.Render("STATS") + "\n" + sb.String())
}

// fmtDur formats a duration as a short human-readable string (e.g. "1.2ms", "870ms").
func fmtDur(d time.Duration) string {
	ms := float64(d) / float64(time.Millisecond)
	if ms < 0.1 {
		return fmt.Sprintf("%.0fµs", float64(d)/float64(time.Microsecond))
	}
	return fmt.Sprintf("%.1fms", ms)
}

// barStr renders a fixed-width cyan block bar for a value in [0,1].
func barStr(v float32, width int) string {
	if v > 1.0 {
		v = 1.0
	}
	if v < 0 {
		v = 0
	}
	filled := int(v * float32(width))
	bar := strings.Repeat("█", filled) + strings.Repeat("░", width-filled)
	return stCyan.Render(bar)
}

// ChunkRMS computes the root mean square of a PCM16 chunk, normalized to [0,1].
// Exported so pipeline.go can call it without importing "math" itself.
func ChunkRMS(chunk []byte) float32 {
	n := len(chunk) / 2
	if n == 0 {
		return 0
	}
	var sum float64
	for i := 0; i < n; i++ {
		s := int16(chunk[i*2]) | int16(chunk[i*2+1])<<8
		f := float64(s) / 32768.0
		sum += f * f
	}
	return float32(math.Sqrt(sum / float64(n)))
}
