// Vox - Atlas Voice Module
//
// Local speech recognition using Whisper.cpp + Silero VAD.
// No API calls, no costs, runs on-device.
//
// Setup:
//
//	make setup    # Install deps, download models
//	make vox      # Run voice assistant
//
// Say "Hey Atlas, <your command>" to interact.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"

	"github.com/ashinsabu/atlas/internal/vox"
	"github.com/ashinsabu/atlas/internal/vox/audio"
	"github.com/ashinsabu/atlas/internal/vox/config"
	"github.com/ashinsabu/atlas/internal/vox/debug"
	"github.com/ashinsabu/atlas/internal/vox/speaker"
	"github.com/ashinsabu/atlas/internal/vox/speech"
	"github.com/ashinsabu/atlas/internal/vox/stt"
)

const (
	colorReset = "\033[0m"
	colorGreen = "\033[32m"
	colorCyan  = "\033[36m"
	colorGray  = "\033[90m"
	colorRed   = "\033[31m"
	colorBold  = "\033[1m"
)

func main() {
	// Load YAML config first so CLI flags can override it.
	cfg, err := config.Load()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Warning: config load failed: %v (using defaults)\n", err)
		cfg = &config.VoxConfig{}
		cfg.ApplyDefaults()
	}

	v := cfg.Atlas.Vox

	// CLI flags — defaults come from YAML config so that CLI always wins.
	whisperModel := flag.String("whisper", v.WhisperModel, "Path to Whisper model (empty = auto-detect)")
	sileroModel := flag.String("silero", v.SileroModel, "Path to Silero VAD model")
	debugMode := flag.Bool("debug", false, "Enable debug TUI")
	modelsFlag := flag.String("models", "", "Comma-separated model names/paths for side-by-side comparison (debug only). First = primary.")
	logLevel := flag.String("log-level", v.LogLevel, "Log level: debug, info, warn, error")
	flag.Parse()

	// Configure structured logging.
	setupSlog(*logLevel)

	// Find Whisper model (auto-detect if not specified).
	modelPath := *whisperModel
	if modelPath == "" {
		modelPath = findWhisperModel()
	}

	// Build the ordered list of model paths and display names.
	// -models overrides -whisper and allows N engines for side-by-side comparison.
	var allModelPaths []string
	var allModelNames []string
	if *modelsFlag != "" {
		for _, raw := range strings.Split(*modelsFlag, ",") {
			p := resolveModelPath(strings.TrimSpace(raw))
			allModelPaths = append(allModelPaths, p)
			allModelNames = append(allModelNames, modelDisplayName(p))
		}
		modelPath = allModelPaths[0] // -models overrides -whisper
	} else {
		if modelPath == "" {
			fmt.Fprintln(os.Stderr, "Error: Whisper model not found")
			fmt.Fprintln(os.Stderr, "Run: make setup")
			os.Exit(1)
		}
		allModelPaths = []string{modelPath}
		allModelNames = []string{modelDisplayName(modelPath)}
	}

	// Build audio config: enable debug logs in debug mode.
	audioCfg := audio.DefaultSourceConfig()
	audioCfg.Debug = *debugMode

	// Build speech config from YAML.
	speechCfg := speech.DetectorConfig{
		SpeechThreshold: float32(v.Speech.Threshold),
		MinSpeechMs:     v.Speech.MinSpeechMs,
		MinSilenceMs:    v.Speech.MinSilenceMs,
		MaxSpeechMs:     v.Speech.MaxSpeechMs,
		SampleRate:      16000,
	}
	// Zero values mean "use Go defaults" which would be wrong; fall back to defaults.
	if speechCfg.SpeechThreshold == 0 {
		speechCfg = speech.DefaultDetectorConfig()
	}

	// Build wake words: explicit config takes priority; otherwise auto-generate from name.
	wakewords := v.Wakewords
	if len(wakewords) == 0 {
		name := strings.ToLower(v.Name)
		wakewords = []string{"hey " + name, "hi " + name, "yo " + name}
	}

	// Build pipeline config (Debugger filled in below based on mode).
	pipelineCfg := vox.PipelineConfig{
		WhisperModelPath: modelPath,
		SileroModelPath:  *sileroModel,
		SpeechConfig:     speechCfg,
		AudioConfig:      audioCfg,
		Language:         v.Language,
		WhisperPrompt:    v.WhisperPrompt,
		WakeWords:        wakewords,
	}

	// Set up speaker verification if enabled in config.
	if v.Speaker.Enabled {
		enc, err := speaker.NewEncoder(v.Speaker.ModelPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Warning: speaker encoder load failed: %v\n", err)
			fmt.Fprintln(os.Stderr, "  Speaker verification disabled.")
		} else {
			profile, err := speaker.LoadProfile(v.Speaker.ProfilePath)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Warning: speaker profile load failed: %v\n", err)
				fmt.Fprintln(os.Stderr, "  Run: make vox-enroll")
				enc.Close()
			} else if profile == nil {
				fmt.Fprintf(os.Stderr, "Warning: no speaker profile at %s\n", v.Speaker.ProfilePath)
				fmt.Fprintln(os.Stderr, "  Run: make vox-enroll")
				enc.Close()
			} else {
				verifier := speaker.NewVerifier(enc, profile, float32(v.Speaker.Threshold))
				verifier.SetShortThreshold(float32(v.Speaker.ShortThreshold), v.Speaker.ShortThresholdS)
				pipelineCfg.Speaker = verifier
				slog.Info("speaker verification enabled (wake-word-gated)", "speaker", profile.SpeakerName, "threshold", v.Speaker.Threshold)
			}
		}
	}

	if *debugMode {
		runDebug(pipelineCfg, *sileroModel, allModelNames, allModelPaths)
	} else {
		runNormal(pipelineCfg, modelPath, *sileroModel, v.Name)
	}
}

// runDebug runs the pipeline with the bubbletea TUI.
// allModelNames[0] and allModelPaths[0] are the primary model.
// allModelPaths[1:] are built into compare STT engines (side-by-side debug comparison).
func runDebug(cfg vox.PipelineConfig, sileroModel string, allModelNames, allModelPaths []string) {
	ctx, cancel := context.WithCancel(context.Background())

	// Signal handling: SIGINT/SIGTERM cancel context, which shuts down the pipeline,
	// which causes the goroutine to call prog.Quit().
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		cancel()
	}()

	// Build TUI + debugger before creating the pipeline so the pipeline can use
	// it from the very first chunk.
	prog, dbg, tracker := debug.New(cancel, allModelNames)
	cfg.Debugger = dbg
	cfg.Monitor = tracker

	fmt.Printf("%sLoading models...%s\n", colorGray, colorReset)
	for i, p := range allModelPaths {
		label := "Whisper"
		if i > 0 {
			label = fmt.Sprintf("Compare[%d]", i)
		}
		fmt.Printf("%s  %s: %s%s\n", colorGray, label, filepath.Base(p), colorReset)
	}
	fmt.Printf("%s  Silero:  %s%s\n", colorGray, filepath.Base(sileroModel), colorReset)

	// Build compare STT engines (nil in single-model mode).
	for i, path := range allModelPaths[1:] {
		eng, err := stt.NewWhisper(stt.WhisperConfig{
			ModelPath:     path,
			Language:      cfg.Language,
			InitialPrompt: cfg.WhisperPrompt,
		})
		if err != nil {
			fmt.Fprintf(os.Stderr, "Warning: compare model %q: %v\n", allModelNames[i+1], err)
			continue
		}
		defer eng.Close()
		cfg.CompareSTTs = append(cfg.CompareSTTs, vox.CompareSTT{
			Name:   allModelNames[i+1],
			Engine: eng,
		})
	}

	pipeline, err := vox.NewPipeline(cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create pipeline: %v\n", err)
		os.Exit(1)
	}
	defer pipeline.Close()

	// Pipeline runs in a background goroutine; TUI runs in the main goroutine.
	go func() {
		if err := pipeline.Run(ctx); err != nil && !errors.Is(err, context.Canceled) {
			prog.Send(debug.TranscriptionErrorMsg{Err: err})
		}
		prog.Quit()
	}()

	if _, err := prog.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "TUI error: %v\n", err)
		os.Exit(1)
	}
}

// runNormal runs the pipeline with plain terminal output.
func runNormal(cfg vox.PipelineConfig, modelPath, sileroModel, name string) {
	cfg.Debugger = debug.NopDebugger{}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		fmt.Println("\nShutting down...")
		cancel()
	}()

	fmt.Printf("%sLoading models...%s\n", colorGray, colorReset)
	fmt.Printf("%s  Whisper: %s%s\n", colorGray, filepath.Base(modelPath), colorReset)
	fmt.Printf("%s  Silero:  %s%s\n", colorGray, filepath.Base(sileroModel), colorReset)

	pipeline, err := vox.NewPipeline(cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create pipeline: %v\n", err)
		os.Exit(1)
	}
	defer pipeline.Close()

	// Track the last speaker so OnText can include it in the output.
	var lastSpeaker string

	pipeline.OnSpeakerVerified = func(name string, _ float32, _ bool) {
		lastSpeaker = name
	}

	pipeline.OnText = func(text string) {
		if lastSpeaker != "" {
			tag := colorGray + "[" + lastSpeaker + "]" + colorReset
			fmt.Printf("%s %s>%s %s\n", tag, colorGreen, colorReset, text)
		} else {
			fmt.Printf("%s>%s %s\n", colorGreen, colorReset, text)
		}
		lastSpeaker = "" // Reset after use
	}

	pipeline.OnCommand = func(cmd string) {
		fmt.Println()
		fmt.Printf("%s%s[Command]%s %s\n", colorBold, colorCyan, colorReset, cmd)
		fmt.Println()
	}

	pipeline.OnError = func(err error) {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
	}

	fmt.Println()
	fmt.Printf("%s%s%s VOX%s\n", colorBold, colorCyan, strings.ToUpper(name), colorReset)
	fmt.Printf("%sSay \"Hey %s\" to wake%s\n", colorGray, name, colorReset)
	fmt.Printf("%sPress Ctrl+C to exit%s\n", colorGray, colorReset)
	fmt.Println()
	fmt.Printf("%sListening...%s\n", colorGreen, colorReset)
	fmt.Println()

	if err := pipeline.Run(ctx); err != nil && !errors.Is(err, context.Canceled) {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

// setupSlog configures the global slog logger.
func setupSlog(level string) {
	var l slog.Level
	switch level {
	case "debug":
		l = slog.LevelDebug
	case "warn":
		l = slog.LevelWarn
	case "error":
		l = slog.LevelError
	default:
		l = slog.LevelInfo
	}
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: l})))
}

// resolveModelPath converts a short model name (e.g. "large-v3") to its file path.
// If the name contains a "/" or ends with ".bin", it is used as-is (full path).
func resolveModelPath(name string) string {
	if strings.Contains(name, "/") || strings.HasSuffix(name, ".bin") {
		return name
	}
	return "models/ggml-" + name + ".bin"
}

// modelDisplayName extracts a short display name from a model file path.
// e.g. "models/ggml-large-v3.bin" → "large-v3"
func modelDisplayName(path string) string {
	base := filepath.Base(path)
	base = strings.TrimPrefix(base, "ggml-")
	return strings.TrimSuffix(base, ".bin")
}

func findWhisperModel() string {
	candidates := []string{
		"models/ggml-large-v3.bin",
		"models/ggml-distil-large-v3.bin",
		"models/ggml-large.bin",
		"models/ggml-medium.bin",
		"models/ggml-small.bin",
		"models/ggml-base.bin",
	}

	for _, path := range candidates {
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}
	return ""
}
