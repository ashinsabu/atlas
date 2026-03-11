// Vox - Atlas Voice Module
//
// Local speech recognition using Whisper.cpp + Silero VAD.
// No API calls, no costs, runs on-device.
//
// Setup:
//   make setup    # Install deps, download models
//   make vox      # Run voice assistant
//
// Say "Hey Atlas, <your command>" to interact.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"

	"github.com/ashinsabu/atlas/internal/vox"
	"github.com/ashinsabu/atlas/internal/vox/audio"
	"github.com/ashinsabu/atlas/internal/vox/debug"
	"github.com/ashinsabu/atlas/internal/vox/speech"
)

const (
	colorReset = "\033[0m"
	colorGreen = "\033[32m"
	colorCyan  = "\033[36m"
	colorGray  = "\033[90m"
	colorBold  = "\033[1m"
)

func main() {
	// CLI flags
	whisperModel := flag.String("whisper", "", "Path to Whisper model")
	sileroModel := flag.String("silero", "models/silero_vad.onnx", "Path to Silero VAD model")
	debugMode := flag.Bool("debug", false, "Enable debug TUI")
	flag.Parse()

	// Find Whisper model
	modelPath := *whisperModel
	if modelPath == "" {
		modelPath = findWhisperModel()
	}
	if modelPath == "" {
		fmt.Fprintln(os.Stderr, "Error: Whisper model not found")
		fmt.Fprintln(os.Stderr, "Run: make setup")
		os.Exit(1)
	}

	// Create pipeline config (Debugger filled in below based on mode)
	cfg := vox.PipelineConfig{
		WhisperModelPath: modelPath,
		SileroModelPath:  *sileroModel,
		SpeechConfig:     speech.DefaultDetectorConfig(),
		AudioConfig:      audio.DefaultSourceConfig(),
		Language:         "en",
	}

	if *debugMode {
		runDebug(cfg, modelPath, *sileroModel)
	} else {
		runNormal(cfg, modelPath, *sileroModel)
	}
}

// runDebug runs the pipeline with the bubbletea TUI.
func runDebug(cfg vox.PipelineConfig, modelPath, sileroModel string) {
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
	prog, dbg := debug.New(cancel)
	cfg.Debugger = dbg

	fmt.Printf("%sLoading models...%s\n", colorGray, colorReset)
	fmt.Printf("%s  Whisper: %s%s\n", colorGray, filepath.Base(modelPath), colorReset)
	fmt.Printf("%s  Silero:  %s%s\n", colorGray, filepath.Base(sileroModel), colorReset)

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
func runNormal(cfg vox.PipelineConfig, modelPath, sileroModel string) {
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

	pipeline.OnText = func(text string) {
		fmt.Printf("%s> %s%s\n", colorGreen, text, colorReset)
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
	fmt.Printf("%s%sATLAS VOX%s\n", colorBold, colorCyan, colorReset)
	fmt.Printf("%sSay \"Hey Atlas\" to wake%s\n", colorGray, colorReset)
	fmt.Printf("%sPress Ctrl+C to exit%s\n", colorGray, colorReset)
	fmt.Println()
	fmt.Printf("%sListening...%s\n", colorGreen, colorReset)
	fmt.Println()

	if err := pipeline.Run(ctx); err != nil && !errors.Is(err, context.Canceled) {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
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
