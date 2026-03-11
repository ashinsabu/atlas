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
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"

	"github.com/ashinsabu/atlas/internal/vox"
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
	debug := flag.Bool("debug", false, "Enable debug output")
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

	// Create pipeline config
	cfg := vox.PipelineConfig{
		WhisperModelPath: modelPath,
		SileroModelPath:  *sileroModel,
		SpeechConfig:     speech.DefaultDetectorConfig(),
		Language:         "en",
		Debug:            *debug,
	}

	// Setup context with signal handling
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		fmt.Println("\nShutting down...")
		cancel()
	}()

	// Create pipeline
	fmt.Printf("%sLoading models...%s\n", colorGray, colorReset)
	fmt.Printf("%s  Whisper: %s%s\n", colorGray, filepath.Base(modelPath), colorReset)
	fmt.Printf("%s  Silero:  %s%s\n", colorGray, filepath.Base(*sileroModel), colorReset)

	pipeline, err := vox.NewPipeline(cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create pipeline: %v\n", err)
		os.Exit(1)
	}
	defer pipeline.Close()

	// Set up callbacks
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

	// Print banner
	fmt.Println()
	fmt.Printf("%s%sATLAS VOX%s\n", colorBold, colorCyan, colorReset)
	fmt.Printf("%sSay \"Hey Atlas\" to wake%s\n", colorGray, colorReset)
	fmt.Printf("%sPress Ctrl+C to exit%s\n", colorGray, colorReset)
	fmt.Println()
	fmt.Printf("%sListening...%s\n", colorGreen, colorReset)
	fmt.Println()

	// Run
	if err := pipeline.Run(ctx); err != nil && err != context.Canceled {
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
