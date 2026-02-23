// Vox - Atlas Voice Module
//
// Fully local speech recognition using Whisper.cpp.
// No API calls, no costs, runs on-device.
//
// Configuration priority: CLI flags > ENV vars > .env file > defaults
//
// Setup:
//   1. Install PortAudio: brew install portaudio
//   2. Download Whisper model: make setup-whisper
//   3. Copy internal/vox/.env.example to internal/vox/.env (optional)
//   4. Run: go run ./cmd/vox
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
	"strings"
	"syscall"

	"github.com/ashinsabu/atlas/internal/vox"
	"github.com/ashinsabu/atlas/internal/vox/config"
)

const (
	colorReset  = "\033[0m"
	colorGreen  = "\033[32m"
	colorYellow = "\033[33m"
	colorCyan   = "\033[36m"
	colorGray   = "\033[90m"
	colorBold   = "\033[1m"
	clearLine   = "\033[2K\r"
)

func main() {
	// CLI flags (override .env and defaults)
	modelPath := flag.String("model", "", "Path to Whisper model (overrides VOX_MODEL_PATH)")
	debug := flag.Bool("debug", false, "Enable debug output (overrides VOX_DEBUG)")
	mode := flag.String("mode", "", "STT mode: streaming|accurate (overrides VOX_MODE)")
	threshold := flag.Float64("threshold", 0, "Energy threshold 0.0-1.0 (overrides VOX_ENERGY_THRESHOLD)")
	flag.Parse()

	// Load config from .env and environment
	cfg, err := config.Load()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Config error: %v\n", err)
		os.Exit(1)
	}

	// Apply CLI overrides
	if *modelPath != "" {
		cfg.ModelPath = *modelPath
	}
	if *debug {
		cfg.Debug = true
	}
	if *mode != "" {
		cfg.Mode = *mode
	}
	if *threshold > 0 {
		cfg.EnergyThreshold = float32(*threshold)
	}

	// Auto-detect model if not specified
	if cfg.ModelPath == "" {
		cfg.ModelPath = findModel()
	}
	if cfg.ModelPath == "" {
		fmt.Println("Error: Whisper model not found")
		fmt.Println("")
		fmt.Println("To set up:")
		fmt.Println("  1. Run: make setup-whisper")
		fmt.Println("  2. Or download manually from:")
		fmt.Println("     https://huggingface.co/ggerganov/whisper.cpp/tree/main")
		fmt.Println("  3. Place in ./models/ directory")
		fmt.Println("  4. Or set VOX_MODEL_PATH in internal/vox/.env")
		os.Exit(1)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle shutdown
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		fmt.Println("\n\nShutting down...")
		cancel()
	}()

	// Initialize Vox from config
	fmt.Printf("%sLoading Whisper model: %s%s\n", colorGray, filepath.Base(cfg.ModelPath), colorReset)

	v, err := vox.NewFromConfig(cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to initialize Vox: %v\n", err)
		os.Exit(1)
	}
	defer v.Close()

	// Set up callbacks
	v.OnListening = func() {
		// Just show a subtle indicator - streaming will show the actual text
		fmt.Printf("%s● Listening...%s", colorYellow, colorReset)
	}

	// Streaming transcription - updates in place like Google
	v.OnPartialTranscript = func(text string, isFinal bool) {
		trimmed := strings.TrimSpace(text)

		// Filter hallucinations
		if isHallucination(trimmed) {
			return
		}

		if isFinal {
			// Final transcript - move to new line with green text
			fmt.Printf("%s%s> %s%s\n", clearLine, colorGreen, trimmed, colorReset)
		} else {
			// Partial transcript - update same line with gray text
			fmt.Printf("%s%s● %s%s", clearLine, colorGray, trimmed, colorReset)
		}
	}

	// Legacy callback (also filtered)
	v.OnTranscript = func(text string) {
		// OnPartialTranscript already handles display, this is just for wake word processing
	}

	v.OnWake = func() {
		fmt.Printf("%s✓ Wake word detected!%s\n", colorGreen, colorReset)
	}

	v.OnCommand = func(cmd string) {
		fmt.Println()
		fmt.Printf("%s╭─ Command ─────────────────────────────%s\n", colorCyan, colorReset)
		fmt.Printf("%s│%s %s\n", colorCyan, colorReset, cmd)
		fmt.Printf("%s╰────────────────────────────────────────%s\n", colorCyan, colorReset)
		fmt.Println()
		fmt.Printf("%s  [TODO: Send to Brain]%s\n", colorGray, colorReset)
		fmt.Println()
	}

	v.OnError = func(err error) {
		fmt.Printf("%s✗ Error: %v%s\n", colorYellow, err, colorReset)
	}

	// Energy visualization (debug mode)
	if cfg.Debug {
		lastBar := ""
		nearThresh := cfg.EnergyThreshold * 0.7
		v.OnEnergy = func(energy float32, isSpeech bool) {
			barLen := int(energy * 400)
			if barLen > 40 {
				barLen = 40
			}

			bar := strings.Repeat("█", barLen) + strings.Repeat("░", 40-barLen)

			color := colorGray
			label := "ambient"
			if isSpeech {
				color = colorGreen
				label = "SPEECH "
			} else if energy > nearThresh {
				color = colorYellow
				label = "near   "
			}

			newBar := fmt.Sprintf("%s%s[%s] %.3f %s%s", clearLine, color, bar, energy, label, colorReset)
			if newBar != lastBar {
				fmt.Print(newBar)
				lastBar = newBar
			}
		}
	}

	// Show startup banner
	printBanner(v.ModelInfo(), cfg)

	// Run
	if err := v.Run(ctx); err != nil && err != context.Canceled {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func printBanner(modelInfo string, cfg *config.Config) {
	fmt.Println()
	fmt.Println(colorCyan + colorBold + "╔══════════════════════════════════════╗" + colorReset)
	fmt.Println(colorCyan + colorBold + "║           ATLAS VOX MODULE           ║" + colorReset)
	fmt.Println(colorCyan + colorBold + "║         Local Whisper STT            ║" + colorReset)
	fmt.Println(colorCyan + colorBold + "╚══════════════════════════════════════╝" + colorReset)
	fmt.Println()
	fmt.Printf("%s  %s | Mode: %s%s\n", colorGray, modelInfo, cfg.Mode, colorReset)
	fmt.Println()
	fmt.Printf("%s  Wake: %q%s\n", colorGray, cfg.WakeWords, colorReset)
	fmt.Println(colorGray + "  Press Ctrl+C to exit" + colorReset)
	fmt.Println()
	if cfg.Debug {
		fmt.Printf("%s  DEBUG: threshold=%.3f silence=%dms min=%dms max=%dms%s\n",
			colorYellow,
			cfg.EnergyThreshold,
			cfg.SilenceTimeout.Milliseconds(),
			cfg.MinSpeechDuration.Milliseconds(),
			cfg.MaxSpeechDuration.Milliseconds(),
			colorReset)
		// Threshold marker
		threshPos := int(float64(cfg.EnergyThreshold) * 400)
		if threshPos > 40 {
			threshPos = 40
		}
		marker := strings.Repeat("░", threshPos) + "|" + strings.Repeat("░", 40-threshPos)
		fmt.Printf("%s  [%s] <- threshold%s\n", colorGray, marker, colorReset)
	}
	fmt.Println()
	fmt.Println(colorGreen + "◉ Ready - waiting for speech..." + colorReset)
	fmt.Println()
}

// isHallucination checks if text is a common Whisper hallucination.
func isHallucination(text string) bool {
	if len(text) < 4 {
		return true
	}

	lower := strings.ToLower(text)

	// Common Whisper hallucinations on short/silent audio
	hallucinations := []string{
		// Silence markers
		"[blank_audio]", "blank_audio", "(blank audio)", "[silence]",
		"[music]", "(music)", "[applause]", "(applause)",
		// YouTube artifacts
		"thanks for watching", "subscribe", "like and subscribe",
		"please subscribe", "don't forget to subscribe",
		// Single word noise
		"thank you.", "bye.", "you.", "the.", "i.", "okay.", "ok.",
		"yeah.", "yes.", "no.", "so.", "um.", "uh.", "hmm.",
		"you", "the", "i", "a", "okay", "ok", "yeah", "yes", "no",
		// Common short phrases from ambient noise
		"...", "..", ".", "-", "--",
	}
	for _, h := range hallucinations {
		if lower == h || lower == h+"." || lower == h+"!" || lower == h+"?" {
			return true
		}
	}

	// Filter text that's just punctuation or whitespace
	cleaned := strings.Trim(lower, " .!?,;:-\"'()[]")
	return len(cleaned) < 2
}

// findModel locates the Whisper model file.
func findModel() string {
	candidates := []string{
		"models/ggml-large.bin",
		"models/ggml-large-v3.bin",
		"models/ggml-medium.bin",
		"models/ggml-small.bin",
		"models/ggml-base.bin",
		"models/ggml-tiny.bin",
	}

	// Add home directory paths
	if home, err := os.UserHomeDir(); err == nil {
		candidates = append(candidates,
			filepath.Join(home, ".cache/whisper/ggml-large.bin"),
			filepath.Join(home, ".cache/whisper/ggml-small.bin"),
		)
	}

	for _, path := range candidates {
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}
	return ""
}
