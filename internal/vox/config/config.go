// Package config provides centralized configuration for the Vox module.
//
// All settings can be overridden via environment variables or a .env file
// placed at internal/vox/.env (gitignored).
//
// Usage:
//
//	cfg, err := config.Load()
//	v, err := vox.NewWithConfig(cfg)
package config

import (
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"time"

	"github.com/joho/godotenv"
)

// Config holds all Vox module configuration.
type Config struct {
	// ----- Whisper STT Settings -----

	// ModelPath: Path to Whisper GGML model file (e.g., "models/ggml-large.bin")
	// Larger models = better accuracy, slower inference
	// Options: ggml-tiny.bin, ggml-base.bin, ggml-small.bin, ggml-medium.bin, ggml-large.bin
	ModelPath string

	// Language: Whisper transcription language code (default: "en")
	// See: https://github.com/openai/whisper#available-models-and-languages
	Language string

	// Mode: STT processing mode
	// "streaming" = 2s max utterance, faster feedback
	// "accurate"  = 5s max utterance, better context for accuracy
	Mode string

	// ----- STT Tuning -----

	// Temperature: Whisper decoding temperature (default: 0.0)
	// 0.0 = deterministic (no hallucinations), higher = more variation
	Temperature float32

	// BeamSize: Number of beam search paths (default: 5)
	// Higher = better accuracy at cost of speed
	BeamSize int

	// InitialPrompt: Bias Whisper toward expected vocabulary
	// Include domain-specific words like "atlas", "planche", etc.
	InitialPrompt string

	// ----- Voice Activity Detection (VAD) -----

	// EnergyThreshold: RMS energy level to trigger speech detection (0.0-1.0)
	// Lower = more sensitive (picks up quiet speech), Higher = less sensitive
	// Typical values: 0.01-0.05 depending on mic/environment
	EnergyThreshold float32

	// SilenceTimeout: Duration of silence before speech segment ends
	// Longer = allows thinking pauses, Shorter = faster response
	SilenceTimeout time.Duration

	// MinSpeechDuration: Minimum speech duration to process (filters noise bursts)
	// Utterances shorter than this are discarded as noise
	MinSpeechDuration time.Duration

	// MaxSpeechDuration: Maximum speech duration before forcing segment end
	// Prevents runaway recording from continuous audio (TV, music, etc.)
	MaxSpeechDuration time.Duration

	// StreamingInterval: How often to emit partial transcripts during speech
	// Set to 0 to disable streaming (wait for complete utterance)
	// Typical: 500ms for responsive streaming
	StreamingInterval time.Duration

	// ----- Audio Capture -----

	// SampleRate: Audio sample rate in Hz (default: 16000)
	// Whisper requires 16kHz mono audio
	SampleRate int

	// Channels: Number of audio channels (default: 1 = mono)
	// Whisper requires mono audio
	Channels int

	// FramesPerBuffer: Audio frames per capture buffer (default: 1600 = 100ms at 16kHz)
	// Smaller = lower latency, larger = more efficient
	FramesPerBuffer int

	// ----- Wake Word -----

	// WakeWords: Comma-separated trigger phrases (default: "hey atlas,atlas")
	// Case-insensitive matching against transcripts
	WakeWords string

	// ListenTimeout: How long to wait for command after wake word (default: 10s)
	ListenTimeout time.Duration

	// ----- Debug -----

	// Debug: Enable verbose debug output including energy meter
	Debug bool
}

// Default returns configuration with sensible defaults.
func Default() *Config {
	return &Config{
		// STT
		ModelPath:   "",   // Auto-detect
		Language:    "en",
		Mode:        "streaming",
		Temperature:   0.0, // Deterministic decoding
		BeamSize:      5,   // Balanced accuracy/speed
		InitialPrompt: "",  // No bias - transcribe naturally

		// VAD
		EnergyThreshold:   0.05,                   // Less sensitive - reduce false triggers
		SilenceTimeout:    800 * time.Millisecond, // 800ms gap = end of utterance
		MinSpeechDuration: 500 * time.Millisecond, // Filter <500ms noise bursts
		MaxSpeechDuration: 5 * time.Second,        // Allow complete sentences
		StreamingInterval: 500 * time.Millisecond, // Update transcript every 500ms while speaking

		// Audio
		SampleRate:      16000, // Required by Whisper
		Channels:        1,     // Required by Whisper (mono)
		FramesPerBuffer: 1600,  // 100ms chunks

		// Wake word
		WakeWords:     "hey atlas,atlas",
		ListenTimeout: 10 * time.Second,

		// Debug
		Debug: false,
	}
}

// Load reads configuration from .env file and environment variables.
// Priority: ENV vars > .env file > defaults
func Load() (*Config, error) {
	cfg := Default()

	// Try to load .env from internal/vox/.env
	envPath := findEnvFile()
	if envPath != "" {
		_ = godotenv.Load(envPath) // Ignore error if file doesn't exist
	}

	// Override from environment
	if v := os.Getenv("VOX_MODEL_PATH"); v != "" {
		cfg.ModelPath = v
	}
	if v := os.Getenv("VOX_LANGUAGE"); v != "" {
		cfg.Language = v
	}
	if v := os.Getenv("VOX_MODE"); v != "" {
		cfg.Mode = v
	}
	if v := os.Getenv("VOX_TEMPERATURE"); v != "" {
		if f, err := strconv.ParseFloat(v, 32); err == nil {
			cfg.Temperature = float32(f)
		}
	}
	if v := os.Getenv("VOX_BEAM_SIZE"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			cfg.BeamSize = n
		}
	}
	if v := os.Getenv("VOX_INITIAL_PROMPT"); v != "" {
		cfg.InitialPrompt = v
	}
	if v := os.Getenv("VOX_ENERGY_THRESHOLD"); v != "" {
		if f, err := strconv.ParseFloat(v, 32); err == nil {
			cfg.EnergyThreshold = float32(f)
		}
	}
	if v := os.Getenv("VOX_SILENCE_TIMEOUT_MS"); v != "" {
		if ms, err := strconv.Atoi(v); err == nil {
			cfg.SilenceTimeout = time.Duration(ms) * time.Millisecond
		}
	}
	if v := os.Getenv("VOX_MIN_SPEECH_MS"); v != "" {
		if ms, err := strconv.Atoi(v); err == nil {
			cfg.MinSpeechDuration = time.Duration(ms) * time.Millisecond
		}
	}
	if v := os.Getenv("VOX_MAX_SPEECH_MS"); v != "" {
		if ms, err := strconv.Atoi(v); err == nil {
			cfg.MaxSpeechDuration = time.Duration(ms) * time.Millisecond
		}
	}
	if v := os.Getenv("VOX_STREAMING_INTERVAL_MS"); v != "" {
		if ms, err := strconv.Atoi(v); err == nil {
			cfg.StreamingInterval = time.Duration(ms) * time.Millisecond
		}
	}
	if v := os.Getenv("VOX_SAMPLE_RATE"); v != "" {
		if sr, err := strconv.Atoi(v); err == nil {
			cfg.SampleRate = sr
		}
	}
	if v := os.Getenv("VOX_CHANNELS"); v != "" {
		if ch, err := strconv.Atoi(v); err == nil {
			cfg.Channels = ch
		}
	}
	if v := os.Getenv("VOX_FRAMES_PER_BUFFER"); v != "" {
		if fpb, err := strconv.Atoi(v); err == nil {
			cfg.FramesPerBuffer = fpb
		}
	}
	if v := os.Getenv("VOX_WAKE_WORDS"); v != "" {
		cfg.WakeWords = v
	}
	if v := os.Getenv("VOX_LISTEN_TIMEOUT_S"); v != "" {
		if s, err := strconv.Atoi(v); err == nil {
			cfg.ListenTimeout = time.Duration(s) * time.Second
		}
	}
	if v := os.Getenv("VOX_DEBUG"); v != "" {
		cfg.Debug = v == "true" || v == "1"
	}

	// Apply mode-based MaxSpeechDuration if not explicitly set
	if os.Getenv("VOX_MAX_SPEECH_MS") == "" {
		switch cfg.Mode {
		case "streaming":
			cfg.MaxSpeechDuration = 5 * time.Second // Allow complete sentences
		case "accurate":
			cfg.MaxSpeechDuration = 10 * time.Second // Extended for longer monologues
		}
	}

	return cfg, nil
}

// findEnvFile looks for .env file in the vox module directory.
func findEnvFile() string {
	// Try relative to this source file
	_, thisFile, _, ok := runtime.Caller(0)
	if ok {
		// internal/vox/config/config.go -> internal/vox/.env
		voxDir := filepath.Dir(filepath.Dir(thisFile))
		envPath := filepath.Join(voxDir, ".env")
		if _, err := os.Stat(envPath); err == nil {
			return envPath
		}
	}

	// Fallback: try current working directory
	if _, err := os.Stat(".env"); err == nil {
		return ".env"
	}

	return ""
}
