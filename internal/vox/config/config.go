// Package config provides YAML-based configuration for Vox.
//
// Config file location (in priority order):
//  1. $ATLAS_CONFIG_DIR/vox.yaml   (env var — overrides all)
//  2. ./vox.yaml                   (local project root)
//  3. built-in defaults
//
// Set ATLAS_CONFIG_DIR to point at a shared directory (e.g. /etc/atlas) for
// multi-environment deployments. Leave unset for local development.
//
// Priority: CLI flags > YAML file > hardcoded defaults.
package config

import (
	"fmt"
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"
)

// SpeechCfg holds VAD/segmentation parameters.
type SpeechCfg struct {
	Threshold    float64 `yaml:"threshold"`
	MinSilenceMs int     `yaml:"min_silence_ms"`
	MinSpeechMs  int     `yaml:"min_speech_ms"`
	MaxSpeechMs  int     `yaml:"max_speech_ms"`
}

// SpeakerCfg holds speaker verification parameters.
type SpeakerCfg struct {
	Enabled          bool    `yaml:"enabled"`
	ProfilePath      string  `yaml:"profile_path"`
	ModelPath        string  `yaml:"model_path"`
	Threshold        float64 `yaml:"threshold"`
	ShortThreshold   float64 `yaml:"short_threshold"`
	ShortThresholdS  float64 `yaml:"short_threshold_secs"`
}

// VoxCfg is the `atlas.vox` namespace.
type VoxCfg struct {
	Name           string     `yaml:"name"`
	WhisperModel   string     `yaml:"whisper_model"`
	SileroModel    string     `yaml:"silero_model"`
	Language       string     `yaml:"language"`
	LogLevel       string     `yaml:"log_level"`
	WhisperPrompt  string     `yaml:"whisper_prompt"`
	Speech         SpeechCfg  `yaml:"speech"`
	Wakewords      []string   `yaml:"wakewords"`
	ListenTimeoutS int        `yaml:"listen_timeout_s"`
	Speaker        SpeakerCfg `yaml:"speaker"`
}

// VoxConfig is the top-level config struct that mirrors the YAML hierarchy.
type VoxConfig struct {
	Atlas struct {
		Vox VoxCfg `yaml:"vox"`
	} `yaml:"atlas"`
}

// ConfigDir returns the atlas configuration directory.
// Priority: $ATLAS_CONFIG_DIR env var → current directory.
func ConfigDir() string {
	if dir := os.Getenv("ATLAS_CONFIG_DIR"); dir != "" {
		return dir
	}
	return "."
}

// Load reads the vox config file and returns a populated VoxConfig.
// Returns defaults if no config file is found.
func Load() (*VoxConfig, error) {
	dir := ConfigDir()
	if cfg, err := loadFile(filepath.Join(dir, "vox.yaml")); err == nil {
		return cfg, nil
	}

	// No config file found — return defaults.
	cfg := &VoxConfig{}
	cfg.ApplyDefaults()
	return cfg, nil
}

func loadFile(path string) (*VoxConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var cfg VoxConfig
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	cfg.ApplyDefaults()
	return &cfg, nil
}

// ApplyDefaults fills zero-valued fields with package-level defaults.
// Called automatically by Load and loadFile.
func (c *VoxConfig) ApplyDefaults() {
	v := &c.Atlas.Vox

	if v.Name == "" {
		v.Name = "atlas"
	}
	if v.SileroModel == "" {
		v.SileroModel = "models/silero_vad.onnx"
	}
	if v.Language == "" {
		v.Language = "en"
	}
	if v.LogLevel == "" {
		v.LogLevel = "info"
	}
	if v.Speech.Threshold == 0 {
		v.Speech.Threshold = 0.5
	}
	if v.Speech.MinSilenceMs == 0 {
		v.Speech.MinSilenceMs = 1500
	}
	if v.Speech.MinSpeechMs == 0 {
		v.Speech.MinSpeechMs = 250
	}
	if v.Speech.MaxSpeechMs == 0 {
		v.Speech.MaxSpeechMs = 30000
	}
	if v.ListenTimeoutS == 0 {
		v.ListenTimeoutS = 10
	}
	if v.Speaker.ProfilePath == "" {
		v.Speaker.ProfilePath = filepath.Join(ConfigDir(), "speaker.json")
	}
	if v.Speaker.ModelPath == "" {
		v.Speaker.ModelPath = "models/wespeaker-resnet34.onnx"
	}
	if v.Speaker.Threshold == 0 {
		// Calibrated at 0.70: TPR=100% on 25 bootstrap recordings (min score 0.719).
		// Raise to 0.75 if false-accept rate is too high after testing with other-speaker data.
		v.Speaker.Threshold = 0.70
	}
	if v.Speaker.ShortThreshold == 0 {
		// Short utterances produce noisier embeddings. Observed enrolled-speaker
		// scores: 0.45–0.59 for <2s segments vs 0.76–0.79 for >2s. Other-speaker
		// scores stay low (max 0.27 observed). 0.40 leaves a clear margin.
		v.Speaker.ShortThreshold = 0.40
	}
	if v.Speaker.ShortThresholdS == 0 {
		v.Speaker.ShortThresholdS = 2.0
	}
	// WhisperPrompt intentionally has no default — an empty string means no prompt,
	// which is the safe choice. Set it explicitly in vox.yaml only if needed.
}
