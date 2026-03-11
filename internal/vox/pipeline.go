// Package vox - Pipeline orchestrates the voice processing stages.
//
// Pipeline flow:
//   Mic → AudioSource → SpeechDetector → STT → WakeWordDetector
//          (chunks)    (speech segments)  (text)    (commands)
//
// Speaker verification will be added later.
package vox

import (
	"context"
	"fmt"
	"strings"
	"sync"

	"github.com/ashinsabu/atlas/internal/vox/audio"
	"github.com/ashinsabu/atlas/internal/vox/speech"
	"github.com/ashinsabu/atlas/internal/vox/stt"
	"github.com/ashinsabu/atlas/internal/vox/wakeword"
)

// Pipeline orchestrates the voice processing stages.
type Pipeline struct {
	audio    audio.AudioSource
	speech   speech.SpeechDetector
	stt      stt.STT
	wakeword *wakeword.Detector

	// Callbacks
	OnCommand func(cmd string)    // Wake word + command detected
	OnText    func(text string)   // Transcription result
	OnError   func(err error)     // Processing error

	Debug bool

	mu     sync.RWMutex
	status string
}

// PipelineConfig holds configuration for creating a Pipeline.
type PipelineConfig struct {
	// Model paths
	WhisperModelPath string
	SileroModelPath  string

	// Speech detection config
	SpeechConfig speech.DetectorConfig

	// Audio config
	AudioConfig audio.SourceConfig

	// Whisper config
	Language string

	// Debug mode
	Debug bool
}

// DefaultPipelineConfig returns sensible defaults.
func DefaultPipelineConfig() PipelineConfig {
	return PipelineConfig{
		WhisperModelPath: "models/ggml-large-v3.bin",
		SileroModelPath:  "models/silero_vad.onnx",
		SpeechConfig:     speech.DefaultDetectorConfig(),
		AudioConfig:      audio.DefaultSourceConfig(),
		Language:         "en",
		Debug:            false,
	}
}

// NewPipeline creates a new voice processing pipeline.
func NewPipeline(cfg PipelineConfig) (*Pipeline, error) {
	// Create audio source
	audioSource, err := audio.NewPortAudioSource(cfg.AudioConfig)
	if err != nil {
		return nil, fmt.Errorf("audio source: %w", err)
	}

	// Create speech detector (Silero VAD)
	speechDetector, err := speech.NewSileroDetector(cfg.SileroModelPath, cfg.SpeechConfig)
	if err != nil {
		audioSource.Close()
		return nil, fmt.Errorf("speech detector: %w", err)
	}

	// Create STT engine
	whisperCfg := stt.WhisperConfig{
		ModelPath: cfg.WhisperModelPath,
		Language:  cfg.Language,
	}
	sttEngine, err := stt.NewWhisper(whisperCfg)
	if err != nil {
		speechDetector.Close()
		audioSource.Close()
		return nil, fmt.Errorf("stt engine: %w", err)
	}

	// Create wake word detector
	wakewordDetector := wakeword.New()

	return &Pipeline{
		audio:    audioSource,
		speech:   speechDetector,
		stt:      sttEngine,
		wakeword: wakewordDetector,
		Debug:    cfg.Debug,
		status:   "initialized",
	}, nil
}

// Run starts the pipeline. Blocks until context is cancelled.
func (p *Pipeline) Run(ctx context.Context) error {
	p.setStatus("running")

	return p.audio.Start(ctx, func(chunk []byte) {
		p.processChunk(chunk)
	})
}

// processChunk handles a single audio chunk through the pipeline.
func (p *Pipeline) processChunk(chunk []byte) {
	segments := p.speech.Process(chunk)

	for _, seg := range segments {
		p.processSegment(&seg)
	}
}

// processSegment handles a complete speech segment.
func (p *Pipeline) processSegment(seg *speech.SpeechSegment) {
	if p.Debug {
		fmt.Printf("[pipeline] speech: %.2fs\n", seg.Seconds())
	}

	// Transcribe
	p.setStatus("transcribing")
	text, err := p.stt.Transcribe(seg.Audio)
	p.setStatus("running")

	if err != nil {
		if p.OnError != nil {
			p.OnError(fmt.Errorf("transcribe: %w", err))
		}
		return
	}

	text = strings.TrimSpace(text)
	if text == "" {
		return
	}

	// Notify with transcription
	if p.OnText != nil {
		p.OnText(text)
	}

	// Check for wake word
	command := p.wakeword.Process(text, true)
	if command != "" && p.OnCommand != nil {
		p.OnCommand(command)
	}
}

// Close releases all pipeline resources.
func (p *Pipeline) Close() error {
	var errs []error

	if p.stt != nil {
		if err := p.stt.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	if p.speech != nil {
		if err := p.speech.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	if p.audio != nil {
		if err := p.audio.Close(); err != nil {
			errs = append(errs, err)
		}
	}

	if len(errs) > 0 {
		return fmt.Errorf("close: %v", errs)
	}
	return nil
}

// Status returns current pipeline status.
func (p *Pipeline) Status() string {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.status
}

func (p *Pipeline) setStatus(s string) {
	p.mu.Lock()
	p.status = s
	p.mu.Unlock()
}

// SetWakeWords configures the wake words.
func (p *Pipeline) SetWakeWords(words ...string) {
	p.wakeword.WithWakeWords(words...)
}
