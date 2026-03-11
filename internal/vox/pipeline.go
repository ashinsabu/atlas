// Package vox - Pipeline orchestrates the voice processing stages.
//
// Pipeline flow:
//
//	Mic → Source → SpeechDetector → STT → WakeWordDetector → SpeakerVerifier
//	       (chunks)    (speech segments)  (text)    (commands)      (wake only)
//
// Speaker verification runs only when the wake word fires, not on every segment.
package vox

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/ashinsabu/atlas/internal/monitor"
	"github.com/ashinsabu/atlas/internal/vox/audio"
	"github.com/ashinsabu/atlas/internal/vox/debug"
	"github.com/ashinsabu/atlas/internal/vox/speaker"
	"github.com/ashinsabu/atlas/internal/vox/speech"
	"github.com/ashinsabu/atlas/internal/vox/stt"
	"github.com/ashinsabu/atlas/internal/vox/wakeword"
)

// CompareSTT holds a named secondary STT engine for debug comparison.
// Only populated in debug mode via PipelineConfig.CompareSTTs.
type CompareSTT struct {
	Name   string
	Engine stt.STT
}

// Pipeline orchestrates the voice processing stages.
type Pipeline struct {
	audio    audio.Source
	speech   speech.SpeechDetector
	stt      stt.STT
	wakeword *wakeword.Detector
	verifier *speaker.Verifier // nil = speaker verification disabled (pass-all)

	compareSTTs []CompareSTT // nil in normal mode; populated in debug multi-model mode

	// segCh: audio callback → relay goroutine. Small buffer; relay drains it instantly.
	segCh chan speech.SpeechSegment
	// sttCh: relay goroutine → STT goroutine. Relay holds a dynamic queue so
	// the audio callback never blocks and no segments are ever dropped.
	sttCh chan speech.SpeechSegment

	// Callbacks
	OnCommand          func(cmd string)                         // Wake word + command detected
	OnText             func(text string)                        // Transcription result
	OnError            func(err error)                          // Processing error
	OnSpeakerVerified  func(name string, score float32, accepted bool) // Speaker verification result

	debugger debug.Debugger
	mon      *monitor.Tracker // nil = monitoring disabled (no-ops)

	// VAD state written by OnFrame (inside speech.Process), read by processChunk.
	// Both accesses happen on the single audio-callback goroutine — no mutex needed.
	vadProb     float32
	vadSpeaking bool

	// wakeJustFired is set by wakeword.OnWake and read in processSegment.
	// Both happen on the sttLoop goroutine — no mutex needed.
	wakeJustFired bool

	// chunkN counts processed chunks; used only by the debugger.
	chunkN int

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
	Language      string
	WhisperPrompt string

	// Speaker is an optional speaker verifier. nil = verification disabled (all speakers pass).
	Speaker *speaker.Verifier

	// Debugger receives pipeline events. Use debug.NopDebugger{} when not debugging.
	Debugger debug.Debugger

	// Monitor records per-stage latencies. nil = monitoring disabled.
	Monitor *monitor.Tracker

	// WakeWords sets the wake phrases. If empty, the detector uses no default wake words.
	WakeWords []string

	// CompareSTTs holds secondary STT engines for debug comparison. Only used in debug mode.
	CompareSTTs []CompareSTT
}

// DefaultPipelineConfig returns sensible defaults.
func DefaultPipelineConfig() PipelineConfig {
	return PipelineConfig{
		WhisperModelPath: "models/ggml-large-v3.bin",
		SileroModelPath:  "models/silero_vad.onnx",
		SpeechConfig:     speech.DefaultDetectorConfig(),
		AudioConfig:      audio.DefaultSourceConfig(),
		Language:         "en",
		Debugger:         debug.NopDebugger{},
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
		ModelPath:     cfg.WhisperModelPath,
		Language:      cfg.Language,
		InitialPrompt: cfg.WhisperPrompt,
	}
	sttEngine, err := stt.NewWhisper(whisperCfg)
	if err != nil {
		speechDetector.Close()
		audioSource.Close()
		return nil, fmt.Errorf("stt engine: %w", err)
	}

	// Create wake word detector
	wakewordDetector := wakeword.New()
	if len(cfg.WakeWords) > 0 {
		wakewordDetector.WithWakeWords(cfg.WakeWords...)
	}

	dbg := cfg.Debugger
	if dbg == nil {
		dbg = debug.NopDebugger{}
	}

	p := &Pipeline{
		audio:       audioSource,
		speech:      speechDetector,
		stt:         sttEngine,
		wakeword:    wakewordDetector,
		verifier:    cfg.Speaker,
		compareSTTs: cfg.CompareSTTs,
		segCh:       make(chan speech.SpeechSegment, 4),
		sttCh:       make(chan speech.SpeechSegment, 1),
		debugger:    dbg,
		mon:         cfg.Monitor,
		status:      "initialized",
	}

	// OnFrame is called inside speech.Process every 32ms.
	// Store VAD state for processChunk to include in the chunk event.
	speechDetector.OnFrame = func(prob float32, speaking bool) {
		p.vadProb = prob
		p.vadSpeaking = speaking
	}

	// OnWake fires on the sttLoop goroutine when the wake phrase is first detected.
	// Set wakeJustFired so processSegment can trigger speaker verification.
	wakewordDetector.OnWake = func() { p.wakeJustFired = true }

	return p, nil
}

// Run starts the pipeline. Blocks until context is cancelled.
func (p *Pipeline) Run(ctx context.Context) error {
	p.setStatus("running")

	// relayLoop bridges the fast audio callback to the slow STT goroutine.
	// It holds an in-memory queue so no segments are ever dropped.
	go p.relayLoop(ctx)
	// sttLoop processes segments one at a time from the relay.
	go p.sttLoop(ctx)

	return p.audio.Start(ctx, func(chunk []byte) {
		p.processChunk(chunk)
	})
}

// relayLoop maintains an unbounded in-memory queue between the audio callback
// (which sends to segCh) and the STT goroutine (which reads from sttCh).
// This is the standard Go pattern for an unbounded channel.
func (p *Pipeline) relayLoop(ctx context.Context) {
	var queue []speech.SpeechSegment
	for {
		if len(queue) == 0 {
			// Nothing queued — just wait for the next segment.
			select {
			case <-ctx.Done():
				return
			case seg := <-p.segCh:
				queue = append(queue, seg)
				p.debugger.OnQueueDepth(len(queue))
			}
		} else {
			// Have segments queued — race between accepting more and forwarding to STT.
			select {
			case <-ctx.Done():
				return
			case seg := <-p.segCh:
				queue = append(queue, seg)
				p.debugger.OnQueueDepth(len(queue))
			case p.sttCh <- queue[0]:
				queue = queue[1:]
				p.debugger.OnQueueDepth(len(queue))
			}
		}
	}
}

// sttLoop transcribes segments sequentially as the relay delivers them.
func (p *Pipeline) sttLoop(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case seg := <-p.sttCh:
			p.processSegment(&seg)
		}
	}
}

// processChunk handles a single audio chunk through the pipeline.
func (p *Pipeline) processChunk(chunk []byte) {
	rms := debug.ChunkRMS(chunk)

	// speech.Process triggers OnFrame which updates p.vadProb / p.vadSpeaking.
	t0vad := time.Now()
	segments := p.speech.Process(chunk)
	if p.mon != nil {
		p.mon.Record("vad", time.Since(t0vad))
	}

	p.chunkN++
	p.debugger.OnChunk(p.chunkN, rms, p.vadProb, p.vadSpeaking)

	for _, seg := range segments {
		p.segCh <- seg
	}
}

// processSegment handles a complete speech segment.
func (p *Pipeline) processSegment(seg *speech.SpeechSegment) {
	p.debugger.OnSegmentStart(seg.Seconds())

	// Transcribe with primary model.
	p.setStatus("transcribing")
	t0 := time.Now()
	text, err := p.stt.Transcribe(seg.Audio)
	elapsed := time.Since(t0)
	if p.mon != nil {
		p.mon.Record("stt", elapsed)
	}
	p.setStatus("running")

	if err != nil {
		p.debugger.OnTranscriptionError(err)
		if p.OnError != nil {
			p.OnError(fmt.Errorf("transcribe: %w", err))
		}
		return
	}

	text = strings.TrimSpace(text)
	if text == "" {
		p.debugger.OnTranscriptionEmpty(elapsed.Milliseconds())
		return
	}

	p.debugger.OnTranscription(text, elapsed.Milliseconds())

	// Run compare STTs sequentially after primary result is delivered.
	for _, cmp := range p.compareSTTs {
		t0 := time.Now()
		cmpText, _ := cmp.Engine.Transcribe(seg.Audio)
		cmpElapsed := time.Since(t0)
		if p.mon != nil {
			p.mon.Record("stt:"+cmp.Name, cmpElapsed)
		}
		p.debugger.OnCompareTranscription(cmp.Name, strings.TrimSpace(cmpText), cmpElapsed.Milliseconds())
	}

	// Check for wake word. OnWake sets p.wakeJustFired when the wake phrase is
	// detected for the first time in this segment (not on follow-up Listening segments).
	p.wakeJustFired = false
	command := p.wakeword.Process(text, true)

	if command != "" {
		// Wake word fired — verify the speaker before acting.
		// Verification runs here (after STT) so we only pay the cost on wake events.
		// TODO: verify Listening follow-ups too if multi-user support is needed.
		if p.wakeJustFired && p.verifier != nil {
			t0spk := time.Now()
			name, score, accepted, verErr := p.verifier.Verify(seg.Audio)
			if p.mon != nil {
				p.mon.Record("spk", time.Since(t0spk))
			}
			p.debugger.OnSpeakerVerified(name, score, accepted)
			if p.OnSpeakerVerified != nil {
				p.OnSpeakerVerified(name, score, accepted)
			}
			if verErr != nil && p.OnError != nil {
				p.OnError(fmt.Errorf("speaker verify: %w", verErr))
			}
			if !accepted {
				return // Unknown speaker — discard command
			}
		}
		if p.OnText != nil {
			p.OnText(text)
		}
		p.debugger.OnWakeWord(command)
		if p.OnCommand != nil {
			p.OnCommand(command)
		}
	} else {
		// No wake word — deliver transcription without verification.
		if p.OnText != nil {
			p.OnText(text)
		}
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
