// speaker-id: Live speaker identification with transcription
//
// Continuously monitors audio, identifies speakers, and transcribes speech.
// Output format: "Speaker: transcript"
//
// Optimized pipeline:
//   Mic → float32 buffer → VAD detection → parallel (Speaker ID + STT) → output
//
// Usage: speaker-id (loads all enrolled profiles)
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/ashinsabu/atlas/internal/vox/speaker"
	"github.com/ashinsabu/atlas/internal/vox/stt"
	"github.com/gordonklaus/portaudio"
)

const (
	sampleRate     = 16000
	channels       = 1
	framesPerBuf   = 512 // Larger buffer = fewer callbacks, less overhead
	unknownSpeaker = "Unknown"

	// VAD parameters (energy-based)
	energyThreshold   = 0.003  // Speech detection threshold
	silenceTimeoutMs  = 800    // ms of silence to end utterance
	minSpeechMs       = 300    // Minimum speech duration (noise filter)
	maxSpeechMs       = 15000  // Maximum utterance length
)

type SpeakerID struct {
	verifier *speaker.MultiProfileVerifier
	whisper  *stt.Whisper
	profiles []string

	// Audio state (protected by mutex)
	mu           sync.Mutex
	audioBuffer  []float32 // Accumulates speech audio
	isSpeaking   bool
	speechStart  time.Time
	silenceStart time.Time

	// Output channel
	outputChan chan utterance
}

type utterance struct {
	audio []float32
}

func main() {
	whisperModel := flag.String("whisper", "models/ggml-large.bin", "Whisper model path")
	speakerModel := flag.String("speaker", "models/wespeaker_en_voxceleb_CAM++.onnx", "Speaker model path")
	threshold := flag.Float64("threshold", 0.55, "Speaker similarity threshold")
	flag.Parse()

	fmt.Println("╔════════════════════════════════════════╗")
	fmt.Println("║     ATLAS Live Speaker ID              ║")
	fmt.Println("╚════════════════════════════════════════╝")
	fmt.Println()

	// Load profiles
	profiles, err := speaker.ListProfiles()
	if err != nil || len(profiles) == 0 {
		fmt.Println("No speaker profiles found.")
		fmt.Println("Run 'make enroll' to create a profile first.")
		os.Exit(1)
	}

	fmt.Printf("Loaded profiles: %s\n", strings.Join(profiles, ", "))
	fmt.Println()

	// Initialize speaker verifier
	spkCfg := speaker.DefaultConfig()
	spkCfg.Threshold = float32(*threshold)

	verifier, err := speaker.NewMultiProfileVerifier(spkCfg, *speakerModel)
	if err != nil {
		fmt.Printf("Failed to create speaker verifier: %v\n", err)
		os.Exit(1)
	}
	defer verifier.Close()

	if err := verifier.LoadAllProfiles(); err != nil {
		fmt.Printf("Failed to load profiles: %v\n", err)
		os.Exit(1)
	}

	// Initialize Whisper
	whisperCfg := stt.DefaultWhisperConfig()
	whisperCfg.ModelPath = *whisperModel

	whisperInst, err := stt.NewWhisper(whisperCfg)
	if err != nil {
		fmt.Printf("Failed to initialize Whisper: %v\n", err)
		os.Exit(1)
	}
	defer whisperInst.Close()

	fmt.Printf("STT: %s\n", whisperInst.ModelInfo())

	sid := &SpeakerID{
		verifier:   verifier,
		whisper:    whisperInst,
		profiles:   profiles,
		outputChan: make(chan utterance, 5),
	}

	// Initialize PortAudio
	if err := portaudio.Initialize(); err != nil {
		fmt.Printf("Failed to initialize audio: %v\n", err)
		os.Exit(1)
	}
	defer portaudio.Terminate()

	fmt.Println()
	fmt.Println("────────────────────────────────────────")
	fmt.Println("Listening... (Ctrl+C to stop)")
	fmt.Println("────────────────────────────────────────")
	fmt.Println()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle signals
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigChan
		fmt.Println("\nStopping...")
		cancel()
	}()

	if err := sid.run(ctx); err != nil && err != context.Canceled {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}
}

func (s *SpeakerID) run(ctx context.Context) error {
	// Start utterance processor (runs Speaker ID + STT in parallel)
	go s.processLoop(ctx)

	// Open audio stream - capture directly as float32
	stream, err := portaudio.OpenDefaultStream(channels, 0, float64(sampleRate), framesPerBuf, func(in []float32) {
		s.processAudioChunk(in)
	})
	if err != nil {
		return fmt.Errorf("open stream: %w", err)
	}
	defer stream.Close()

	if err := stream.Start(); err != nil {
		return fmt.Errorf("start stream: %w", err)
	}
	defer stream.Stop()

	<-ctx.Done()
	return ctx.Err()
}

// processAudioChunk handles incoming audio with inline VAD
// Optimized: no unnecessary allocations, direct float32 processing
func (s *SpeakerID) processAudioChunk(samples []float32) {
	energy := calculateEnergy(samples)
	isSpeech := energy > energyThreshold
	now := time.Now()

	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.isSpeaking {
		// Currently silent
		if isSpeech {
			// Speech started
			s.isSpeaking = true
			s.speechStart = now
			s.silenceStart = time.Time{}
			// Pre-allocate for ~10s of audio
			s.audioBuffer = make([]float32, 0, sampleRate*10)
			s.audioBuffer = append(s.audioBuffer, samples...)
		}
		return
	}

	// Currently speaking - accumulate audio
	s.audioBuffer = append(s.audioBuffer, samples...)
	speechDuration := now.Sub(s.speechStart)

	// Check max duration
	if speechDuration >= maxSpeechMs*time.Millisecond {
		s.emitUtterance()
		return
	}

	if isSpeech {
		// Reset silence timer
		s.silenceStart = time.Time{}
	} else {
		// Track silence
		if s.silenceStart.IsZero() {
			s.silenceStart = now
		}

		silenceDuration := now.Sub(s.silenceStart)
		if silenceDuration >= silenceTimeoutMs*time.Millisecond {
			// Silence timeout reached
			if speechDuration >= minSpeechMs*time.Millisecond {
				s.emitUtterance()
			} else {
				// Too short, discard (noise)
				s.isSpeaking = false
				s.audioBuffer = nil
			}
		}
	}
}

// emitUtterance sends completed speech to processing (must be called with lock held)
func (s *SpeakerID) emitUtterance() {
	audio := s.audioBuffer
	s.audioBuffer = nil
	s.isSpeaking = false

	// Non-blocking send
	select {
	case s.outputChan <- utterance{audio: audio}:
	default:
		// Channel full, drop utterance
	}
}

// processLoop handles utterances with parallel Speaker ID + STT
func (s *SpeakerID) processLoop(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case utt := <-s.outputChan:
			s.processUtterance(utt.audio)
		}
	}
}

// processUtterance runs Speaker ID and STT in parallel
func (s *SpeakerID) processUtterance(audio []float32) {
	var wg sync.WaitGroup
	var speakerName string
	var transcript string

	// Run Speaker ID and STT in parallel
	wg.Add(2)

	// Speaker ID (uses float32 directly)
	go func() {
		defer wg.Done()
		speakerName = s.identifySpeaker(audio)
	}()

	// STT (needs PCM bytes)
	go func() {
		defer wg.Done()
		pcmBytes := float32ToPCM(audio)
		text, err := s.whisper.TranscribeBytes(pcmBytes)
		if err == nil {
			transcript = strings.TrimSpace(text)
		}
	}()

	wg.Wait()

	if transcript == "" {
		return
	}

	// Format speaker name (capitalize first letter)
	displayName := speakerName
	if len(displayName) > 0 {
		displayName = strings.ToUpper(string(displayName[0])) + displayName[1:]
	}

	fmt.Printf("%s: %s\n", displayName, transcript)
}

func (s *SpeakerID) identifySpeaker(samples []float32) string {
	result, err := s.verifier.Verify(samples)
	if err != nil || !result.Matched {
		return unknownSpeaker
	}
	return result.MatchedBy
}

// calculateEnergy computes RMS energy of float32 audio
func calculateEnergy(samples []float32) float32 {
	if len(samples) == 0 {
		return 0
	}

	var sum float64
	for _, s := range samples {
		sum += float64(s) * float64(s)
	}

	rms := float32(sum / float64(len(samples)))
	if rms > 1.0 {
		rms = 1.0
	}
	return rms
}

// float32ToPCM converts float32 audio to 16-bit PCM bytes
func float32ToPCM(samples []float32) []byte {
	pcm := make([]byte, len(samples)*2)
	for i, s := range samples {
		// Clamp to [-1, 1]
		if s > 1.0 {
			s = 1.0
		} else if s < -1.0 {
			s = -1.0
		}
		// Convert to int16
		sample := int16(s * 32767)
		pcm[i*2] = byte(sample)
		pcm[i*2+1] = byte(sample >> 8)
	}
	return pcm
}
