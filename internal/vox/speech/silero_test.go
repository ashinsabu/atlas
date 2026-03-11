package speech

import (
	"math"
	"os"
	"testing"
)

const testModelPath = "../../../models/silero_vad.onnx"
const testSpeechWAV = "../../../test/stt/recordings/script_1_rec_1.wav"

// makePCM16 generates a PCM16 sine wave at the given frequency for durationMs.
func makePCM16(freqHz float64, durationMs int) []byte {
	numSamples := SileroSampleRate * durationMs / 1000
	buf := make([]byte, numSamples*2)
	for i := 0; i < numSamples; i++ {
		// Amplitude 0.5 to stay well within int16 range
		v := 0.5 * math.Sin(2*math.Pi*freqHz*float64(i)/float64(SileroSampleRate))
		s := int16(v * 32767)
		buf[i*2] = byte(s)
		buf[i*2+1] = byte(s >> 8)
	}
	return buf
}

// makeSilencePCM16 generates silence (all-zero PCM16) for durationMs.
func makeSilencePCM16(durationMs int) []byte {
	numSamples := SileroSampleRate * durationMs / 1000
	return make([]byte, numSamples*2)
}

// TestSileroInference checks that the model runs without errors and produces
// plausible probabilities: near 0 for silence, higher for a speech-like signal.
func TestSileroInference(t *testing.T) {
	cfg := DefaultDetectorConfig()
	d, err := NewSileroDetector(testModelPath, cfg)
	if err != nil {
		t.Fatalf("NewSileroDetector: %v", err)
	}
	defer d.Close()

	// Collect VAD probs via the OnFrame callback
	var silenceProbs, speechProbs []float32
	d.OnFrame = func(prob float32, speaking bool) {
		// This callback is only used to collect probs; we set it outside of debug mode too
	}

	// Feed 1s of silence — expect no segments and low prob
	silenceAudio := makeSilencePCM16(1000)
	var frameCount int
	var totalProb float32
	d.OnFrame = func(prob float32, speaking bool) {
		silenceProbs = append(silenceProbs, prob)
		totalProb += prob
		frameCount++
	}

	segs := d.Process(silenceAudio)
	if len(segs) != 0 {
		t.Errorf("silence: expected 0 segments, got %d", len(segs))
	}
	if frameCount == 0 {
		t.Fatal("OnFrame was never called — inference is not running")
	}
	avgSilenceProb := totalProb / float32(frameCount)
	t.Logf("silence: %d frames, avg VAD prob = %.4f", frameCount, avgSilenceProb)
	if avgSilenceProb > 0.3 {
		t.Errorf("silence: avg VAD prob %.4f is unexpectedly high (expected < 0.3)", avgSilenceProb)
	}

	// Feed a 300Hz tone — Silero may or may not flag it as speech (it's not
	// real speech), but the inference must not error and probs must be in [0,1].
	toneAudio := makePCM16(300, 500)
	frameCount = 0
	totalProb = 0
	d.OnFrame = func(prob float32, speaking bool) {
		speechProbs = append(speechProbs, prob)
		if prob < 0 || prob > 1 {
			t.Errorf("out-of-range VAD prob: %f", prob)
		}
		totalProb += prob
		frameCount++
	}
	d.Process(toneAudio)
	if frameCount > 0 {
		t.Logf("300Hz tone: %d frames, avg VAD prob = %.4f", frameCount, totalProb/float32(frameCount))
	}
}

// TestSileroRealSpeech feeds an actual speech recording through the VAD and
// asserts that speech is detected and at least one segment is emitted.
func TestSileroRealSpeech(t *testing.T) {
	if _, err := os.Stat(testSpeechWAV); err != nil {
		t.Skipf("speech recording not found (%s): %v", testSpeechWAV, err)
	}

	// Load raw PCM bytes from WAV (skip 44-byte header)
	raw, err := os.ReadFile(testSpeechWAV)
	if err != nil {
		t.Fatalf("read wav: %v", err)
	}
	if len(raw) <= 44 {
		t.Fatal("wav file too small")
	}
	pcm := raw[44:] // strip standard WAV header

	cfg := DefaultDetectorConfig()
	cfg.MinSilenceMs = 500 // shorter for test speed
	cfg.MinSpeechMs = 100

	d, err := NewSileroDetector(testModelPath, cfg)
	if err != nil {
		t.Fatalf("NewSileroDetector: %v", err)
	}
	defer d.Close()

	var maxProb float32
	var frameCount int
	var speechFrames int
	d.OnFrame = func(prob float32, speaking bool) {
		frameCount++
		if prob > maxProb {
			maxProb = prob
		}
		if speaking {
			speechFrames++
		}
	}

	// Feed in 512-sample (1024-byte) chunks matching Silero's window size
	chunkBytes := SileroWindowSamples * 2
	var segs []SpeechSegment
	for i := 0; i+chunkBytes <= len(pcm); i += chunkBytes {
		segs = append(segs, d.Process(pcm[i:i+chunkBytes])...)
	}
	segs = append(segs, d.Flush()...)

	t.Logf("frames processed: %d", frameCount)
	t.Logf("max VAD prob:     %.4f", maxProb)
	t.Logf("speech frames:    %d / %d", speechFrames, frameCount)
	t.Logf("segments emitted: %d", len(segs))
	for i, s := range segs {
		t.Logf("  segment %d: %.2fs", i, s.Seconds())
	}

	if maxProb < 0.5 {
		t.Errorf("max VAD prob %.4f too low — VAD is not detecting speech in the recording", maxProb)
	}
	if len(segs) == 0 {
		t.Error("no segments emitted from real speech recording")
	}
}
