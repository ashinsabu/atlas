package speaker

import (
	"bytes"
	"math"
	"testing"
)

// generateTestAudio creates deterministic test audio.
func generateTestAudio(seed float64, duration float64) []float32 {
	sampleRate := 16000
	numSamples := int(duration * float64(sampleRate))
	samples := make([]float32, numSamples)

	// Generate a simple waveform with the seed as frequency modifier
	for i := 0; i < numSamples; i++ {
		t := float64(i) / float64(sampleRate)
		// Mix of harmonics seeded by the input
		samples[i] = float32(
			0.3*math.Sin(2*math.Pi*200*(1+seed*0.1)*t) +
				0.2*math.Sin(2*math.Pi*400*(1+seed*0.15)*t) +
				0.1*math.Sin(2*math.Pi*800*(1+seed*0.2)*t),
		)
	}
	return samples
}

func TestExtractor_Extract(t *testing.T) {
	extractor := NewExtractor(128)

	audio := generateTestAudio(1.0, 1.0) // 1 second

	embedding, err := extractor.Extract(audio)
	if err != nil {
		t.Fatalf("Extract failed: %v", err)
	}

	if len(embedding) != 128 {
		t.Errorf("Expected embedding dim 128, got %d", len(embedding))
	}

	// Check embedding is normalized (L2 norm ~1.0)
	var sumSq float32
	for _, v := range embedding {
		sumSq += v * v
	}
	norm := math.Sqrt(float64(sumSq))
	if math.Abs(norm-1.0) > 0.01 {
		t.Errorf("Expected L2 norm ~1.0, got %f", norm)
	}
}

func TestExtractor_TooShort(t *testing.T) {
	extractor := NewExtractor(128)

	audio := generateTestAudio(1.0, 0.1) // 100ms - too short

	_, err := extractor.Extract(audio)
	if err != ErrAudioTooShort {
		t.Errorf("Expected ErrAudioTooShort, got %v", err)
	}
}

func TestVerifier_EnrollAndVerify(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MinEnrollSamples = 3
	cfg.Threshold = 0.5

	verifier := NewVerifier(cfg)

	// Create similar samples (same "speaker")
	samples := [][]float32{
		generateTestAudio(1.0, 1.0),
		generateTestAudio(1.05, 1.0),
		generateTestAudio(0.95, 1.0),
	}

	if err := verifier.Enroll(samples); err != nil {
		t.Fatalf("Enroll failed: %v", err)
	}

	if !verifier.IsEnrolled() {
		t.Error("Expected IsEnrolled to be true after enrollment")
	}

	// Test with similar audio (should accept)
	testSample := generateTestAudio(1.02, 1.0)
	isOwner, confidence, err := verifier.Verify(testSample)
	if err != nil {
		t.Fatalf("Verify failed: %v", err)
	}

	t.Logf("Similar sample: isOwner=%v, confidence=%f", isOwner, confidence)

	// Test with different audio (should have lower confidence)
	differentSample := generateTestAudio(5.0, 1.0) // Very different seed
	isOwner2, confidence2, err := verifier.Verify(differentSample)
	if err != nil {
		t.Fatalf("Verify failed: %v", err)
	}

	t.Logf("Different sample: isOwner=%v, confidence=%f", isOwner2, confidence2)

	// Similar should have higher confidence than different
	if confidence <= confidence2 {
		t.Errorf("Expected similar audio to have higher confidence (%f) than different (%f)",
			confidence, confidence2)
	}
}

func TestVerifier_NotEnrolled(t *testing.T) {
	verifier := NewVerifier(DefaultConfig())

	audio := generateTestAudio(1.0, 1.0)
	_, _, err := verifier.Verify(audio)

	if err != ErrNotEnrolled {
		t.Errorf("Expected ErrNotEnrolled, got %v", err)
	}
}

func TestProfile_SaveLoad(t *testing.T) {
	// Create a profile
	embedding := make([]float32, 128)
	for i := range embedding {
		embedding[i] = float32(i) / 128.0
	}
	normalizeL2(embedding)

	profile, err := EnrollFromEmbeddings([][]float32{embedding})
	if err != nil {
		t.Fatalf("EnrollFromEmbeddings failed: %v", err)
	}

	// Write to buffer
	var buf bytes.Buffer
	if err := profile.WriteTo(&buf); err != nil {
		t.Fatalf("WriteTo failed: %v", err)
	}

	// Read back
	loaded, err := ReadProfileFrom(&buf)
	if err != nil {
		t.Fatalf("ReadProfileFrom failed: %v", err)
	}

	// Verify
	if loaded.Version != profile.Version {
		t.Errorf("Version mismatch: %d vs %d", loaded.Version, profile.Version)
	}
	if loaded.SampleCount != profile.SampleCount {
		t.Errorf("SampleCount mismatch: %d vs %d", loaded.SampleCount, profile.SampleCount)
	}
	if len(loaded.Embedding) != len(profile.Embedding) {
		t.Fatalf("Embedding length mismatch: %d vs %d", len(loaded.Embedding), len(profile.Embedding))
	}

	for i := range profile.Embedding {
		if math.Abs(float64(loaded.Embedding[i]-profile.Embedding[i])) > 1e-6 {
			t.Errorf("Embedding[%d] mismatch: %f vs %f", i, loaded.Embedding[i], profile.Embedding[i])
		}
	}
}

func TestCosineSimilarity(t *testing.T) {
	// Identical vectors should have similarity 1.0
	a := []float32{1, 0, 0}
	b := []float32{1, 0, 0}
	sim := cosineSimilarity(a, b)
	if math.Abs(float64(sim)-1.0) > 0.001 {
		t.Errorf("Identical vectors: expected 1.0, got %f", sim)
	}

	// Orthogonal vectors should have similarity 0.0
	c := []float32{1, 0, 0}
	d := []float32{0, 1, 0}
	sim = cosineSimilarity(c, d)
	if math.Abs(float64(sim)) > 0.001 {
		t.Errorf("Orthogonal vectors: expected 0.0, got %f", sim)
	}

	// Opposite vectors should have similarity -1.0
	e := []float32{1, 0, 0}
	f := []float32{-1, 0, 0}
	sim = cosineSimilarity(e, f)
	if math.Abs(float64(sim)+1.0) > 0.001 {
		t.Errorf("Opposite vectors: expected -1.0, got %f", sim)
	}
}

func TestEnrollmentSession(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MinEnrollSamples = 3

	session := NewEnrollmentSession(cfg)

	if session.IsComplete() {
		t.Error("Session should not be complete with 0 samples")
	}

	if session.RemainingCount() != 3 {
		t.Errorf("Expected 3 remaining, got %d", session.RemainingCount())
	}

	// Add samples
	for i := 0; i < 3; i++ {
		audio := generateTestAudio(float64(i)*0.05+1.0, 1.0)
		idx, err := session.AddSample(audio)
		if err != nil {
			t.Fatalf("AddSample %d failed: %v", i, err)
		}
		if idx != i {
			t.Errorf("Expected index %d, got %d", i, idx)
		}
	}

	if !session.IsComplete() {
		t.Error("Session should be complete with 3 samples")
	}

	if session.RemainingCount() != 0 {
		t.Errorf("Expected 0 remaining, got %d", session.RemainingCount())
	}

	// Complete enrollment
	profile, err := session.Complete()
	if err != nil {
		t.Fatalf("Complete failed: %v", err)
	}

	if profile.SampleCount != 3 {
		t.Errorf("Expected 3 samples in profile, got %d", profile.SampleCount)
	}

	// Check quality scores
	quality := session.OverallQuality()
	t.Logf("Overall quality: %f", quality)
	// Quality should be reasonable for similar samples
	if quality < 0.3 {
		t.Errorf("Quality too low for similar samples: %f", quality)
	}
}
