package speaker

import (
	"math"
	"testing"

	"github.com/ashinsabu/atlas/internal/vox/audio"
)

const (
	testModelPath    = "../../../models/wespeaker-resnet34.onnx"
	testRecordingDir = "../../../test/stt/recordings"
)

// TestEncoder_Determinism verifies that encoding the same audio twice
// produces identical embeddings (deterministic inference).
func TestEncoder_Determinism(t *testing.T) {
	enc, err := NewEncoder(testModelPath)
	if err != nil {
		t.Skipf("speaker model not available: %v", err)
	}
	defer enc.Close()

	samples, err := audio.LoadWAV(testRecordingDir + "/script_1_rec_1.wav")
	if err != nil {
		t.Skipf("test recording not available: %v", err)
	}

	emb1, err := enc.Encode(samples)
	if err != nil {
		t.Fatalf("Encode (1st): %v", err)
	}
	emb2, err := enc.Encode(samples)
	if err != nil {
		t.Fatalf("Encode (2nd): %v", err)
	}

	if len(emb1) != len(emb2) {
		t.Fatalf("embedding length mismatch: %d vs %d", len(emb1), len(emb2))
	}
	for i := range emb1 {
		if emb1[i] != emb2[i] {
			t.Errorf("embedding[%d] differs: %f vs %f", i, emb1[i], emb2[i])
			break
		}
	}
}

// TestEncoder_UnitNorm verifies that the model output is L2-normalized.
// WeSpeaker ResNet34-LM normalizes embeddings internally.
func TestEncoder_UnitNorm(t *testing.T) {
	enc, err := NewEncoder(testModelPath)
	if err != nil {
		t.Skipf("speaker model not available: %v", err)
	}
	defer enc.Close()

	samples, err := audio.LoadWAV(testRecordingDir + "/script_1_rec_1.wav")
	if err != nil {
		t.Skipf("test recording not available: %v", err)
	}

	emb, err := enc.Encode(samples)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}

	var norm float64
	for _, v := range emb {
		norm += float64(v) * float64(v)
	}
	norm = math.Sqrt(norm)

	const tolerance = 1e-3
	if math.Abs(norm-1.0) > tolerance {
		t.Errorf("embedding L2 norm = %f, want 1.0 ± %f", norm, tolerance)
	}
}

// TestEncoder_EmbeddingDim verifies the output dimension matches EmbeddingDim.
func TestEncoder_EmbeddingDim(t *testing.T) {
	enc, err := NewEncoder(testModelPath)
	if err != nil {
		t.Skipf("speaker model not available: %v", err)
	}
	defer enc.Close()

	samples, err := audio.LoadWAV(testRecordingDir + "/script_1_rec_1.wav")
	if err != nil {
		t.Skipf("test recording not available: %v", err)
	}

	emb, err := enc.Encode(samples)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}

	if len(emb) != EmbeddingDim {
		t.Errorf("embedding length = %d, want %d", len(emb), EmbeddingDim)
	}
}
