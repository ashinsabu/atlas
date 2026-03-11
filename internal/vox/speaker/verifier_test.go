package speaker

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/ashinsabu/atlas/internal/vox/audio"
)

const (
	userRecordingsDir = "../../../test/stt/recordings"
	otherSpeakersDir  = "../../../test/speaker/other"
	defaultThreshold  = float32(0.75)
)

// TestVerifier_UserRecordings enrolls from script_1 recordings, then verifies
// all available recordings from the same speaker. Expects all to be accepted.
func TestVerifier_UserRecordings(t *testing.T) {
	enc, err := NewEncoder(testModelPath)
	if err != nil {
		t.Skipf("speaker model not available: %v", err)
	}
	defer enc.Close()

	enrollPaths := globFiles(t, userRecordingsDir, "script_1_rec_*.wav")
	if len(enrollPaths) == 0 {
		t.Skip("no enrollment recordings found")
	}

	profile, err := EnrollFromFiles(enc, enrollPaths, "owner")
	if err != nil {
		t.Fatalf("EnrollFromFiles: %v", err)
	}

	v := NewVerifier(enc, profile, defaultThreshold)

	// Test all user recordings (same speaker as enrolled).
	allPaths := globFiles(t, userRecordingsDir, "*.wav")
	accepted, total := 0, 0
	for _, path := range allPaths {
		pcm, err := audio.LoadWAVBytes(path)
		if err != nil {
			t.Logf("skip %s: %v", filepath.Base(path), err)
			continue
		}
		total++
		_, score, ok, verifyErr := v.Verify(pcm)
		if verifyErr != nil {
			t.Errorf("Verify(%s): %v", filepath.Base(path), verifyErr)
			continue
		}
		if ok {
			accepted++
		} else {
			t.Logf("REJECTED %s  score=%.3f (threshold=%.2f)", filepath.Base(path), score, defaultThreshold)
		}
	}

	if total == 0 {
		t.Skip("no recordings to test")
	}

	tpr := float64(accepted) / float64(total) * 100
	t.Logf("User recordings: %d/%d accepted (TPR: %.1f%%)", accepted, total, tpr)

	if tpr < 90.0 {
		t.Errorf("TPR = %.1f%%, want >= 90%% — consider lowering threshold", tpr)
	}
}

// TestVerifier_OtherSpeakers verifies that other speakers are rejected.
// Only runs if test/speaker/other/ contains WAV files (from make setup-speaker-test).
func TestVerifier_OtherSpeakers(t *testing.T) {
	enc, err := NewEncoder(testModelPath)
	if err != nil {
		t.Skipf("speaker model not available: %v", err)
	}
	defer enc.Close()

	otherPaths := globFiles(t, otherSpeakersDir, "*.wav")
	if len(otherPaths) == 0 {
		// Try subdirectory pattern.
		otherPaths = globFilesDeep(t, otherSpeakersDir, "wav")
	}
	if len(otherPaths) == 0 {
		t.Skip("no other-speaker recordings found (run 'make setup-speaker-test')")
	}

	enrollPaths := globFiles(t, userRecordingsDir, "script_1_rec_*.wav")
	if len(enrollPaths) == 0 {
		t.Skip("no enrollment recordings found")
	}

	profile, err := EnrollFromFiles(enc, enrollPaths, "owner")
	if err != nil {
		t.Fatalf("EnrollFromFiles: %v", err)
	}

	v := NewVerifier(enc, profile, defaultThreshold)

	rejected, total := 0, 0
	for _, path := range otherPaths {
		pcm, err := audio.LoadWAVBytes(path)
		if err != nil {
			t.Logf("skip %s: %v", filepath.Base(path), err)
			continue
		}
		total++
		_, score, ok, verifyErr := v.Verify(pcm)
		if verifyErr != nil {
			t.Logf("Verify(%s) error: %v", filepath.Base(path), verifyErr)
			continue
		}
		if !ok {
			rejected++
		} else {
			t.Logf("ACCEPTED (false positive) %s  score=%.3f", filepath.Base(path), score)
		}
	}

	if total == 0 {
		t.Skip("no other-speaker recordings to test")
	}

	tnr := float64(rejected) / float64(total) * 100
	t.Logf("Other speakers: %d/%d rejected (TNR: %.1f%%)", rejected, total, tnr)

	if tnr < 85.0 {
		t.Errorf("TNR = %.1f%%, want >= 85%% — consider raising threshold", tnr)
	}
}

// TestCosineSimilarity verifies the math for known vectors.
func TestCosineSimilarity(t *testing.T) {
	// Identical unit vectors → similarity = 1.0
	a := []float32{1, 0, 0}
	if got := CosineSimilarity(a, a); math.Abs(float64(got-1.0)) > 1e-6 {
		t.Errorf("identical vectors: similarity = %f, want 1.0", got)
	}

	// Orthogonal unit vectors → similarity = 0.0
	b := []float32{0, 1, 0}
	if got := CosineSimilarity(a, b); math.Abs(float64(got)) > 1e-6 {
		t.Errorf("orthogonal vectors: similarity = %f, want 0.0", got)
	}

	// Opposite unit vectors → similarity = -1.0
	c := []float32{-1, 0, 0}
	if got := CosineSimilarity(a, c); math.Abs(float64(got+1.0)) > 1e-6 {
		t.Errorf("opposite vectors: similarity = %f, want -1.0", got)
	}
}

// TestL2Normalize verifies normalization to unit length.
func TestL2Normalize(t *testing.T) {
	v := []float32{3, 4} // |v| = 5
	L2Normalize(v)
	norm := math.Sqrt(float64(v[0]*v[0]) + float64(v[1]*v[1]))
	if math.Abs(norm-1.0) > 1e-6 {
		t.Errorf("after L2Normalize: norm = %f, want 1.0", norm)
	}
}

// TestVerifier_NilProfile verifies safe fallback when no profile is loaded.
func TestVerifier_NilProfile(t *testing.T) {
	v := NewVerifier(nil, nil, defaultThreshold)
	name, score, accepted, err := v.Verify(make([]byte, 1000))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if name != "Unknown" || score != 0 || accepted {
		t.Errorf("nil profile: got name=%q score=%f accepted=%v, want Unknown/0/false", name, score, accepted)
	}
}

// globFiles returns files matching pattern in dir. Non-fatal if dir doesn't exist.
func globFiles(t *testing.T, dir, pattern string) []string {
	t.Helper()
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		return nil
	}
	matches, _ := filepath.Glob(filepath.Join(dir, pattern))
	return matches
}

// globFilesDeep walks dir recursively and returns files with the given extension.
func globFilesDeep(t *testing.T, dir, ext string) []string {
	t.Helper()
	var paths []string
	_ = filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		if !info.IsDir() && filepath.Ext(path) == "."+ext {
			paths = append(paths, path)
		}
		return nil
	})
	return paths
}
