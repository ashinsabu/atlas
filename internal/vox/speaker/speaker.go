// Package speaker provides speaker verification for Atlas.
//
// This module enables single-user voice authentication by:
// 1. Enrolling the owner's voice (collecting multiple samples → averaged embedding)
// 2. Verifying incoming audio against the enrolled profile
//
// The embedding extraction is currently a placeholder using audio statistics.
// It will be replaced with an ONNX model (e.g., ECAPA-TDNN) for production use.
//
// Usage:
//
//	v := speaker.NewVerifier(speaker.DefaultConfig())
//	err := v.Enroll(audioSamples)  // Enroll with multiple samples
//	v.SaveProfile("~/.atlas/speaker_profile.bin")
//
//	// Later:
//	v.LoadProfile("~/.atlas/speaker_profile.bin")
//	isOwner, confidence, err := v.Verify(newSample)
package speaker

import (
	"errors"
	"fmt"
	"os"
	"sync"
)

// ErrNotEnrolled is returned when verification is attempted without enrollment.
var ErrNotEnrolled = errors.New("speaker not enrolled: no profile loaded")

// ErrNoSamples is returned when enrollment is attempted with no samples.
var ErrNoSamples = errors.New("no audio samples provided for enrollment")

// Verifier defines the speaker verification interface.
// Implementations must be safe for concurrent use.
type Verifier interface {
	// Enroll creates a speaker profile from multiple audio samples.
	// samples: slice of float32 audio (16kHz mono, normalized [-1.0, 1.0])
	// At least 3 samples recommended for robust enrollment.
	Enroll(samples [][]float32) error

	// Verify checks if audio matches the enrolled speaker.
	// Returns: (isOwner, confidence score 0.0-1.0, error)
	// Confidence represents cosine similarity to enrolled profile.
	Verify(sample []float32) (bool, float32, error)

	// LoadProfile loads a speaker profile from disk.
	LoadProfile(path string) error

	// SaveProfile saves the current speaker profile to disk.
	SaveProfile(path string) error

	// IsEnrolled returns true if a speaker profile is loaded.
	IsEnrolled() bool

	// EmbeddingDim returns the dimension of speaker embeddings.
	EmbeddingDim() int
}

// Config holds verifier configuration.
type Config struct {
	// Threshold for verification (cosine similarity).
	// Higher = stricter verification.
	// Typical range: 0.5-0.7 depending on security needs.
	Threshold float32

	// EmbeddingDim is the size of speaker embeddings.
	// Must match the embedding model output size.
	// WeSpeaker CAM++: 512 dimensions
	EmbeddingDim int

	// MinEnrollSamples is the minimum samples required for enrollment.
	MinEnrollSamples int

	// ModelPath is the path to the ONNX speaker embedding model.
	// Leave empty to use placeholder (non-ONNX) extractor.
	ModelPath string
}

// DefaultConfig returns sensible defaults for speaker verification.
func DefaultConfig() Config {
	return Config{
		Threshold:        0.60, // Balanced threshold for ONNX model
		EmbeddingDim:     512,  // WeSpeaker CAM++ output dimension
		MinEnrollSamples: 3,    // At least 3 samples for robust enrollment
		ModelPath:        "models/wespeaker_en_voxceleb_CAM++.onnx",
	}
}

// DefaultVerifier is the standard speaker verifier implementation.
type DefaultVerifier struct {
	config    Config
	extractor *Extractor
	profile   *Profile

	mu sync.RWMutex
}

// NewVerifier creates a new speaker verifier with the given config.
func NewVerifier(cfg Config) *DefaultVerifier {
	return &DefaultVerifier{
		config:    cfg,
		extractor: NewExtractor(cfg.EmbeddingDim),
	}
}

// Enroll creates a speaker profile from multiple audio samples.
func (v *DefaultVerifier) Enroll(samples [][]float32) error {
	if len(samples) == 0 {
		return ErrNoSamples
	}
	if len(samples) < v.config.MinEnrollSamples {
		return fmt.Errorf("need at least %d samples, got %d", v.config.MinEnrollSamples, len(samples))
	}

	// Extract embeddings for all samples
	embeddings := make([][]float32, len(samples))
	for i, sample := range samples {
		emb, err := v.extractor.Extract(sample)
		if err != nil {
			return fmt.Errorf("extract embedding for sample %d: %w", i, err)
		}
		embeddings[i] = emb
	}

	// Create profile with averaged embeddings
	profile, err := EnrollFromEmbeddings(embeddings)
	if err != nil {
		return fmt.Errorf("create profile: %w", err)
	}

	v.mu.Lock()
	v.profile = profile
	v.mu.Unlock()

	return nil
}

// Verify checks if audio matches the enrolled speaker.
func (v *DefaultVerifier) Verify(sample []float32) (bool, float32, error) {
	v.mu.RLock()
	profile := v.profile
	v.mu.RUnlock()

	if profile == nil {
		return false, 0, ErrNotEnrolled
	}

	// Extract embedding from sample
	emb, err := v.extractor.Extract(sample)
	if err != nil {
		return false, 0, fmt.Errorf("extract embedding: %w", err)
	}

	// Compute cosine similarity
	similarity := cosineSimilarity(emb, profile.Embedding)

	isOwner := similarity >= v.config.Threshold
	return isOwner, similarity, nil
}

// LoadProfile loads a speaker profile from disk.
func (v *DefaultVerifier) LoadProfile(path string) error {
	profile, err := LoadProfileFromFile(path)
	if err != nil {
		return err
	}

	// Validate embedding dimension
	if len(profile.Embedding) != v.config.EmbeddingDim {
		return fmt.Errorf("profile embedding dim %d != config %d", len(profile.Embedding), v.config.EmbeddingDim)
	}

	v.mu.Lock()
	v.profile = profile
	v.mu.Unlock()

	return nil
}

// SaveProfile saves the current speaker profile to disk.
func (v *DefaultVerifier) SaveProfile(path string) error {
	v.mu.RLock()
	profile := v.profile
	v.mu.RUnlock()

	if profile == nil {
		return ErrNotEnrolled
	}

	return profile.SaveToFile(path)
}

// IsEnrolled returns true if a speaker profile is loaded.
func (v *DefaultVerifier) IsEnrolled() bool {
	v.mu.RLock()
	defer v.mu.RUnlock()
	return v.profile != nil
}

// EmbeddingDim returns the dimension of speaker embeddings.
func (v *DefaultVerifier) EmbeddingDim() int {
	return v.config.EmbeddingDim
}

// GetThreshold returns the current verification threshold.
func (v *DefaultVerifier) GetThreshold() float32 {
	return v.config.Threshold
}

// SetThreshold updates the verification threshold.
func (v *DefaultVerifier) SetThreshold(t float32) {
	v.config.Threshold = t
}

// cosineSimilarity computes cosine similarity between two vectors.
// Returns value in [-1.0, 1.0], where 1.0 means identical direction.
func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var dotProduct, normA, normB float32
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (sqrt32(normA) * sqrt32(normB))
}

// sqrt32 is a simple float32 square root using Newton's method.
func sqrt32(x float32) float32 {
	if x <= 0 {
		return 0
	}
	z := x / 2
	for i := 0; i < 10; i++ {
		z = (z + x/z) / 2
	}
	return z
}

// NewAutoVerifier creates a verifier, preferring ONNX if model exists.
// Falls back to placeholder embeddings if ONNX unavailable.
func NewAutoVerifier(cfg Config) (Verifier, func(), error) {
	modelPath := cfg.ModelPath
	if modelPath == "" {
		modelPath = "models/wespeaker_en_voxceleb_CAM++.onnx"
	}

	// Try ONNX verifier if model exists
	if _, err := os.Stat(modelPath); err == nil {
		verifier, err := NewONNXVerifier(cfg, modelPath)
		if err == nil {
			cleanup := func() { verifier.Close() }
			return verifier, cleanup, nil
		}
		// ONNX failed, fall through to placeholder
	}

	// Fallback to placeholder
	verifier := NewVerifier(cfg)
	return verifier, func() {}, nil
}
