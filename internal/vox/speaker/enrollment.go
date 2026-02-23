// Package speaker - enrollment logic.
//
// Enrollment creates a speaker profile by averaging embeddings from multiple
// audio samples. More samples = more robust profile.

package speaker

import (
	"fmt"
	"time"
)

// Profile represents an enrolled speaker.
type Profile struct {
	// Embedding is the averaged speaker embedding.
	Embedding []float32

	// SampleCount is the number of samples used for enrollment.
	SampleCount int

	// CreatedAt is when the profile was created.
	CreatedAt time.Time

	// Version allows for profile format upgrades.
	Version int
}

// CurrentProfileVersion is the current profile format version.
const CurrentProfileVersion = 1

// EnrollFromEmbeddings creates a speaker profile from pre-computed embeddings.
// Embeddings are averaged to create a centroid embedding.
func EnrollFromEmbeddings(embeddings [][]float32) (*Profile, error) {
	if len(embeddings) == 0 {
		return nil, ErrNoSamples
	}

	dim := len(embeddings[0])
	if dim == 0 {
		return nil, fmt.Errorf("embedding dimension is zero")
	}

	// Verify all embeddings have same dimension
	for i, emb := range embeddings {
		if len(emb) != dim {
			return nil, fmt.Errorf("embedding %d has dimension %d, expected %d", i, len(emb), dim)
		}
	}

	// Average embeddings (centroid)
	averaged := make([]float32, dim)
	for _, emb := range embeddings {
		for i, v := range emb {
			averaged[i] += v
		}
	}
	for i := range averaged {
		averaged[i] /= float32(len(embeddings))
	}

	// L2 normalize the averaged embedding
	normalizeL2(averaged)

	return &Profile{
		Embedding:   averaged,
		SampleCount: len(embeddings),
		CreatedAt:   time.Now(),
		Version:     CurrentProfileVersion,
	}, nil
}

// EnrollmentSession manages the enrollment process.
// Useful for interactive enrollment with feedback.
type EnrollmentSession struct {
	config     Config
	extractor  *Extractor
	embeddings [][]float32
}

// NewEnrollmentSession creates a new enrollment session.
func NewEnrollmentSession(cfg Config) *EnrollmentSession {
	return &EnrollmentSession{
		config:     cfg,
		extractor:  NewExtractor(cfg.EmbeddingDim),
		embeddings: make([][]float32, 0, cfg.MinEnrollSamples),
	}
}

// AddSample adds an audio sample to the enrollment session.
// Returns the sample index and any error.
func (s *EnrollmentSession) AddSample(audio []float32) (int, error) {
	emb, err := s.extractor.Extract(audio)
	if err != nil {
		return -1, err
	}
	s.embeddings = append(s.embeddings, emb)
	return len(s.embeddings) - 1, nil
}

// SampleCount returns the number of samples collected.
func (s *EnrollmentSession) SampleCount() int {
	return len(s.embeddings)
}

// IsComplete returns true if enough samples have been collected.
func (s *EnrollmentSession) IsComplete() bool {
	return len(s.embeddings) >= s.config.MinEnrollSamples
}

// RemainingCount returns how many more samples are needed.
func (s *EnrollmentSession) RemainingCount() int {
	remaining := s.config.MinEnrollSamples - len(s.embeddings)
	if remaining < 0 {
		return 0
	}
	return remaining
}

// Complete finalizes enrollment and returns the profile.
// Returns error if not enough samples collected.
func (s *EnrollmentSession) Complete() (*Profile, error) {
	if !s.IsComplete() {
		return nil, fmt.Errorf("need %d more samples", s.RemainingCount())
	}
	return EnrollFromEmbeddings(s.embeddings)
}

// SampleQuality estimates the quality of a sample embedding.
// Returns a score from 0.0 (poor) to 1.0 (excellent).
// Uses consistency with other samples if available.
func (s *EnrollmentSession) SampleQuality(sampleIdx int) float32 {
	if sampleIdx < 0 || sampleIdx >= len(s.embeddings) {
		return 0
	}

	if len(s.embeddings) < 2 {
		// Can't compute consistency with only one sample
		return 0.5
	}

	// Compute average similarity to other samples
	var totalSim float32
	count := 0
	for i, emb := range s.embeddings {
		if i == sampleIdx {
			continue
		}
		sim := cosineSimilarity(s.embeddings[sampleIdx], emb)
		totalSim += sim
		count++
	}

	if count == 0 {
		return 0.5
	}

	avgSim := totalSim / float32(count)

	// Map similarity to quality score
	// Similarity range roughly [-1, 1], typical good range [0.6, 1.0]
	// Map [0.5, 1.0] -> [0.0, 1.0]
	quality := (avgSim - 0.5) * 2
	if quality < 0 {
		quality = 0
	}
	if quality > 1 {
		quality = 1
	}

	return quality
}

// OverallQuality returns the overall enrollment quality.
// Based on average consistency between all samples.
func (s *EnrollmentSession) OverallQuality() float32 {
	if len(s.embeddings) < 2 {
		return 0.5
	}

	var totalSim float32
	count := 0

	for i := 0; i < len(s.embeddings); i++ {
		for j := i + 1; j < len(s.embeddings); j++ {
			sim := cosineSimilarity(s.embeddings[i], s.embeddings[j])
			totalSim += sim
			count++
		}
	}

	if count == 0 {
		return 0.5
	}

	avgSim := totalSim / float32(count)

	// Map to quality score
	quality := (avgSim - 0.5) * 2
	if quality < 0 {
		quality = 0
	}
	if quality > 1 {
		quality = 1
	}

	return quality
}
