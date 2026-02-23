// Package speaker - multi-profile verification.
//
// Supports verifying audio against multiple enrolled speaker profiles.
// Use case: Allow multiple family members to use the voice assistant.

package speaker

import (
	"fmt"
	"sync"
)

// MultiProfileVerifier verifies audio against multiple enrolled profiles.
type MultiProfileVerifier struct {
	config     Config
	extractor  *ONNXExtractor
	profiles   map[string]*Profile // name -> profile
	authorized map[string]bool     // which profiles are active for verification

	mu sync.RWMutex
}

// MultiVerifyResult contains the result of multi-profile verification.
type MultiVerifyResult struct {
	Matched    bool    // True if any authorized profile matched
	MatchedBy  string  // Name of the matched profile (empty if no match)
	Confidence float32 // Confidence score of the best match
	AllScores  map[string]float32 // Scores against all authorized profiles
}

// NewMultiProfileVerifier creates a new multi-profile verifier.
func NewMultiProfileVerifier(cfg Config, modelPath string) (*MultiProfileVerifier, error) {
	extractor, err := NewONNXExtractor(modelPath)
	if err != nil {
		return nil, err
	}

	cfg.EmbeddingDim = extractor.EmbeddingDim()

	return &MultiProfileVerifier{
		config:     cfg,
		extractor:  extractor,
		profiles:   make(map[string]*Profile),
		authorized: make(map[string]bool),
	}, nil
}

// LoadProfile loads a named profile from the profiles directory.
func (v *MultiProfileVerifier) LoadProfile(name string) error {
	path, err := NamedProfilePath(name)
	if err != nil {
		return err
	}

	profile, err := LoadProfileFromFile(path)
	if err != nil {
		return fmt.Errorf("load profile %s: %w", name, err)
	}

	if len(profile.Embedding) != v.config.EmbeddingDim {
		return fmt.Errorf("profile %s dim %d != expected %d", name, len(profile.Embedding), v.config.EmbeddingDim)
	}

	v.mu.Lock()
	v.profiles[name] = profile
	v.authorized[name] = true // Authorize by default when loaded
	v.mu.Unlock()

	return nil
}

// LoadAllProfiles loads all profiles from the profiles directory.
func (v *MultiProfileVerifier) LoadAllProfiles() error {
	names, err := ListProfiles()
	if err != nil {
		return err
	}

	for _, name := range names {
		if err := v.LoadProfile(name); err != nil {
			return err
		}
	}
	return nil
}

// LoadProfiles loads specific named profiles.
func (v *MultiProfileVerifier) LoadProfiles(names []string) error {
	for _, name := range names {
		if err := v.LoadProfile(name); err != nil {
			return err
		}
	}
	return nil
}

// Authorize enables verification against a profile.
func (v *MultiProfileVerifier) Authorize(name string) error {
	v.mu.Lock()
	defer v.mu.Unlock()

	if _, exists := v.profiles[name]; !exists {
		return fmt.Errorf("profile %s not loaded", name)
	}
	v.authorized[name] = true
	return nil
}

// Deauthorize disables verification against a profile.
func (v *MultiProfileVerifier) Deauthorize(name string) {
	v.mu.Lock()
	v.authorized[name] = false
	v.mu.Unlock()
}

// AuthorizeOnly sets exactly which profiles are authorized.
func (v *MultiProfileVerifier) AuthorizeOnly(names []string) error {
	v.mu.Lock()
	defer v.mu.Unlock()

	// Clear all
	for name := range v.authorized {
		v.authorized[name] = false
	}

	// Authorize specified
	for _, name := range names {
		if _, exists := v.profiles[name]; !exists {
			return fmt.Errorf("profile %s not loaded", name)
		}
		v.authorized[name] = true
	}
	return nil
}

// Verify checks if audio matches any authorized profile.
func (v *MultiProfileVerifier) Verify(samples []float32) (MultiVerifyResult, error) {
	result := MultiVerifyResult{
		AllScores: make(map[string]float32),
	}

	emb, err := v.extractor.Extract(samples)
	if err != nil {
		return result, fmt.Errorf("extract embedding: %w", err)
	}

	v.mu.RLock()
	defer v.mu.RUnlock()

	var bestScore float32
	var bestName string

	for name, profile := range v.profiles {
		if !v.authorized[name] {
			continue
		}

		score := cosineSimilarity(emb, profile.Embedding)
		result.AllScores[name] = score

		if score > bestScore {
			bestScore = score
			bestName = name
		}
	}

	result.Confidence = bestScore
	if bestScore >= v.config.Threshold {
		result.Matched = true
		result.MatchedBy = bestName
	}

	return result, nil
}

// ListLoaded returns the names of all loaded profiles.
func (v *MultiProfileVerifier) ListLoaded() []string {
	v.mu.RLock()
	defer v.mu.RUnlock()

	names := make([]string, 0, len(v.profiles))
	for name := range v.profiles {
		names = append(names, name)
	}
	return names
}

// ListAuthorized returns the names of authorized profiles.
func (v *MultiProfileVerifier) ListAuthorized() []string {
	v.mu.RLock()
	defer v.mu.RUnlock()

	var names []string
	for name, auth := range v.authorized {
		if auth {
			names = append(names, name)
		}
	}
	return names
}

// Close releases resources.
func (v *MultiProfileVerifier) Close() error {
	return v.extractor.Close()
}

// GetThreshold returns the verification threshold.
func (v *MultiProfileVerifier) GetThreshold() float32 {
	return v.config.Threshold
}

// SetThreshold sets the verification threshold.
func (v *MultiProfileVerifier) SetThreshold(t float32) {
	v.config.Threshold = t
}
