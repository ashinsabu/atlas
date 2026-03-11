package speaker

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// Profile stores an enrolled speaker's averaged embedding.
// Saved to disk as JSON; loaded at startup if speaker verification is enabled.
type Profile struct {
	Version         int       `json:"version"`
	SpeakerName     string    `json:"speaker_name"`
	Embedding       []float32 `json:"embedding"`
	EnrollmentCount int       `json:"enrollment_count"` // number of recordings used
	CreatedAt       time.Time `json:"created_at"`
}

// LoadProfile reads a speaker profile from disk.
// Returns nil, nil if the file does not exist (caller treats as "not enrolled").
func LoadProfile(path string) (*Profile, error) {
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read profile: %w", err)
	}

	var p Profile
	if err := json.Unmarshal(data, &p); err != nil {
		return nil, fmt.Errorf("parse profile: %w", err)
	}
	return &p, nil
}

// SaveProfile writes a speaker profile to disk, creating parent directories.
func SaveProfile(path string, p *Profile) error {
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		return fmt.Errorf("create profile dir: %w", err)
	}

	data, err := json.MarshalIndent(p, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal profile: %w", err)
	}

	if err := os.WriteFile(path, data, 0600); err != nil {
		return fmt.Errorf("write profile: %w", err)
	}
	return nil
}
