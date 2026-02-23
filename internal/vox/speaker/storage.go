// Package speaker - profile storage.
//
// Provides binary serialization for speaker profiles.
// Format:
//   - Magic bytes: "ATLV" (4 bytes) - Atlas Voice profile
//   - Version: uint32 (4 bytes)
//   - Sample count: uint32 (4 bytes)
//   - Created timestamp: int64 (8 bytes, Unix seconds)
//   - Embedding dimension: uint32 (4 bytes)
//   - Embedding data: float32 * dim (dim * 4 bytes)

package speaker

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"
)

// Magic bytes for profile file format.
var profileMagic = []byte("ATLV")

// SaveToFile saves the profile to a binary file.
func (p *Profile) SaveToFile(path string) error {
	// Ensure directory exists
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0700); err != nil {
		return fmt.Errorf("create directory: %w", err)
	}

	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create file: %w", err)
	}
	defer f.Close()

	return p.WriteTo(f)
}

// WriteTo writes the profile to a writer.
func (p *Profile) WriteTo(w io.Writer) error {
	// Magic bytes
	if _, err := w.Write(profileMagic); err != nil {
		return fmt.Errorf("write magic: %w", err)
	}

	// Version
	if err := binary.Write(w, binary.LittleEndian, uint32(p.Version)); err != nil {
		return fmt.Errorf("write version: %w", err)
	}

	// Sample count
	if err := binary.Write(w, binary.LittleEndian, uint32(p.SampleCount)); err != nil {
		return fmt.Errorf("write sample count: %w", err)
	}

	// Created timestamp
	if err := binary.Write(w, binary.LittleEndian, p.CreatedAt.Unix()); err != nil {
		return fmt.Errorf("write timestamp: %w", err)
	}

	// Embedding dimension
	if err := binary.Write(w, binary.LittleEndian, uint32(len(p.Embedding))); err != nil {
		return fmt.Errorf("write embedding dim: %w", err)
	}

	// Embedding data
	for i, v := range p.Embedding {
		if err := binary.Write(w, binary.LittleEndian, v); err != nil {
			return fmt.Errorf("write embedding[%d]: %w", i, err)
		}
	}

	return nil
}

// LoadProfileFromFile loads a profile from a binary file.
func LoadProfileFromFile(path string) (*Profile, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open file: %w", err)
	}
	defer f.Close()

	return ReadProfileFrom(f)
}

// ReadProfileFrom reads a profile from a reader.
func ReadProfileFrom(r io.Reader) (*Profile, error) {
	// Magic bytes
	magic := make([]byte, 4)
	if _, err := io.ReadFull(r, magic); err != nil {
		return nil, fmt.Errorf("read magic: %w", err)
	}
	if string(magic) != string(profileMagic) {
		return nil, fmt.Errorf("invalid profile magic: got %q, expected %q", magic, profileMagic)
	}

	// Version
	var version uint32
	if err := binary.Read(r, binary.LittleEndian, &version); err != nil {
		return nil, fmt.Errorf("read version: %w", err)
	}

	// For future: handle version migration here
	if version > CurrentProfileVersion {
		return nil, fmt.Errorf("profile version %d is newer than supported %d", version, CurrentProfileVersion)
	}

	// Sample count
	var sampleCount uint32
	if err := binary.Read(r, binary.LittleEndian, &sampleCount); err != nil {
		return nil, fmt.Errorf("read sample count: %w", err)
	}

	// Created timestamp
	var timestamp int64
	if err := binary.Read(r, binary.LittleEndian, &timestamp); err != nil {
		return nil, fmt.Errorf("read timestamp: %w", err)
	}

	// Embedding dimension
	var dim uint32
	if err := binary.Read(r, binary.LittleEndian, &dim); err != nil {
		return nil, fmt.Errorf("read embedding dim: %w", err)
	}

	// Sanity check dimension
	if dim == 0 || dim > 4096 {
		return nil, fmt.Errorf("invalid embedding dimension: %d", dim)
	}

	// Embedding data
	embedding := make([]float32, dim)
	for i := uint32(0); i < dim; i++ {
		if err := binary.Read(r, binary.LittleEndian, &embedding[i]); err != nil {
			return nil, fmt.Errorf("read embedding[%d]: %w", i, err)
		}
	}

	return &Profile{
		Embedding:   embedding,
		SampleCount: int(sampleCount),
		CreatedAt:   time.Unix(timestamp, 0),
		Version:     int(version),
	}, nil
}

// ProfilePath returns the default profile path (~/.atlas/speaker_profile.bin).
func ProfilePath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("get home dir: %w", err)
	}
	return filepath.Join(home, ".atlas", "speaker_profile.bin"), nil
}

// ProfileExists checks if a profile exists at the default location.
func ProfileExists() bool {
	path, err := ProfilePath()
	if err != nil {
		return false
	}
	_, err = os.Stat(path)
	return err == nil
}

// ProfilesDir returns the profiles directory path.
func ProfilesDir() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("get home dir: %w", err)
	}
	return filepath.Join(home, ".atlas", "profiles"), nil
}

// NamedProfilePath returns the path for a named profile.
func NamedProfilePath(name string) (string, error) {
	dir, err := ProfilesDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, name+".bin"), nil
}

// ListProfiles returns all available profile names.
func ListProfiles() ([]string, error) {
	dir, err := ProfilesDir()
	if err != nil {
		return nil, err
	}

	entries, err := os.ReadDir(dir)
	if os.IsNotExist(err) {
		return nil, nil // No profiles yet
	}
	if err != nil {
		return nil, fmt.Errorf("read profiles dir: %w", err)
	}

	var names []string
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if filepath.Ext(name) == ".bin" {
			names = append(names, name[:len(name)-4]) // Remove .bin extension
		}
	}
	return names, nil
}

// DeleteProfile removes a named profile.
func DeleteProfile(name string) error {
	path, err := NamedProfilePath(name)
	if err != nil {
		return err
	}
	return os.Remove(path)
}
