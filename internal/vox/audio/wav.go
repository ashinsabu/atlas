// Package audio - WAV file utilities.
//
// Provides functions for loading and saving WAV files in the format
// expected by Atlas (16kHz mono 16-bit PCM).

package audio

import (
	"encoding/binary"
	"fmt"
	"os"
)

// WAV format constants
const (
	SampleRate      = 16000
	Channels        = 1
	BitsPerSample   = 16
	BytesPerSample  = BitsPerSample / 8
)

// LoadWAV loads a 16kHz mono 16-bit WAV file and returns float32 samples.
// Samples are normalized to [-1.0, 1.0].
func LoadWAV(path string) ([]float32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	// Read RIFF header (44 bytes standard)
	header := make([]byte, 44)
	if _, err := f.Read(header); err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}

	// Verify RIFF/WAVE
	if string(header[0:4]) != "RIFF" || string(header[8:12]) != "WAVE" {
		return nil, fmt.Errorf("not a valid WAV file")
	}

	// Read format chunk
	numChannels := int(binary.LittleEndian.Uint16(header[22:24]))
	wavSampleRate := int(binary.LittleEndian.Uint32(header[24:28]))
	bitsPerSample := int(binary.LittleEndian.Uint16(header[34:36]))

	// Validate format
	if numChannels != Channels {
		return nil, fmt.Errorf("expected mono, got %d channels", numChannels)
	}
	if wavSampleRate != SampleRate {
		return nil, fmt.Errorf("expected %dHz, got %d Hz", SampleRate, wavSampleRate)
	}
	if bitsPerSample != BitsPerSample {
		return nil, fmt.Errorf("expected %d-bit, got %d-bit", BitsPerSample, bitsPerSample)
	}

	// Data chunk size
	dataSize := int(binary.LittleEndian.Uint32(header[40:44]))
	numSamples := dataSize / BytesPerSample

	// Read audio data
	audioBytes := make([]byte, dataSize)
	if _, err := f.Read(audioBytes); err != nil {
		return nil, fmt.Errorf("read audio data: %w", err)
	}

	// Convert to float32
	samples := make([]float32, numSamples)
	for i := range numSamples {
		sample := int16(audioBytes[i*2]) | int16(audioBytes[i*2+1])<<8
		samples[i] = float32(sample) / 32768.0
	}

	return samples, nil
}

// LoadWAVBytes loads a 16kHz mono 16-bit WAV file and returns raw PCM bytes.
func LoadWAVBytes(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	// Read RIFF header
	header := make([]byte, 44)
	if _, err := f.Read(header); err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}

	// Verify RIFF/WAVE
	if string(header[0:4]) != "RIFF" || string(header[8:12]) != "WAVE" {
		return nil, fmt.Errorf("not a valid WAV file")
	}

	// Data chunk size
	dataSize := int(binary.LittleEndian.Uint32(header[40:44]))

	// Read audio data
	audioBytes := make([]byte, dataSize)
	if _, err := f.Read(audioBytes); err != nil {
		return nil, fmt.Errorf("read audio data: %w", err)
	}

	return audioBytes, nil
}

// SaveWAV saves float32 samples to a WAV file.
func SaveWAV(path string, samples []float32) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	dataSize := len(samples) * BytesPerSample
	fileSize := 36 + dataSize

	// Write RIFF header
	f.Write([]byte("RIFF"))
	binary.Write(f, binary.LittleEndian, uint32(fileSize))
	f.Write([]byte("WAVE"))

	// Write fmt chunk
	f.Write([]byte("fmt "))
	binary.Write(f, binary.LittleEndian, uint32(16)) // chunk size
	binary.Write(f, binary.LittleEndian, uint16(1))  // PCM format
	binary.Write(f, binary.LittleEndian, uint16(Channels))
	binary.Write(f, binary.LittleEndian, uint32(SampleRate))
	binary.Write(f, binary.LittleEndian, uint32(SampleRate*Channels*BytesPerSample)) // byte rate
	binary.Write(f, binary.LittleEndian, uint16(Channels*BytesPerSample))            // block align
	binary.Write(f, binary.LittleEndian, uint16(BitsPerSample))

	// Write data chunk
	f.Write([]byte("data"))
	binary.Write(f, binary.LittleEndian, uint32(dataSize))

	// Write samples
	for _, s := range samples {
		sample := int16(s * 32767)
		binary.Write(f, binary.LittleEndian, sample)
	}

	return nil
}

// SaveWAVInt16 saves int16 samples to a WAV file.
func SaveWAVInt16(path string, samples []int16) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	dataSize := len(samples) * BytesPerSample
	fileSize := 36 + dataSize

	// Write RIFF header
	f.Write([]byte("RIFF"))
	binary.Write(f, binary.LittleEndian, uint32(fileSize))
	f.Write([]byte("WAVE"))

	// Write fmt chunk
	f.Write([]byte("fmt "))
	binary.Write(f, binary.LittleEndian, uint32(16))
	binary.Write(f, binary.LittleEndian, uint16(1))
	binary.Write(f, binary.LittleEndian, uint16(Channels))
	binary.Write(f, binary.LittleEndian, uint32(SampleRate))
	binary.Write(f, binary.LittleEndian, uint32(SampleRate*Channels*BytesPerSample))
	binary.Write(f, binary.LittleEndian, uint16(Channels*BytesPerSample))
	binary.Write(f, binary.LittleEndian, uint16(BitsPerSample))

	// Write data chunk
	f.Write([]byte("data"))
	binary.Write(f, binary.LittleEndian, uint32(dataSize))

	// Write samples
	for _, s := range samples {
		binary.Write(f, binary.LittleEndian, s)
	}

	return nil
}

// Int16ToFloat32 converts int16 samples to float32.
func Int16ToFloat32(samples []int16) []float32 {
	result := make([]float32, len(samples))
	for i, s := range samples {
		result[i] = float32(s) / 32768.0
	}
	return result
}
