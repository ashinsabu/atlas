// Package audio - PortAudio implementation of Source.
//
// Captures audio at the device's native sample rate and resamples to 16kHz
// for Whisper/Silero compatibility.
package audio

import (
	"context"
	"fmt"
	"sync"

	"github.com/gordonklaus/portaudio"
)

// Config defines audio capture parameters.
// Deprecated: Use SourceConfig instead.
type Config = SourceConfig

// PortAudioSource implements Source using PortAudio.
type PortAudioSource struct {
	config     SourceConfig
	nativeRate int // Actual device sample rate
	stream     *portaudio.Stream
	mu         sync.Mutex
	closed     bool
}

// Compile-time check that PortAudioSource implements Source.
var _ Source = (*PortAudioSource)(nil)

// NewPortAudioSource creates a new PortAudio-based audio source.
func NewPortAudioSource(cfg SourceConfig) (*PortAudioSource, error) {
	if err := portaudio.Initialize(); err != nil {
		return nil, fmt.Errorf("portaudio init: %w", err)
	}

	// Get default input device to check native sample rate
	device, err := portaudio.DefaultInputDevice()
	if err != nil {
		portaudio.Terminate()
		return nil, fmt.Errorf("no input device: %w", err)
	}

	nativeRate := int(device.DefaultSampleRate)
	if cfg.Debug {
		fmt.Printf("[audio] Device: %s, native rate: %d Hz, max inputs: %d\n",
			device.Name, nativeRate, device.MaxInputChannels)
	}

	return &PortAudioSource{
		config:     cfg,
		nativeRate: nativeRate,
	}, nil
}

// Start begins capturing audio and calls handler with each chunk.
// Blocks until context is cancelled or error occurs.
// Audio is captured at native rate and resampled to 16kHz.
func (s *PortAudioSource) Start(ctx context.Context, handler func([]byte)) error {
	// Calculate buffer size for native rate
	// We want to output 512 samples at 16kHz (32ms)
	// So at 48kHz, we need 512 * (48000/16000) = 1536 samples
	ratio := float64(s.nativeRate) / float64(s.config.SampleRate)
	nativeBufferSize := int(float64(s.config.FramesPerBuffer) * ratio)

	if s.config.Debug {
		fmt.Printf("[audio] Opening stream: channels=%d, rate=%d, buffer=%d\n",
			s.config.Channels, s.nativeRate, nativeBufferSize)
	}

	buffer := make([]int16, nativeBufferSize)

	stream, err := portaudio.OpenDefaultStream(
		s.config.Channels,
		0, // output channels (none)
		float64(s.nativeRate),
		nativeBufferSize,
		buffer,
	)
	if err != nil {
		return fmt.Errorf("open stream: %w", err)
	}

	s.mu.Lock()
	s.stream = stream
	s.mu.Unlock()

	if err := stream.Start(); err != nil {
		stream.Close()
		return fmt.Errorf("start stream: %w", err)
	}

	defer func() {
		stream.Stop()
		stream.Close()
	}()

	// Pre-allocate output buffer
	outputSize := s.config.FramesPerBuffer
	resampled := make([]int16, outputSize)

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			if err := stream.Read(); err != nil {
				return fmt.Errorf("read stream: %w", err)
			}

			// Resample if needed
			var output []int16
			if s.nativeRate == s.config.SampleRate {
				output = buffer
			} else {
				resample(buffer, resampled, s.nativeRate, s.config.SampleRate)
				output = resampled
			}

			// Convert int16 samples to bytes (little-endian)
			data := int16ToBytes(output)
			handler(data)
		}
	}
}

// Close releases PortAudio resources.
func (s *PortAudioSource) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return nil
	}
	s.closed = true

	if s.stream != nil {
		s.stream.Close()
	}
	return portaudio.Terminate()
}

// resample converts samples from srcRate to dstRate using linear interpolation.
// Simple but effective for speech (not music).
func resample(src []int16, dst []int16, srcRate, dstRate int) {
	ratio := float64(srcRate) / float64(dstRate)

	for i := range dst {
		srcPos := float64(i) * ratio
		srcIdx := int(srcPos)
		frac := srcPos - float64(srcIdx)

		if srcIdx+1 < len(src) {
			// Linear interpolation
			dst[i] = int16(float64(src[srcIdx])*(1-frac) + float64(src[srcIdx+1])*frac)
		} else if srcIdx < len(src) {
			dst[i] = src[srcIdx]
		}
	}
}

// int16ToBytes converts 16-bit samples to little-endian bytes.
func int16ToBytes(samples []int16) []byte {
	data := make([]byte, len(samples)*2)
	for i, s := range samples {
		data[i*2] = byte(s)
		data[i*2+1] = byte(s >> 8)
	}
	return data
}
