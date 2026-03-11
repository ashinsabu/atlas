// Package speaker - FBANK (mel filterbank) feature extraction.
//
// Computes 80-dimensional mel filterbank features for speaker embedding models.
// Based on standard speech processing parameters:
// - 16kHz sample rate
// - 25ms frame size (400 samples)
// - 10ms hop size (160 samples)
// - 80 mel filterbanks
// - Pre-emphasis filter
// - Hamming window

package speaker

import (
	"math"
)

// FBankConfig holds filterbank extraction parameters.
type FBankConfig struct {
	SampleRate      int
	FrameSize       int     // Samples per frame (25ms = 400 at 16kHz)
	HopSize         int     // Samples between frames (10ms = 160 at 16kHz)
	NumMelBins      int     // Number of mel filterbanks (typically 80)
	LowFreq         float64 // Lowest frequency for mel banks
	HighFreq        float64 // Highest frequency (0 = Nyquist, negative = offset from Nyquist)
	PreEmphasis     float64 // Pre-emphasis coefficient (0.97 typical)
	UseLogMel  bool // Apply log to mel energies
	ApplyCMN   bool // Apply cepstral mean normalization (WeSpeaker)
	AmplitudeScale  float32 // Scale factor for audio amplitude (WeSpeaker uses 32768)
	UsePoveyWindow  bool    // Use Povey window (Kaldi-style) instead of Hamming
	RemoveDCOffset  bool    // Remove DC offset from audio before processing
}

// DefaultFBankConfig returns standard parameters for speaker embedding.
// Matches sherpa-onnx/Kaldi defaults: 80 mel bins, 25ms frame, 10ms hop, Povey window.
func DefaultFBankConfig() FBankConfig {
	return FBankConfig{
		SampleRate:     16000,
		FrameSize:      400,    // 25ms at 16kHz
		HopSize:        160,    // 10ms at 16kHz
		NumMelBins:     80,
		LowFreq:        20.0,
		HighFreq:       -400.0, // Nyquist - 400 = 7600Hz (sherpa-onnx default)
		PreEmphasis: 0.97,
		UseLogMel:   true,
		ApplyCMN:    true, // Mean normalization (WeSpeaker standard)
		AmplitudeScale: 32768,  // WeSpeaker expects int16 scale audio
		UsePoveyWindow: true,   // Kaldi-style window (required for WeSpeaker)
		RemoveDCOffset: true,   // Remove DC offset (sherpa-onnx default)
	}
}

// ComputeFBank extracts mel filterbank features from audio samples.
// Input: float32 audio samples, 16kHz mono, normalized to [-1.0, 1.0]
// Output: 2D array [num_frames][num_mel_bins]
func ComputeFBank(samples []float32, cfg FBankConfig) [][]float32 {
	if len(samples) < cfg.FrameSize {
		return nil
	}

	// Set high frequency (negative means offset from Nyquist)
	nyquist := float64(cfg.SampleRate) / 2
	highFreq := cfg.HighFreq
	if highFreq <= 0 {
		highFreq = nyquist + highFreq // e.g., -400 -> 7600 Hz
	}
	if highFreq > nyquist {
		highFreq = nyquist
	}

	// Make a working copy
	working := make([]float32, len(samples))
	copy(working, samples)

	// Remove DC offset if configured
	if cfg.RemoveDCOffset {
		var sum float64
		for _, s := range working {
			sum += float64(s)
		}
		mean := float32(sum / float64(len(working)))
		for i := range working {
			working[i] -= mean
		}
	}

	// Scale amplitude if configured (WeSpeaker expects int16 range)
	if cfg.AmplitudeScale > 0 {
		for i := range working {
			working[i] *= cfg.AmplitudeScale
		}
	}

	// Pre-emphasis
	preEmph := make([]float32, len(working))
	preEmph[0] = working[0]
	for i := 1; i < len(working); i++ {
		preEmph[i] = working[i] - float32(cfg.PreEmphasis)*working[i-1]
	}

	// Compute number of frames
	numFrames := (len(preEmph) - cfg.FrameSize) / cfg.HopSize + 1
	if numFrames < 1 {
		return nil
	}

	// FFT size (power of 2 >= frame size)
	fftSize := 1
	for fftSize < cfg.FrameSize {
		fftSize *= 2
	}

	// Create window function
	var window []float64
	if cfg.UsePoveyWindow {
		window = makePoveyWindow(cfg.FrameSize)
	} else {
		window = makeHammingWindow(cfg.FrameSize)
	}

	// Create mel filterbank
	melFilter := createMelFilterbank(cfg.NumMelBins, fftSize, cfg.SampleRate, cfg.LowFreq, highFreq)

	// Extract features for each frame
	features := make([][]float32, numFrames)
	for i := 0; i < numFrames; i++ {
		start := i * cfg.HopSize
		frame := preEmph[start : start+cfg.FrameSize]

		// Apply window
		windowed := make([]float64, fftSize)
		for j := 0; j < cfg.FrameSize; j++ {
			windowed[j] = float64(frame[j]) * window[j]
		}

		// Compute power spectrum
		powerSpec := computePowerSpectrum(windowed)

		// Apply mel filterbank
		melEnergies := applyMelFilterbank(powerSpec, melFilter)

		// Apply log
		if cfg.UseLogMel {
			for j := range melEnergies {
				if melEnergies[j] < 1e-10 {
					melEnergies[j] = 1e-10
				}
				melEnergies[j] = float32(math.Log(float64(melEnergies[j])))
			}
		}

		features[i] = melEnergies
	}

	// Apply normalization (per-utterance)
	if len(features) > 0 && cfg.ApplyCMN {
		features = applyCMN(features)
	}

	return features
}

// makeHammingWindow creates a Hamming window of the given size.
func makeHammingWindow(size int) []float64 {
	window := make([]float64, size)
	for i := 0; i < size; i++ {
		window[i] = 0.54 - 0.46*math.Cos(2*math.Pi*float64(i)/float64(size-1))
	}
	return window
}

// makePoveyWindow creates a Povey window (Kaldi-style).
// Povey window is a Hann window raised to the power of 0.85.
// Formula: w[n] = (0.5 - 0.5 * cos(2*pi*n/(N-1)))^0.85
func makePoveyWindow(size int) []float64 {
	window := make([]float64, size)
	for i := 0; i < size; i++ {
		// Hann window value
		hann := 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(size-1))
		// Raise to power 0.85 for Povey window
		window[i] = math.Pow(hann, 0.85)
	}
	return window
}

// hzToMel converts frequency in Hz to mel scale.
func hzToMel(hz float64) float64 {
	return 2595.0 * math.Log10(1.0+hz/700.0)
}

// melToHz converts mel scale to frequency in Hz.
func melToHz(mel float64) float64 {
	return 700.0 * (math.Pow(10.0, mel/2595.0) - 1.0)
}

// createMelFilterbank creates a mel filterbank matrix.
// Returns: [numMelBins][fftSize/2+1] weights
func createMelFilterbank(numMelBins, fftSize, sampleRate int, lowFreq, highFreq float64) [][]float32 {
	numBins := fftSize/2 + 1

	// Compute mel points
	lowMel := hzToMel(lowFreq)
	highMel := hzToMel(highFreq)

	melPoints := make([]float64, numMelBins+2)
	for i := 0; i < numMelBins+2; i++ {
		melPoints[i] = lowMel + float64(i)*(highMel-lowMel)/float64(numMelBins+1)
	}

	// Convert to FFT bin indices
	bins := make([]int, numMelBins+2)
	for i := range melPoints {
		hz := melToHz(melPoints[i])
		bins[i] = int(math.Floor(float64(fftSize+1) * hz / float64(sampleRate)))
		if bins[i] < 0 {
			bins[i] = 0
		}
		if bins[i] >= numBins {
			bins[i] = numBins - 1
		}
	}

	// Create filterbank
	filterbank := make([][]float32, numMelBins)
	for i := 0; i < numMelBins; i++ {
		filterbank[i] = make([]float32, numBins)

		for j := bins[i]; j < bins[i+1]; j++ {
			if bins[i+1]-bins[i] > 0 {
				filterbank[i][j] = float32(j-bins[i]) / float32(bins[i+1]-bins[i])
			}
		}
		for j := bins[i+1]; j < bins[i+2]; j++ {
			if bins[i+2]-bins[i+1] > 0 {
				filterbank[i][j] = float32(bins[i+2]-j) / float32(bins[i+2]-bins[i+1])
			}
		}
	}

	return filterbank
}

// computePowerSpectrum computes the power spectrum using DFT.
// For simplicity, uses direct DFT rather than FFT.
func computePowerSpectrum(frame []float64) []float32 {
	n := len(frame)
	numBins := n/2 + 1
	spectrum := make([]float32, numBins)

	for k := 0; k < numBins; k++ {
		var real, imag float64
		for j := 0; j < n; j++ {
			angle := -2.0 * math.Pi * float64(k) * float64(j) / float64(n)
			real += frame[j] * math.Cos(angle)
			imag += frame[j] * math.Sin(angle)
		}
		spectrum[k] = float32(real*real + imag*imag)
	}

	return spectrum
}

// applyMelFilterbank applies mel filterbank to power spectrum.
func applyMelFilterbank(powerSpec []float32, filterbank [][]float32) []float32 {
	numMelBins := len(filterbank)
	melEnergies := make([]float32, numMelBins)

	for i := 0; i < numMelBins; i++ {
		var energy float32
		for j := 0; j < len(powerSpec) && j < len(filterbank[i]); j++ {
			energy += powerSpec[j] * filterbank[i][j]
		}
		melEnergies[i] = energy
	}

	return melEnergies
}

// applyCMN applies cepstral mean normalization per utterance (WeSpeaker standard).
// Only subtracts mean, does NOT normalize variance.
func applyCMN(features [][]float32) [][]float32 {
	if len(features) == 0 {
		return features
	}

	numFrames := len(features)
	numBins := len(features[0])

	// Compute mean across time dimension (dim 0)
	mean := make([]float64, numBins)
	for i := 0; i < numFrames; i++ {
		for j := 0; j < numBins; j++ {
			mean[j] += float64(features[i][j])
		}
	}
	for j := 0; j < numBins; j++ {
		mean[j] /= float64(numFrames)
	}

	// Subtract mean only (no variance normalization)
	result := make([][]float32, numFrames)
	for i := 0; i < numFrames; i++ {
		result[i] = make([]float32, numBins)
		for j := 0; j < numBins; j++ {
			result[i][j] = float32(float64(features[i][j]) - mean[j])
		}
	}

	return result
}
