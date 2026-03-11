// Package speaker provides speaker verification using a pre-trained ONNX model.
//
// Data flow:
//
//	PCM16 bytes → float32 [-1,1] → mel-filterbank [T×80] → ONNX encoder → embedding [256]
//
// The fbank parameters match WeSpeaker ResNet34-LM training config exactly.
// Feeding wrong feature types (e.g. raw PCM floats) to the ONNX model will
// produce nonsense embeddings — the fbank extraction step is mandatory.
package speaker

import (
	"fmt"
	"math"
)

// FbankConfig defines mel-filterbank extraction parameters.
// Default values match WeSpeaker ResNet34-LM training configuration.
type FbankConfig struct {
	SampleRate   int     // 16000
	NumMelBins   int     // 80
	FrameLenMs   float64 // 25.0 ms = 400 samples at 16kHz
	FrameShiftMs float64 // 10.0 ms = 160 samples at 16kHz
	LowFreq      float64 // 20.0 Hz
	HighFreq     float64 // 7600.0 Hz
	PreEmphasis  float64 // 0.97
}

// DefaultFbankConfig returns parameters matching WeSpeaker training.
func DefaultFbankConfig() FbankConfig {
	return FbankConfig{
		SampleRate:   16000,
		NumMelBins:   80,
		FrameLenMs:   25.0,
		FrameShiftMs: 10.0,
		LowFreq:      20.0,
		HighFreq:     7600.0,
		PreEmphasis:  0.97,
	}
}

// ExtractFbank converts float32 PCM samples (normalized to [-1, 1]) into
// Kaldi-compatible mel-filterbank features.
//
// Returns:
//   - features: flat []float32 of length T*NumMelBins (row-major: frame0_bin0..79, frame1_bin0..79, ...)
//   - numFrames: T
//   - err: non-nil if audio is too short to produce at least one frame
func ExtractFbank(samples []float32, cfg FbankConfig) (features []float32, numFrames int, err error) {
	frameLen := int(float64(cfg.SampleRate) * cfg.FrameLenMs / 1000.0)   // 400
	frameShift := int(float64(cfg.SampleRate) * cfg.FrameShiftMs / 1000.0) // 160
	fftSize := nextPow2(frameLen)                                           // 512

	if len(samples) <= frameLen {
		return nil, 0, fmt.Errorf("audio too short: %d samples, need > %d for at least one frame", len(samples), frameLen)
	}

	numFrames = (len(samples)-frameLen)/frameShift + 1

	// 1. Pre-emphasis: y[n] = x[n] - α*x[n-1] (first sample unchanged)
	emphasized := make([]float32, len(samples))
	emphasized[0] = samples[0]
	for i := 1; i < len(samples); i++ {
		emphasized[i] = samples[i] - float32(cfg.PreEmphasis)*samples[i-1]
	}

	// Hann window: w[n] = 0.5*(1 - cos(2π*n/(N-1))), N = frameLen
	hannWindow := make([]float32, frameLen)
	for n := 0; n < frameLen; n++ {
		hannWindow[n] = float32(0.5 * (1.0 - math.Cos(2.0*math.Pi*float64(n)/float64(frameLen-1))))
	}

	// Pre-compute mel filterbank matrix: [numMelBins][fftSize/2+1]
	filterBank := buildMelFilterBank(cfg.SampleRate, fftSize, cfg.NumMelBins, cfg.LowFreq, cfg.HighFreq)

	features = make([]float32, numFrames*cfg.NumMelBins)

	// Reuse FFT buffers across frames to avoid allocations.
	fftReal := make([]float32, fftSize)
	fftImag := make([]float32, fftSize)
	numBins := fftSize/2 + 1 // 257 for fftSize=512

	for t := 0; t < numFrames; t++ {
		start := t * frameShift

		// Zero-initialize FFT buffers (reused each frame).
		for i := range fftReal {
			fftReal[i] = 0
		}
		for i := range fftImag {
			fftImag[i] = 0
		}

		// 2+3. Extract frame, apply Hann window, zero-pad to fftSize.
		for n := 0; n < frameLen; n++ {
			fftReal[n] = emphasized[start+n] * hannWindow[n]
		}
		// Indices frameLen..fftSize-1 remain zero (zero-padding for FFT).

		// 4. FFT (in-place, radix-2 DIT, 512-point).
		ditFFT(fftReal, fftImag)

		// 5. Power spectrum: |X[k]|^2 for k=0..256 (lower half only).
		power := make([]float32, numBins)
		for k := 0; k < numBins; k++ {
			power[k] = fftReal[k]*fftReal[k] + fftImag[k]*fftImag[k]
		}

		// 6. Apply mel filterbank, log-compress each filter output.
		base := t * cfg.NumMelBins
		for f := 0; f < cfg.NumMelBins; f++ {
			energy := float32(0)
			for k := 0; k < numBins; k++ {
				energy += filterBank[f][k] * power[k]
			}
			// Floor energy to avoid log(0); 1e-6 matches Kaldi/torchaudio.
			features[base+f] = float32(math.Log(float64(energy) + 1e-6))
		}
	}

	// 7. Per-utterance mean normalization: subtract per-bin mean across time.
	// This removes channel/microphone effects and improves speaker embeddings.
	for f := 0; f < cfg.NumMelBins; f++ {
		var sum float32
		for t := 0; t < numFrames; t++ {
			sum += features[t*cfg.NumMelBins+f]
		}
		mean := sum / float32(numFrames)
		for t := 0; t < numFrames; t++ {
			features[t*cfg.NumMelBins+f] -= mean
		}
	}

	return features, numFrames, nil
}

// buildMelFilterBank computes the [numMelBins][numBins] triangular filter matrix.
// Linearly spaced on the mel scale between lowFreq and highFreq.
// Uses Kaldi/torchaudio bin conversion: bin = floor((fftSize+1)*hz/sampleRate).
func buildMelFilterBank(sampleRate, fftSize, numMelBins int, lowFreq, highFreq float64) [][]float32 {
	numBins := fftSize/2 + 1

	melLow := hz2mel(lowFreq)
	melHigh := hz2mel(highFreq)

	// numMelBins+2 linearly-spaced mel points (includes left and right boundaries).
	melPoints := make([]float64, numMelBins+2)
	for i := range melPoints {
		melPoints[i] = melLow + float64(i)*(melHigh-melLow)/float64(numMelBins+1)
	}

	// Convert mel points to FFT bin indices.
	bins := make([]int, numMelBins+2)
	for i, m := range melPoints {
		hz := mel2hz(m)
		b := int(math.Floor(float64(fftSize+1) * hz / float64(sampleRate)))
		if b >= numBins {
			b = numBins - 1
		}
		bins[i] = b
	}

	// Build triangular filter matrix.
	filterBank := make([][]float32, numMelBins)
	for i := 0; i < numMelBins; i++ {
		filterBank[i] = make([]float32, numBins)
		left := bins[i]
		center := bins[i+1]
		right := bins[i+2]

		// Rising slope: left → center
		if center > left {
			for k := left; k < center && k < numBins; k++ {
				filterBank[i][k] = float32(float64(k-left) / float64(center-left))
			}
		}
		// Falling slope: center → right
		if right > center {
			for k := center; k <= right && k < numBins; k++ {
				filterBank[i][k] = float32(float64(right-k) / float64(right-center))
			}
		}
	}

	return filterBank
}

// hz2mel converts frequency in Hz to the mel scale.
func hz2mel(hz float64) float64 {
	return 2595.0 * math.Log10(1.0+hz/700.0)
}

// mel2hz converts mel scale value back to Hz.
func mel2hz(mel float64) float64 {
	return 700.0 * (math.Pow(10.0, mel/2595.0) - 1.0)
}

// nextPow2 returns the smallest power of 2 >= n.
func nextPow2(n int) int {
	p := 1
	for p < n {
		p <<= 1
	}
	return p
}

// ditFFT performs an in-place radix-2 DIT Cooley-Tukey FFT.
// Both slices must be the same power-of-2 length.
// For real-only input, set imag to all zeros before calling.
func ditFFT(real, imag []float32) {
	n := len(real)

	// Bit-reversal permutation: rearranges elements so butterflies are in-order.
	j := 0
	for i := 1; i < n; i++ {
		bit := n >> 1
		for ; j&bit != 0; bit >>= 1 {
			j ^= bit
		}
		j ^= bit
		if i < j {
			real[i], real[j] = real[j], real[i]
			imag[i], imag[j] = imag[j], imag[i]
		}
	}

	// Cooley-Tukey butterfly: process each FFT stage.
	// Stage s handles groups of length 2^s; twiddle factors are e^(-2πi*k/length).
	for length := 2; length <= n; length <<= 1 {
		halfLen := length >> 1
		angle := -2.0 * math.Pi / float64(length)
		// wReal+i*wImag is the base twiddle factor for this stage.
		wReal := float32(math.Cos(angle))
		wImag := float32(math.Sin(angle))

		for i := 0; i < n; i += length {
			// uReal+i*uImag: running twiddle factor, starts at 1.
			uReal := float32(1.0)
			uImag := float32(0.0)
			for k := 0; k < halfLen; k++ {
				// Butterfly: a = x[i+k] + u*x[i+k+h], b = x[i+k] - u*x[i+k+h]
				tReal := uReal*real[i+k+halfLen] - uImag*imag[i+k+halfLen]
				tImag := uReal*imag[i+k+halfLen] + uImag*real[i+k+halfLen]
				real[i+k+halfLen] = real[i+k] - tReal
				imag[i+k+halfLen] = imag[i+k] - tImag
				real[i+k] += tReal
				imag[i+k] += tImag
				// Advance twiddle factor: u *= w
				newU := uReal*wReal - uImag*wImag
				uImag = uReal*wImag + uImag*wReal
				uReal = newU
			}
		}
	}
}
