package speaker

import (
	"math"
	"testing"
)

// TestExtractFbank_Shape verifies output dimensions for 1s of audio.
func TestExtractFbank_Shape(t *testing.T) {
	cfg := DefaultFbankConfig()
	samples := makeSine(300.0, 1.0, cfg.SampleRate) // 1s, 16000 samples

	features, T, err := ExtractFbank(samples, cfg)
	if err != nil {
		t.Fatalf("ExtractFbank: %v", err)
	}

	// Expected: (16000 - 400) / 160 + 1 = 98 frames
	wantT := (len(samples)-400)/160 + 1
	if T != wantT {
		t.Errorf("numFrames = %d, want %d", T, wantT)
	}
	if len(features) != T*cfg.NumMelBins {
		t.Errorf("len(features) = %d, want %d", len(features), T*cfg.NumMelBins)
	}
}

// TestExtractFbank_Determinism verifies identical output on repeated calls.
func TestExtractFbank_Determinism(t *testing.T) {
	cfg := DefaultFbankConfig()
	samples := makeSine(300.0, 1.0, cfg.SampleRate)

	f1, T1, err1 := ExtractFbank(samples, cfg)
	f2, T2, err2 := ExtractFbank(samples, cfg)

	if err1 != nil || err2 != nil {
		t.Fatalf("ExtractFbank errors: %v, %v", err1, err2)
	}
	if T1 != T2 {
		t.Fatalf("different frame counts: %d vs %d", T1, T2)
	}
	for i := range f1 {
		if f1[i] != f2[i] {
			t.Errorf("feature[%d] differs: %f vs %f", i, f1[i], f2[i])
			break
		}
	}
}

// TestExtractFbank_ValueRange checks that feature values are in a reasonable range.
//
// After per-utterance mean normalization, values should be centered near zero
// with relatively small absolute magnitude (log-energy residuals).
// This catches obvious bugs like all-zero output or unconstrained values.
func TestExtractFbank_ValueRange(t *testing.T) {
	cfg := DefaultFbankConfig()
	samples := makeSine(300.0, 1.0, cfg.SampleRate)

	features, _, err := ExtractFbank(samples, cfg)
	if err != nil {
		t.Fatalf("ExtractFbank: %v", err)
	}

	var maxAbs float32
	allZero := true
	for _, v := range features {
		if v != 0 {
			allZero = false
		}
		if v < 0 {
			v = -v
		}
		if v > maxAbs {
			maxAbs = v
		}
	}

	if allZero {
		t.Error("all features are zero — fbank extraction produced no output")
	}
	// After mean normalization, absolute values should be well below 100.
	// Unreasonably large values indicate a bug in the log or filter computation.
	if maxAbs > 50 {
		t.Errorf("max |feature| = %f, expected < 50 after mean normalization", maxAbs)
	}
}

// TestExtractFbank_FilterBankFreq verifies the filter bank maps 300Hz to bin ~10.
// This tests the filter bank construction separately from mean normalization.
func TestExtractFbank_FilterBankFreq(t *testing.T) {
	cfg := DefaultFbankConfig()
	frameLen := 400
	fftSize := 512
	numBins := fftSize/2 + 1

	filterBank := buildMelFilterBank(cfg.SampleRate, fftSize, cfg.NumMelBins, cfg.LowFreq, cfg.HighFreq)

	// The FFT bin for 300Hz: bin = floor((fftSize+1)*300/16000) = floor(9.5625) = 9
	targetBin := 9

	// Find which mel filter has the highest weight for FFT bin 9.
	bestFilter := -1
	var bestWeight float32
	for f := 0; f < cfg.NumMelBins; f++ {
		if f >= len(filterBank) || targetBin >= numBins {
			break
		}
		if filterBank[f][targetBin] > bestWeight {
			bestWeight = filterBank[f][targetBin]
			bestFilter = f
		}
	}
	_ = frameLen

	// 300Hz maps to mel filter ~10 (0-indexed).
	// mel(300) ≈ 402; range [31.75..2787]; step ≈ 34 → index ≈ 10-11.
	if bestFilter < 7 || bestFilter > 15 {
		t.Errorf("300Hz maps to mel filter %d, expected 7..15", bestFilter)
	}
}

// TestExtractFbank_TooShort verifies error on sub-frame audio.
func TestExtractFbank_TooShort(t *testing.T) {
	cfg := DefaultFbankConfig()
	samples := make([]float32, 100) // 100 samples < 400 frame length
	_, _, err := ExtractFbank(samples, cfg)
	if err == nil {
		t.Error("expected error for too-short audio, got nil")
	}
}

// TestNextPow2 exercises the utility function.
func TestNextPow2(t *testing.T) {
	cases := [][2]int{
		{1, 1}, {2, 2}, {3, 4}, {400, 512}, {512, 512}, {513, 1024},
	}
	for _, c := range cases {
		if got := nextPow2(c[0]); got != c[1] {
			t.Errorf("nextPow2(%d) = %d, want %d", c[0], got, c[1])
		}
	}
}

// TestDitFFT_KnownValues validates the FFT against a manually computed result.
// For a 4-point FFT of [1, 0, 0, 0], output should be [1, 1, 1, 1].
func TestDitFFT_KnownValues(t *testing.T) {
	real := []float32{1, 0, 0, 0}
	imag := []float32{0, 0, 0, 0}
	ditFFT(real, imag)
	for i, v := range real {
		if math.Abs(float64(v-1)) > 1e-5 {
			t.Errorf("real[%d] = %f, want 1.0", i, v)
		}
	}
	for i, v := range imag {
		if math.Abs(float64(v)) > 1e-5 {
			t.Errorf("imag[%d] = %f, want 0.0", i, v)
		}
	}
}

// TestDitFFT_Sine checks that a sine wave produces a peak at the correct bin.
// 8-point FFT of sin(2π*k/8) at k=0..7 should have energy mainly at bin 1.
func TestDitFFT_Sine(t *testing.T) {
	n := 8
	real := make([]float32, n)
	imag := make([]float32, n)
	for i := range real {
		real[i] = float32(math.Sin(2 * math.Pi * float64(i) / float64(n)))
	}
	ditFFT(real, imag)

	// |X[1]| should be significantly larger than all other bins.
	mag1 := math.Sqrt(float64(real[1]*real[1] + imag[1]*imag[1]))
	for k := 0; k < n; k++ {
		if k == 1 || k == n-1 {
			continue // conjugate pair for real sine
		}
		magK := math.Sqrt(float64(real[k]*real[k] + imag[k]*imag[k]))
		if magK > mag1*0.1 {
			t.Errorf("unexpected energy at bin %d: |X[%d]|=%.3f vs |X[1]|=%.3f", k, k, magK, mag1)
		}
	}
}

// makeSine generates freqHz sine wave at sampleRate for durationSecs.
func makeSine(freqHz, durationSecs float64, sampleRate int) []float32 {
	n := int(durationSecs * float64(sampleRate))
	samples := make([]float32, n)
	for i := range samples {
		samples[i] = float32(math.Sin(2 * math.Pi * freqHz * float64(i) / float64(sampleRate)))
	}
	return samples
}
