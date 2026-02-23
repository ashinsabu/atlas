// Package speaker - embedding extraction.
//
// This file contains the embedding extraction logic. Currently uses a placeholder
// implementation based on audio statistics (energy, zero-crossings, spectral features).
//
// TODO: Replace with ONNX model (e.g., ECAPA-TDNN, x-vector) for production accuracy.

package speaker

import (
	"errors"
	"math"
)

// ErrAudioTooShort is returned when audio is too short for embedding extraction.
var ErrAudioTooShort = errors.New("audio too short for embedding extraction")

// MinAudioSamples is the minimum number of samples required for extraction.
// At 16kHz, this is ~0.5 seconds of audio.
const MinAudioSamples = 8000

// Extractor extracts speaker embeddings from audio.
// Currently uses placeholder audio statistics; will be replaced with ONNX.
type Extractor struct {
	dim int
}

// NewExtractor creates an embedding extractor.
// dim: output embedding dimension (must match model when ONNX is added)
func NewExtractor(dim int) *Extractor {
	return &Extractor{dim: dim}
}

// Extract computes a speaker embedding from audio samples.
// Input: float32 audio samples, 16kHz mono, normalized to [-1.0, 1.0]
// Output: embedding vector of length Extractor.dim
//
// PLACEHOLDER IMPLEMENTATION:
// Computes audio statistics in overlapping frames:
// - Frame energy (RMS)
// - Zero-crossing rate
// - Spectral centroid approximation
// - Delta features (first derivative of above)
//
// This provides some speaker discrimination but is not production-ready.
// Real speaker embeddings require a trained neural network.
func (e *Extractor) Extract(samples []float32) ([]float32, error) {
	if len(samples) < MinAudioSamples {
		return nil, ErrAudioTooShort
	}

	// Frame parameters (match typical speech processing)
	frameSize := 400   // 25ms at 16kHz
	hopSize := 160     // 10ms hop
	numFrames := (len(samples) - frameSize) / hopSize

	if numFrames < 10 {
		return nil, ErrAudioTooShort
	}

	// Compute frame-level features
	frameEnergies := make([]float32, numFrames)
	frameZCR := make([]float32, numFrames)
	frameSpectral := make([]float32, numFrames)

	for i := 0; i < numFrames; i++ {
		start := i * hopSize
		frame := samples[start : start+frameSize]

		frameEnergies[i] = computeRMS(frame)
		frameZCR[i] = computeZCR(frame)
		frameSpectral[i] = computeSpectralCentroid(frame)
	}

	// Aggregate statistics into embedding
	embedding := make([]float32, e.dim)
	idx := 0

	// Energy statistics (8 values)
	stats := computeStats(frameEnergies)
	idx = copyStats(embedding, idx, stats)

	// ZCR statistics (8 values)
	stats = computeStats(frameZCR)
	idx = copyStats(embedding, idx, stats)

	// Spectral centroid statistics (8 values)
	stats = computeStats(frameSpectral)
	idx = copyStats(embedding, idx, stats)

	// Delta features (first derivative)
	deltaEnergy := computeDeltas(frameEnergies)
	stats = computeStats(deltaEnergy)
	idx = copyStats(embedding, idx, stats)

	deltaZCR := computeDeltas(frameZCR)
	stats = computeStats(deltaZCR)
	idx = copyStats(embedding, idx, stats)

	deltaSpectral := computeDeltas(frameSpectral)
	stats = computeStats(deltaSpectral)
	idx = copyStats(embedding, idx, stats)

	// Pitch-related features (simplified)
	pitchFeatures := computePitchFeatures(samples)
	idx = copySlice(embedding, idx, pitchFeatures)

	// Temporal dynamics
	tempFeatures := computeTemporalFeatures(frameEnergies, frameZCR)
	idx = copySlice(embedding, idx, tempFeatures)

	// Fill remaining with derived features (histogram bins, etc.)
	histEnergy := computeHistogram(frameEnergies, 8)
	idx = copySlice(embedding, idx, histEnergy)

	histZCR := computeHistogram(frameZCR, 8)
	idx = copySlice(embedding, idx, histZCR)

	histSpectral := computeHistogram(frameSpectral, 8)
	idx = copySlice(embedding, idx, histSpectral)

	// Cross-correlation features
	corrFeatures := computeCrossCorrelation(frameEnergies, frameZCR, frameSpectral)
	idx = copySlice(embedding, idx, corrFeatures)

	// Fill any remaining dimensions with zeros (padding for ONNX compatibility)
	for i := idx; i < e.dim; i++ {
		embedding[i] = 0
	}

	// L2 normalize the embedding
	normalizeL2(embedding)

	return embedding, nil
}

// computeRMS computes root mean square energy of a frame.
func computeRMS(frame []float32) float32 {
	var sum float32
	for _, s := range frame {
		sum += s * s
	}
	return float32(math.Sqrt(float64(sum / float32(len(frame)))))
}

// computeZCR computes zero-crossing rate of a frame.
func computeZCR(frame []float32) float32 {
	var crossings int
	for i := 1; i < len(frame); i++ {
		if (frame[i] >= 0) != (frame[i-1] >= 0) {
			crossings++
		}
	}
	return float32(crossings) / float32(len(frame)-1)
}

// computeSpectralCentroid approximates spectral centroid using DFT magnitude.
// Simplified version - computes weighted average frequency.
func computeSpectralCentroid(frame []float32) float32 {
	// Apply simple DFT at a few frequency bins
	numBins := 16
	magnitudes := make([]float32, numBins)

	for k := 0; k < numBins; k++ {
		var real, imag float64
		freq := float64(k) * 2 * math.Pi / float64(numBins)
		for n, s := range frame {
			angle := freq * float64(n)
			real += float64(s) * math.Cos(angle)
			imag += float64(s) * math.Sin(angle)
		}
		magnitudes[k] = float32(math.Sqrt(real*real + imag*imag))
	}

	// Compute centroid
	var sumWeighted, sumMag float32
	for k, mag := range magnitudes {
		sumWeighted += float32(k) * mag
		sumMag += mag
	}

	if sumMag == 0 {
		return 0
	}
	return sumWeighted / sumMag
}

// statisticsResult holds computed statistics.
type statisticsResult struct {
	mean, std, min, max           float32
	q25, median, q75, dynamicRange float32
}

// computeStats computes summary statistics for a feature sequence.
func computeStats(values []float32) statisticsResult {
	if len(values) == 0 {
		return statisticsResult{}
	}

	// Sort for percentiles
	sorted := make([]float32, len(values))
	copy(sorted, values)
	sortFloat32(sorted)

	// Basic stats
	var sum, sumSq float32
	minVal, maxVal := sorted[0], sorted[len(sorted)-1]

	for _, v := range values {
		sum += v
		sumSq += v * v
	}

	mean := sum / float32(len(values))
	variance := sumSq/float32(len(values)) - mean*mean
	std := float32(math.Sqrt(float64(variance)))

	// Percentiles
	q25 := sorted[len(sorted)*25/100]
	median := sorted[len(sorted)*50/100]
	q75 := sorted[len(sorted)*75/100]

	dynamicRange := maxVal - minVal

	return statisticsResult{
		mean: mean, std: std, min: minVal, max: maxVal,
		q25: q25, median: median, q75: q75, dynamicRange: dynamicRange,
	}
}

// computeDeltas computes first-order differences (delta features).
func computeDeltas(values []float32) []float32 {
	if len(values) < 2 {
		return []float32{0}
	}
	deltas := make([]float32, len(values)-1)
	for i := 1; i < len(values); i++ {
		deltas[i-1] = values[i] - values[i-1]
	}
	return deltas
}

// computePitchFeatures extracts simplified pitch-related features.
// Real pitch detection would use autocorrelation or CREPE.
func computePitchFeatures(samples []float32) []float32 {
	// Simplified autocorrelation at typical pitch periods (50-400 Hz)
	// At 16kHz: period 40-320 samples
	features := make([]float32, 8)

	// Sample autocorrelation at a few lags
	lags := []int{40, 60, 80, 100, 120, 160, 200, 320}
	for i, lag := range lags {
		if lag >= len(samples) {
			continue
		}
		var corr float32
		for j := 0; j < len(samples)-lag; j++ {
			corr += samples[j] * samples[j+lag]
		}
		features[i] = corr / float32(len(samples)-lag)
	}

	return features
}

// computeTemporalFeatures extracts temporal dynamics.
func computeTemporalFeatures(energy, zcr []float32) []float32 {
	features := make([]float32, 8)

	// Energy modulation rate
	if len(energy) > 1 {
		var modulation float32
		for i := 1; i < len(energy); i++ {
			diff := energy[i] - energy[i-1]
			modulation += diff * diff
		}
		features[0] = float32(math.Sqrt(float64(modulation / float32(len(energy)-1))))
	}

	// ZCR modulation rate
	if len(zcr) > 1 {
		var modulation float32
		for i := 1; i < len(zcr); i++ {
			diff := zcr[i] - zcr[i-1]
			modulation += diff * diff
		}
		features[1] = float32(math.Sqrt(float64(modulation / float32(len(zcr)-1))))
	}

	// Cross-correlation between energy and ZCR
	if len(energy) == len(zcr) && len(energy) > 0 {
		var sumE, sumZ, sumEZ, sumE2, sumZ2 float32
		for i := range energy {
			sumE += energy[i]
			sumZ += zcr[i]
			sumEZ += energy[i] * zcr[i]
			sumE2 += energy[i] * energy[i]
			sumZ2 += zcr[i] * zcr[i]
		}
		n := float32(len(energy))
		num := n*sumEZ - sumE*sumZ
		den := float32(math.Sqrt(float64((n*sumE2-sumE*sumE)*(n*sumZ2-sumZ*sumZ))))
		if den > 0 {
			features[2] = num / den
		}
	}

	// Segment the audio into thirds and compare energy
	third := len(energy) / 3
	if third > 0 {
		var e1, e2, e3 float32
		for i := 0; i < third; i++ {
			e1 += energy[i]
		}
		for i := third; i < 2*third; i++ {
			e2 += energy[i]
		}
		for i := 2 * third; i < len(energy); i++ {
			e3 += energy[i]
		}
		e1 /= float32(third)
		e2 /= float32(third)
		e3 /= float32(len(energy) - 2*third)

		features[3] = e2 - e1 // Attack
		features[4] = e3 - e2 // Decay
		features[5] = e1 + e2 + e3 // Total energy
	}

	return features
}

// computeHistogram computes a normalized histogram of values.
func computeHistogram(values []float32, bins int) []float32 {
	hist := make([]float32, bins)
	if len(values) == 0 {
		return hist
	}

	// Find range
	minVal, maxVal := values[0], values[0]
	for _, v := range values {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}

	// Handle edge case
	rangeVal := maxVal - minVal
	if rangeVal == 0 {
		hist[0] = 1.0
		return hist
	}

	// Bin values
	for _, v := range values {
		bin := int(float64(v-minVal) / float64(rangeVal) * float64(bins-1))
		if bin >= bins {
			bin = bins - 1
		}
		hist[bin]++
	}

	// Normalize
	for i := range hist {
		hist[i] /= float32(len(values))
	}

	return hist
}

// computeCrossCorrelation computes cross-correlations between feature streams.
func computeCrossCorrelation(energy, zcr, spectral []float32) []float32 {
	features := make([]float32, 16)
	n := len(energy)
	if n == 0 || len(zcr) != n || len(spectral) != n {
		return features
	}

	// Lag-0 correlations
	features[0] = lagCorrelation(energy, zcr, 0)
	features[1] = lagCorrelation(energy, spectral, 0)
	features[2] = lagCorrelation(zcr, spectral, 0)

	// Lag-1 correlations
	features[3] = lagCorrelation(energy, zcr, 1)
	features[4] = lagCorrelation(energy, spectral, 1)
	features[5] = lagCorrelation(zcr, spectral, 1)

	// Lag-2 correlations
	features[6] = lagCorrelation(energy, zcr, 2)
	features[7] = lagCorrelation(energy, spectral, 2)
	features[8] = lagCorrelation(zcr, spectral, 2)

	// Feature ratios
	statsE := computeStats(energy)
	statsZ := computeStats(zcr)
	statsS := computeStats(spectral)

	if statsZ.mean > 0 {
		features[9] = statsE.mean / statsZ.mean
	}
	if statsS.mean > 0 {
		features[10] = statsE.mean / statsS.mean
		features[11] = statsZ.mean / statsS.mean
	}

	// Standard deviation ratios
	if statsZ.std > 0 {
		features[12] = statsE.std / statsZ.std
	}
	if statsS.std > 0 {
		features[13] = statsE.std / statsS.std
		features[14] = statsZ.std / statsS.std
	}

	return features
}

// lagCorrelation computes correlation between two signals at a given lag.
func lagCorrelation(a, b []float32, lag int) float32 {
	n := len(a) - lag
	if n <= 0 || len(b) < n {
		return 0
	}

	var sumA, sumB, sumAB, sumA2, sumB2 float32
	for i := 0; i < n; i++ {
		aVal := a[i+lag]
		bVal := b[i]
		sumA += aVal
		sumB += bVal
		sumAB += aVal * bVal
		sumA2 += aVal * aVal
		sumB2 += bVal * bVal
	}

	nf := float32(n)
	num := nf*sumAB - sumA*sumB
	den := float32(math.Sqrt(float64((nf*sumA2 - sumA*sumA) * (nf*sumB2 - sumB*sumB))))
	if den == 0 {
		return 0
	}
	return num / den
}

// normalizeL2 normalizes a vector to unit length.
func normalizeL2(v []float32) {
	var sumSq float32
	for _, x := range v {
		sumSq += x * x
	}
	if sumSq == 0 {
		return
	}
	norm := float32(math.Sqrt(float64(sumSq)))
	for i := range v {
		v[i] /= norm
	}
}

// sortFloat32 sorts a float32 slice in place (simple insertion sort).
func sortFloat32(a []float32) {
	for i := 1; i < len(a); i++ {
		j := i
		for j > 0 && a[j-1] > a[j] {
			a[j-1], a[j] = a[j], a[j-1]
			j--
		}
	}
}

// copyStats copies statistics to embedding starting at idx, returns next idx.
func copyStats(embedding []float32, idx int, stats statisticsResult) int {
	if idx+8 > len(embedding) {
		return len(embedding)
	}
	embedding[idx] = stats.mean
	embedding[idx+1] = stats.std
	embedding[idx+2] = stats.min
	embedding[idx+3] = stats.max
	embedding[idx+4] = stats.q25
	embedding[idx+5] = stats.median
	embedding[idx+6] = stats.q75
	embedding[idx+7] = stats.dynamicRange
	return idx + 8
}

// copySlice copies src to embedding starting at idx, returns next idx.
func copySlice(embedding []float32, idx int, src []float32) int {
	for i, v := range src {
		if idx+i >= len(embedding) {
			return len(embedding)
		}
		embedding[idx+i] = v
	}
	return idx + len(src)
}
