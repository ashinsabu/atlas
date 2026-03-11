package speaker

import (
	"fmt"
	"math"
	"time"

	"github.com/ashinsabu/atlas/internal/vox/audio"
)

// Verifier gates pipeline commands on speaker identity.
// If no profile is loaded, Verify always returns ("Unknown", 0, false, nil) — safe default.
type Verifier struct {
	profile   *Profile
	encoder   *Encoder
	threshold float32

	// Duration-aware threshold: short utterances produce noisier embeddings due to
	// fewer fbank frames for per-utterance mean normalization. shortThreshold is used
	// when audio duration < shortSecs. Zero means disabled (always use threshold).
	shortThreshold float32
	shortSecs      float32
}

// NewVerifier creates a Verifier. profile may be nil (no verification, all rejected).
func NewVerifier(encoder *Encoder, profile *Profile, threshold float32) *Verifier {
	return &Verifier{
		profile:   profile,
		encoder:   encoder,
		threshold: threshold,
	}
}

// SetShortThreshold configures a lower acceptance threshold for short utterances.
// Segments shorter than secs seconds use shortThreshold instead of the normal threshold.
func (v *Verifier) SetShortThreshold(shortThreshold float32, secs float64) {
	v.shortThreshold = shortThreshold
	v.shortSecs = float32(secs)
}

// Verify checks whether pcmBytes (raw PCM16, 16kHz mono, little-endian) belongs
// to the enrolled speaker.
//
// Returns:
//   - name: profile.SpeakerName if accepted, "Unknown" if rejected or no profile
//   - score: cosine similarity to profile embedding (0 if no profile)
//   - accepted: true if score >= threshold
//   - err: non-nil only on encoding failure
func (v *Verifier) Verify(pcmBytes []byte) (name string, score float32, accepted bool, err error) {
	if v.profile == nil {
		return "Unknown", 0, false, nil
	}

	// Convert PCM16 (int16 little-endian) to float32 [-1, 1].
	n := len(pcmBytes) / 2
	samples := make([]float32, n)
	for i := 0; i < n; i++ {
		s := int16(pcmBytes[i*2]) | int16(pcmBytes[i*2+1])<<8
		samples[i] = float32(s) / 32768.0
	}

	embedding, err := v.encoder.Encode(samples)
	if err != nil {
		return "Unknown", 0, false, fmt.Errorf("encode: %w", err)
	}

	score = CosineSimilarity(embedding, v.profile.Embedding)

	// Pick threshold based on audio duration. Short utterances have noisier embeddings
	// (fewer fbank frames → less stable per-utterance mean normalization), so they need
	// a lower bar. Other-speaker scores stay low regardless of duration.
	threshold := v.threshold
	if v.shortSecs > 0 {
		durationSecs := float32(len(pcmBytes)) / (2 * 16000) // PCM16 @ 16kHz
		if durationSecs < v.shortSecs {
			threshold = v.shortThreshold
		}
	}
	accepted = score >= threshold

	name = "Unknown"
	if accepted {
		name = v.profile.SpeakerName
	}
	return name, score, accepted, nil
}

// CosineSimilarity computes cosine similarity between two L2-normalized embeddings.
// Both must be the same length. Since both are unit-norm, this equals their dot product.
func CosineSimilarity(a, b []float32) float32 {
	var dot float32
	for i := range a {
		dot += a[i] * b[i]
	}
	return dot
}

// L2Normalize normalizes v in-place to unit length.
func L2Normalize(v []float32) {
	var norm float64
	for _, x := range v {
		norm += float64(x) * float64(x)
	}
	norm = math.Sqrt(norm)
	if norm > 0 {
		for i := range v {
			v[i] /= float32(norm)
		}
	}
}

// EnrollFromSamples encodes each float32 PCM recording, averages the embeddings,
// and returns an L2-normalized speaker profile.
// recordings[i] must be float32 normalized to [-1,1] at 16kHz mono.
func EnrollFromSamples(enc *Encoder, recordings [][]float32, speakerName string) (*Profile, error) {
	if len(recordings) == 0 {
		return nil, fmt.Errorf("no recordings provided for enrollment")
	}

	embeddings := make([][]float32, 0, len(recordings))
	for i, samples := range recordings {
		embedding, err := enc.Encode(samples)
		if err != nil {
			return nil, fmt.Errorf("encode recording %d: %w", i+1, err)
		}
		embeddings = append(embeddings, embedding)
	}

	return averageEmbeddings(embeddings, speakerName, len(recordings)), nil
}

// EnrollFromFiles loads WAV files, encodes them, and returns an L2-normalized profile.
// WAV files must be 16kHz mono 16-bit PCM.
func EnrollFromFiles(enc *Encoder, paths []string, speakerName string) (*Profile, error) {
	if len(paths) == 0 {
		return nil, fmt.Errorf("no WAV files provided for enrollment")
	}

	recordings := make([][]float32, 0, len(paths))
	for _, path := range paths {
		samples, err := audio.LoadWAV(path)
		if err != nil {
			return nil, fmt.Errorf("load %s: %w", path, err)
		}
		recordings = append(recordings, samples)
	}

	return EnrollFromSamples(enc, recordings, speakerName)
}

// averageEmbeddings averages a list of embeddings and L2-normalizes the result.
// Averaging unit-norm vectors reduces noise across takes; re-normalizing keeps it on the hypersphere.
func averageEmbeddings(embeddings [][]float32, speakerName string, count int) *Profile {
	embDim := len(embeddings[0])
	avg := make([]float32, embDim)
	for _, emb := range embeddings {
		for i, v := range emb {
			avg[i] += v
		}
	}
	for i := range avg {
		avg[i] /= float32(len(embeddings))
	}
	L2Normalize(avg)

	return &Profile{
		Version:         1,
		SpeakerName:     speakerName,
		Embedding:       avg,
		EnrollmentCount: count,
		CreatedAt:       time.Now(),
	}
}
