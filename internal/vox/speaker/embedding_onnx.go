// Package speaker - ONNX-based embedding extraction.
//
// Uses WeSpeaker CAM++ model for speaker embeddings.
// Model: wespeaker_en_voxceleb_CAM++.onnx (~29MB, 512-dim output)
//
// Requirements:
// - ONNX Runtime shared library (brew install onnxruntime on macOS)
// - Speaker model downloaded via `make setup-speaker`

package speaker

import (
	"fmt"
	"os"
	"runtime"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

var (
	onnxInitOnce sync.Once
	onnxInitErr  error
)

func init() {
	// Set ONNX Runtime shared library path based on platform
	var libPath string
	switch runtime.GOOS {
	case "darwin":
		// macOS: check Homebrew paths
		paths := []string{
			"/opt/homebrew/lib/libonnxruntime.dylib", // ARM64 Homebrew
			"/usr/local/lib/libonnxruntime.dylib",    // Intel Homebrew
		}
		for _, p := range paths {
			if _, err := os.Stat(p); err == nil {
				libPath = p
				break
			}
		}
	case "linux":
		libPath = "/usr/lib/libonnxruntime.so"
	}

	if libPath != "" {
		ort.SetSharedLibraryPath(libPath)
	}
}

// initONNX initializes the ONNX runtime exactly once.
func initONNX() error {
	onnxInitOnce.Do(func() {
		onnxInitErr = ort.InitializeEnvironment()
	})
	return onnxInitErr
}

// ONNXExtractor uses an ONNX model for speaker embedding extraction.
type ONNXExtractor struct {
	session   *ort.DynamicAdvancedSession
	modelPath string
	dim       int

	mu sync.Mutex
}

// NewONNXExtractor creates an ONNX-based embedding extractor.
// modelPath: path to the .onnx model file
// Returns error if model cannot be loaded.
func NewONNXExtractor(modelPath string) (*ONNXExtractor, error) {
	// Check model exists
	if _, err := os.Stat(modelPath); err != nil {
		return nil, fmt.Errorf("model not found: %s (run 'make setup-speaker')", modelPath)
	}

	// Initialize ONNX Runtime (thread-safe, idempotent via sync.Once)
	if err := initONNX(); err != nil {
		return nil, fmt.Errorf("init onnx runtime: %w", err)
	}

	// Create session options
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("create session options: %w", err)
	}
	defer opts.Destroy()

	// Create dynamic session (handles variable-length input)
	session, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{"feats"},  // Input names for WeSpeaker
		[]string{"embs"},   // Output names for WeSpeaker
		opts,
	)
	if err != nil {
		return nil, fmt.Errorf("create onnx session: %w", err)
	}

	return &ONNXExtractor{
		session:   session,
		modelPath: modelPath,
		dim:       512, // WeSpeaker CAM++ output dimension
	}, nil
}

// Extract computes a speaker embedding from audio samples.
// Input: float32 audio samples, 16kHz mono, normalized to [-1.0, 1.0]
// Output: 512-dimensional L2-normalized embedding vector.
func (e *ONNXExtractor) Extract(samples []float32) ([]float32, error) {
	if len(samples) < MinAudioSamples {
		return nil, ErrAudioTooShort
	}

	// Compute FBANK features from raw audio
	fbankCfg := DefaultFBankConfig()
	features := ComputeFBank(samples, fbankCfg)
	if len(features) == 0 {
		return nil, fmt.Errorf("failed to compute FBANK features")
	}

	// Flatten features for ONNX input: [1, num_frames, 80]
	numFrames := len(features)
	numBins := fbankCfg.NumMelBins
	flatFeatures := make([]float32, numFrames*numBins)
	for i := 0; i < numFrames; i++ {
		for j := 0; j < numBins && j < len(features[i]); j++ {
			flatFeatures[i*numBins+j] = features[i][j]
		}
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	// WeSpeaker expects input shape [batch, num_frames, num_mel_bins]
	inputShape := ort.NewShape(1, int64(numFrames), int64(numBins))
	inputTensor, err := ort.NewTensor(inputShape, flatFeatures)
	if err != nil {
		return nil, fmt.Errorf("create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	// Create output tensor for embeddings [1, 512]
	outputShape := ort.NewShape(1, int64(e.dim))
	outputData := make([]float32, e.dim)
	outputTensor, err := ort.NewTensor(outputShape, outputData)
	if err != nil {
		return nil, fmt.Errorf("create output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	// Run inference
	err = e.session.Run(
		[]ort.ArbitraryTensor{inputTensor},
		[]ort.ArbitraryTensor{outputTensor},
	)
	if err != nil {
		return nil, fmt.Errorf("onnx inference: %w", err)
	}

	// Copy embedding from output tensor
	embedding := make([]float32, e.dim)
	copy(embedding, outputTensor.GetData())

	// L2 normalize (should already be normalized, but ensure)
	normalizeL2(embedding)

	return embedding, nil
}

// EmbeddingDim returns the output embedding dimension.
func (e *ONNXExtractor) EmbeddingDim() int {
	return e.dim
}

// Close releases ONNX resources.
func (e *ONNXExtractor) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.session != nil {
		if err := e.session.Destroy(); err != nil {
			return fmt.Errorf("destroy session: %w", err)
		}
		e.session = nil
	}
	return nil
}

// ONNXVerifier is a speaker verifier using ONNX-based embeddings.
type ONNXVerifier struct {
	config    Config
	extractor *ONNXExtractor
	profile   *Profile

	mu sync.RWMutex
}

// NewONNXVerifier creates a new speaker verifier with ONNX embeddings.
func NewONNXVerifier(cfg Config, modelPath string) (*ONNXVerifier, error) {
	extractor, err := NewONNXExtractor(modelPath)
	if err != nil {
		return nil, err
	}

	// Override config embedding dim to match model
	cfg.EmbeddingDim = extractor.EmbeddingDim()

	return &ONNXVerifier{
		config:    cfg,
		extractor: extractor,
	}, nil
}

// Enroll creates a speaker profile from multiple audio samples.
func (v *ONNXVerifier) Enroll(samples [][]float32) error {
	if len(samples) == 0 {
		return ErrNoSamples
	}
	if len(samples) < v.config.MinEnrollSamples {
		return fmt.Errorf("need at least %d samples, got %d", v.config.MinEnrollSamples, len(samples))
	}

	embeddings := make([][]float32, len(samples))
	for i, sample := range samples {
		emb, err := v.extractor.Extract(sample)
		if err != nil {
			return fmt.Errorf("extract embedding for sample %d: %w", i, err)
		}
		embeddings[i] = emb
	}

	profile, err := EnrollFromEmbeddings(embeddings)
	if err != nil {
		return fmt.Errorf("create profile: %w", err)
	}

	v.mu.Lock()
	v.profile = profile
	v.mu.Unlock()

	return nil
}

// Verify checks if audio matches the enrolled speaker.
func (v *ONNXVerifier) Verify(sample []float32) (bool, float32, error) {
	v.mu.RLock()
	profile := v.profile
	v.mu.RUnlock()

	if profile == nil {
		return false, 0, ErrNotEnrolled
	}

	emb, err := v.extractor.Extract(sample)
	if err != nil {
		return false, 0, fmt.Errorf("extract embedding: %w", err)
	}

	similarity := cosineSimilarity(emb, profile.Embedding)
	isOwner := similarity >= v.config.Threshold
	return isOwner, similarity, nil
}

// LoadProfile loads a speaker profile from disk.
func (v *ONNXVerifier) LoadProfile(path string) error {
	profile, err := LoadProfileFromFile(path)
	if err != nil {
		return err
	}

	if len(profile.Embedding) != v.config.EmbeddingDim {
		return fmt.Errorf("profile embedding dim %d != model dim %d", len(profile.Embedding), v.config.EmbeddingDim)
	}

	v.mu.Lock()
	v.profile = profile
	v.mu.Unlock()

	return nil
}

// SaveProfile saves the current speaker profile to disk.
func (v *ONNXVerifier) SaveProfile(path string) error {
	v.mu.RLock()
	profile := v.profile
	v.mu.RUnlock()

	if profile == nil {
		return ErrNotEnrolled
	}

	return profile.SaveToFile(path)
}

// IsEnrolled returns true if a speaker profile is loaded.
func (v *ONNXVerifier) IsEnrolled() bool {
	v.mu.RLock()
	defer v.mu.RUnlock()
	return v.profile != nil
}

// EmbeddingDim returns the dimension of speaker embeddings.
func (v *ONNXVerifier) EmbeddingDim() int {
	return v.config.EmbeddingDim
}

// GetThreshold returns the current verification threshold.
func (v *ONNXVerifier) GetThreshold() float32 {
	return v.config.Threshold
}

// SetThreshold updates the verification threshold.
func (v *ONNXVerifier) SetThreshold(t float32) {
	v.config.Threshold = t
}

// Close releases ONNX resources.
func (v *ONNXVerifier) Close() error {
	return v.extractor.Close()
}
