package speaker

import (
	"fmt"
	"os"

	ort "github.com/yalue/onnxruntime_go"

	ortinit "github.com/ashinsabu/atlas/internal/vox/ort"
)

// EmbeddingDim is the output embedding size for WeSpeaker ResNet34-LM.
// The model outputs a 256-dimensional L2-normalized speaker embedding.
const EmbeddingDim = 256

// Encoder wraps the WeSpeaker ResNet34-LM ONNX model.
// Input: mel-filterbank features [1, T, 80].
// Output: L2-normalized speaker embedding [256] (normalized in Encode, not by the model).
type Encoder struct {
	session  *ort.DynamicAdvancedSession
	fbankCfg FbankConfig
}

// NewEncoder loads the WeSpeaker ONNX speaker encoder.
// Expected tensor names: input "feats" [1,T,80], output "embs" [1,256].
// Returns a descriptive error if the model cannot be loaded.
func NewEncoder(modelPath string) (*Encoder, error) {
	if _, err := os.Stat(modelPath); err != nil {
		return nil, fmt.Errorf("speaker model not found: %s (run 'make setup-wespeaker')", modelPath)
	}

	if err := ortinit.Initialize(); err != nil {
		return nil, fmt.Errorf("init onnx runtime: %w", err)
	}

	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("create session options: %w", err)
	}
	defer opts.Destroy()

	// Tensor names for WeSpeaker ResNet34-LM.
	// Run 'make vox-enroll -inspect' to verify if these change with a different model.
	session, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{"feats"},
		[]string{"embs"},
		opts,
	)
	if err != nil {
		return nil, fmt.Errorf(
			"load speaker model: %w\n"+
				"  Hint: expected input='feats' output='embs'. Run 'make vox-enroll -inspect' to verify tensor names.",
			err,
		)
	}

	return &Encoder{
		session:  session,
		fbankCfg: DefaultFbankConfig(),
	}, nil
}

// Encode extracts a speaker embedding from float32 PCM samples.
// Samples must be normalized to [-1, 1] at 16kHz mono.
// Returns an L2-normalized embedding of length EmbeddingDim (256).
func (e *Encoder) Encode(samples []float32) ([]float32, error) {
	// Extract mel-filterbank features: [T, 80] stored as flat [T*80].
	features, T, err := ExtractFbank(samples, e.fbankCfg)
	if err != nil {
		return nil, fmt.Errorf("fbank extraction: %w", err)
	}

	// Build input tensor: shape [1, T, 80].
	inputTensor, err := ort.NewTensor(ort.NewShape(1, int64(T), 80), features)
	if err != nil {
		return nil, fmt.Errorf("create feats tensor: %w", err)
	}
	defer inputTensor.Destroy()

	// Run inference. Let ORT allocate the output tensor.
	outputs := []ort.Value{nil}
	if err := e.session.Run([]ort.Value{inputTensor}, outputs); err != nil {
		return nil, fmt.Errorf("speaker encoder inference: %w", err)
	}
	defer outputs[0].Destroy()

	outTensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("unexpected output tensor type: %T", outputs[0])
	}

	// Copy embedding before the tensor is destroyed.
	// GetData() returns a slice backed by the tensor's memory — copy is mandatory.
	raw := outTensor.GetData()
	embedding := make([]float32, len(raw))
	copy(embedding, raw)

	// L2-normalize so that cosine similarity = dot product.
	// The model does not normalize internally (verified: raw scores exceed 1.0).
	L2Normalize(embedding)

	return embedding, nil
}

// Close releases ONNX session resources.
func (e *Encoder) Close() error {
	if e.session != nil {
		if err := e.session.Destroy(); err != nil {
			return fmt.Errorf("destroy speaker session: %w", err)
		}
		e.session = nil
	}
	return nil
}

