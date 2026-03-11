package speaker

import (
	"fmt"
	"os"

	ort "github.com/yalue/onnxruntime_go"

	ortinit "github.com/ashinsabu/atlas/internal/vox/ort"
)

// InspectModel prints ONNX model input/output tensor names and shapes.
// Used by 'make vox-enroll -inspect' to validate model compatibility before
// hardcoding tensor names in encoder.go.
//
// If the configured tensor names are wrong, ORT will report the actual names
// in the error message returned by NewDynamicAdvancedSession.
func InspectModel(modelPath string) error {
	if _, err := os.Stat(modelPath); err != nil {
		return fmt.Errorf("model not found: %s", modelPath)
	}

	if err := ortinit.Initialize(); err != nil {
		return fmt.Errorf("init onnx: %w", err)
	}

	opts, err := ort.NewSessionOptions()
	if err != nil {
		return fmt.Errorf("session options: %w", err)
	}
	defer opts.Destroy()

	// Attempt to load with configured names. If wrong, ORT error reveals actual names.
	fmt.Printf("Model: %s\n\n", modelPath)
	fmt.Printf("Configured tensor names (encoder.go):\n")
	fmt.Printf("  Input:  \"feats\"  shape [1, T, 80]  (mel-filterbank features)\n")
	fmt.Printf("  Output: \"embs\"   shape [1, 256]    (L2-normalized speaker embedding)\n\n")

	session, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{"feats"},
		[]string{"embs"},
		opts,
	)
	if err != nil {
		fmt.Printf("✗ Session creation failed — tensor names may be wrong:\n  %v\n\n", err)
		fmt.Printf("To inspect actual names: use netron.app (open the .onnx file in browser)\n")
		fmt.Printf("Then update tensor names in internal/vox/speaker/encoder.go\n")
		return nil // Not a fatal error — user can inspect and fix
	}
	session.Destroy()

	fmt.Printf("✓ Session created successfully with configured tensor names.\n")
	fmt.Printf("  Model appears compatible with encoder.go.\n")
	return nil
}
