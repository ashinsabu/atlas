// Package ort provides a shared ONNX Runtime initialization point.
//
// Both the speech (Silero VAD) and speaker (WeSpeaker) packages use ONNX Runtime.
// ort.InitializeEnvironment must be called exactly once per process — calling it
// a second time returns an error. This package centralizes that call behind a
// sync.Once so any number of callers from any package can safely invoke Initialize().
package ort

import (
	"os"
	"runtime"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

var (
	once    sync.Once
	initErr error
)

func init() {
	// Set the shared library path once, before any Initialize() call.
	var libPath string
	switch runtime.GOOS {
	case "darwin":
		for _, p := range []string{
			"/opt/homebrew/lib/libonnxruntime.dylib", // ARM64 Homebrew
			"/usr/local/lib/libonnxruntime.dylib",    // Intel Homebrew
		} {
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

// Initialize calls ort.InitializeEnvironment exactly once across all packages.
// Subsequent calls return the same error (or nil) as the first call.
func Initialize() error {
	once.Do(func() { initErr = ort.InitializeEnvironment() })
	return initErr
}
