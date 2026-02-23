// speaker-test: Test speaker verification against enrolled profile
//
// Usage:
//   speaker-test                         # Record from microphone
//   speaker-test -dir test/stt/recordings # Test directory
//   speaker-test -threshold 0.6          # Custom threshold
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/ashinsabu/atlas/internal/vox/audio"
	"github.com/ashinsabu/atlas/internal/vox/speaker"
)

func main() {
	wavFile := flag.String("file", "", "Single WAV file to test")
	wavDir := flag.String("dir", "", "Directory of WAV files to test")
	profilePath := flag.String("profile", "", "Profile path (default: ~/.atlas/speaker_profile.bin)")
	threshold := flag.Float64("threshold", 0.60, "Verification threshold")
	modelPath := flag.String("model", "models/wespeaker_en_voxceleb_CAM++.onnx", "ONNX model path")
	flag.Parse()

	// Load profile
	profPath := *profilePath
	if profPath == "" {
		var err error
		profPath, err = speaker.ProfilePath()
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			os.Exit(1)
		}
	}

	if _, err := os.Stat(profPath); os.IsNotExist(err) {
		fmt.Printf("Profile not found: %s\nRun 'make speaker-enroll' first.\n", profPath)
		os.Exit(1)
	}

	cfg := speaker.DefaultConfig()
	cfg.Threshold = float32(*threshold)
	cfg.ModelPath = *modelPath

	// Create verifier
	verifier, cleanup, err := speaker.NewAutoVerifier(cfg)
	if err != nil {
		fmt.Printf("Failed to create verifier: %v\n", err)
		os.Exit(1)
	}
	defer cleanup()

	if err := verifier.LoadProfile(profPath); err != nil {
		fmt.Printf("Failed to load profile: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Profile: %s\nThreshold: %.2f\n\n", profPath, *threshold)

	// Test mode
	if *wavFile != "" {
		testFile(verifier, *wavFile, float32(*threshold))
	} else if *wavDir != "" {
		testDirectory(verifier, *wavDir, float32(*threshold))
	} else {
		testMicrophone(verifier, float32(*threshold))
	}
}

func testFile(verifier speaker.Verifier, path string, threshold float32) {
	data, err := audio.LoadWAV(path)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}

	isOwner, confidence, err := verifier.Verify(data)
	if err != nil {
		fmt.Printf("Verification error: %v\n", err)
		os.Exit(1)
	}

	printResult(filepath.Base(path), isOwner, confidence, threshold)
}

func testDirectory(verifier speaker.Verifier, dir string, threshold float32) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}

	var tested, passed int
	var totalConf float32

	fmt.Println("Testing all WAV files...")
	fmt.Println(strings.Repeat("-", 60))

	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(strings.ToLower(entry.Name()), ".wav") {
			continue
		}

		path := filepath.Join(dir, entry.Name())
		data, err := audio.LoadWAV(path)
		if err != nil {
			fmt.Printf("%-30s  ERROR: %v\n", entry.Name(), err)
			continue
		}

		isOwner, confidence, err := verifier.Verify(data)
		if err != nil {
			fmt.Printf("%-30s  ERROR: %v\n", entry.Name(), err)
			continue
		}

		tested++
		totalConf += confidence
		if isOwner {
			passed++
		}

		status := "REJECT"
		if isOwner {
			status = "ACCEPT"
		}
		fmt.Printf("%-30s  %s  (%.3f)\n", entry.Name(), status, confidence)
	}

	fmt.Println(strings.Repeat("-", 60))
	fmt.Printf("Results: %d/%d accepted (%.1f%%)\n", passed, tested, float64(passed)/float64(tested)*100)
	if tested > 0 {
		fmt.Printf("Average confidence: %.3f\n", totalConf/float32(tested))
	}
}

func testMicrophone(verifier speaker.Verifier, threshold float32) {
	recorder, err := audio.NewRecorder()
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}
	defer recorder.Close()

	fmt.Println("=== Speaker Verification Test ===")
	fmt.Println("Press Ctrl+C to exit.")
	fmt.Println()

	for i := 1; ; i++ {
		fmt.Printf("Test %d: Press ENTER to record...\n", i)

		data, err := recorder.RecordUntilEnter()
		if err != nil {
			fmt.Printf("Error: %v\n\n", err)
			continue
		}

		isOwner, confidence, err := verifier.Verify(data)
		if err != nil {
			fmt.Printf("Error: %v\n\n", err)
			continue
		}

		printResult(fmt.Sprintf("Test %d", i), isOwner, confidence, threshold)
		fmt.Println()
	}
}

func printResult(name string, isOwner bool, confidence, threshold float32) {
	status := "REJECTED"
	if isOwner {
		status = "ACCEPTED"
	}

	bar := confidenceBar(confidence)
	fmt.Printf("%s: %s\n", name, status)
	fmt.Printf("  Confidence: %.3f %s\n", confidence, bar)
	fmt.Printf("  Threshold:  %.3f\n", threshold)
}

func confidenceBar(confidence float32) string {
	width := 20
	filled := int(confidence * float32(width))
	if filled > width {
		filled = width
	}
	if filled < 0 {
		filled = 0
	}
	return "[" + strings.Repeat("#", filled) + strings.Repeat("-", width-filled) + "]"
}
