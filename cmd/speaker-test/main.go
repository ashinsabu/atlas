// speaker-test evaluates speaker verification accuracy against labeled test data.
//
// Usage:
//
//	make speaker-test
//	# or
//	./bin/speaker-test -profile ~/.atlas/speaker.json -model models/wespeaker-resnet34.onnx
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/ashinsabu/atlas/internal/vox/config"
	"github.com/ashinsabu/atlas/internal/vox/speaker"
)

func main() {
	profilePath := flag.String("profile", "", "Speaker profile path (default: $ATLAS_CONFIG_DIR/speaker.json or ./speaker.json)")
	modelPath := flag.String("model", "models/wespeaker-resnet34.onnx", "WeSpeaker ONNX model path")
	userDir := flag.String("user", "test/stt/recordings", "Directory of user (positive) WAV files")
	otherDir := flag.String("other", "test/speaker/other", "Directory of other-speaker (negative) WAV files")
	threshold := flag.Float64("threshold", 0.70, "Cosine similarity threshold for acceptance")
	flag.Parse()

	// Resolve profile path.
	if *profilePath == "" {
		*profilePath = filepath.Join(config.ConfigDir(), "speaker.json")
	}

	// Load profile.
	profile, err := speaker.LoadProfile(*profilePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: load profile: %v\n", err)
		os.Exit(1)
	}
	if profile == nil {
		fmt.Fprintf(os.Stderr, "Error: no profile at %s\n", *profilePath)
		fmt.Fprintln(os.Stderr, "Run: make vox-enroll")
		os.Exit(1)
	}

	// Load encoder.
	enc, err := speaker.NewEncoder(*modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: load encoder: %v\n", err)
		os.Exit(1)
	}
	defer enc.Close()

	thr := float32(*threshold)
	v := speaker.NewVerifier(enc, profile, thr)

	fmt.Printf("Speaker: %s  |  Threshold: %.2f  |  Profile: %s\n\n", profile.SpeakerName, thr, *profilePath)

	// --- Positive test: user recordings ---
	userPaths := glob(*userDir, "*.wav")
	userAccepted, userTotal := 0, 0
	var minUserScore, maxUserScore float32 = 1.0, -1.0
	if len(userPaths) > 0 {
		fmt.Printf("User recordings (%s):\n", *userDir)
		for _, path := range userPaths {
			pcm, err := loadWAVBytes(path)
			if err != nil {
				fmt.Printf("  [skip] %s: %v\n", filepath.Base(path), err)
				continue
			}
			userTotal++
			_, score, ok, verifyErr := v.Verify(pcm)
			if verifyErr != nil {
				fmt.Printf("  [err]  %s: %v\n", filepath.Base(path), verifyErr)
				continue
			}
			status := "✓"
			if !ok {
				status = "✗"
			} else {
				userAccepted++
			}
			if score < minUserScore {
				minUserScore = score
			}
			if score > maxUserScore {
				maxUserScore = score
			}
			fmt.Printf("  %s  %-40s  score=%.3f\n", status, filepath.Base(path), score)
		}
		fmt.Println()
	}

	// --- Negative test: other speakers ---
	otherPaths := globDeep(*otherDir, ".wav")
	otherRejected, otherTotal := 0, 0
	var maxOtherScore float32 = -1.0
	if len(otherPaths) > 0 {
		fmt.Printf("Other speakers (%s):\n", *otherDir)
		for _, path := range otherPaths {
			pcm, err := loadWAVBytes(path)
			if err != nil {
				fmt.Printf("  [skip] %s: %v\n", filepath.Base(path), err)
				continue
			}
			otherTotal++
			_, score, ok, verifyErr := v.Verify(pcm)
			if verifyErr != nil {
				fmt.Printf("  [err]  %s: %v\n", filepath.Base(path), verifyErr)
				continue
			}
			status := "✓ rejected"
			if ok {
				status = "✗ ACCEPTED (false positive)"
			} else {
				otherRejected++
			}
			if score > maxOtherScore {
				maxOtherScore = score
			}
			fmt.Printf("  %s  %-40s  score=%.3f\n", status, filepath.Base(path), score)
		}
		fmt.Println()
	}

	// --- Summary ---
	fmt.Printf("═══ Summary ═══════════════════════════════════════\n")
	if userTotal > 0 {
		tpr := float64(userAccepted) / float64(userTotal) * 100
		fmt.Printf("User recordings:    %d/%d accepted  (TPR: %.1f%%)\n", userAccepted, userTotal, tpr)
		fmt.Printf("Min user score:     %.3f\n", minUserScore)
		fmt.Printf("Max user score:     %.3f\n", maxUserScore)
	} else {
		fmt.Printf("User recordings:    (none found in %s)\n", *userDir)
	}
	if otherTotal > 0 {
		tnr := float64(otherRejected) / float64(otherTotal) * 100
		fmt.Printf("Other speakers:     %d/%d rejected  (TNR: %.1f%%)\n", otherRejected, otherTotal, tnr)
		fmt.Printf("Max other score:    %.3f\n", maxOtherScore)
	} else {
		fmt.Printf("Other speakers:     (none found in %s)\n", *otherDir)
		fmt.Printf("                    Run 'make setup-speaker-test' to download LibriSpeech samples\n")
	}
	fmt.Printf("Threshold used:     %.2f\n", thr)
}

func glob(dir, pattern string) []string {
	matches, _ := filepath.Glob(filepath.Join(dir, pattern))
	return matches
}

func globDeep(dir, ext string) []string {
	var paths []string
	_ = filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return nil
		}
		if filepath.Ext(path) == ext {
			paths = append(paths, path)
		}
		return nil
	})
	return paths
}

func loadWAVBytes(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	header := make([]byte, 44)
	if _, err := f.Read(header); err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}
	if string(header[0:4]) != "RIFF" || string(header[8:12]) != "WAVE" {
		return nil, fmt.Errorf("not a WAV file")
	}

	dataSize := int(uint32(header[40]) | uint32(header[41])<<8 | uint32(header[42])<<16 | uint32(header[43])<<24)
	audioBytes := make([]byte, dataSize)
	if _, err := f.Read(audioBytes); err != nil {
		return nil, fmt.Errorf("read audio: %w", err)
	}
	return audioBytes, nil
}
