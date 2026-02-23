// stt-test: Test STT accuracy against recordings
//
// Usage:
//   stt-test                    # Test all recordings
//   stt-test -script 1          # Test only script 1 recordings
//   stt-test -verbose           # Show detailed output
//
// Compares transcriptions against expected scripts and reports accuracy.
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/ashinsabu/atlas/internal/vox/stt"
)

type TestResult struct {
	Recording    string
	Expected     string
	Transcribed  string
	WordAccuracy float64
	Match        bool
}

func main() {
	scriptNum := flag.Int("script", 0, "Test specific script (0 = all)")
	modelPath := flag.String("model", "models/ggml-large.bin", "Whisper model path")
	verbose := flag.Bool("verbose", false, "Show detailed output")
	flag.Parse()

	// Initialize Whisper with optimized config
	fmt.Printf("Loading model: %s\n", *modelPath)
	cfg := stt.DefaultWhisperConfig()
	cfg.ModelPath = *modelPath
	whisper, err := stt.NewWhisper(cfg)
	if err != nil {
		fmt.Printf("Failed to load Whisper: %v\n", err)
		os.Exit(1)
	}
	defer whisper.Close()

	fmt.Printf("Model loaded: %s\n", whisper.ModelInfo())
	fmt.Printf("Config: temp=%.1f, beam=%d, prompt=%q\n\n",
		cfg.Temperature, cfg.BeamSize, truncate(cfg.InitialPrompt, 50))

	// Find recordings
	recordings, err := filepath.Glob("test/stt/recordings/script_*_rec_*.wav")
	if err != nil || len(recordings) == 0 {
		fmt.Println("No recordings found in test/stt/recordings/")
		fmt.Println("Run: go run ./cmd/stt-record -script 1 -take 1")
		os.Exit(1)
	}

	// Filter by script if specified
	if *scriptNum > 0 {
		var filtered []string
		prefix := fmt.Sprintf("script_%d_", *scriptNum)
		for _, r := range recordings {
			if strings.Contains(filepath.Base(r), prefix) {
				filtered = append(filtered, r)
			}
		}
		recordings = filtered
	}

	fmt.Printf("Testing %d recordings...\n\n", len(recordings))

	var results []TestResult
	totalWords := 0
	correctWords := 0

	for _, recPath := range recordings {
		// Parse script number from filename
		base := filepath.Base(recPath)
		var sNum int
		fmt.Sscanf(base, "script_%d_", &sNum)

		// Load expected text
		scriptPath := fmt.Sprintf("test/stt/scripts/script_%d.txt", sNum)
		expectedBytes, err := os.ReadFile(scriptPath)
		if err != nil {
			fmt.Printf("Skip %s: no script file\n", base)
			continue
		}
		expected := strings.TrimSpace(string(expectedBytes))

		// Load audio
		audioData, err := loadWAV(recPath)
		if err != nil {
			fmt.Printf("Skip %s: %v\n", base, err)
			continue
		}

		// Transcribe
		transcribed, err := whisper.TranscribeBytes(audioData)
		if err != nil {
			fmt.Printf("Skip %s: transcription failed: %v\n", base, err)
			continue
		}

		// Calculate word accuracy
		accuracy, match := compareText(expected, transcribed)

		expWords := len(strings.Fields(normalize(expected)))
		totalWords += expWords
		correctWords += int(float64(expWords) * accuracy)

		result := TestResult{
			Recording:    base,
			Expected:     expected,
			Transcribed:  transcribed,
			WordAccuracy: accuracy,
			Match:        match,
		}
		results = append(results, result)

		// Print result
		status := "PASS"
		if !match {
			status = "FAIL"
		}
		if *verbose {
			fmt.Printf("[%s] %s\n", status, base)
			fmt.Printf("  Expected:    %s\n", expected)
			fmt.Printf("  Transcribed: %s\n", transcribed)
			fmt.Printf("  Accuracy:    %.1f%%\n\n", accuracy*100)
		} else {
			fmt.Printf("[%s] %s (%.0f%%)\n", status, base, accuracy*100)
		}
	}

	// Summary
	fmt.Println()
	fmt.Println("=== SUMMARY ===")
	passed := 0
	for _, r := range results {
		if r.Match {
			passed++
		}
	}
	fmt.Printf("Tests:    %d/%d passed\n", passed, len(results))
	if totalWords > 0 {
		fmt.Printf("Words:    %d/%d correct (%.1f%%)\n", correctWords, totalWords, float64(correctWords)/float64(totalWords)*100)
	}

	// Save results
	resultsPath := "test/stt/results/latest.txt"
	f, _ := os.Create(resultsPath)
	if f != nil {
		defer f.Close()
		for _, r := range results {
			status := "PASS"
			if !r.Match {
				status = "FAIL"
			}
			fmt.Fprintf(f, "[%s] %s\n", status, r.Recording)
			fmt.Fprintf(f, "  Expected:    %s\n", r.Expected)
			fmt.Fprintf(f, "  Transcribed: %s\n", r.Transcribed)
			fmt.Fprintf(f, "  Accuracy:    %.1f%%\n\n", r.WordAccuracy*100)
		}
		fmt.Printf("\nResults saved: %s\n", resultsPath)
	}
}

// truncate shortens a string for display
func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + "..."
}

// normalize removes punctuation and lowercases for comparison
func normalize(s string) string {
	s = strings.ToLower(s)
	re := regexp.MustCompile(`[^\w\s]`)
	s = re.ReplaceAllString(s, "")
	return strings.TrimSpace(s)
}

// compareText returns word accuracy and exact match status
func compareText(expected, transcribed string) (accuracy float64, match bool) {
	expNorm := normalize(expected)
	transNorm := normalize(transcribed)

	if expNorm == transNorm {
		return 1.0, true
	}

	expWords := strings.Fields(expNorm)
	transWords := strings.Fields(transNorm)

	if len(expWords) == 0 {
		return 0, false
	}

	// Simple word matching (order-sensitive)
	matches := 0
	transSet := make(map[string]int)
	for _, w := range transWords {
		transSet[w]++
	}
	for _, w := range expWords {
		if transSet[w] > 0 {
			matches++
			transSet[w]--
		}
	}

	accuracy = float64(matches) / float64(len(expWords))
	// Consider >90% accuracy as a pass
	return accuracy, accuracy >= 0.9
}

// loadWAV reads PCM data from a WAV file
func loadWAV(path string) ([]byte, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	// Simple WAV parser - find "data" chunk
	if len(data) < 44 {
		return nil, fmt.Errorf("file too small")
	}
	if string(data[0:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return nil, fmt.Errorf("not a WAV file")
	}

	// Find data chunk
	for i := 12; i < len(data)-8; i++ {
		if string(data[i:i+4]) == "data" {
			size := int(data[i+4]) | int(data[i+5])<<8 | int(data[i+6])<<16 | int(data[i+7])<<24
			start := i + 8
			end := start + size
			if end > len(data) {
				end = len(data)
			}
			return data[start:end], nil
		}
	}

	return nil, fmt.Errorf("no data chunk found")
}
