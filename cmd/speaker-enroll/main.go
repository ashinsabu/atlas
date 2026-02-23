// speaker-enroll: Interactive voice enrollment for Atlas
//
// Usage: speaker-enroll (no arguments needed - fully interactive)
package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/gordonklaus/portaudio"

	"github.com/ashinsabu/atlas/internal/vox/speaker"
)

// Enrollment scripts - natural phrases for voice capture
var enrollmentScripts = []string{
	"Hey Atlas, what should I do now?",
	"Hey Atlas, how is my progress looking?",
	"Remember, I need to call the dentist tomorrow morning.",
	"Prioritize fitness this week.",
	"What's my net worth looking like?",
}

const (
	sampleRate    = 16000
	takesPerScript = 2
	maxSeconds    = 10
	minSamples    = 8000 // 0.5s minimum
)

func main() {
	modelPath := flag.String("model", "models/wespeaker_en_voxceleb_CAM++.onnx", "ONNX model path")
	flag.Parse()

	fmt.Println("╔════════════════════════════════════════╗")
	fmt.Println("║       ATLAS Voice Enrollment           ║")
	fmt.Println("╚════════════════════════════════════════╝")
	fmt.Println()

	// Get profile name
	name := promptName()
	if name == "" {
		fmt.Println("Enrollment cancelled.")
		return
	}

	// Check if profile exists
	profilePath, err := speaker.NamedProfilePath(name)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}

	if _, err := os.Stat(profilePath); err == nil {
		fmt.Printf("\nProfile '%s' already exists. Overwrite? [y/N]: ", name)
		reader := bufio.NewReader(os.Stdin)
		response, _ := reader.ReadString('\n')
		response = strings.TrimSpace(strings.ToLower(response))
		if response != "y" && response != "yes" {
			fmt.Println("Enrollment cancelled.")
			return
		}
	}

	// Initialize PortAudio
	if err := portaudio.Initialize(); err != nil {
		fmt.Printf("Failed to initialize audio: %v\n", err)
		os.Exit(1)
	}
	defer portaudio.Terminate()

	// Collect voice samples
	fmt.Println("\n┌────────────────────────────────────────┐")
	fmt.Println("│ Voice Sample Collection                │")
	fmt.Println("├────────────────────────────────────────┤")
	fmt.Printf("│ Scripts: %d                             │\n", len(enrollmentScripts))
	fmt.Printf("│ Takes per script: %d                    │\n", takesPerScript)
	fmt.Printf("│ Total recordings: %d                   │\n", len(enrollmentScripts)*takesPerScript)
	fmt.Println("└────────────────────────────────────────┘")
	fmt.Println()
	fmt.Println("Instructions:")
	fmt.Println("  1. Read each phrase naturally")
	fmt.Println("  2. Press ENTER when ready to record")
	fmt.Println("  3. Speak the phrase")
	fmt.Println("  4. Press ENTER when done")
	fmt.Println()

	samples, err := collectSamples()
	if err != nil {
		fmt.Printf("Error collecting samples: %v\n", err)
		os.Exit(1)
	}

	if len(samples) < 5 {
		fmt.Printf("Not enough samples collected: %d (need at least 5)\n", len(samples))
		os.Exit(1)
	}

	// Create profile
	fmt.Printf("\n\nCreating voice profile for '%s'...\n", name)

	cfg := speaker.DefaultConfig()
	cfg.ModelPath = *modelPath
	cfg.MinEnrollSamples = 5

	verifier, cleanup, err := speaker.NewAutoVerifier(cfg)
	if err != nil {
		fmt.Printf("Failed to create verifier: %v\n", err)
		os.Exit(1)
	}
	defer cleanup()

	if err := verifier.Enroll(samples); err != nil {
		fmt.Printf("Enrollment failed: %v\n", err)
		os.Exit(1)
	}

	if err := verifier.SaveProfile(profilePath); err != nil {
		fmt.Printf("Failed to save profile: %v\n", err)
		os.Exit(1)
	}

	fmt.Println()
	fmt.Println("╔════════════════════════════════════════╗")
	fmt.Printf("║  Profile '%s' created successfully!\n", name)
	fmt.Println("╚════════════════════════════════════════╝")
	fmt.Println()
	fmt.Println("You can now use: make speaker-id")
}

func promptName() string {
	reader := bufio.NewReader(os.Stdin)

	// Show existing profiles
	profiles, _ := speaker.ListProfiles()
	if len(profiles) > 0 {
		fmt.Println("Existing profiles:")
		for _, p := range profiles {
			fmt.Printf("  • %s\n", p)
		}
		fmt.Println()
	}

	fmt.Print("Enter your name: ")
	name, _ := reader.ReadString('\n')
	name = strings.TrimSpace(name)

	// Validate name
	if name == "" {
		return ""
	}

	// Simple validation - alphanumeric and underscores only
	for _, c := range name {
		if !((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_') {
			fmt.Println("Name can only contain letters, numbers, and underscores.")
			return promptName()
		}
	}

	return strings.ToLower(name)
}

func collectSamples() ([][]float32, error) {
	reader := bufio.NewReader(os.Stdin)
	var samples [][]float32
	totalRecordings := len(enrollmentScripts) * takesPerScript
	recordingNum := 0

	for scriptIdx, script := range enrollmentScripts {
		for take := 1; take <= takesPerScript; take++ {
			recordingNum++
			fmt.Println()
			fmt.Printf("─── Recording %d/%d ───────────────────────\n", recordingNum, totalRecordings)
			fmt.Printf("Script %d, Take %d:\n", scriptIdx+1, take)
			fmt.Println()
			fmt.Printf("  \"%s\"\n", script)
			fmt.Println()
			fmt.Print("Press ENTER when ready to record...")
			reader.ReadString('\n')

			sample, err := recordSample()
			if err != nil {
				fmt.Printf("Recording failed: %v\n", err)
				fmt.Print("Try again? [Y/n]: ")
				response, _ := reader.ReadString('\n')
				response = strings.TrimSpace(strings.ToLower(response))
				if response == "n" || response == "no" {
					continue
				}
				// Retry
				sample, err = recordSample()
				if err != nil {
					fmt.Printf("Skipping this recording: %v\n", err)
					continue
				}
			}

			samples = append(samples, sample)
			fmt.Printf("✓ Recorded (%.1fs)\n", float64(len(sample))/sampleRate)
		}
	}

	return samples, nil
}

func recordSample() ([]float32, error) {
	maxSamples := sampleRate * maxSeconds
	buffer := make([]int16, maxSamples)
	recorded := 0

	stream, err := portaudio.OpenDefaultStream(1, 0, float64(sampleRate), 256, func(in []int16) {
		if recorded+len(in) <= maxSamples {
			copy(buffer[recorded:], in)
			recorded += len(in)
		}
	})
	if err != nil {
		return nil, fmt.Errorf("open stream: %w", err)
	}
	defer stream.Close()

	fmt.Println("🎤 RECORDING... Press ENTER to stop")

	if err := stream.Start(); err != nil {
		return nil, fmt.Errorf("start stream: %w", err)
	}

	reader := bufio.NewReader(os.Stdin)
	reader.ReadString('\n')

	if err := stream.Stop(); err != nil {
		return nil, fmt.Errorf("stop stream: %w", err)
	}

	// Convert to float32
	samples := make([]float32, recorded)
	for i := 0; i < recorded; i++ {
		samples[i] = float32(buffer[i]) / 32768.0
	}

	if len(samples) < minSamples {
		return nil, fmt.Errorf("too short: need at least 0.5s")
	}

	return samples, nil
}
