// speaker-profile: Manage speaker profiles for Atlas
//
// Usage:
//   speaker-profile list                  # List all profiles
//   speaker-profile delete <name>         # Delete a profile
//   speaker-profile test -users alice,bob # Test multi-profile verification
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
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	switch os.Args[1] {
	case "list":
		listProfiles()
	case "delete":
		if len(os.Args) < 3 {
			fmt.Println("Usage: speaker-profile delete <name>")
			os.Exit(1)
		}
		deleteProfile(os.Args[2])
	case "test":
		testCmd := flag.NewFlagSet("test", flag.ExitOnError)
		users := testCmd.String("users", "", "Comma-separated profile names")
		wavDir := testCmd.String("dir", "", "Directory of WAV files")
		wavFile := testCmd.String("file", "", "Single WAV file")
		threshold := testCmd.Float64("threshold", 0.60, "Verification threshold")
		modelPath := testCmd.String("model", "models/wespeaker_en_voxceleb_CAM++.onnx", "ONNX model path")
		testCmd.Parse(os.Args[2:])
		testMultiProfile(*users, *wavDir, *wavFile, float32(*threshold), *modelPath)
	default:
		fmt.Printf("Unknown command: %s\n", os.Args[1])
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Println("speaker-profile: Manage speaker profiles")
	fmt.Println("\nCommands:")
	fmt.Println("  list              List all enrolled profiles")
	fmt.Println("  delete <name>     Delete a profile")
	fmt.Println("  test -users a,b   Test multi-profile verification")
}

func listProfiles() {
	profiles, err := speaker.ListProfiles()
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}

	if len(profiles) == 0 {
		fmt.Println("No profiles enrolled.")
		fmt.Println("Use 'make profile-add NAME=yourname' to create one.")
		return
	}

	fmt.Println("Enrolled profiles:")
	fmt.Println(strings.Repeat("-", 40))
	for _, name := range profiles {
		path, _ := speaker.NamedProfilePath(name)
		info, err := os.Stat(path)
		if err != nil {
			fmt.Printf("  %-20s (error)\n", name)
			continue
		}
		fmt.Printf("  %-20s %s\n", name, info.ModTime().Format("2006-01-02 15:04"))
	}
	fmt.Println(strings.Repeat("-", 40))
	fmt.Printf("Total: %d profile(s)\n", len(profiles))
}

func deleteProfile(name string) {
	if err := speaker.DeleteProfile(name); err != nil {
		if os.IsNotExist(err) {
			fmt.Printf("Profile '%s' not found.\n", name)
		} else {
			fmt.Printf("Error: %v\n", err)
		}
		os.Exit(1)
	}
	fmt.Printf("Profile '%s' deleted.\n", name)
}

func testMultiProfile(users, wavDir, wavFile string, threshold float32, modelPath string) {
	if users == "" {
		fmt.Println("Error: -users required (e.g., -users alice,bob)")
		os.Exit(1)
	}

	userList := strings.Split(users, ",")
	for i := range userList {
		userList[i] = strings.TrimSpace(userList[i])
	}

	cfg := speaker.DefaultConfig()
	cfg.Threshold = threshold

	verifier, err := speaker.NewMultiProfileVerifier(cfg, modelPath)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}
	defer verifier.Close()

	for _, name := range userList {
		if err := verifier.LoadProfile(name); err != nil {
			fmt.Printf("Failed to load '%s': %v\n", name, err)
			os.Exit(1)
		}
	}

	fmt.Printf("Profiles: %s\nThreshold: %.2f\n\n", strings.Join(userList, ", "), threshold)

	if wavFile != "" {
		testSingleFile(verifier, wavFile)
	} else if wavDir != "" {
		testDirectory(verifier, wavDir)
	} else {
		testMicrophone(verifier)
	}
}

func testSingleFile(verifier *speaker.MultiProfileVerifier, path string) {
	data, err := audio.LoadWAV(path)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}

	result, err := verifier.Verify(data)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}

	printMultiResult(filepath.Base(path), result, verifier.GetThreshold())
}

func testDirectory(verifier *speaker.MultiProfileVerifier, dir string) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}

	var tested, passed int

	fmt.Println("Testing all WAV files...")
	fmt.Println(strings.Repeat("-", 70))

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

		result, err := verifier.Verify(data)
		if err != nil {
			fmt.Printf("%-30s  ERROR: %v\n", entry.Name(), err)
			continue
		}

		tested++
		if result.Matched {
			passed++
			fmt.Printf("%-30s  MATCH: %-10s (%.3f)\n", entry.Name(), result.MatchedBy, result.Confidence)
		} else {
			fmt.Printf("%-30s  NO MATCH          (%.3f)\n", entry.Name(), result.Confidence)
		}
	}

	fmt.Println(strings.Repeat("-", 70))
	fmt.Printf("Results: %d/%d matched (%.1f%%)\n", passed, tested, float64(passed)/float64(tested)*100)
}

func testMicrophone(verifier *speaker.MultiProfileVerifier) {
	recorder, err := audio.NewRecorder()
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}
	defer recorder.Close()

	fmt.Println("=== Multi-Profile Verification ===")
	fmt.Printf("Authorized: %s\n", strings.Join(verifier.ListAuthorized(), ", "))
	fmt.Println("Press Ctrl+C to exit.")
	fmt.Println()

	for i := 1; ; i++ {
		fmt.Printf("Test %d: Press ENTER to record...\n", i)

		data, err := recorder.RecordUntilEnter()
		if err != nil {
			fmt.Printf("Error: %v\n\n", err)
			continue
		}

		result, err := verifier.Verify(data)
		if err != nil {
			fmt.Printf("Error: %v\n\n", err)
			continue
		}

		printMultiResult(fmt.Sprintf("Test %d", i), result, verifier.GetThreshold())
		fmt.Println()
	}
}

func printMultiResult(name string, result speaker.MultiVerifyResult, threshold float32) {
	if result.Matched {
		fmt.Printf("%s: ACCEPTED by '%s'\n", name, result.MatchedBy)
	} else {
		fmt.Printf("%s: REJECTED\n", name)
	}
	fmt.Printf("  Confidence: %.3f (threshold: %.2f)\n", result.Confidence, threshold)
	if len(result.AllScores) > 1 {
		fmt.Print("  All: ")
		for n, s := range result.AllScores {
			fmt.Printf("%s=%.3f ", n, s)
		}
		fmt.Println()
	}
}
