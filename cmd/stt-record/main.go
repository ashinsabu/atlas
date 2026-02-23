// stt-record: Record audio for STT testing
//
// Usage:
//   stt-record -script 1 -take 1    # Records test/stt/recordings/script_1_rec_1.wav
//   stt-record -script 2 -take 3    # Records test/stt/recordings/script_2_rec_3.wav
//
// Press Enter to start recording, Enter again to stop.
package main

import (
	"bufio"
	"encoding/binary"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/gordonklaus/portaudio"
)

const (
	sampleRate = 16000
	channels   = 1
	bitsPerSample = 16
)

func main() {
	scriptNum := flag.Int("script", 1, "Script number (1-5)")
	takeNum := flag.Int("take", 1, "Take/recording number")
	flag.Parse()

	// Show the script to read
	scriptPath := fmt.Sprintf("test/stt/scripts/script_%d.txt", *scriptNum)
	scriptContent, err := os.ReadFile(scriptPath)
	if err != nil {
		fmt.Printf("Error reading script: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("=== STT Recording Tool ===")
	fmt.Println()
	fmt.Printf("Script %d: %s", *scriptNum, string(scriptContent))
	fmt.Println()
	fmt.Println("Press ENTER to start recording...")

	reader := bufio.NewReader(os.Stdin)
	reader.ReadLine()

	// Initialize PortAudio
	if err := portaudio.Initialize(); err != nil {
		fmt.Printf("PortAudio init failed: %v\n", err)
		os.Exit(1)
	}
	defer portaudio.Terminate()

	// Buffer for recording (max 10 seconds)
	maxSamples := sampleRate * 10
	buffer := make([]int16, maxSamples)
	recorded := 0

	// Create stream
	stream, err := portaudio.OpenDefaultStream(channels, 0, float64(sampleRate), 256, func(in []int16) {
		if recorded+len(in) <= maxSamples {
			copy(buffer[recorded:], in)
			recorded += len(in)
		}
	})
	if err != nil {
		fmt.Printf("Open stream failed: %v\n", err)
		os.Exit(1)
	}
	defer stream.Close()

	fmt.Println(">>> RECORDING... Press ENTER to stop <<<")

	if err := stream.Start(); err != nil {
		fmt.Printf("Start stream failed: %v\n", err)
		os.Exit(1)
	}

	reader.ReadLine()

	if err := stream.Stop(); err != nil {
		fmt.Printf("Stop stream failed: %v\n", err)
	}

	fmt.Printf("Recorded %d samples (%.2f seconds)\n", recorded, float64(recorded)/float64(sampleRate))

	// Save to WAV
	outPath := fmt.Sprintf("test/stt/recordings/script_%d_rec_%d.wav", *scriptNum, *takeNum)
	if err := os.MkdirAll(filepath.Dir(outPath), 0755); err != nil {
		fmt.Printf("Create dir failed: %v\n", err)
		os.Exit(1)
	}

	if err := saveWAV(outPath, buffer[:recorded]); err != nil {
		fmt.Printf("Save WAV failed: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Saved: %s\n", outPath)
}

func saveWAV(path string, samples []int16) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	dataSize := len(samples) * 2
	fileSize := 36 + dataSize

	// RIFF header
	f.WriteString("RIFF")
	binary.Write(f, binary.LittleEndian, uint32(fileSize))
	f.WriteString("WAVE")

	// fmt chunk
	f.WriteString("fmt ")
	binary.Write(f, binary.LittleEndian, uint32(16)) // chunk size
	binary.Write(f, binary.LittleEndian, uint16(1))  // PCM
	binary.Write(f, binary.LittleEndian, uint16(channels))
	binary.Write(f, binary.LittleEndian, uint32(sampleRate))
	binary.Write(f, binary.LittleEndian, uint32(sampleRate*channels*bitsPerSample/8)) // byte rate
	binary.Write(f, binary.LittleEndian, uint16(channels*bitsPerSample/8))            // block align
	binary.Write(f, binary.LittleEndian, uint16(bitsPerSample))

	// data chunk
	f.WriteString("data")
	binary.Write(f, binary.LittleEndian, uint32(dataSize))
	return binary.Write(f, binary.LittleEndian, samples)
}
