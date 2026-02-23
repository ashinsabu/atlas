# ATLAS - Claude Reference

## What Is This

Voice-first AI secretary for ADHD productivity. Not an AGI buddy - a competent executive secretary.

## Architecture

Two modules:
- **Brain** (daemon): Intent parsing, goals, LLM, patterns, scheduling. Receives text, returns text.
- **Vox** (voice): Wake word, speaker ID, STT (Whisper), TTS (natural). Audio I/O only.

Transport: Interface-based. Local = channels. Remote = gRPC. Vox on Pi, Brain on server.

## Tech Stack

- Go 1.24+
- SQLite (goals, habits, schedules)
- Qdrant (semantic search)
- Whisper.cpp via CGO (local STT)
- Claude API (LLM)
- gRPC (remote IPC)
- PortAudio (audio capture, float32)
- ONNX Runtime (speaker verification)

## Key Constraints

- Local-first, privacy-respecting
- <$20/month LLM budget
- <2s simple queries, <5s complex
- Must work offline for core features
- Single user (N=1 pattern learning)
- STT accuracy target: 90%+

---

## Quick Start

```bash
make setup       # Install deps, build whisper, download models
make enroll      # Enroll your voice (interactive)
make vox         # Start voice assistant
```

Say **"Hey Atlas, what should I do now?"** to interact.

Example output:
```
> Just talking normally
> Hey Atlas, what should I do now?
✓ Wake word detected!
╭─ Command ─────────────────────────────
│ what should I do now?
╰────────────────────────────────────────
```

---

## Primary Commands

| Command | Description |
|---------|-------------|
| `make setup` | Full setup: deps + whisper lib + models |
| `make vox` | Main voice assistant (wake word mode) |
| `make vox-debug` | With energy level visualization |
| `make enroll` | Interactive voice enrollment |
| `make profiles` | List enrolled speakers |
| `make speaker-id` | Speaker identification demo |
| `make build` | Build all binaries |
| `make test` | Run tests |
| `make stt-test` | Test STT accuracy |
| `make help` | Show all commands |

---

## Vox Pipeline (`cmd/vox/`)

Main voice assistant pipeline:

```
Mic → VAD → Whisper STT → Wake Word Detection → [Brain]
                ↓                    ↓
         Speaker Verify        "Hey Atlas" trigger
```

**Components:**
- **Audio Capture**: PortAudio, 16kHz mono
- **VAD**: Energy-based voice activity detection
- **STT**: Whisper.cpp via CGO (96% accuracy)
- **Wake Word**: Detects "Hey Atlas" or "Atlas"
- **Speaker Verify**: ONNX-based (optional, for multi-user)

## Speaker ID Demo (`cmd/speaker-id/`)

Utility for testing speaker identification. Shows who is speaking in real-time.

```
Mic (float32) → VAD → [Speaker ID + STT in parallel] → "Name: transcript"
```

**Optimizations in this demo:**
- float32 throughout (no int16 conversions)
- Parallel Speaker ID + STT (2 goroutines)
- Inline VAD (no separate module)

---

## Interactive Enrollment (`cmd/speaker-enroll/`)

Fully interactive enrollment flow:

1. Run `make enroll`
2. Enter your name when prompted
3. Read 5 phrases, twice each (10 recordings total):
   - "Hey Atlas, what should I do now?"
   - "Hey Atlas, how is my progress looking?"
   - "Remember, I need to call the dentist tomorrow morning."
   - "Prioritize fitness this week."
   - "What's my net worth looking like?"
4. Profile saved to `~/.atlas/profiles/<name>.bin`

---

## Implemented Entities

### Speaker ID (`cmd/speaker-id/main.go`)
- Live speaker identification with transcription
- Captures float32 audio directly
- Inline VAD (energy-based)
- Parallel Speaker ID + STT processing
- Output: `<Speaker>: <transcript>`

### Speaker Enrollment (`cmd/speaker-enroll/main.go`)
- Interactive enrollment flow
- Prompts for name
- Shows scripts to read
- Records 10 samples (5 scripts × 2 takes)
- Creates profile automatically

### Vox Module (`internal/vox/`)

#### Audio Capture (`internal/vox/audio/`)
- `Capture` struct wraps PortAudio
- 16kHz mono, float32
- Callback-based streaming

#### VAD (`internal/vox/vad/`)
- `VAD` struct with energy-based voice activity detection
- Config: `SilenceTimeout`, `MaxSpeechDuration`, energy thresholds
- Callbacks: `OnSpeechStart`, `OnSpeechEnd(pcmData []byte)`

#### STT (`internal/vox/stt/`)
**IMPORTANT: Uses CGO bindings, NOT CLI subprocess**

```go
type WhisperConfig struct {
    ModelPath     string   // Required: path to ggml model
    Language      string   // Default: "en"
    Temperature   float32  // Default: 0.0 (deterministic)
    BeamSize      int      // Default: 5
    InitialPrompt string   // Default: "" (no bias)
}

func NewWhisper(cfg WhisperConfig) (*Whisper, error)
func (w *Whisper) TranscribeBytes(pcmData []byte) (string, error)
func (w *Whisper) Close() error
```

#### Speaker Module (`internal/vox/speaker/`)

| File | Purpose |
|------|---------|
| `speaker.go` | Verifier interface, DefaultVerifier, Config |
| `embedding_onnx.go` | ONNXExtractor, ONNXVerifier |
| `fbank.go` | 80-dim mel filterbank features |
| `enrollment.go` | Profile creation |
| `storage.go` | Binary serialization, named profiles |
| `multi.go` | MultiProfileVerifier for multi-user |

#### Wake Word (`internal/vox/wakeword/`)
- `Detector` struct with state machine: `idle` → `listening` → `processing`
- Configurable wake words (default: "hey atlas", "atlas")

---

## CGO Build System

### Why CGO?
Whisper.cpp Go bindings call native C/C++. Benefits:
- Single self-contained binary
- No external CLI dependency
- Direct memory sharing
- Accelerate/BLAS optimizations on Apple Silicon

### Build Dependencies (macOS ARM64)
```bash
xcode-select --install  # Clang, linker, headers
brew install cmake portaudio onnxruntime
```

### CGO Environment (set by Makefile)
```makefile
export CGO_ENABLED := 1
export C_INCLUDE_PATH := $(PWD)/third_party/whisper.cpp/include:...
export LIBRARY_PATH := $(PWD)/third_party/whisper.cpp/build/src:...
```

**CRITICAL**: Must use `make build`, `make vox`, etc. Direct `go build` fails without env vars.

### Metal GPU
**Currently DISABLED** - whisper.cpp Metal backend uses macOS 15+ APIs.
Falls back to CPU with Accelerate/BLAS - still fast on Apple Silicon.

---

## Directory Structure

```
atlas/
├── cmd/
│   ├── vox/              # Wake word + command mode
│   ├── speaker-enroll/   # Interactive voice enrollment
│   ├── speaker-id/       # Live speaker identification (optimized)
│   ├── speaker-profile/  # Profile management
│   ├── stt-record/       # Test recording tool
│   └── stt-test/         # STT accuracy tester
├── internal/
│   ├── audio/            # Shared audio utilities
│   │   ├── wav.go        # LoadWAV, LoadWAVBytes
│   │   ├── record.go     # RecordSample, PortAudio
│   │   └── directory.go  # LoadWAVDirectory
│   └── vox/
│       ├── audio/        # PortAudio capture
│       ├── config/       # Configuration
│       ├── stt/          # Whisper CGO bindings
│       ├── speaker/      # Speaker verification (ONNX)
│       ├── vad/          # Voice activity detection
│       ├── wakeword/     # Wake word detection
│       └── vox.go        # Coordinator
├── models/               # ML models (gitignored)
├── test/stt/             # STT test data
├── third_party/          # whisper.cpp (gitignored)
├── ~/.atlas/profiles/    # User voice profiles
└── Makefile
```

---

## go.mod Key Dependencies

```go
require (
    github.com/ggerganov/whisper.cpp/bindings/go v0.0.0
    github.com/gordonklaus/portaudio v0.0.0-...
    github.com/joho/godotenv v1.5.1
    github.com/yalue/onnxruntime_go v1.26.0
)

// Local whisper bindings
replace github.com/ggerganov/whisper.cpp/bindings/go => ./third_party/whisper.cpp/bindings/go
```

---

## Build Status

✅ **Speaker ID Pipeline** (NEW - optimized)
- Float32 audio capture
- Inline VAD
- Parallel Speaker ID + STT
- Output: `"Name: transcript"`

✅ **Interactive Enrollment** (NEW)
- Prompts for name
- Guides through 10 recordings
- Creates profile automatically

✅ **Vox Module**
- Audio capture via PortAudio
- VAD for speech segmentation
- STT via whisper.cpp CGO
- Wake word detection

✅ **STT Test Infrastructure**
- 96.5% word accuracy (21/25 tests pass)

✅ **Speaker Verification**
- ONNX-based (WeSpeaker CAM++)
- Multi-user profiles

⏳ **Next Up**
- TTS (text-to-speech response)
- Brain CLI (text in/out)
- Goal CRUD + atlas_state.yaml

---

## Speaker Verification Details

### Model
- WeSpeaker CAM++ (29MB ONNX, trained on VoxCeleb)
- Input: 80-dim mel filterbank features
- Output: 512-dim speaker embedding
- Comparison: Cosine similarity, threshold 0.55

### Profile Storage
- Location: `~/.atlas/profiles/<name>.bin`
- Format: Binary with magic "ATLV", version, sample count, timestamp, embedding

### MultiProfileVerifier API
```go
verifier, _ := speaker.NewMultiProfileVerifier(cfg, modelPath)
verifier.LoadAllProfiles()  // Loads all from ~/.atlas/profiles/

result, _ := verifier.Verify(float32Samples)
if result.Matched {
    fmt.Println("Speaker:", result.MatchedBy)
}
```

---

## User Preferences

- No over-engineering
- Direct answers, no sycophancy
- Build-to-think: requirements will evolve
- Iterate fast, validate assumptions early
- User-first UX (interactive, guided flows)

---

## Session Progress Log

### Session: Feb 2024 - Live Speaker ID & UX Overhaul

**Goal**: Simplify UX, add live speaker identification

**Achieved**:
- Interactive enrollment (`make enroll`)
- Live speaker ID (`make speaker-id`)
- Optimized pipeline (float32, parallel processing)
- Simplified Makefile (removed 10+ redundant commands)

**Changes Made**:
| File | Change |
|------|--------|
| `cmd/speaker-enroll/main.go` | Complete rewrite: interactive, guided flow |
| `cmd/speaker-id/main.go` | New: optimized live speaker ID |
| `Makefile` | Simplified to `enroll`, `speaker-id`, `profiles` |
| `README.md` | Updated quick start |
| `UNDERSTANDING.md` | Updated architecture docs |
| `ARCHITECTURE.md` | Added optimized pipeline diagram |
| `CLAUDE.md` | This update |

**Key Decisions**:
- float32 throughout pipeline (industry standard for DSP)
- Parallel Speaker ID + STT (reduces latency ~50%)
- Inline VAD (no separate module overhead)
- 5 scripts × 2 takes = 10 recordings (good voice variety)

---

### Session: Feb 2024 - STT Optimization

**Achieved**: 96.5% word accuracy

---

### Session: Feb 2024 - Speaker Verification

**Achieved**: Working multi-user speaker verification (threshold 0.55)

---

### Session: Feb 2024 - DRY Refactor

**Achieved**: Created shared `internal/audio/` package, eliminated ~300 lines duplication
