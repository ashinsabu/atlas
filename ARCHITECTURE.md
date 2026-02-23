# ATLAS Architecture

## Overview

ATLAS is a voice-first AI secretary built as two primary modules communicating over an adaptive transport layer.

```
┌─────────────────────────────────────────────────────────────┐
│                        ATLAS                                │
│                                                             │
│   ┌───────────────┐          ┌───────────────────────────┐ │
│   │     Vox       │          │          Brain            │ │
│   │               │  text    │                           │ │
│   │ - wake word   │ ◄──────► │ - intent parsing          │ │
│   │ - speaker ID  │          │ - goal hierarchy          │ │
│   │ - STT         │          │ - LLM orchestration       │ │
│   │ - TTS         │          │ - pattern learning        │ │
│   │               │          │ - scheduling              │ │
│   └───────────────┘          └───────────────────────────┘ │
│          ▲                              ▲                   │
│          │                              │                   │
│       audio                        CLI / API                │
└─────────────────────────────────────────────────────────────┘
```

## Vox Pipeline

Main voice assistant pipeline:

```
┌─────────────────────────────────────────────────────────────┐
│                    Vox Voice Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Mic → VAD → Whisper STT → Wake Word Detection → [Brain]    │
│                    ↓                                        │
│             Speaker Verify (optional)                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Components:**
- **Audio Capture**: PortAudio, 16kHz mono
- **VAD**: Energy-based voice activity detection
- **STT**: Whisper.cpp via CGO (96% accuracy)
- **Wake Word**: Detects "Hey Atlas" or "Atlas"
- **Speaker Verify**: ONNX-based (optional, for multi-user)

## Modules

### Brain (Daemon)

The central intelligence. Receives text prompts, returns text responses. Stateless per-request but maintains persistent storage connections.

**Responsibilities:**
- Intent classification and parsing
- Hierarchical goal management (L1 → L2 → L3)
- LLM API orchestration (Claude)
- Pattern detection and storage
- Rule-based scheduling
- Progress tracking and metrics

**Interfaces:**
- gRPC server (for remote Vox instances)
- Local Go interface (for co-located Vox)
- CLI interface (direct text interaction)
- HTTP/REST (future: web dashboard)

**Storage:**
- `atlas_state.yaml`: Dynamic state file (goals, metrics, progressions) - Brain reads AND writes
- SQLite: historical data, retros, voice note metadata
- Qdrant: semantic search over voice notes

### Vox (Voice Module)

Audio I/O specialist. Handles everything between microphone and speaker.

**Responsibilities:**
- Wake word detection ("Hey Atlas")
- Speaker identification (identify who is speaking)
- Speech-to-text (Whisper, local)
- Text-to-speech (natural cadence, not robotic)
- Command serialization (one active session at a time)

**Deployment modes:**
- Co-located: Same machine as Brain, uses Go channels
- Remote: Raspberry Pi in another room, connects via gRPC

## Quick Start

```bash
make setup       # Install deps, build whisper, download models
make enroll      # Enroll your voice (interactive)
make vox         # Start voice assistant
```

Say **"Hey Atlas, what should I do now?"** to interact.

Example output from `make vox`:
```
> Just talking normally
> Hey Atlas, what should I do now?
✓ Wake word detected!
╭─ Command ─────────────────────────────
│ what should I do now?
╰────────────────────────────────────────
```

## Directory Structure

```
atlas/
├── cmd/
│   ├── vox/              # Main voice assistant
│   ├── speaker-enroll/   # Interactive voice enrollment
│   ├── speaker-id/       # Speaker identification demo
│   ├── speaker-profile/  # Profile management
│   ├── stt-record/       # Record test samples
│   └── stt-test/         # STT accuracy test
├── internal/
│   ├── audio/            # Shared audio utilities
│   │   ├── wav.go        # WAV file loading
│   │   ├── record.go     # Microphone recording
│   │   └── directory.go  # Directory scanning
│   ├── vox/
│   │   ├── audio/        # PortAudio capture
│   │   ├── config/       # Configuration
│   │   ├── stt/          # Whisper CGO bindings
│   │   ├── speaker/      # Speaker verification (ONNX)
│   │   ├── vad/          # Voice activity detection
│   │   ├── wakeword/     # Wake word detection
│   │   └── vox.go        # Coordinator
│   ├── brain/            # (future) Brain core logic
│   └── transport/        # (future) IPC layer
├── models/               # ML models (gitignored)
├── test/stt/             # STT test data
├── third_party/          # whisper.cpp (gitignored)
├── ~/.atlas/profiles/    # User voice profiles
└── go.mod
```

## Data Flow

### Speaker ID Mode (make speaker-id)

```
User speaks
    │
    ▼
[Mic] float32 audio captured
    │
    ▼
[VAD] Energy > threshold? Start accumulating
    │
    ▼
[VAD] Silence timeout reached? Emit segment
    │
    ├──────────────────────┐
    │                      │
    ▼                      ▼
[Speaker ID]          [Whisper STT]
 (parallel)             (parallel)
    │                      │
    └──────────┬───────────┘
               │
               ▼
        "Name: transcript"
```

### Vox Mode (make vox)

```
User speaks
    │
    ▼
[Vox] Wake word detected → Speaker verified → STT
    │
    │ text prompt
    ▼
[Brain] Intent parsed → Storage queried → LLM called (if needed)
    │
    │ text response
    ▼
[Vox] TTS with natural cadence
    │
    ▼
User hears response
```

## External Dependencies

### Whisper.cpp (STT)

C++ port of OpenAI Whisper. We use Go bindings via CGO, linking against static `libwhisper.a`.

**Build**: `make whisper-lib` clones and builds. Makefile sets CGO env vars automatically.

**Models**: GGML-quantized from HuggingFace. `large-v3` for accuracy, `small` for speed.

### PortAudio (Audio Capture)

Cross-platform audio I/O. Captures 16kHz mono as float32.

**Install**: `brew install portaudio`

### ONNX Runtime (Speaker Verification)

Runs WeSpeaker CAM++ neural network for speaker embeddings.

**Model**: `wespeaker_en_voxceleb_CAM++.onnx` (~29MB)
- Input: 80-dim mel filterbank features
- Output: 512-dim speaker embedding
- Comparison: Cosine similarity, threshold 0.55

**Install**: `brew install onnxruntime`

## Non-Functional Requirements

- **Privacy**: All data local, minimal context to LLM APIs
- **Offline**: Core features work without internet
- **Cost**: <$20/month LLM budget with hard limits
- **Latency**: <2s for speaker-id output (parallel processing)
