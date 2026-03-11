# ATLAS - Claude Reference

## What Is This

Voice-first AI secretary for ADHD productivity. Not an AGI buddy - a competent executive secretary.

## Architecture

Two modules:
- **Brain** (daemon): Intent parsing, goals, LLM, patterns, scheduling. Receives text, returns text.
- **Vox** (voice): Wake word, STT (Whisper), TTS (natural). Audio I/O only.

Transport: Interface-based. Local = channels. Remote = gRPC. Vox on Pi, Brain on server.

## Tech Stack

- Go 1.24+, SQLite, Qdrant, Claude API, gRPC
- Whisper.cpp via CGO (local STT)
- Silero VAD via ONNX (speech detection)
- PortAudio (audio capture)
- ONNX Runtime (VAD, future speaker verification)

## Key Constraints

- Local-first, privacy-respecting, <$20/month LLM budget
- <2s simple queries, <5s complex, must work offline for core features
- Single user. STT accuracy target: 90%+
- **Vox must be Pi-deployable**: ~8GB RAM, ARM CPU, no GPU. Use Whisper small/base model.

## Quick Start

```bash
make setup       # Install deps, download models
make vox         # Start voice assistant
make vox-debug   # With pipeline visualization
```

Say **"Hey Atlas, what should I do now?"** to interact.

## Vox Pipeline

```
Mic → AudioSource → SpeechDetector → STT → WakeWordDetector
       (32ms chunks)   (Silero VAD)    (Whisper)   (text match)
```

## Directory Structure

```
cmd/vox/              # Entry point
internal/vox/
  audio/
    source.go         # AudioSource interface
    portaudio.go      # PortAudio implementation
    wav.go            # WAV load/save utilities (currently unused, kept for future use)
  speech/
    detector.go       # SpeechDetector interface + DetectorConfig
    segment.go        # SpeechSegment type
    silero.go         # Silero VAD (ONNX) — has OnFrame debug callback field
  stt/
    whisper.go        # Whisper.cpp via CGO
  wakeword/
    detector.go       # Wake word detection
  pipeline.go         # Main orchestrator — vox-debug visualization lives here
models/               # gitignored - whisper + silero models
third_party/          # gitignored - whisper.cpp source
```

## Build System

**Must use Makefile** — CGO needs env vars (C_INCLUDE_PATH, LIBRARY_PATH).

## Key Technical Decisions

| Decision | Why |
|----------|-----|
| Whisper via CGO | Single binary, no subprocess, direct memory |
| Silero VAD | Neural speech detection, not energy-based |
| ONNX Runtime | Runs on Pi, no GPU required |
| Interface-based pipeline | Clean, testable, swappable implementations |

## Current Status

Vox pipeline working end-to-end: PortAudio → Silero VAD → Whisper STT → wake word detection.

Debug mode (`make vox-debug`) shows real-time per-chunk pipeline meter:
```
  #0042  MIC ████░░░░░░ 0.24 | VAD ██████████ 0.89  ◉ 1.2s
```
Then on segment detection:
```
  ▸ seg  1.4s  →  transcribing...
  ✓ "Hey Atlas, what time is it?"  (287ms)
  ◆ wake: what time is it?
```

## Next Priorities

1. Speaker verification (build from scratch using ONNX-compatible model)
2. gRPC transport (Vox ↔ Brain)
3. TTS for responses
4. Brain module (intent parsing, LLM)
