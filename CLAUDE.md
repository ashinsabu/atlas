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
- WeSpeaker ResNet34-LM via ONNX (speaker verification)
- PortAudio (audio capture)
- ONNX Runtime (VAD + speaker verification)

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
Mic → AudioSource → SpeechDetector → STT → WakeWordDetector → SpeakerVerifier
       (32ms chunks)   (Silero VAD)    (Whisper)   (text match)    (wake only)
```

Speaker verification runs **only when the wake word fires**. Non-wake speech goes to `OnText` unverified. This matches Siri's approach: authenticate at the wake phrase, trust the command that follows.

## Directory Structure

```
cmd/vox/              # Entry point (normal + debug modes)
cmd/vox-enroll/       # Speaker enrollment CLI (live mic recording)
cmd/speaker-test/     # Speaker verification accuracy test harness
internal/vox/
  audio/
    source.go         # AudioSource interface
    portaudio.go      # PortAudio implementation (Debug flag gates device info)
    wav.go            # WAV load/save utilities (used by speaker enrollment)
  speech/
    detector.go       # SpeechDetector interface + DetectorConfig
    segment.go        # SpeechSegment type
    silero.go         # Silero VAD (ONNX) — has OnFrame debug callback field
  stt/
    whisper.go        # Whisper.cpp via CGO
  wakeword/
    detector.go       # Wake word detection
  speaker/
    fbank.go          # Mel-filterbank feature extraction (pure Go, Kaldi-compatible)
    encoder.go        # WeSpeaker ONNX encoder: PCM → fbank → embedding
    verifier.go       # Enrollment + real-time speaker verification
    profile.go        # JSON speaker profile (load/save)
    inspect.go        # ONNX tensor name inspection utility
  config/
    config.go         # YAML config (atlas.vox namespace); ConfigDir() for profile paths
  debug/
    debugger.go       # Debugger interface + NopDebugger + UIDebugger
    events.go         # Tea messages: ChunkMsg, SegmentMsg, STTMsg, SpeakerMsg
    ui.go             # Bubbletea TUI (4 panes: AUDIO, PIPELINE, TRANSCRIPT, STATS)
    styles.go         # Lipgloss styles
    debug.go          # debug.New() entry point (returns prog, dbg, *monitor.Tracker)
  pipeline.go         # Main orchestrator; verifier field; OnSpeakerVerified callback
internal/monitor/
  tracker.go          # Concurrent-safe per-stage latency recorder; Go runtime snapshot
models/               # gitignored — whisper, silero, wespeaker models
third_party/          # gitignored — whisper.cpp source
docs/
  engineering-concepts.md  # Low-level systems reference for this codebase
```

## Build System

**Must use Makefile** — CGO needs env vars (C_INCLUDE_PATH, LIBRARY_PATH).

## Key Technical Decisions

| Decision | Why |
|----------|-----|
| Whisper via CGO | Single binary, no subprocess, direct memory |
| Silero VAD | Neural speech detection, not energy-based |
| WeSpeaker ResNet34-LM | Speaker embeddings via ONNX; Pi-deployable, no GPU |
| Fbank in pure Go | No Python dependency; matches WeSpeaker training exactly |
| ONNX Runtime | Runs on Pi, no GPU required |
| Interface-based pipeline | Clean, testable, swappable implementations |
| Config via YAML + CLI flags | YAML becomes flag defaults; CLI overrides automatically |

## Current Status

**Beta.** Full pipeline working end-to-end with speaker verification.

Debug mode (`make vox-debug`) shows real-time per-chunk pipeline meter:
```
  #0042  MIC ████░░░░░░ 0.24 | VAD ██████████ 0.89  ◉ 1.2s
```
On segment detection:
```
  ▸ seg  1.4s  →  transcribing...
  ✓ "Hey Atlas, what time is it?"  (287ms)
  SPK  [Ashin] 0.81 ✓
  ◆ wake: what time is it?
```

Speaker verification stats (Ashin, 25 recordings):
- TPR: 100% at threshold 0.70
- Score range: 0.719 – 0.855
- Enable in `vox.yaml`: `atlas.vox.speaker.enabled: true`

Config priority: `$ATLAS_CONFIG_DIR/vox.yaml` → `./vox.yaml` → hardcoded defaults.

## Speaker Verification Setup

```bash
make setup-wespeaker   # Download WeSpeaker ResNet34-LM ONNX (~25MB)
make vox-enroll        # Interactive mic enrollment (5 phrases × 2 takes)
make speaker-test      # Accuracy report (TPR + TNR)
```

Profile saved to `./speaker.json` (or `$ATLAS_CONFIG_DIR/speaker.json`).

## Next Priorities

1. gRPC transport (Vox ↔ Brain)
2. TTS for responses
3. Brain module (intent parsing, LLM)
4. Pi deployment validation
