# Atlas

Voice-first AI secretary for ADHD productivity. Local-first, privacy-respecting.

## Quick Start

```bash
make setup       # Install dependencies + download models
make enroll      # Enroll your voice (interactive)
make vox         # Start voice assistant
```

Say **"Hey Atlas, what should I do now?"** to interact.

## What It Does

Atlas is a voice-controlled executive secretary that:
- Listens for wake word ("Hey Atlas")
- Transcribes your commands locally (Whisper)
- Identifies who is speaking (multi-user support)
- Sends commands to Brain for processing (coming soon)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        ATLAS                                │
│                                                             │
│   ┌───────────────┐          ┌───────────────────────────┐ │
│   │     Vox       │  text    │          Brain            │ │
│   │ - wake word   │ ◄──────► │ - intent parsing          │ │
│   │ - speaker ID  │          │ - goal tracking           │ │
│   │ - STT/TTS     │          │ - LLM orchestration       │ │
│   └───────────────┘          └───────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Commands

```bash
# Setup
make setup       # Install everything

# Voice Assistant
make vox         # Main voice assistant (wake word mode)
make vox-debug   # With energy level visualization

# Speaker Enrollment
make enroll      # Enroll your voice (interactive)
make profiles    # List enrolled speakers

# Utilities
make speaker-id  # Live speaker identification demo
make build       # Build all binaries
make test        # Run tests
make help        # Show all commands
```

## Example Output

`make vox` (main assistant):
```
> Just talking normally
> Hey Atlas, what should I do now?
✓ Wake word detected!
╭─ Command ─────────────────────────────
│ what should I do now?
╰────────────────────────────────────────
```

`make speaker-id` (speaker identification demo):
```
Ashin: Hey, what's up?
Rahul: Not much, just working.
Unknown: What are you two talking about?
```

## Features

- **Local STT**: 96% accuracy via Whisper.cpp (no cloud APIs)
- **Speaker Verification**: Only responds to enrolled users
- **Multi-User Support**: Identifies different speakers
- **Privacy-First**: All processing on-device

## Requirements

- macOS (M-series recommended)
- Homebrew
- Microphone access

## Documentation

- `CLAUDE.md` - Technical reference
- `UNDERSTANDING.md` - Technical handoff guide
- `ARCHITECTURE.md` - System architecture
- `ATLAS.md` - Product specification

## Status

| Module | Status |
|--------|--------|
| Vox (Voice) | ✅ Working |
| Brain (Intelligence) | ⏳ Not started |
| TTS (Speech output) | ⏳ Not started |

## License

Private - not for distribution.
