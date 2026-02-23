# Atlas Makefile
#
# Quick reference:
#   make setup      - Install all dependencies + build whisper lib
#   make enroll     - Enroll your voice (interactive)
#   make speaker-id - Live speaker identification
#   make vox        - Run the Vox voice module

.PHONY: setup setup-deps setup-whisper setup-speaker whisper-lib vox build test clean enroll speaker-id profiles

# Default model (large-v3 for best accuracy)
WHISPER_MODEL ?= large-v3

# Whisper.cpp paths
WHISPER_DIR := third_party/whisper.cpp
WHISPER_LIB := $(WHISPER_DIR)/build/src/libwhisper.a
WHISPER_INCLUDE := $(WHISPER_DIR)/include
GGML_INCLUDE := $(WHISPER_DIR)/ggml/include

# Speaker embedding model (WeSpeaker CAM++ trained on VoxCeleb)
SPEAKER_MODEL := models/wespeaker_en_voxceleb_CAM++.onnx
SPEAKER_MODEL_URL := https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_CAM++.onnx

# CGO flags for whisper
export CGO_ENABLED := 1
export C_INCLUDE_PATH := $(PWD)/$(WHISPER_INCLUDE):$(PWD)/$(GGML_INCLUDE)
export LIBRARY_PATH := $(PWD)/$(WHISPER_DIR)/build/src:$(PWD)/$(WHISPER_DIR)/build/ggml/src:$(PWD)/$(WHISPER_DIR)/build/ggml/src/ggml-blas
export MACOSX_DEPLOYMENT_TARGET := 15.0

#══════════════════════════════════════════════════════════════
# Setup
#══════════════════════════════════════════════════════════════

setup: setup-deps whisper-lib setup-whisper setup-speaker
	@echo ""
	@echo "╔════════════════════════════════════════╗"
	@echo "║         Setup complete!                ║"
	@echo "╚════════════════════════════════════════╝"
	@echo ""
	@echo "Next steps:"
	@echo "  1. make enroll     - Enroll your voice"
	@echo "  2. make speaker-id - Start live speaker ID"

setup-deps:
	@echo "Installing system dependencies..."
	@which brew > /dev/null || (echo "Error: Homebrew required" && exit 1)
	brew install portaudio cmake onnxruntime
	@echo "Installing Go dependencies..."
	go mod download
	go mod tidy

setup-whisper:
	@echo "Downloading Whisper model ($(WHISPER_MODEL))..."
	@mkdir -p models
	@if [ ! -f "models/ggml-$(WHISPER_MODEL).bin" ]; then \
		curl -L -o "models/ggml-$(WHISPER_MODEL).bin" \
			"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-$(WHISPER_MODEL).bin"; \
		echo "Model downloaded: models/ggml-$(WHISPER_MODEL).bin"; \
	else \
		echo "Model already exists: models/ggml-$(WHISPER_MODEL).bin"; \
	fi

setup-speaker:
	@echo "Downloading speaker embedding model..."
	@mkdir -p models
	@if [ ! -f "$(SPEAKER_MODEL)" ]; then \
		curl -L -o "$(SPEAKER_MODEL)" "$(SPEAKER_MODEL_URL)"; \
		echo "Speaker model downloaded: $(SPEAKER_MODEL)"; \
	else \
		echo "Speaker model already exists: $(SPEAKER_MODEL)"; \
	fi

#══════════════════════════════════════════════════════════════
# Whisper.cpp Library
#══════════════════════════════════════════════════════════════

whisper-clone:
	@if [ ! -d "$(WHISPER_DIR)" ]; then \
		echo "Cloning whisper.cpp..."; \
		mkdir -p third_party; \
		git clone --depth 1 https://github.com/ggerganov/whisper.cpp.git $(WHISPER_DIR); \
	else \
		echo "whisper.cpp already cloned"; \
	fi

whisper-lib: whisper-clone
	@echo "Building whisper.cpp library..."
	@if [ ! -f "$(WHISPER_LIB)" ]; then \
		cd $(WHISPER_DIR) && \
		cmake -B build \
			-DBUILD_SHARED_LIBS=OFF \
			-DWHISPER_BUILD_EXAMPLES=OFF \
			-DWHISPER_BUILD_TESTS=OFF \
			-DGGML_METAL=OFF \
			-DCMAKE_C_FLAGS="-I$$(pwd)/ggml/include" \
			-DCMAKE_CXX_FLAGS="-I$$(pwd)/ggml/include" && \
		cmake --build build --config Release -j$$(nproc 2>/dev/null || sysctl -n hw.ncpu); \
		echo "Whisper library built: $(WHISPER_LIB)"; \
	else \
		echo "Whisper library already exists: $(WHISPER_LIB)"; \
	fi

whisper-update:
	@echo "Updating whisper.cpp..."
	cd $(WHISPER_DIR) && git pull
	rm -rf $(WHISPER_DIR)/build
	$(MAKE) whisper-lib

#══════════════════════════════════════════════════════════════
# Speaker Enrollment & Identification (Primary Commands)
#══════════════════════════════════════════════════════════════

# Interactive voice enrollment - prompts for name, guides through recording
enroll:
	@echo ""
	go run ./cmd/speaker-enroll

# Live speaker identification with transcription
# Shows: "Speaker: transcript" for each utterance
speaker-id: whisper-lib
	go run ./cmd/speaker-id

# List enrolled profiles
profiles:
	@go run ./cmd/speaker-profile list

#══════════════════════════════════════════════════════════════
# Vox Voice Module
#══════════════════════════════════════════════════════════════

vox: whisper-lib
	@echo "Starting Vox (streaming mode - 2s chunks)..."
	go run ./cmd/vox -mode streaming

vox-accurate: whisper-lib
	@echo "Starting Vox (accurate mode - 5s chunks)..."
	go run ./cmd/vox -mode accurate

vox-debug: whisper-lib
	go run ./cmd/vox -debug -mode streaming

#══════════════════════════════════════════════════════════════
# Build
#══════════════════════════════════════════════════════════════

build: whisper-lib
	@echo "Building..."
	@mkdir -p bin
	go build -o bin/vox ./cmd/vox
	go build -o bin/speaker-enroll ./cmd/speaker-enroll
	go build -o bin/speaker-id ./cmd/speaker-id
	@echo "Built: bin/vox, bin/speaker-enroll, bin/speaker-id"

#══════════════════════════════════════════════════════════════
# Test
#══════════════════════════════════════════════════════════════

test: whisper-lib
	go test ./...

test-verbose: whisper-lib
	go test -v ./...

# STT accuracy testing
stt-test: whisper-lib
	go run ./cmd/stt-test -verbose

stt-test-quick: whisper-lib
	go run ./cmd/stt-test

# Record test samples: make stt-record SCRIPT=1 TAKE=1
SCRIPT ?= 1
TAKE ?= 1
stt-record: whisper-lib
	@cat test/stt/scripts/script_$(SCRIPT).txt 2>/dev/null || echo "Script $(SCRIPT) not found"
	go run ./cmd/stt-record -script $(SCRIPT) -take $(TAKE)

#══════════════════════════════════════════════════════════════
# Clean
#══════════════════════════════════════════════════════════════

clean:
	rm -rf bin/
	rm -rf models/
	go clean

clean-whisper:
	rm -rf $(WHISPER_DIR)/build

clean-all: clean clean-whisper
	rm -rf third_party/

#══════════════════════════════════════════════════════════════
# Help
#══════════════════════════════════════════════════════════════

help:
	@echo "Atlas Makefile"
	@echo ""
	@echo "Getting Started:"
	@echo "  make setup      - Install dependencies + download models"
	@echo "  make enroll     - Enroll your voice (interactive)"
	@echo "  make speaker-id - Live speaker identification"
	@echo ""
	@echo "Voice Module:"
	@echo "  make vox        - Start Vox (wake word + commands)"
	@echo "  make vox-debug  - Start with debug output"
	@echo ""
	@echo "Profiles:"
	@echo "  make profiles   - List enrolled speaker profiles"
	@echo ""
	@echo "Build & Test:"
	@echo "  make build      - Build all binaries to ./bin/"
	@echo "  make test       - Run all tests"
	@echo "  make stt-test   - Test STT accuracy"
	@echo ""
	@echo "Clean:"
	@echo "  make clean      - Remove binaries and models"
	@echo "  make clean-all  - Remove everything including third_party/"
