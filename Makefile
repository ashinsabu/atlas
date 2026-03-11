# Atlas Makefile
#
# Quick reference:
#   make setup      - Install all dependencies + build whisper lib
#   make enroll     - Enroll your voice (interactive)
#   make vox        - Run the Vox voice module

.PHONY: setup setup-deps setup-whisper setup-whisper-fast setup-speaker whisper-lib vox vox-fast build test clean enroll profiles speaker-enroll speaker-list speaker-delete

# Whisper models
# large-v3: Best accuracy (3GB) - use for accuracy-critical tasks
# distil-large-v3: 6x faster, 1% WER increase (756MB) - recommended for daily use
WHISPER_MODEL ?= large-v3
WHISPER_MODEL_FAST := distil-large-v3

# Whisper.cpp paths
WHISPER_DIR := third_party/whisper.cpp
WHISPER_LIB := $(WHISPER_DIR)/build/src/libwhisper.a
WHISPER_INCLUDE := $(WHISPER_DIR)/include
GGML_INCLUDE := $(WHISPER_DIR)/ggml/include

# Speaker embedding model (WeSpeaker CAM++ trained on VoxCeleb)
SPEAKER_MODEL := models/wespeaker_en_voxceleb_CAM++.onnx
SPEAKER_MODEL_URL := https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_CAM++.onnx

# Silero VAD model (neural voice activity detection)
SILERO_MODEL := models/silero_vad.onnx
SILERO_MODEL_URL := https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx

# CGO flags for whisper
export CGO_ENABLED := 1
export C_INCLUDE_PATH := $(PWD)/$(WHISPER_INCLUDE):$(PWD)/$(GGML_INCLUDE)
export LIBRARY_PATH := $(PWD)/$(WHISPER_DIR)/build/src:$(PWD)/$(WHISPER_DIR)/build/ggml/src:$(PWD)/$(WHISPER_DIR)/build/ggml/src/ggml-blas
export MACOSX_DEPLOYMENT_TARGET := 15.0

#══════════════════════════════════════════════════════════════
# Setup
#══════════════════════════════════════════════════════════════

setup: setup-deps whisper-lib setup-whisper setup-speaker setup-silero
	@echo ""
	@echo "╔════════════════════════════════════════╗"
	@echo "║         Setup complete!                ║"
	@echo "╚════════════════════════════════════════╝"
	@echo ""
	@echo "Next steps:"
	@echo "  1. make enroll      - Enroll your voice"
	@echo "  2. make vox         - Start voice assistant"

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

# Download distil-large-v3 (6x faster, 756MB, ~1% WER increase)
setup-whisper-fast:
	@echo "Downloading fast Whisper model ($(WHISPER_MODEL_FAST))..."
	@mkdir -p models
	@if [ ! -f "models/ggml-$(WHISPER_MODEL_FAST).bin" ]; then \
		curl -L -o "models/ggml-$(WHISPER_MODEL_FAST).bin" \
			"https://huggingface.co/distil-whisper/distil-large-v3-ggml/resolve/main/ggml-distil-large-v3.bin"; \
		echo "Model downloaded: models/ggml-$(WHISPER_MODEL_FAST).bin"; \
	else \
		echo "Model already exists: models/ggml-$(WHISPER_MODEL_FAST).bin"; \
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

setup-silero:
	@echo "Downloading Silero VAD model..."
	@mkdir -p models
	@if [ ! -f "$(SILERO_MODEL)" ]; then \
		curl -L -o "$(SILERO_MODEL)" "$(SILERO_MODEL_URL)"; \
		echo "Silero VAD model downloaded: $(SILERO_MODEL)"; \
	else \
		echo "Silero VAD model already exists: $(SILERO_MODEL)"; \
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
# Speaker Enrollment
#══════════════════════════════════════════════════════════════

speaker-enroll:
	go run ./cmd/speaker enroll

speaker-list:
	go run ./cmd/speaker list

speaker-delete:
	@if [ -z "$(NAME)" ]; then \
		echo "Usage: make speaker-delete NAME=<profile-name>"; \
		exit 1; \
	fi
	go run ./cmd/speaker delete $(NAME)

# Aliases for convenience
enroll: speaker-enroll
profiles: speaker-list

#══════════════════════════════════════════════════════════════
# Vox Voice Module
#══════════════════════════════════════════════════════════════

vox: whisper-lib
	go run ./cmd/vox -mode streaming

# Fast mode: 6x faster using distil-large-v3 (requires: make setup-whisper-fast)
vox-fast: whisper-lib
	@if [ ! -f "models/ggml-$(WHISPER_MODEL_FAST).bin" ]; then \
		echo "Error: Fast model not found. Run 'make setup-whisper-fast' first."; \
		exit 1; \
	fi
	VOX_MODEL_PATH=models/ggml-$(WHISPER_MODEL_FAST).bin go run ./cmd/vox -mode streaming

vox-accurate: whisper-lib
	VOX_MODEL_PATH=models/ggml-$(WHISPER_MODEL).bin go run ./cmd/vox -mode accurate

vox-debug: whisper-lib
	go run ./cmd/vox -debug -mode streaming

#══════════════════════════════════════════════════════════════
# Build
#══════════════════════════════════════════════════════════════

build: whisper-lib
	@echo "Building..."
	@mkdir -p bin
	go build -o bin/vox ./cmd/vox
	go build -o bin/speaker ./cmd/speaker
	@echo "Built: bin/vox, bin/speaker"

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
	@echo "  make setup              - Install dependencies + download models"
	@echo "  make setup-whisper-fast - Download fast model (distil-large-v3, 6x faster)"
	@echo "  make enroll             - Enroll your voice (interactive)"
	@echo ""
	@echo "Voice Module:"
	@echo "  make vox          - Start Vox (streaming mode)"
	@echo "  make vox-fast     - Start Vox with distil model (6x faster)"
	@echo "  make vox-accurate - Start Vox (accurate mode, 5s chunks)"
	@echo "  make vox-debug    - Start with debug output + energy bars"
	@echo ""
	@echo "Models:"
	@echo "  large-v3        - Best accuracy (3GB) - default"
	@echo "  distil-large-v3 - 6x faster, ~1%% WER increase (756MB) - recommended"
	@echo ""
	@echo "Speaker Profiles:"
	@echo "  make enroll               - Enroll your voice (interactive)"
	@echo "  make profiles             - List enrolled speaker profiles"
	@echo "  make speaker-delete NAME=x - Delete a profile"
	@echo ""
	@echo "Build & Test:"
	@echo "  make build    - Build all binaries to ./bin/"
	@echo "  make test     - Run all tests"
	@echo "  make stt-test - Test STT accuracy"
	@echo ""
	@echo "Clean:"
	@echo "  make clean     - Remove binaries and models"
	@echo "  make clean-all - Remove everything including third_party/"
