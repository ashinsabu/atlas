# Atlas Makefile
#
# Quick reference:
#   make setup   - Install dependencies + download models
#   make vox     - Run the voice assistant

.PHONY: setup setup-deps setup-whisper setup-whisper-fast setup-silero whisper-lib vox vox-debug build test clean help

# Whisper models
WHISPER_MODEL ?= small
WHISPER_MODEL_FAST := small

# Whisper.cpp paths
WHISPER_DIR := third_party/whisper.cpp
WHISPER_LIB := $(WHISPER_DIR)/build/src/libwhisper.a
WHISPER_INCLUDE := $(WHISPER_DIR)/include
GGML_INCLUDE := $(WHISPER_DIR)/ggml/include

# Silero VAD model
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

setup: setup-deps whisper-lib setup-whisper setup-silero
	@echo ""
	@echo "Setup complete!"
	@echo ""
	@echo "Run: make vox"

setup-deps:
	@echo "Installing system dependencies..."
	@which brew > /dev/null || (echo "Error: Homebrew required" && exit 1)
	brew install portaudio cmake onnxruntime
	go mod download
	go mod tidy

setup-whisper:
	@echo "Downloading Whisper model ($(WHISPER_MODEL))..."
	@mkdir -p models
	@if [ ! -f "models/ggml-$(WHISPER_MODEL).bin" ]; then \
		curl -L -o "models/ggml-$(WHISPER_MODEL).bin" \
			"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-$(WHISPER_MODEL).bin"; \
	else \
		echo "Model exists: models/ggml-$(WHISPER_MODEL).bin"; \
	fi

setup-whisper-fast:
	@echo "Downloading fast Whisper model ($(WHISPER_MODEL_FAST))..."
	@mkdir -p models
	@if [ ! -f "models/ggml-$(WHISPER_MODEL_FAST).bin" ]; then \
		curl -L -o "models/ggml-$(WHISPER_MODEL_FAST).bin" \
			"https://huggingface.co/distil-whisper/distil-large-v3-ggml/resolve/main/ggml-distil-large-v3.bin"; \
	else \
		echo "Model exists: models/ggml-$(WHISPER_MODEL_FAST).bin"; \
	fi

setup-silero:
	@echo "Downloading Silero VAD model..."
	@mkdir -p models
	@if [ ! -f "$(SILERO_MODEL)" ]; then \
		curl -L -o "$(SILERO_MODEL)" "$(SILERO_MODEL_URL)"; \
	else \
		echo "Model exists: $(SILERO_MODEL)"; \
	fi

#══════════════════════════════════════════════════════════════
# Whisper.cpp Library
#══════════════════════════════════════════════════════════════

whisper-clone:
	@if [ ! -d "$(WHISPER_DIR)" ]; then \
		echo "Cloning whisper.cpp..."; \
		mkdir -p third_party; \
		git clone --depth 1 https://github.com/ggerganov/whisper.cpp.git $(WHISPER_DIR); \
	fi

whisper-lib: whisper-clone
	@if [ ! -f "$(WHISPER_LIB)" ]; then \
		echo "Building whisper.cpp..."; \
		cd $(WHISPER_DIR) && \
		cmake -B build \
			-DBUILD_SHARED_LIBS=OFF \
			-DWHISPER_BUILD_EXAMPLES=OFF \
			-DWHISPER_BUILD_TESTS=OFF \
			-DGGML_METAL=OFF \
			-DCMAKE_C_FLAGS="-I$$(pwd)/ggml/include" \
			-DCMAKE_CXX_FLAGS="-I$$(pwd)/ggml/include" && \
		cmake --build build --config Release -j$$(nproc 2>/dev/null || sysctl -n hw.ncpu); \
	fi

#══════════════════════════════════════════════════════════════
# Vox
#══════════════════════════════════════════════════════════════

WHISPER_MODEL_PATH ?= models/ggml-small.bin

vox: whisper-lib
	go run ./cmd/vox -whisper $(WHISPER_MODEL_PATH)

vox-debug: whisper-lib
	go run ./cmd/vox -debug -whisper $(WHISPER_MODEL_PATH)

#══════════════════════════════════════════════════════════════
# Build & Test
#══════════════════════════════════════════════════════════════

build: whisper-lib
	@mkdir -p bin
	go build -o bin/vox ./cmd/vox
	@echo "Built: bin/vox"

test: whisper-lib
	go test ./...

stt-test: whisper-lib
	go run ./cmd/stt-test -verbose

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
	@echo "Atlas Voice Assistant"
	@echo ""
	@echo "Setup:"
	@echo "  make setup       - Install dependencies + download models"
	@echo ""
	@echo "Run:"
	@echo "  make vox         - Start voice assistant"
	@echo "  make vox-debug   - Start with debug output"
	@echo ""
	@echo "Build:"
	@echo "  make build       - Build binary to ./bin/"
	@echo "  make test        - Run tests"
	@echo ""
	@echo "Clean:"
	@echo "  make clean       - Remove binaries and models"
	@echo "  make clean-all   - Remove everything"
