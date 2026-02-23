module github.com/ashinsabu/atlas

go 1.24.2

require (
	github.com/ggerganov/whisper.cpp/bindings/go v0.0.0
	github.com/gordonklaus/portaudio v0.0.0-20260203164431-765aa7dfa631
)

require github.com/joho/godotenv v1.5.1

require github.com/yalue/onnxruntime_go v1.26.0

// Use local whisper.cpp bindings (cloned via make whisper-lib)
replace github.com/ggerganov/whisper.cpp/bindings/go => ./third_party/whisper.cpp/bindings/go
