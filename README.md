# Whisper Speech-to-Text Inference API

A high-performance API for real-time speech-to-text transcription using CTranslate2-backed Whisper models. Provides both batch and streaming transcription with GPU acceleration and intelligent batching.

## Features

- 🚀 **Ultra Low-Latency**: Optimized for real-time transcription with sub-second response times
- 📊 **Intelligent Batching**: Coalesces multiple requests for efficient GPU utilization
- 🔄 **Streaming Support**: WebSocket-based streaming transcription with VAD (Voice Activity Detection)
- 💾 **Model Caching**: Automatic model downloading and caching from HuggingFace
- 🐳 **Docker Ready**: Complete Dockerfile for easy deployment
- ⚡ **GPU Optimized**: Leverages CTranslate2 backend for maximum performance

## Architecture

- **Main API** (`main.py`): FastAPI application, loads the model once at startup and mounts routes
- **Transcription** (`transcribe.py`): Batch transcription endpoint for audio files
- **Streaming** (`streaming.py`): WebSocket streaming with VAD and real-time transcription
- **Model Manager** (`model_manager.py`): Handles model loading and caching
- **Batcher** (`batcher.py`): Coalesces requests for batch processing
- **Warmup** (`warmup.py`): Keeps the model warm for faster inference

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Docker (optional, for containerized deployment)

### Setup

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables in `.env`:
```
HUGGINGFACE_TOKEN=your_hf_token_here
MODEL_REPO_ID=./models/tara-ct2
```

`MODEL_REPO_ID` can be a HuggingFace repo id (downloaded and cached under `./models/`) or a path to a local CTranslate2 model directory.

If your model isn't already in CTranslate2 format, convert it once:
```bash
./scripts/convert_to_ct2.sh <hf-repo-id> ./models/model-ct2 float16
```

## Usage

### Starting the Server

```bash
sh setup.sh
```

### API Endpoints

#### Health Check
```bash
GET /health
```

Returns the health status of the API.

#### Batch Transcription
```bash
POST /transcribe
```

**Headers:**
- `language`: Language code (default: "fr")

**Body:**
- `audio_file`: Audio file (multipart/form-data)

**Response:**
```json
{
  "transcription": "Transcribed text here"
}
```

**Example:**
```bash
curl -X POST "http://localhost:8080/transcribe" \
  -H "language: fr" \
  -F "audio_file=@audio.wav"
```

#### Streaming Transcription
```bash
WS /stream
```

**Headers:**
- `sample-rate`: Audio sample rate (default: 8000)
- `vad-threshold`: VAD threshold (default: 0.3)
- `min-silence-duration`: Minimum silence duration in seconds (default: 0.3)
- `stt-interval-ms`: STT interval in milliseconds (default: 200)
- `language`: Language code (default: "fr")

**Message Format:**
- Send audio chunks as binary WebSocket messages
- Receive transcription events as JSON:
```json
{
  "type": "transcription",
  "text": "Transcribed text",
  "final": true,
  "silence_duration": 0.5,
  "avg_logprob": -0.2,
  "infer": 150,
  "stream_id": "ABC1"
}
```

## Performance Optimization

- **Coalescing Batching**: Groups up to 32 requests with a maximum delay of 5ms
- **Model Warming**: Automatically warms the model every 20 seconds to reduce cold start latency
- **VAD Integration**: Reduces unnecessary processing on silent audio

## Benchmarking latency & cost

- `python testing/test_charge.py` — concurrency sweep (edit `CONCURRENCY_LEVELS` env var), reports p50/mean/max latency and throughput per level.
- `python testing/monitor_gpu.py` — run alongside a load test to see GPU util/VRAM/power, tells you whether you're GPU-bound.

## Testing

Test scripts are available in the `testing/` directory:

- `test_gpu.py`: GPU performance testing
- `test_streaming.py`: Streaming functionality testing
- `test_streaming_random.py`: Streaming with randomized/synthetic audio
- `test_charge.py`: Concurrency/throughput testing
- `monitor_gpu.py`: GPU monitoring utilities
- `test_env_loader.py`: Unit test for `.env` loading

## Project Structure

```
inference/
├── optimized_api/
│   ├── main.py              # FastAPI application
│   ├── transcribe.py        # Batch transcription endpoint
│   ├── streaming.py         # WebSocket streaming endpoint
│   ├── model_manager.py     # Model loading and caching
│   ├── batcher.py           # Request batching logic
│   ├── warmup.py            # Model warming utilities
│   └── env_loader.py        # Minimal .env loader
├── testing/                 # Test scripts
├── scripts/                 # Model conversion helpers
├── dockerfile                # Docker configuration
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables
└── README.md                # This file
```

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Uses [whisper-s2t](https://github.com/shashikg/WhisperS2T/)
- Powered by [CTranslate2](https://github.com/OpenNMT/CTranslate2)
