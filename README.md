# Finetuned Whisper Large V3 Turbo - Multiples models inference API

A high-performance, production-ready API for real-time speech-to-text transcription using custom Whisper models. This system provides both batch and streaming transcription capabilities with GPU acceleration, intelligent batching, and quota management.
This system can handle multiples models on the same GPU.

### Response Time (Latency)
| STT Service | Average Latency | Improvement vs Latice |
|------------|----------------|---------------------|
| Latice STT | 0.360s | Reference |
| Competitors | 0.409s - 1.553s | +13% to +332% slower |


## Features

- 🚀 **Ultra Low-Latency**: Optimized for real-time transcription with sub-second response times
- 📊 **Intelligent Batching**: Coalesces multiple requests for efficient GPU utilization
- 🔄 **Streaming Support**: WebSocket-based streaming transcription with VAD (Voice Activity Detection)
- 💾 **Model Caching**: Automatic model downloading and caching from HuggingFace
- 📈 **Quota Management**: Built-in quota tracking and usage monitoring
- 🐳 **Docker Ready**: Complete Dockerfile for easy deployment
- ⚡ **GPU Optimized**: Leverages CTranslate2 backend for maximum performance

## Architecture

The system consists of several key components:

- **Main API** (`main.py`): FastAPI application with health checks and route mounting
- **Transcription** (`transcribe.py`): Batch transcription endpoint for audio files
- **Streaming** (`streaming.py`): WebSocket streaming with VAD and real-time transcription
- **Model Manager** (`model_manager.py`): Handles model loading and caching
- **Batcher** (`batcher.py`): Coalesces requests for batch processing
- **Quota Manager** (`quota_manager.py`): Manages user quotas and usage tracking
- **Warmup** (`warmup.py`): Keeps models warm for faster inference

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Docker (optional, for containerized deployment)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd deploy_instance
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install whisper-s2t:
```bash
pip install git+https://github.com/LATICE-AI/whisper-s2t.git
```

5. Configure environment variables:
```bash
cp env.example .env
# Edit .env with your configuration
```

## Running Trelis/tara standalone (no Latice backend)

This repo originally called Latice's own SaaS backend (`app.latice.ai`) to map
`api-key`/`model-id` headers to a HuggingFace repo and to check quota. That backend
is gone. `LOCAL_MODE=true` (default, see `env.example`) skips it entirely: any
api-key/model-id is accepted, quota is unlimited, and every request is served by
`LOCAL_MODEL_REPO_ID`.

`Trelis/tara` ships as HF Transformers safetensors, not the CTranslate2 format
this API loads. Convert once:

```bash
./scripts/convert_to_ct2.sh Trelis/tara ./models/tara-ct2 float16
```

Then set in `.env`:
```
LOCAL_MODE=true
LOCAL_MODEL_REPO_ID=./models/tara-ct2
```

`sh setup.sh` as usual. `/transcribe` and `/stream` work with any api-key/model-id header value.

## Benchmarking latency & cost

- `python testing/test_charge.py` — concurrency sweep (edit `CONCURRENCY_LEVELS` env var), reports p50/mean/max latency and throughput per level.
- `python testing/monitor_gpu.py` — run alongside a load test to see GPU util/VRAM/power, tells you whether you're GPU-bound.
- GPU rental cost per audio-hour = `(hourly $ rate) / (concurrent streams sustainable at real-time) `. Read it off `test_charge.py`'s throughput at the concurrency where `monitor_gpu.py` shows GPU util >90%; below that you're leaving throughput on the table.

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
- `api-key`: Your API key
- `model-id`: Model identifier
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
  -H "api-key: your-api-key" \
  -H "model-id: your-model-id" \
  -H "language: fr" \
  -F "audio_file=@audio.wav"
```

#### Streaming Transcription
```bash
WS /stream
```

**Headers:**
- `api-key`: Your API key
- `model-id`: Model identifier
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

The system includes several optimizations:

- **Coalescing Batching**: Groups up to 32 requests with a maximum delay of 5ms
- **Model Warming**: Automatically warms models every 20 seconds to reduce cold start latency
- **GPU Quota Management**: Efficient GPU resource allocation
- **VAD Integration**: Reduces unnecessary processing on silent audio

## Testing

Test scripts are available in the `testing/` directory:

- `test_gpu.py`: GPU performance testing
- `test_streaming.py`: Streaming functionality testing
- `test_concurrent.py`: Concurrent request testing
- `monitor_gpu.py`: GPU monitoring utilities

## Project Structure

```
deploy_instance/
├── optimized_api/
│   ├── main.py              # FastAPI application
│   ├── transcribe.py        # Batch transcription endpoint
│   ├── streaming.py         # WebSocket streaming endpoint
│   ├── model_manager.py     # Model loading and caching
│   ├── batcher.py           # Request batching logic
│   ├── quota_manager.py     # Quota and usage management
│   └── warmup.py            # Model warming utilities
├── testing/                 # Test scripts
├── dockerfile               # Docker configuration
├── requirements.txt         # Python dependencies
├── env.example             # Environment variables template
└── README.md               # This file
```

## Author

Antoine Marcel (https://www.linkedin.com/in/antoine-marcel/)

Michael Charhon (https://www.linkedin.com/in/micha%C3%ABl-charhon/)

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Uses [whisper-s2t](https://github.com/shashikg/WhisperS2T/)
- Powered by [CTranslate2](https://github.com/OpenNMT/CTranslate2)