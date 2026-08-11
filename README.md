# Tara STT (faster-whisper)

Simple FastAPI server for **Trelis/tara** Hindi/Hinglish ASR.

## Setup

```bash
# 1) convert model once
./scripts/convert_to_ct2.sh Trelis/tara ./models/tara-ct2 float16

# 2) install
pip install -r requirements.txt

# 3) run
export MODEL_PATH=./models/tara-ct2
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## API

### Health
```bash
curl http://localhost:8080/health
```

### HTTP transcript
```bash
curl -X POST http://localhost:8080/transcribe \
  -H "language: hi" \
  -F "audio_file=@clip.wav"
```

```json
{
  "transcription": "...",
  "language": "hi",
  "avg_logprob": -0.1,
  "infer_ms": 120.5
}
```

### WebSocket stream (client VAD — Pipecat owns endpoints)
```
WS /stream
```

Headers: `sample-rate` (16000), `language` (hi)

```
Client → binary PCM16LE mono
Client → {"event":"finalize"}   # Pipecat VAD stop OR bot start (flush residual)
Client → {"event":"clear"}      # rare hard reset only — do not use on bot start
Server → {"type":"transcription","text":"...","final":true,...}
Client → "close"
```

No live server VAD. Pipecat owns endpoints. On bot TTS start the plugin **finalizes** residual user speech (never clear-drops it). Startup runs a silent warmup so first decode is not cold (~9s).

## Agents (techladder)

```bash
STT_PROVIDER=tara
TARA_STT_WS_URL=ws://YOUR_HOST:8080/stream
TARA_STT_API_KEY=x
TARA_STT_MODEL_ID=x
TARA_STT_LANGUAGE=hi
```

(Service ignores api-key; plugin still sends headers.)

## Env

| Name | Default | Meaning |
|------|---------|---------|
| `MODEL_PATH` | `./models/tara-ct2` | CT2 model folder |
| `DEVICE` | `cuda` | cuda / cpu |
| `COMPUTE_TYPE` | `float16` | float16 / int8 |
| `MIXED_CODE` | `true` | Tara Hinglish mode |
| `PORT` | `8080` | listen port |

## Layout

```
app/
  main.py    # FastAPI + HTTP + WebSocket
  model.py   # faster-whisper + mixed-code
scripts/convert_to_ct2.sh
```
