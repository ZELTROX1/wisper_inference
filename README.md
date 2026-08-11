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

### WebSocket stream
```
WS /stream
```

Headers (all optional):
- `sample-rate` (default 16000)
- `language` (default hi)
- `vad-threshold` (default 0.3)
- `min-silence-duration` (default 0.3)

Client sends raw **PCM16LE mono** binary frames.  
Server replies with finals:

```json
{
  "type": "transcription",
  "text": "...",
  "final": true,
  "silence_duration": 0.35,
  "avg_logprob": -0.15,
  "infer": 140,
  "stream_id": "AB12"
}
```

Send text `close` to hang up.

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
