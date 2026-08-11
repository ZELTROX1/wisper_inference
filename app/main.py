"""Tara STT HTTP API — decode path matched to Pipecat Whisper.

Primary API (used by SegmentedSTT plugin)
-----------------------------------------
  POST /transcribe
    multipart: audio_file (WAV or raw PCM16 from SegmentedSTTService)
    header:    language: hi

This mirrors Pipecat ``WhisperSTTService.run_stt``: one clip → one text.
Pipeline VAD (SegmentedSTTService) owns utterance boundaries; this process
only runs faster-whisper.

Optional legacy WebSocket ``/stream`` remains for older clients (buffer +
finalize). Prefer HTTP for the agent plugin.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import string
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Header, UploadFile, WebSocket, WebSocketDisconnect

_env = Path(__file__).resolve().parents[1] / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip())

from app.model import SAMPLE_RATE, load_model, transcribe, warmup

_executor = ThreadPoolExecutor(max_workers=int(os.getenv("WORKERS", "2")))
MAX_UTTERANCE_S = float(os.getenv("MAX_UTTERANCE_S", "30"))
MIN_AUDIO_S = float(os.getenv("MIN_AUDIO_S", "0.2"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_executor, warmup)
    yield
    _executor.shutdown(wait=False)


app = FastAPI(
    title="Tara STT",
    version="1.4.0",
    description="Pipecat-aligned faster-whisper API for Trelis/tara",
)


async def _run_asr(audio: bytes, language: str) -> dict:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _executor, lambda: transcribe(audio, language=language)
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "1.4.0",
        "backend": "faster-whisper",
        "decode": "pipecat_whisper_style",
        "mixed_code": os.getenv("MIXED_CODE", "true"),
    }


@app.post("/transcribe")
async def http_transcribe(
    audio_file: UploadFile = File(...),
    language: str = Header("hi"),
):
    """One VAD segment → one transcript (Pipecat Whisper equivalent)."""
    data = await audio_file.read()
    if not data:
        return {"transcription": "", "language": language or "hi", "infer_ms": 0.0}

    lang = (language or "hi").strip().lower().replace("_", "-").split("-", 1)[0]
    t0 = time.time()
    result = await _run_asr(data, lang)
    wall = (time.time() - t0) * 1000

    text = (result.get("text") or "").strip()
    print(
        f"HTTP /transcribe lang={lang} text={text!r} "
        f"infer={result.get('infer_ms', 0):.0f}ms wall={wall:.0f}ms "
        f"bytes={len(data)}"
    )
    return {
        "transcription": text,
        "language": result.get("language", lang),
        "avg_logprob": result.get("avg_logprob", 0.0),
        "infer_ms": result.get("infer_ms", 0.0),
    }


# --- Legacy WS (optional; agent plugin uses HTTP) ---


@app.websocket("/stream")
async def stream(ws: WebSocket):
    """Legacy continuous buffer + finalize protocol."""
    await ws.accept()
    sid = "".join(random.choices(string.ascii_uppercase + string.digits, k=4))
    sample_rate = int(ws.headers.get("sample-rate", str(SAMPLE_RATE)))
    language = (ws.headers.get("language") or "hi").strip().lower()
    language = language.replace("_", "-").split("-", 1)[0]

    max_buf = int(MAX_UTTERANCE_S * sample_rate * 2)
    min_bytes = int(MIN_AUDIO_S * sample_rate * 2)
    buf = bytearray()
    lock = asyncio.Lock()
    decode_lock = asyncio.Lock()

    print(f"[{sid}] open sr={sample_rate} lang={language} mode=legacy_ws")

    async def finalize_utterance(reason: str = "finalize") -> None:
        nonlocal buf
        async with decode_lock:
            async with lock:
                if len(buf) < min_bytes:
                    print(f"[{sid}] {reason} skip short buf={len(buf)}")
                    buf.clear()
                    return
                snap = bytes(buf)
                buf.clear()

            # Segmented path sends WAV; WS path sends raw PCM → wrap as WAV-equivalent
            # by decoding as raw float inside transcribe (raw PCM branch).
            try:
                r = await _run_asr(snap, language)
            except Exception as e:
                await ws.send_text(json.dumps({"type": "error", "error": str(e)}))
                return

            text = (r.get("text") or "").strip()
            print(
                f"[{sid}] stt={text!r} infer={r.get('infer_ms', 0):.0f}ms "
                f"bytes={len(snap)} reason={reason}"
            )
            if not text:
                return
            await ws.send_text(
                json.dumps(
                    {
                        "type": "transcription",
                        "text": text,
                        "final": True,
                        "avg_logprob": r.get("avg_logprob", 0.0),
                        "infer": int(r.get("infer_ms") or 0),
                        "stream_id": sid,
                    }
                )
            )

    try:
        while True:
            msg = await ws.receive()
            if msg.get("bytes") is not None:
                chunk = msg["bytes"] or b""
                if not chunk:
                    continue
                async with lock:
                    buf.extend(chunk)
                    if len(buf) > max_buf:
                        buf[:] = buf[-max_buf:]
            elif msg.get("text") is not None:
                raw = (msg["text"] or "").strip()
                if raw.lower() == "close":
                    await ws.close()
                    break
                event = raw.lower()
                try:
                    obj = json.loads(raw)
                    if isinstance(obj, dict):
                        event = str(obj.get("event") or "").lower()
                except json.JSONDecodeError:
                    pass
                if event in ("finalize", "flush", "end"):
                    await finalize_utterance(event)
                elif event in ("clear", "reset"):
                    async with lock:
                        buf.clear()
            elif msg.get("type") == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"type": "error", "error": str(e)}))
        except Exception:
            pass
    finally:
        print(f"[{sid}] closed")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8080")),
        reload=False,
    )
