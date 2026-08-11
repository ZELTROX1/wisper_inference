"""Tara faster-whisper STT — simple FastAPI service.

Endpoints
  GET  /health
  POST /transcribe
  WS   /stream   — client-VAD protocol (Pipecat-driven)

WebSocket protocol
------------------
  Client → binary PCM16LE mono @ sample-rate (default 16 kHz)
  Client → {"event":"finalize"}   end of utterance (VAD stop OR bot start)
  Client → {"event":"clear"}      drop buffer without decoding (rare)
  Client → "close"
  Server → {"type":"transcription","text":...,"final":true,...}
  Server → {"type":"error","error":...}
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
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

from app.model import (
    SAMPLE_RATE,
    collapse_repeats,
    is_repeat_loop,
    load_model,
    pcm16_to_wav,
    transcribe,
    warmup,
)

_executor = ThreadPoolExecutor(max_workers=int(os.getenv("WORKERS", "2")))
MAX_UTTERANCE_S = float(os.getenv("MAX_UTTERANCE_S", "12"))
MIN_AUDIO_S = float(os.getenv("MIN_AUDIO_S", "0.3"))
MIN_FINAL_CHARS = int(os.getenv("MIN_FINAL_CHARS", "2"))
LOGPROB_MIN = float(os.getenv("LOGPROB_THRESHOLD", "-0.55"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    # Warm CUDA kernels so first real finalize is not ~9s
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_executor, warmup)
    yield
    _executor.shutdown(wait=False)


app = FastAPI(title="Tara STT", version="1.2.0", lifespan=lifespan)


async def _run_asr(audio: bytes, language: str, sample_rate: int = SAMPLE_RATE) -> dict:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _executor, lambda: transcribe(audio, language=language, sample_rate=sample_rate)
    )


def _clean_text(text: str) -> str:
    text = text.replace("\ufffd", "").replace("�", "")
    text = re.sub(r"\s+", " ", text).strip()
    return collapse_repeats(text, max_run=2)


def _is_junk(text: str) -> bool:
    t = _clean_text(text)
    if len(t) < MIN_FINAL_CHARS:
        return True
    if re.fullmatch(r"[\W_]+", t, flags=re.UNICODE):
        return True
    words = t.split()
    if len(words) >= 4 and len(set(w.lower() for w in words)) == 1:
        return True
    if is_repeat_loop(t):
        return True
    low = t.lower()
    for s in ("subtitle", "sous-titrage", "thanks for watching", "www.", "subscribe"):
        if s in low:
            return True
    return False


@app.get("/health")
async def health():
    return {"status": "ok", "mode": "client_vad", "version": "1.2.0"}


@app.post("/transcribe")
async def http_transcribe(
    audio_file: UploadFile = File(...),
    language: str = Header("hi"),
):
    data = await audio_file.read()
    if not data:
        return {"transcription": "", "error": "empty file"}
    lang = (language or "hi").strip().lower().replace("_", "-").split("-", 1)[0]
    result = await _run_asr(data, lang)
    text = _clean_text(result["text"])
    if _is_junk(text):
        text = ""
    return {
        "transcription": text,
        "language": result["language"],
        "avg_logprob": result["avg_logprob"],
        "infer_ms": result["infer_ms"],
    }


@app.websocket("/stream")
async def stream(ws: WebSocket):
    """Buffer PCM until client sends finalize (Pipecat VAD / bot-start)."""
    await ws.accept()

    sid = "".join(random.choices(string.ascii_uppercase + string.digits, k=4))
    sample_rate = int(ws.headers.get("sample-rate", str(SAMPLE_RATE)))
    language = (ws.headers.get("language") or "hi").strip().lower()
    language = language.replace("_", "-").split("-", 1)[0]

    bytes_per_s = sample_rate * 2
    max_buf = int(MAX_UTTERANCE_S * bytes_per_s)
    min_bytes = int(MIN_AUDIO_S * bytes_per_s)

    buf = bytearray()
    lock = asyncio.Lock()
    # Serialize decodes so two finalizes don't interleave
    decode_lock = asyncio.Lock()

    print(f"[{sid}] open sr={sample_rate} lang={language} mode=client_vad")

    async def finalize_utterance(reason: str = "finalize") -> None:
        nonlocal buf
        async with decode_lock:
            async with lock:
                if len(buf) < min_bytes:
                    print(
                        f"[{sid}] {reason} skipped (too short: {len(buf)} bytes)"
                    )
                    buf.clear()
                    return
                snap = bytes(buf)
                buf.clear()

            wav = pcm16_to_wav(snap, sample_rate)
            t0 = time.time()
            try:
                r = await _run_asr(wav, language, sample_rate)
            except Exception as e:
                print(f"[{sid}] stt error: {e}")
                try:
                    await ws.send_text(json.dumps({"type": "error", "error": str(e)}))
                except Exception:
                    pass
                return

            text = _clean_text(r["text"])
            wall = (time.time() - t0) * 1000
            print(
                f"[{sid}] stt={text!r} lp={r['avg_logprob']:.3f} "
                f"infer={r['infer_ms']:.0f}ms wall={wall:.0f}ms "
                f"bytes={len(snap)} reason={reason}"
            )

            if _is_junk(text):
                print(f"[{sid}] junk dropped reason={reason}")
                return
            if r["avg_logprob"] < LOGPROB_MIN and not r["good_prob"]:
                print(f"[{sid}] low confidence dropped")
                return

            payload = {
                "type": "transcription",
                "text": text,
                "final": True,
                "silence_duration": 0.0,
                "avg_logprob": r["avg_logprob"],
                "infer": int(r["infer_ms"]),
                "stream_id": sid,
                "reason": reason,
            }
            print(f"[{sid}] FINAL {text!r}")
            await ws.send_text(json.dumps(payload))

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
                continue

            if msg.get("text") is not None:
                raw = (msg["text"] or "").strip()
                if not raw:
                    continue
                if raw.lower() == "close":
                    await ws.close()
                    break

                event = raw.lower()
                try:
                    obj = json.loads(raw)
                    if isinstance(obj, dict):
                        event = str(obj.get("event") or obj.get("type") or "").lower()
                except json.JSONDecodeError:
                    pass

                if event in ("finalize", "flush", "end"):
                    await finalize_utterance(reason=event)
                elif event in ("clear", "reset"):
                    async with lock:
                        n = len(buf)
                        buf.clear()
                    print(f"[{sid}] buffer cleared ({n} bytes dropped)")
                else:
                    print(f"[{sid}] ignore text: {raw[:80]!r}")
                continue

            if msg.get("type") == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[{sid}] error: {e}")
        try:
            await ws.send_text(json.dumps({"type": "error", "error": str(e)}))
            await ws.close(code=1011)
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
