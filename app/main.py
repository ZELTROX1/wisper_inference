"""Tara faster-whisper STT — simple FastAPI service.

Endpoints
  GET  /health
  POST /transcribe          multipart audio file
  WS   /stream              raw PCM16LE + VAD finals (for agents)

Run
  uvicorn app.main:app --host 0.0.0.0 --port 8080
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

import numpy as np
import torch
from fastapi import FastAPI, File, Header, UploadFile, WebSocket, WebSocketDisconnect
from silero_vad import load_silero_vad

# load .env if present (no python-dotenv dependency)
_env = Path(__file__).resolve().parents[1] / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip())

from app.model import SAMPLE_RATE, load_model, pcm16_to_wav, transcribe

_executor = ThreadPoolExecutor(max_workers=int(os.getenv("WORKERS", "2")))

# Streaming tuning (can override via env)
MAX_UTTERANCE_S = float(os.getenv("MAX_UTTERANCE_S", "12"))  # hard cap buffer
MIN_SPEECH_S = float(os.getenv("MIN_SPEECH_S", "0.35"))  # ignore blips
MIN_FINAL_CHARS = int(os.getenv("MIN_FINAL_CHARS", "2"))
LOGPROB_MIN = float(os.getenv("LOGPROB_THRESHOLD", "-0.45"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    _executor.shutdown(wait=False)


app = FastAPI(title="Tara STT", version="1.0.1", lifespan=lifespan)

_vad = load_silero_vad()
_vad.eval()
_vad_device = "cuda" if torch.cuda.is_available() else "cpu"
_vad.to(_vad_device)


def _vad_prob(chunk: bytes, sample_rate: int) -> float:
    if not chunk or len(chunk) < 2:
        return 0.0
    need = 256 if sample_rate == 8000 else 512
    x = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
    if x.size == 0:
        return 0.0
    if x.size < need:
        p = np.zeros(need, dtype=np.float32)
        p[: x.size] = x
        x = p
    else:
        x = x[-need:]
    with torch.inference_mode():
        t = torch.from_numpy(x).unsqueeze(0).to(_vad_device)
        return float(_vad(t, sample_rate).item())


async def _run_asr(audio: bytes, language: str, sample_rate: int = SAMPLE_RATE) -> dict:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _executor, lambda: transcribe(audio, language=language, sample_rate=sample_rate)
    )


def _clean_text(text: str) -> str:
    # drop replacement chars and collapse whitespace
    text = text.replace("\ufffd", "").replace("�", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_junk(text: str) -> bool:
    """Filter Whisper hallucinations / empty noise."""
    t = _clean_text(text)
    if len(t) < MIN_FINAL_CHARS:
        return True
    # pure punctuation
    if re.fullmatch(r"[\W_]+", t, flags=re.UNICODE):
        return True
    # repeated single word 4+ times (हाँ हाँ हाँ हाँ)
    words = t.split()
    if len(words) >= 4 and len(set(w.lower() for w in words)) == 1:
        return True
    low = t.lower()
    for s in (
        "subtitle",
        "sous-titrage",
        "thanks for watching",
        "www.",
        "subscribe",
    ):
        if s in low:
            return True
    return False


# ---------- HTTP ----------


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/transcribe")
async def http_transcribe(
    audio_file: UploadFile = File(...),
    language: str = Header("hi"),
):
    data = await audio_file.read()
    if not data:
        return {"text": "", "error": "empty file"}
    lang = (language or "hi").strip().lower().replace("_", "-").split("-", 1)[0]
    result = await _run_asr(data, lang)
    text = _clean_text(result["text"])
    return {
        "transcription": text,
        "language": result["language"],
        "avg_logprob": result["avg_logprob"],
        "infer_ms": result["infer_ms"],
    }


# ---------- WebSocket streaming ----------


@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()

    sid = "".join(random.choices(string.ascii_uppercase + string.digits, k=4))
    sample_rate = int(ws.headers.get("sample-rate", str(SAMPLE_RATE)))
    vad_threshold = float(ws.headers.get("vad-threshold", "0.45"))
    min_silence = float(ws.headers.get("min-silence-duration", "0.4"))
    language = (ws.headers.get("language") or "hi").strip().lower()
    language = language.replace("_", "-").split("-", 1)[0]

    bytes_per_s = sample_rate * 2  # PCM16 mono
    max_buf = int(MAX_UTTERANCE_S * bytes_per_s)
    min_speech_bytes = int(MIN_SPEECH_S * bytes_per_s)

    buf = bytearray()
    speech_bytes = 0  # speech-only counter (for min length)
    last_silence: float | None = None
    have_speech = False
    last_text = ""
    last_lp: float | None = None
    last_good = False
    last_infer = 0
    stt_task: asyncio.Task | None = None
    stt_lock = asyncio.Lock()

    print(
        f"[{sid}] open sr={sample_rate} lang={language} "
        f"vad={vad_threshold} min_sil={min_silence}"
    )

    def _trim_buf() -> None:
        nonlocal buf
        if len(buf) > max_buf:
            # keep last MAX_UTTERANCE_S only
            buf = bytearray(buf[-max_buf:])

    async def do_stt():
        nonlocal last_text, last_lp, last_good, last_infer
        async with stt_lock:
            if len(buf) < min_speech_bytes:
                return
            snap = bytes(buf)
            wav = pcm16_to_wav(snap, sample_rate)
            try:
                r = await _run_asr(wav, language, sample_rate)
                text = _clean_text(r["text"])
                if _is_junk(text):
                    print(f"[{sid}] stt junk dropped={text!r} lp={r['avg_logprob']:.3f}")
                    last_text = ""
                    last_lp = None
                    last_good = False
                    return
                last_text = text
                last_lp = r["avg_logprob"]
                last_good = bool(r["good_prob"]) and last_lp is not None and last_lp > LOGPROB_MIN
                last_infer = int(r["infer_ms"])
                print(
                    f"[{sid}] stt={last_text!r} lp={last_lp:.3f} "
                    f"good={last_good} infer={last_infer}ms bytes={len(snap)}"
                )
            except Exception as e:
                print(f"[{sid}] stt error: {e}")

    async def maybe_final():
        nonlocal last_text, buf, last_silence, have_speech, last_lp, last_good, speech_bytes
        if not last_text.strip() or last_lp is None or last_silence is None:
            return
        silence = time.time() - last_silence
        if silence < min_silence:
            return
        if speech_bytes < min_speech_bytes:
            return
        if not last_good or last_lp <= LOGPROB_MIN:
            return
        if _is_junk(last_text):
            return

        if stt_task and not stt_task.done():
            await stt_task

        # re-check after awaiting in-flight STT
        if not last_text.strip() or last_lp is None or _is_junk(last_text):
            return

        payload = {
            "type": "transcription",
            "text": last_text,
            "final": True,
            "silence_duration": silence,
            "avg_logprob": last_lp,
            "infer": last_infer,
            "stream_id": sid,
        }
        print(f"[{sid}] FINAL {last_text!r}")
        await ws.send_text(json.dumps(payload))

        last_text = ""
        buf.clear()
        speech_bytes = 0
        last_silence = None
        have_speech = False
        last_lp = None
        last_good = False

    try:
        while True:
            msg = await ws.receive()
            if msg.get("bytes"):
                chunk = msg["bytes"]
                buf.extend(chunk)
                _trim_buf()

                prob = _vad_prob(chunk, sample_rate)
                speaking = prob > vad_threshold

                if speaking:
                    speech_bytes += len(chunk)
                    if last_silence is not None:
                        print(f"[{sid}] speech (vad={prob:.2f})")
                    last_silence = None
                    have_speech = True
                else:
                    if last_silence is None and have_speech:
                        last_silence = time.time()
                        print(f"[{sid}] silence start (vad={prob:.2f})")
                        # kick STT once at silence onset
                        if speech_bytes >= min_speech_bytes:
                            if not stt_task or stt_task.done():
                                stt_task = asyncio.create_task(do_stt())
                    if last_silence is not None:
                        await maybe_final()

            elif msg.get("text"):
                if msg["text"].strip().lower() == "close":
                    await ws.close()
                    break
            elif msg.get("type") == "websocket.disconnect":
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
