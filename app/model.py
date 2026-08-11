"""Faster-whisper model + Tara mixed-code prompt."""

from __future__ import annotations

import io
import os
import time
import wave
from typing import List, Optional

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.tokenizer import Tokenizer

SAMPLE_RATE = 16_000
MIXEDCODE = "<|mixedcode|>"

_model: WhisperModel | None = None


def enable_mixedcode(model: WhisperModel) -> None:
    """Inject <|mixedcode|> after language token (Tara Hinglish mode)."""

    def get_prompt(
        tokenizer: Tokenizer,
        previous_tokens: List[int],
        without_timestamps: bool = False,
        prefix: Optional[str] = None,
        hotwords: Optional[str] = None,
    ) -> List[int]:
        prompt: List[int] = []

        if previous_tokens or (hotwords and not prefix):
            prompt.append(tokenizer.sot_prev)
            if hotwords and not prefix:
                toks = tokenizer.encode(" " + hotwords.strip())
                if len(toks) >= model.max_length // 2:
                    toks = toks[: model.max_length // 2 - 1]
                prompt.extend(toks)
            if previous_tokens:
                prompt.extend(previous_tokens[-(model.max_length // 2 - 1) :])

        prompt.append(tokenizer.sot)
        if tokenizer.language is not None:
            prompt.append(tokenizer.language)

        mc = tokenizer.tokenizer.token_to_id(MIXEDCODE)
        if mc is None:
            raise RuntimeError(
                f"{MIXEDCODE} missing — convert model with --copy_files tokenizer.json"
            )
        prompt.append(mc)

        if tokenizer.task is not None:
            prompt.append(tokenizer.task)
        if without_timestamps:
            prompt.append(tokenizer.no_timestamps)

        if prefix:
            toks = tokenizer.encode(" " + prefix.strip())
            if len(toks) >= model.max_length // 2:
                toks = toks[: model.max_length // 2 - 1]
            if not without_timestamps:
                prompt.append(tokenizer.timestamp_begin)
            prompt.extend(toks)

        return prompt

    model.get_prompt = get_prompt  # type: ignore[method-assign]


def load_model() -> WhisperModel:
    global _model
    if _model is not None:
        return _model

    path = os.getenv("MODEL_PATH", "./models/tara-ct2")
    device = os.getenv("DEVICE", "cuda")
    compute_type = os.getenv("COMPUTE_TYPE", "float16")
    mixed = os.getenv("MIXED_CODE", "true").lower() in ("1", "true", "yes")

    print(f"Loading {path} device={device} compute_type={compute_type}")
    model = WhisperModel(path, device=device, compute_type=compute_type)
    if mixed:
        enable_mixedcode(model)
        print("mixed-code ON")
    _model = model
    print("model ready")
    return model


def get_model() -> WhisperModel:
    if _model is None:
        return load_model()
    return _model


def _to_float32(audio: bytes, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    if len(audio) >= 12 and audio[:4] == b"RIFF" and audio[8:12] == b"WAVE":
        with wave.open(io.BytesIO(audio), "rb") as wf:
            ch, width, rate = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
            frames = wf.readframes(wf.getnframes())
        if width != 2:
            raise ValueError("only 16-bit PCM supported")
        x = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if ch > 1:
            x = x.reshape(-1, ch).mean(axis=1)
        if rate != sample_rate and x.size:
            n = int(x.shape[0] * sample_rate / rate)
            x = np.interp(
                np.linspace(0, 1, n, endpoint=False),
                np.linspace(0, 1, x.shape[0], endpoint=False),
                x,
            ).astype(np.float32)
        return x
    return np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0


def pcm16_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def transcribe(
    audio: bytes,
    language: str = "hi",
    sample_rate: int = SAMPLE_RATE,
) -> dict:
    """Run ASR. `audio` = WAV bytes or raw PCM16LE mono."""
    model = get_model()
    wave_f = _to_float32(audio, sample_rate)
    if wave_f.size == 0:
        return {
            "text": "",
            "language": language,
            "avg_logprob": 0.0,
            "infer_ms": 0.0,
            "good_prob": False,
        }

    t0 = time.time()
    segments, _info = model.transcribe(
        wave_f,
        language=language or "hi",
        task="transcribe",
        beam_size=int(os.getenv("BEAM_SIZE", "5")),
        word_timestamps=True,
        without_timestamps=True,
        condition_on_previous_text=False,
        vad_filter=False,
    )
    segs = list(segments)
    infer_ms = (time.time() - t0) * 1000

    texts, lps = [], []
    good = False
    for s in segs:
        t = (s.text or "").strip()
        if t:
            texts.append(t)
        if s.avg_logprob is not None:
            lps.append(float(s.avg_logprob))
        if s.words:
            good = good or any(
                w.probability is not None and float(w.probability) > 0.8 for w in s.words
            )

    avg = float(sum(lps) / len(lps)) if lps else 0.0
    return {
        "text": " ".join(texts).strip(),
        "language": language,
        "avg_logprob": avg,
        "infer_ms": infer_ms,
        "good_prob": good or (avg > -0.5 and bool(texts)),
    }
