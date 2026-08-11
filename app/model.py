"""Tara faster-whisper inference — aligned with Pipecat WhisperSTTService.

Pipecat source of truth
-----------------------
``pipecat.services.whisper.stt.WhisperSTTService.run_stt``:

    audio_float = pcm_i16.astype(float32) / 32768.0
    segments, _ = model.transcribe(audio_float, language=language)
    for segment in segments:
        if segment.no_speech_prob < no_speech_prob:   # default 0.4
            text += segment.text

That is intentionally minimal: no beam/penalty maze. Pipeline VAD
(SegmentedSTTService) already cuts the utterance; the model only decodes
one clean clip.

Tara-only extras
----------------
  * load CT2 path from MODEL_PATH
  * inject ``<|mixedcode|>`` for Hinglish (optional, MIXED_CODE=true)
  * strip known junk tails (दीनदयाल / कौशल योजना) so real place names stay
"""

from __future__ import annotations

import io
import os
import re
import time
import wave
from typing import List, Optional

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.tokenizer import Tokenizer

SAMPLE_RATE = 16_000
MIXEDCODE = "<|mixedcode|>"

# Pipecat WhisperSTTSettings default
DEFAULT_NO_SPEECH_PROB = 0.4

_model: WhisperModel | None = None
_warmed = False

# Junk often appended after real speech on noisy telephony (not on clean HF demos).
_HALLUCINATION_MARKERS = (
    "प्रतिक्रिया के लिए संस्कृति",
    "दीनदयाल",
    "दीन दयाल",
    "ग्रामीण कौशल",
    "कौशल योजना",
    "विशिष्टता को समर्थन",
    "subscribe",
    "thanks for watching",
    "sous-titrage",
    "www.",
)


def enable_mixedcode(model: WhisperModel) -> None:
    """Tara mixed-code: sot → language → <|mixedcode|> → task → …"""

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
                f"{MIXEDCODE} missing — convert with --copy_files tokenizer.json"
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
    # Same constructor style as Pipecat WhisperSTTService._load
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


def warmup() -> None:
    global _warmed
    if _warmed:
        return
    silence = np.zeros(int(SAMPLE_RATE * 0.5), dtype=np.float32)
    model = get_model()
    t0 = time.time()
    # Match Pipecat call shape: transcribe(float, language=...)
    list(model.transcribe(silence, language="hi")[0])
    _warmed = True
    print(f"warmup done in {(time.time() - t0) * 1000:.0f}ms")


def audio_bytes_to_float32(audio: bytes, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """WAV or raw PCM16LE → float32 [-1, 1], same normalization as Pipecat Whisper."""
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
            n = max(1, int(x.shape[0] * sample_rate / rate))
            x = np.interp(
                np.linspace(0, 1, n, endpoint=False),
                np.linspace(0, 1, x.shape[0], endpoint=False),
                x,
            ).astype(np.float32)
        return x
    # Raw PCM (Pipecat SegmentedSTTService can also pass raw in some paths)
    return np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0


def strip_hallucination_tail(text: str) -> str:
    """Keep real speech; cut known junk appended after it (telephony / noise)."""
    if not text:
        return text
    cut_at = len(text)
    low = text.lower()
    for marker in _HALLUCINATION_MARKERS:
        idx = low.find(marker.lower())
        if idx >= 0:
            cut_at = min(cut_at, idx)
    if cut_at < len(text):
        kept = text[:cut_at].rstrip(" ।.,;:-–— \t")
        print(f"hallucination tail stripped → kept={kept!r}")
        return kept.strip()
    return text


def _empty(language: str, infer_ms: float = 0.0) -> dict:
    return {
        "text": "",
        "language": language,
        "avg_logprob": 0.0,
        "infer_ms": infer_ms,
        "good_prob": False,
        "no_speech": True,
    }


def transcribe(
    audio: bytes,
    language: str = "hi",
    sample_rate: int = SAMPLE_RATE,
    no_speech_prob: float | None = None,
) -> dict:
    """Decode one VAD segment the way Pipecat WhisperSTTService does.

    Pipecat:
        segments, _ = model.transcribe(audio_float, language=language)
        keep segment if segment.no_speech_prob < no_speech_prob  (default 0.4)
    """
    model = get_model()
    audio_float = audio_bytes_to_float32(audio, sample_rate)
    if audio_float.size == 0:
        return _empty(language)

    nsp_threshold = (
        no_speech_prob
        if no_speech_prob is not None
        else float(os.getenv("NO_SPEECH_PROB", str(DEFAULT_NO_SPEECH_PROB)))
    )

    t0 = time.time()
    # Minimal call — same as Pipecat (language only; faster-whisper defaults for the rest)
    segments, _info = model.transcribe(audio_float, language=language or "hi")
    segs = list(segments)
    infer_ms = (time.time() - t0) * 1000

    # Exact Pipecat filter: keep segment when no_speech_prob is *below* threshold
    text_parts: list[str] = []
    logprobs: list[float] = []
    for segment in segs:
        if segment.no_speech_prob < nsp_threshold:
            part = (segment.text or "").strip()
            if part:
                text_parts.append(part)
            if segment.avg_logprob is not None:
                logprobs.append(float(segment.avg_logprob))

    text = " ".join(text_parts).strip()
    text = re.sub(r"\s+", " ", text.replace("\ufffd", "").replace("�", "")).strip()
    text = strip_hallucination_tail(text)

    if not text:
        return _empty(language, infer_ms)

    avg = float(sum(logprobs) / len(logprobs)) if logprobs else 0.0
    return {
        "text": text,
        "language": language,
        "avg_logprob": avg,
        "infer_ms": infer_ms,
        "good_prob": True,
        "no_speech": False,
    }
