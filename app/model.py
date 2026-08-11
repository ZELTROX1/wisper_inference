"""Tara faster-whisper inference for SegmentedSTT (Pipecat Whisper pattern).

Flow (same as Pipecat WhisperSTTService + SegmentedSTTService)
--------------------------------------------------------------
  Agent buffers one VAD utterance (with ~1s pre-roll) → POST /transcribe
  Server: float32 = pcm / 32768 → model.transcribe(...) → filter segments

Pipecat calls ``transcribe(audio, language=...)`` with defaults only. That is
fine on clean desktop mics; telephony needs a few explicit safer defaults
(condition_on_previous_text=False, temperature=0, clip VAD) or quality
collapses into hallucinations like "coworking / कौशल योजना / ?".

Tara-only: optional ``<|mixedcode|>`` prompt + strip known junk tails.
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

# Pipecat WhisperSTTSettings default: keep segment if no_speech_prob < this
DEFAULT_NO_SPEECH_PROB = float(os.getenv("NO_SPEECH_PROB", "0.4"))

_model: WhisperModel | None = None
_warmed = False

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
    "coworking",
    "працювати",  # random non-HI/EN garbage seen in logs
)


def enable_mixedcode(model: WhisperModel) -> None:
    """Tara: sot → lang → <|mixedcode|> → task → notimestamps."""

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

    print(f"Loading {path} device={device} compute_type={compute_type}", flush=True)
    model = WhisperModel(path, device=device, compute_type=compute_type)
    if mixed:
        enable_mixedcode(model)
        print("mixed-code ON", flush=True)
    _model = model
    print("model ready", flush=True)
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
    list(
        model.transcribe(
            silence,
            language="hi",
            beam_size=1,
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=False,
        )[0]
    )
    _warmed = True
    print(f"warmup done in {(time.time() - t0) * 1000:.0f}ms", flush=True)


def audio_bytes_to_float32(audio: bytes, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """WAV or raw PCM16LE → float32 [-1, 1] (Pipecat: i16 / 32768)."""
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
    return np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0


def strip_hallucination_tail(text: str) -> str:
    if not text:
        return text
    cut_at = len(text)
    low = text.lower()
    for marker in _HALLUCINATION_MARKERS:
        idx = low.find(marker.lower())
        if idx >= 0:
            cut_at = min(cut_at, idx)
    if cut_at < len(text):
        kept = text[:cut_at].rstrip(" ।.,;:-–— \t?")
        print(f"hallucination tail stripped → kept={kept!r}", flush=True)
        return kept.strip()
    return text


def is_useless_text(text: str) -> bool:
    """Drop punctuation-only noise. Single words (हाँ / yes / ok) are valid."""
    t = text.strip()
    if not t:
        return True
    if re.fullmatch(r"[\W_]+", t, flags=re.UNICODE):
        return True
    return False


def _collect_text(
    segs,
    nsp_threshold: float,
) -> tuple[str, list[float]]:
    """Pipecat rule: keep segment when no_speech_prob < threshold."""
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
    return text, logprobs


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
    """One SegmentedSTT utterance → text (Pipecat Whisper + telephony-safe opts)."""
    model = get_model()
    audio_float = audio_bytes_to_float32(audio, sample_rate)
    if audio_float.size == 0:
        return _empty(language)

    duration_s = float(audio_float.size) / float(sample_rate)
    nsp_threshold = (
        no_speech_prob if no_speech_prob is not None else DEFAULT_NO_SPEECH_PROB
    )

    # Short utterances (single word / yes-no): clip VAD often eats the word
    # inside the ~1s SegmentedSTT pre-roll. Prefer no clip VAD on short clips.
    short_utt = duration_s < 2.5
    env_vad = os.getenv("CLIP_VAD_FILTER", "true").lower() in ("1", "true", "yes")
    use_clip_vad = env_vad and not short_utt
    # Single-word clips often score no_speech_prob in 0.4–0.7; loosen filter.
    if short_utt:
        nsp_threshold = max(nsp_threshold, 0.75)

    beam = int(os.getenv("BEAM_SIZE", "5"))

    t0 = time.time()
    segments, _info = model.transcribe(
        audio_float,
        language=language or "hi",
        task="transcribe",
        beam_size=beam,
        best_of=beam,
        temperature=0.0,
        condition_on_previous_text=False,
        without_timestamps=True,
        vad_filter=use_clip_vad,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        word_timestamps=False,
    )
    segs = list(segments)
    infer_ms = (time.time() - t0) * 1000

    text, logprobs = _collect_text(segs, nsp_threshold)
    text = strip_hallucination_tail(text)

    # Fallback: short clips emptied by strict nsp — take best non-empty segments
    if not text and short_utt and segs:
        text, logprobs = _collect_text(segs, nsp_threshold=0.9)
        text = strip_hallucination_tail(text)
        if text:
            print(
                f"short-utt fallback kept={text!r} dur={duration_s:.2f}s",
                flush=True,
            )

    if is_useless_text(text):
        print(
            f"empty/useless after filter dur={duration_s:.2f}s segs={len(segs)} "
            f"nsp_th={nsp_threshold} clip_vad={use_clip_vad}",
            flush=True,
        )
        return _empty(language, infer_ms)

    if any(m.lower() in text.lower() for m in _HALLUCINATION_MARKERS):
        print(f"junk still present, drop: {text[:80]!r}", flush=True)
        return _empty(language, infer_ms)

    # Density guard only for long invented monologues (not 1–2 word answers)
    n_words = len(text.split())
    if duration_s > 0 and n_words / duration_s > 5.0 and n_words >= 12:
        print(
            f"density drop: {n_words} words / {duration_s:.2f}s → {text[:60]!r}",
            flush=True,
        )
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
