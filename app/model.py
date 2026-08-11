"""Faster-whisper model + Tara mixed-code prompt.

Hallucination control is architectural where possible:
  - client only sends speech windows
  - here we refuse to invent text on silence / no-speech / impossible density
"""

from __future__ import annotations

import io
import os
import re
import time
import wave
from collections import Counter
from typing import List, Optional

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.tokenizer import Tokenizer

SAMPLE_RATE = 16_000
MIXEDCODE = "<|mixedcode|>"

_model: WhisperModel | None = None
_warmed = False

# Phrases Whisper invents on noise in Indic telephony (seen in production logs).
_HALLUCINATION_MARKERS = (
    "दीनदयाल",
    "दीन दयाल",
    "ग्रामीण कौशल",
    "प्रतिक्रिया के लिए संस्कृति",
    "विशिष्टता को समर्थन",
    "subscribe",
    "thanks for watching",
    "sous-titrage",
)


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
            without_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=False,
        )[0]
    )
    _warmed = True
    print(f"warmup done in {(time.time() - t0) * 1000:.0f}ms")


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


def rms_energy(wave_f: np.ndarray) -> float:
    if wave_f.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(wave_f), dtype=np.float64)))


def has_speech_energy(wave_f: np.ndarray) -> bool:
    """True if waveform is loud enough to be speech (not silence/noise floor)."""
    # Absolute RMS gate — silence / line noise fails; real speech passes.
    # Tuned for 16-bit PCM normalized to [-1, 1].
    return rms_energy(wave_f) >= 0.01


def collapse_repeats(text: str, max_run: int = 2) -> str:
    words = text.split()
    if not words:
        return text
    out: list[str] = []
    prev = None
    run = 0
    for w in words:
        key = w.lower()
        if key == prev:
            run += 1
            if run <= max_run:
                out.append(w)
        else:
            prev = key
            run = 1
            out.append(w)
    return " ".join(out)


def is_repeat_loop(text: str, ratio: float = 0.45) -> bool:
    words = [w.lower() for w in text.split() if w.strip()]
    if len(words) < 6:
        return False
    _word, count = Counter(words).most_common(1)[0]
    return (count / len(words)) >= ratio


def is_impossible_density(text: str, duration_s: float) -> bool:
    """Reject when model dumps more words than a human could have spoken."""
    if duration_s <= 0:
        return True
    words = text.split()
    if not words:
        return False
    # ~4 words/sec is already fast speech; above that is invented text on noise.
    return (len(words) / duration_s) > 4.5


def looks_like_known_hallucination(text: str) -> bool:
    low = text.lower()
    return any(m.lower() in low for m in _HALLUCINATION_MARKERS)


def _empty_result(language: str, infer_ms: float = 0.0) -> dict:
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
) -> dict:
    """Run ASR. Returns empty text when input is silence or model is inventing."""
    model = get_model()
    wave_f = _to_float32(audio, sample_rate)
    if wave_f.size == 0:
        return _empty_result(language)

    duration_s = float(wave_f.size) / float(sample_rate)

    # Do not call the decoder on silence — Whisper invents long Hindi phrases.
    if not has_speech_energy(wave_f):
        print(f"skip decode: low energy rms={rms_energy(wave_f):.5f} dur={duration_s:.2f}s")
        return _empty_result(language)

    vad_filter = os.getenv("CLIP_VAD_FILTER", "true").lower() in ("1", "true", "yes")
    beam = int(os.getenv("BEAM_SIZE", "5"))

    t0 = time.time()
    segments, _info = model.transcribe(
        wave_f,
        language=language or "hi",
        task="transcribe",
        beam_size=beam,
        temperature=0.0,
        word_timestamps=True,
        without_timestamps=True,
        condition_on_previous_text=False,
        vad_filter=vad_filter,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
    )
    segs = list(segments)
    infer_ms = (time.time() - t0) * 1000

    if not segs:
        return _empty_result(language, infer_ms)

    # If every segment looks like no-speech, refuse the text.
    if all(float(s.no_speech_prob or 0.0) >= 0.6 for s in segs):
        print("skip decode: no_speech_prob high on all segments")
        return _empty_result(language, infer_ms)

    texts, lps = [], []
    good = False
    max_no_speech = 0.0
    max_comp = 0.0
    for s in segs:
        max_no_speech = max(max_no_speech, float(s.no_speech_prob or 0.0))
        max_comp = max(max_comp, float(s.compression_ratio or 0.0))
        # Drop individual segments that are themselves no-speech inventions
        if float(s.no_speech_prob or 0.0) >= 0.6:
            continue
        t = (s.text or "").strip()
        if t:
            texts.append(t)
        if s.avg_logprob is not None:
            lps.append(float(s.avg_logprob))
        if s.words:
            good = good or any(
                w.probability is not None and float(w.probability) > 0.8 for w in s.words
            )

    text = " ".join(texts).strip()
    text = re.sub(r"\s+", " ", text.replace("\ufffd", "").replace("�", "")).strip()
    text = collapse_repeats(text, max_run=2)

    if not text:
        return _empty_result(language, infer_ms)

    if is_repeat_loop(text) or looks_like_known_hallucination(text):
        print(f"hallucination rejected: {text[:100]!r}")
        return _empty_result(language, infer_ms)

    if is_impossible_density(text, duration_s):
        print(
            f"density rejected: {len(text.split())} words in {duration_s:.2f}s → {text[:80]!r}"
        )
        return _empty_result(language, infer_ms)

    if max_comp >= 2.4:
        print(f"compression_ratio rejected: {max_comp:.2f}")
        return _empty_result(language, infer_ms)

    avg = float(sum(lps) / len(lps)) if lps else 0.0
    return {
        "text": text,
        "language": language,
        "avg_logprob": avg,
        "infer_ms": infer_ms,
        "good_prob": good or (avg > -0.5 and bool(text)),
        "no_speech": False,
    }
