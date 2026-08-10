import time
import tempfile
import os
import wave
from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Header
from pydantic import BaseModel

router = APIRouter()

class TranscriptionResponse(BaseModel):
    transcription: str

def _calculate_duration(audio_bytes: bytes) -> float:
    """Calculates audio duration in seconds"""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_file_path = tmp_file.name

    try:
        with wave.open(tmp_file_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            return frames / float(sample_rate)
    except:
        # Fallback: estimate from file size
        file_size_mb = len(audio_bytes) / (1024 * 1024)
        return file_size_mb * 6
    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

def get_audio_duration_ms(audio_bytes: bytes) -> int:
    """Calculates actual audio duration in milliseconds"""
    duration_seconds = _calculate_duration(audio_bytes)
    return int(duration_seconds * 1000)

async def handle_bytes(audio_bytes: bytes, request: Request, language: str = "fr"):
    total_start = time.time()

    t0 = time.time()
    duration_ms = get_audio_duration_ms(audio_bytes)
    duration_calc = (time.time() - t0) * 1000

    t0 = time.time()
    batcher = request.app.state.batcher
    batcher_get = (time.time() - t0) * 1000

    t0 = time.time()
    text, infer, bs, avg_logprob, good_prob = await batcher.enqueue(audio_bytes, language)
    enqueue = (time.time() - t0) * 1000

    total_ms = (time.time() - total_start) * 1000
    print(f"TOTAL={total_ms:.1f}ms | duration_calc={duration_calc:.1f}ms | batcher_get={batcher_get:.1f}ms | enqueue={enqueue:.1f}ms (infer={infer:.1f}ms)")

    return TranscriptionResponse(transcription=text)

@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    audio_file: UploadFile = File(...),
    language: str = Header("fr", alias="language", description="Language to use"),
    request: Request = None
):
    """Transcribe an audio file."""
    request_start = time.time()
    try:
        audio_bytes = await audio_file.read()
    except:
        raise HTTPException(400, "Failed to read file")
    result = await handle_bytes(audio_bytes, request, language)
    request_total = (time.time() - request_start) * 1000
    print(f"REQUEST TOTAL: {request_total:.1f}ms (from HTTP entry point)")
    return result
