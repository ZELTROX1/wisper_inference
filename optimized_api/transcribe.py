import gzip, time
import io
import tempfile
import os
import wave
from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Header
from pydantic import BaseModel

from request_context import resolve_request_context

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
        # Fallback: estimation basée sur la taille
        file_size_mb = len(audio_bytes) / (1024 * 1024)
        return file_size_mb * 6
    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

def get_audio_duration_ms(audio_bytes: bytes) -> int:
    """Calculates actual audio duration in milliseconds"""
    duration_seconds = _calculate_duration(audio_bytes)
    return int(duration_seconds * 1000)

async def handle_bytes(audio_bytes: bytes, model_id: str, request: Request, api_key: str = None, language: str = "fr"):
    total_start = time.time()
    requested_at = time.time()
    api_key, model_id = resolve_request_context(api_key, model_id)
    
    # duration_calc
    t0 = time.time()
    duration_ms = get_audio_duration_ms(audio_bytes)
    duration_calc = (time.time() - t0) * 1000
    
    quota_check = 0.0
    try:
        # quota_check
        t0 = time.time()
        gpu_quota_manager = request.app.state.gpu_quota_manager
        if not gpu_quota_manager.can_transcribe(api_key, duration_ms):
            raise HTTPException(402, f"Insufficient quota. Required: {duration_ms}ms, Available: {gpu_quota_manager.get_available_milliseconds(api_key)}ms")
        
        # Récupérer le repo_id (local_mode: toujours LOCAL_MODEL_REPO_ID)
        repo_id = gpu_quota_manager.resolve_repo_id(api_key, model_id)

        if not repo_id:
            raise HTTPException(400, f"model-id header not found for this user")
        quota_check = (time.time() - t0) * 1000
    except Exception as e:
        raise HTTPException(500, f"Internal error: {e}")
    
    # batcher_get
    t0 = time.time()
    batcher = request.app.state.gpu_quota_manager.get_batcher(repo_id)
    batcher_get = (time.time() - t0) * 1000

    # enqueue (inclut infer)
    t0 = time.time()
    text, infer, bs, avg_logprob, good_prob = await batcher.enqueue(audio_bytes, language)
    enqueue = (time.time() - t0) * 1000
    
    try:
        gpu_quota_manager = request.app.state.gpu_quota_manager
        gpu_quota_manager.consume_quota(api_key, duration_ms)
        cost = (duration_ms / 1000) * 0.00015
        gpu_quota_manager.record_usage(
            api_key=api_key,
            cost=cost,
            duration_seconds=duration_ms / 1000,
            latency_ms=int(infer),
            requested_at=requested_at,
            streaming=False
        )
    except Exception as e:
        print(f"Error recording usage: {e}")
    
    total_ms = (time.time() - total_start) * 1000
    print(f"TOTAL={total_ms:.1f}ms | duration_calc={duration_calc:.1f}ms | quota_check={quota_check:.1f}ms | batcher_get={batcher_get:.1f}ms | enqueue={enqueue:.1f}ms (infer={infer:.1f}ms)")
    
    return TranscriptionResponse(transcription=text)

@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    audio_file: UploadFile = File(...), 
    api_key: str = Header(None, alias="api-key", description="API key for authentication"),
    model_id: str = Header(None, alias="model-id", description="Model ID to use"),
    language: str = Header("fr", alias="language", description="Language to use") ,
    request: Request = None
):
    """Transcribe an audio file with specified model ID."""
    request_start = time.time()
    try:
        audio_bytes = await audio_file.read()
    except: 
        raise HTTPException(400, "Failed to read file")
    result = await handle_bytes(audio_bytes, model_id, request, api_key, language)
    request_total = (time.time() - request_start) * 1000
    print(f"REQUEST TOTAL: {request_total:.1f}ms (from HTTP entry point)")
    return result
