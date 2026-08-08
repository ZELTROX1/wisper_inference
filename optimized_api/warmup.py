import asyncio, time, io, wave

_warm_last_ts: float = 0.0
_warm_in_progress: bool = False
_warm_interval_s: int = 20

def _generate_silent_wav_bytes(duration_seconds: float = 0.2, sample_rate: int = 16000) -> bytes:
    num_samples = int(duration_seconds * sample_rate)
    silent_frames = (b"\x00\x00") * num_samples
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(silent_frames)
    return buf.getvalue()

async def _warm_model(app, repo_id: str):
    batcher = app.state.gpu_quota_manager.get_batcher(repo_id)
    try:
        wav_bytes = _generate_silent_wav_bytes()
        await batcher.enqueue(wav_bytes)
        print(f"Warmed {repo_id}")
    except Exception:
        pass

async def _warm_all_models(app):
    global _warm_in_progress, _warm_last_ts
    if _warm_in_progress:
        return
    _warm_in_progress = True
    try:
        repo_ids = app.state.gpu_quota_manager.get_all_existing_models()
        print(f"Warming {len(repo_ids)} models")
        for rid in repo_ids:
            await _warm_model(app, rid)
    finally:
        _warm_last_ts = time.time()
        _warm_in_progress = False

async def health_tick(app):
    now = time.time()
    print(f"Health tick: {now} - {_warm_last_ts} >= {_warm_interval_s} and not _warm_in_progress: {_warm_in_progress}")
    if (now - _warm_last_ts) >= _warm_interval_s and not _warm_in_progress:
        asyncio.create_task(_warm_all_models(app))

