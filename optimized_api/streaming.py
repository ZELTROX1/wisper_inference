import json, time, os, wave, tempfile, asyncio, numpy as np, torch, base64, random, string
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from silero_vad import load_silero_vad

router = APIRouter()

_VAD_MODEL = load_silero_vad()
_VAD_MODEL.eval()
_VAD_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_VAD_MODEL.to(_VAD_DEVICE)


def vad_prob(chunk: bytes, sample_rate: int) -> float:
    if not chunk or len(chunk) < 2:
        return 0.0
    silero_frame_samples = 256 if sample_rate == 8000 else 512
    audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
    if audio.size == 0:
        return 0.0
    if audio.size < silero_frame_samples:
        padded = np.zeros(silero_frame_samples, dtype=np.float32)
        padded[:audio.size] = audio
        audio = padded
    elif audio.size > silero_frame_samples:
        audio = audio[-silero_frame_samples:]
    with torch.inference_mode():
        tens = torch.from_numpy(audio).unsqueeze(0).to(_VAD_DEVICE)
        return float(_VAD_MODEL(tens, sample_rate).item())

@router.websocket("/stream")
async def stream_audio(websocket: WebSocket):
    await websocket.accept()
    
    stream_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))

    api_key = websocket.headers.get("api-key")
    model_id = websocket.headers.get("model-id")
    if not api_key or not model_id:
        await websocket.send_text(json.dumps({"type": "error", "error": "Missing api-key or model-id header"}))
        await websocket.close(code=4403)
        return

    try:
        gpu_quota_manager = websocket.app.state.gpu_quota_manager
        if not gpu_quota_manager.can_transcribe(api_key, 1000):
            await websocket.send_text(json.dumps({"type": "error", "error": f"Insufficient quota. Required: 1000ms, Available: {gpu_quota_manager.get_available_milliseconds(api_key)}ms"}))
            await websocket.close(code=402)
            return
        repo_id = gpu_quota_manager.resolve_repo_id(api_key, model_id)
        if not repo_id:
            raise RuntimeError(f"Model ID {model_id} not found for this user")
    except Exception as e:
        await websocket.send_text(json.dumps({"type": "error", "error": str(e)}))
        await websocket.close(code=4500)
        return

    batcher = gpu_quota_manager.get_batcher(repo_id)

    sample_rate = int(websocket.headers.get("sample-rate", "8000"))
    vad_threshold = float(websocket.headers.get('vad-threshold', '0.3'))
    min_silence_duration = float(websocket.headers.get('min-silence-duration', '0.3'))
    stt_interval_ms = int(websocket.headers.get('stt-interval-ms', '200'))
    language = websocket.headers.get("language", "fr")

    audio_buffer = bytearray()
    current_transcript = ""
    last_silence_time = None
    model_warmed = False
    last_avg_logprob = None
    last_good_prob = False
    last_wav_base64 = None
    warm_launched = False
    total_streamed_bytes = 0
    last_infer_ms = 0
    have_spoke = False
    stt_task = None
    requested_at = time.time()
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_path = temp_wav.name
    temp_wav.close()

    def write_wav_file():
        with wave.open(temp_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_buffer)

    async def warm_model():
        nonlocal model_warmed
        write_wav_file()
        try:
            with open(temp_path, "rb") as f:
                await batcher.enqueue(f.read(), language)
            model_warmed = True
            print(f"[{stream_id}] Model warmed up")
        except Exception as e:
            print(f"[{stream_id}] Warmup error: {e}")

    async def run_stt():
        nonlocal current_transcript, last_infer_ms, last_avg_logprob, last_good_prob, last_wav_base64, last_silence_time, language
        if len(audio_buffer) < sample_rate * 0.2:
            return
        start_time = time.time()
        write_wav_file()
        print(f"[{stream_id}] time to write wav file: {time.time() - start_time}")
        try:
            with open(temp_path, "rb") as f:
                wav_data = f.read()
                t0 = time.time()
                text, infer, _, avg_logprob, good_prob = await batcher.enqueue(wav_data, language)
                infer = (time.time() - t0) * 1000
            print(f"[{stream_id}] STT: {text} (avg_logprob={avg_logprob:.3f}) (good_prob={good_prob})")
            current_transcript = text
            last_avg_logprob = avg_logprob
            last_good_prob = good_prob
            last_infer_ms = int(infer)
            last_wav_base64 = base64.b64encode(wav_data).decode('utf-8')
        except Exception as e:
            print(f"[{stream_id}] STT error: {e}")

    async def check_end_of_turn():
        nonlocal current_transcript, audio_buffer, last_silence_time, have_spoke, last_avg_logprob, last_good_prob, last_wav_base64, stt_task
        if not current_transcript.strip() or last_avg_logprob is None:
            return False
        silence_duration = (time.time() - last_silence_time) if last_silence_time else 0
        print(f"[{stream_id}] Silence: {silence_duration:.2f}s {last_silence_time} {min_silence_duration} {silence_duration >= min_silence_duration} {last_good_prob} {last_avg_logprob} {current_transcript}")
        if silence_duration >= min_silence_duration:
            if last_good_prob and last_avg_logprob > -0.5 and not "Sous-titrage" in current_transcript:
                if stt_task and not stt_task.done():
                    print(f"[{stream_id}] Waiting for STT to finish before sending result...")
                    await stt_task
                print(f"[{stream_id}] SEND RESULT {time.time()} {current_transcript}")
                await websocket.send_text(json.dumps({
                    "type": "transcription",
                    "text": current_transcript,
                    "final": True,
                    "silence_duration": silence_duration,
                    "avg_logprob": last_avg_logprob,
                    "infer": last_infer_ms,
                    "stream_id": stream_id,
                    # "audio_wav_base64": last_wav_base64,
                }))
                current_transcript = ""
                audio_buffer.clear()
                last_silence_time = None
                have_spoke = False
                last_avg_logprob = None
                last_wav_base64 = None
                return True
        return False
    i = 0
    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"]:
                i = i + 1
                chunk = message["bytes"]
                if not warm_launched:
                    await warm_model()
                    warm_launched = True
                
                vad_prob_value = vad_prob(chunk, sample_rate)
                is_speaking = vad_prob_value > vad_threshold
                print(f"[{stream_id}] VAD: {vad_prob_value:.3f}")
                total_streamed_bytes += len(chunk)
                audio_buffer.extend(chunk)
                
                if is_speaking:
                    if last_silence_time is not None:
                        print(f"[{stream_id}] Speech started")
                    last_silence_time = None
                    have_spoke = True
                else:
                    if last_silence_time is None:
                        last_silence_time = time.time()
                        print(f"[{stream_id}] Silence started")
                        if model_warmed and len(audio_buffer) >= sample_rate * 0.2:
                            if not stt_task or stt_task.done():
                                stt_task = asyncio.create_task(run_stt())
                if last_silence_time is not None:
                    await check_end_of_turn()
            elif "text" in message and message["text"]:
                if message["text"].strip().lower() == "close":
                    await websocket.close()
                    break
            elif message.get("type") == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"type": "error", "error": str(e)}))
            await websocket.close(code=1011)
        except Exception:
            pass
    finally:
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        except Exception:
            pass
        try:
            if api_key:
                duration_ms = int((total_streamed_bytes / (sample_rate * 2)) * 1000.0)
                if duration_ms > 0:
                    gpu_quota_manager.consume_quota(api_key, duration_ms)
                    cost = (duration_ms / 1000) * 0.00015
                    gpu_quota_manager.record_usage(
                        api_key=api_key,
                        cost=cost,
                        duration_seconds=duration_ms / 1000,
                        latency_ms=last_infer_ms,
                        requested_at=requested_at,
                        streaming=True
                    )
        except Exception as e:
            print(f"[{stream_id}] Error recording usage (stream): {e}")
