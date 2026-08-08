import asyncio
import json
import os
import random
import time
import wave
import audioop
import websockets

# Config
API_URL = os.getenv("WS_API_URL", "ws://localhost:8080/stream")
API_KEY = os.getenv("API_KEY", "your-api-key-here")
MODEL_ID = os.getenv("MODEL_ID", "your-model-id-here")
LANGUAGE = os.getenv("LANGUAGE", "fr")
AUDIO_DIR = os.getenv("AUDIO_DIR", "./audio_samples/")
OUT_RATE = int(os.getenv("SAMPLE_RATE", "8000"))  # VAD supports 8000 or 16000
CHUNK_MS = int(os.getenv("CHUNK_MS", "100"))


def pick_random_wav(dir_path: str) -> str:
    files = [f for f in os.listdir(dir_path) if f.lower().endswith(".wav")]
    if not files:
        return ""
    return os.path.join(dir_path, random.choice(files))


def load_as_mono_16bit_8k(path: str) -> bytes:
    with wave.open(path, "rb") as wf:
        fr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    # mono
    if ch > 1:
        frames = audioop.tomono(frames, sw, 0.5, 0.5)
        ch = 1
    # 16-bit
    if sw != 2:
        frames = audioop.lin2lin(frames, sw, 2)
        sw = 2
    # 8kHz
    if fr != OUT_RATE:
        frames, _ = audioop.ratecv(frames, 2, 1, fr, OUT_RATE, None)
    return frames


async def run_test(test_id, wav_path=None):
    if wav_path is None:
        wav_path = pick_random_wav(AUDIO_DIR)
        if not wav_path:
            raise ValueError(f"No WAV files found in {AUDIO_DIR}")
    
    test_start = time.time()
    print(f"[Test {test_id}] START at {test_start}")
    
    audio = load_as_mono_16bit_8k(wav_path)
    chunk_bytes = int(OUT_RATE * 2 * (CHUNK_MS / 1000.0))
    chunk_dur = chunk_bytes / (OUT_RATE * 2)

    uri = API_URL
    headers = [
        ("api-key", API_KEY),
        ("model-id", MODEL_ID),
        ("sample-rate", str(OUT_RATE)),
        ("language", LANGUAGE),
    ]

    result = {"test_id": test_id, "file": os.path.basename(wav_path), "status": "fail", "latency_ms": None, "response": None, "chunks_sent": 0, "send_duration": 0, "stopped_early": False}
    done = asyncio.Event()
    t0 = None
    chunks_sent = 0

    async def sender(ws):
        nonlocal t0, chunks_sent
        send_start = time.time()
        actual_times = []
        sleep_start_times = []
        sleep_end_times = []
        
        for i in range(0, len(audio), chunk_bytes):
            if done.is_set():
                print(f"[Test {test_id}] Sender STOPPED EARLY - done event set at chunk {chunks_sent}")
                break
                
            chunk = audio[i : i + chunk_bytes]
            send_time = time.time()
            await ws.send(chunk)
            actual_times.append(send_time)
            chunks_sent += 1
            t0 = time.time()
            
            if chunks_sent > 1:
                actual_gap = actual_times[-1] - actual_times[-2]
                if abs(actual_gap - chunk_dur) > 0.02:
                    print(f"[Test {test_id}] WARNING chunk {chunks_sent}: Gap {actual_gap:.3f}s (expected {chunk_dur:.3f}s, diff {actual_gap - chunk_dur:.3f}s)")
            
            sleep_start = time.time()
            await asyncio.sleep(chunk_dur)
            sleep_end = time.time()
            sleep_actual = sleep_end - sleep_start
            if abs(sleep_actual - chunk_dur) > 0.02:
                print(f"[Test {test_id}] WARNING sleep drift: {sleep_actual:.3f}s (expected {chunk_dur:.3f}s)")
        
        if not done.is_set():
            # 2s de silence
            silence = b"\x00" * chunk_bytes
            total = 0.0
            while total < 2.0 and not done.is_set():
                await ws.send(silence)
                chunks_sent += 1
                await asyncio.sleep(chunk_dur)
                total += chunk_dur
        
        send_end = time.time()
        result["chunks_sent"] = chunks_sent
        result["send_duration"] = round(send_end - send_start, 2)
        result["stopped_early"] = done.is_set()
        print(f"[Test {test_id}] Sender finished: {chunks_sent} chunks in {result['send_duration']:.2f}s (stopped_early={done.is_set()})")

    async def receiver(ws):
        nonlocal t0
        first_response = None
        while True:
            msg = await ws.recv()
            recv_time = time.time()
            try:
                data = json.loads(msg)
                if first_response is None:
                    first_response = recv_time
            except Exception:
                continue
            if data.get("type") == "error":
                result["status"] = "fail"
                result["response"] = data.get("error")
                done.set()
                return
            if data.get("type") == "transcription" and data.get("final"):
                latency = int((time.time() - (t0 or time.time())) * 1000)
                result["status"] = "success"
                result["latency_ms"] = latency
                result["infer_ms"] = data.get("infer", 0)
                result["response"] = data.get("text", "")
                done.set()
                return

    try:
        async with websockets.connect(uri, max_size=None, additional_headers=headers) as ws:
            send_task = asyncio.create_task(sender(ws))
            recv_task = asyncio.create_task(receiver(ws))
            try:
                await asyncio.wait_for(done.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                result["status"] = "fail"
                result["response"] = "timeout"
            finally:
                for t in (send_task, recv_task):
                    if not t.done():
                        t.cancel()
                try:
                    await ws.close()
                except Exception:
                    pass
    except Exception as e:
        result["status"] = "fail"
        result["response"] = f"conn_error: {e}"
    return result


if __name__ == "__main__":
    import sys
    async def main():
        num_tests = int(os.getenv("NUM_TESTS", "5"))
        wav_path = sys.argv[1] if len(sys.argv) > 1 else None
        print(f"Starting {num_tests} concurrent tests")
        tasks = [asyncio.create_task(run_test(i+1, wav_path)) for i in range(num_tests)]
        await asyncio.gather(*tasks)
        results = [task.result() for task in tasks]
        if results:
            latencies = [r['latency_ms'] for r in results if r.get('latency_ms')]
            infers = [r['infer_ms'] for r in results if r.get('infer_ms')]
            if latencies:
                print(f"Avg latency: {sum(latencies) / len(latencies):.2f}ms")
            if infers:
                print(f"Avg infer: {sum(infers) / len(infers):.2f}ms")
    
    asyncio.run(main())

