import asyncio
import json
import websockets
import pyaudio

import os
API_URL = os.getenv("WS_API_URL", "ws://localhost:8080/stream")
API_KEY = os.getenv("API_KEY", "your-api-key-here")
MODEL_ID = os.getenv("MODEL_ID", "your-model-id-here")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "8000"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1600"))

async def stream_from_microphone():
    uri = API_URL
    
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )
    headers = [
        ("api-key", API_KEY),
        ("model-id", MODEL_ID),
        ("sample-rate", str(SAMPLE_RATE))
    ]
    try:
        async with websockets.connect(uri, max_size=None, additional_headers=headers) as ws:
            print("Connection established. Speak into the microphone...")
            
            async def sender():
                while True:
                    chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                    await ws.send(chunk)
                    await asyncio.sleep(0.001)
            
            async def receiver():
                while True:
                    msg = await ws.recv()
                    print(msg)
                    
            
            await asyncio.gather(sender(), receiver())
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

asyncio.run(stream_from_microphone())
