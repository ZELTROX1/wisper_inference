import requests
import uuid
import time

CALL_NUMBER = 30
DELAY_RANGE = 500

def test_latice(audio_bytes):
    import os
    api_url = os.getenv("API_URL", "http://localhost:8080/transcribe")
    headers = {
        "API-Key": os.getenv("API_KEY", "your-api-key-here"),
        "Accept": "application/json",
        "Model-Id": os.getenv("MODEL_ID", "your-model-id-here")
    }
    files = {
        "audio_file": (f"audio_{uuid.uuid4()}.mp3", audio_bytes, "audio/mp3")
    }
    response = requests.post(f"{api_url}", files=files, headers=headers, timeout=500)
    return response.json()['transcription']

if __name__ == "__main__":
        import sys
        if len(sys.argv) < 2:
            print("Usage: python test_gpu.py <audio_file_path>")
            sys.exit(1)
        
        PATH = sys.argv[1]
        with open(PATH, "rb") as f:
            audio_bytes = f.read()
        
        start_time = time.time()
        print(test_latice(audio_bytes)) 
        print("Time taken:", time.time() - start_time)