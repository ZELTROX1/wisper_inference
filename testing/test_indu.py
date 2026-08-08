import os
import sys
import requests
import uuid
import time

def test_latice(audio_bytes, api_url=None, api_key=None, model_id=None):
    """Sends audio file to transcription API and returns transcription."""
    api_url = api_url or os.getenv("API_URL", "http://localhost:8080/transcribe")
    api_key = api_key or os.getenv("API_KEY", "your-api-key-here")
    model_id = model_id or os.getenv("MODEL_ID", "your-model-id-here")
    
    headers = {
        "API-Key": api_key,
        "Accept": "application/json",
        "Model-Id": model_id
    }
    files = {
        "audio_file": (f"audio_{uuid.uuid4()}.mp3", audio_bytes, "audio/mp3")
    }
    response = requests.post(f"{api_url}", files=files, headers=headers, timeout=500)
    response.raise_for_status()
    return response.json()['transcription']


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_indu.py <audio_file_path> [num_tests]")
        print("Or set environment variables: API_URL, API_KEY, MODEL_ID")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    test_number = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.getenv("TEST_NUMBER", "10"))
    
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)
    
    latencies = []
    for i in range(test_number):
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        
        start_time = time.time()
        try:
            transcription = test_latice(audio_bytes)
            latency = time.time() - start_time
            latencies.append(latency)
            print(f"Test {i+1}: {latency:.3f}s - {transcription[:50]}...")
        except Exception as e:
            print(f"Test {i+1} failed: {e}")
    
    if latencies:
        print(f"\nAverage latency: {sum(latencies) / len(latencies):.3f}s")
        print(f"Min latency: {min(latencies):.3f}s")
        print(f"Max latency: {max(latencies):.3f}s")
    else:
        print("No successful tests to calculate average latency.")