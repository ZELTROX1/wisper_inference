import os
from fastapi import FastAPI
from transcribe import router as transcribe_router
from streaming import router as streaming_router
from warmup import health_tick
from model_manager import get_model
from batcher import CoalescingBatcher
from env_loader import load_env_file

load_env_file()

app = FastAPI(title="Whisper S2T Ultra Low-Latency", version="2.0.0")

# Mount routes
app.include_router(transcribe_router)
app.include_router(streaming_router)

# Single model loaded once at startup, shared by every request.
MODEL_REPO_ID = os.getenv("MODEL_REPO_ID", "./models/tara-ct2")
model = get_model(MODEL_REPO_ID)
app.state.batcher = CoalescingBatcher(model)

@app.get("/health")
async def health():
    await health_tick(app)
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
