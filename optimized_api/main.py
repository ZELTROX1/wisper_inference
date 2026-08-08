from fastapi import FastAPI
from transcribe import router as transcribe_router
from streaming import router as streaming_router
from warmup import health_tick
from quota_manager import GPUQuotaManager

app = FastAPI(title="Whisper S2T Ultra Low-Latency", version="2.0.0")

# Mount routes
app.include_router(transcribe_router)
app.include_router(streaming_router)

# Shared global state
app.state.gpu_quota_manager = GPUQuotaManager()

@app.get("/health")
async def health():
    # Flush usage records for all users (checks if > 1min since last send)
    app.state.gpu_quota_manager._flush_usage_records()
    await health_tick(app)
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
