FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=./models/tara-ct2 \
    DEVICE=cuda \
    COMPUTE_TYPE=float16 \
    MIXED_CODE=true \
    HOST=0.0.0.0 \
    PORT=8080

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt \
    && pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128

COPY app/ ./app/
COPY scripts/ ./scripts/

EXPOSE 8080
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
