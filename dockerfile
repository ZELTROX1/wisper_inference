FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies, Consul, and ffmpeg in one layer
RUN apt update && apt install -y --no-install-recommends \
    unzip curl jq wget \
    python3 python3-pip python3-venv \
    git ca-certificates \
    lsof ffmpeg \
    && wget -q https://releases.hashicorp.com/consul/1.19.0/consul_1.19.0_linux_amd64.zip \
    && unzip consul_1.19.0_linux_amd64.zip \
    && mv consul /usr/local/bin/ \
    && rm consul_1.19.0_linux_amd64.zip \
    && mkdir -p /etc/consul.d /opt/consul \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create virtual environment (layer rarely changes)
RUN python3 -m venv /opt/venv --system-site-packages \
    && /opt/venv/bin/pip install --upgrade pip \
    && rm -rf /tmp/* /var/tmp/*

# Install PyTorch separately (changes rarely, so better cache reuse)
RUN /opt/venv/bin/pip install --no-cache-dir \
        torch==2.8.0 \
        torchaudio==2.8.0 \
        --index-url https://download.pytorch.org/whl/cu128 \
    && rm -rf /tmp/* /var/tmp/*

# Install other Python dependencies (changes more often)
RUN /opt/venv/bin/pip install --no-cache-dir \
        git+https://github.com/LATICE-AI/whisper-s2t.git \
        fastapi \
        uvicorn \
        huggingface-hub \
        pydantic \
        python-multipart \
        silero-vad \
        websockets \
        requests \
        numpy \
    && rm -rf /tmp/* /var/tmp/* ~/.cache/pip



