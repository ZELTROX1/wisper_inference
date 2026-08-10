#!/bin/bash

python3 -m venv /opt/venv --system-site-packages
. /opt/venv/bin/activate

# Install dependencies
# pip install -r requirements.txt

sudo apt update && sudo apt install -y lsof ffmpeg

# Load .env so the Python process sees the same configuration values.
if [ -f .env ]; then
    set -a
    . ./.env
    set +a
fi


# # Wait for VAST_TCP_PORT_8080 environment variable to be available
# echo "Waiting for VAST_TCP_PORT_8080 environment variable..."
# while [ -z "${VAST_TCP_PORT_8080}" ]; do
#     echo "Still waiting for VAST_TCP_PORT_8080..."
#     sleep 1
# done
# echo "VAST_TCP_PORT_8080 is set to: ${VAST_TCP_PORT_8080}"

sudo lsof -ti:8080 | xargs kill -9
nohup python optimized_api/main.py > /var/log/gpu-stt.log 2>&1 &
# sh consul.sh

# Optional: follow logs
# tail -f /var/log/gpu-stt.log
# tail -n 1000 /var/log/gpu-stt.log | grep '\[T4U7\]' > test.txt