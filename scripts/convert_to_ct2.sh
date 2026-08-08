#!/bin/bash
# Converts a HuggingFace Transformers Whisper checkpoint (safetensors) to the
# CTranslate2 format whisper-s2t / this API expects.
#
# Trelis/tara ships as safetensors (openai/whisper-large-v3 architecture), so
# it must go through this conversion once before optimized_api/model_manager.py
# can load it. Output dir doubles as LOCAL_MODEL_REPO_ID.
#
# Usage:
#   ./scripts/convert_to_ct2.sh Trelis/tara ./models/tara-ct2 [float16|int8_float16|int8]
set -euo pipefail

SRC_MODEL="${1:?usage: convert_to_ct2.sh <hf_repo_id> <output_dir> [quantization]}"
OUT_DIR="${2:?usage: convert_to_ct2.sh <hf_repo_id> <output_dir> [quantization]}"
QUANT="${3:-float16}"   # float16 (fastest on modern GPUs), int8_float16 (smaller/cheaper), int8 (CPU)

pip show ctranslate2 >/dev/null 2>&1 || pip install "ctranslate2>=4.5.0"

ct2-transformers-converter \
    --model "$SRC_MODEL" \
    --output_dir "$OUT_DIR" \
    --copy_files tokenizer.json preprocessor_config.json \
    --quantization "$QUANT" \
    --force

echo "Converted. Set LOCAL_MODEL_REPO_ID=$OUT_DIR before starting the API."
