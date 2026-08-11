#!/bin/bash
# Convert Trelis/tara (or any Whisper HF model) → CTranslate2 for faster-whisper
# Usage: ./scripts/convert_to_ct2.sh Trelis/tara ./models/tara-ct2 float16
set -euo pipefail

SRC="${1:-Trelis/tara}"
OUT="${2:-./models/tara-ct2}"
QUANT="${3:-float16}"

pip show ctranslate2 >/dev/null 2>&1 || pip install ctranslate2
pip show transformers >/dev/null 2>&1 || pip install transformers torch

ct2-transformers-converter \
  --model "$SRC" \
  --output_dir "$OUT" \
  --copy_files tokenizer.json preprocessor_config.json \
  --quantization "$QUANT" \
  --force

echo "Done. MODEL_PATH=$OUT"
