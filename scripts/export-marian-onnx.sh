#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

UV=(uv)
if [[ "$(uname -m)" == "x86_64" ]] && command -v arch >/dev/null 2>&1; then
  UV=(arch -arm64 uv)
fi

"${UV[@]}" python install 3.12 >/dev/null

exec "${UV[@]}" run --python 3.12 --isolated \
  --with 'numpy<2' \
  --with 'torch>=2.5' \
  --with 'transformers==4.46.3' \
  --with sentencepiece \
  --with onnx \
  --with onnxscript \
  --with onnxruntime \
  --with huggingface_hub \
  --with tokenizers \
  -- python "${ROOT}/scripts/setup_marian.py" "$@"
