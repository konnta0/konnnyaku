#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/models/asr"
ARCHIVE="sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2"
URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${ARCHIVE}"
EXTRACTED="sherpa-onnx-streaming-zipformer-en-2023-06-26"
TARGET="${DEST}/zipformer-en"
ENCODER="${TARGET}/encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx"

if [[ -f "${ENCODER}" ]]; then
  echo "ASR model already present at ${TARGET}"
  exit 0
fi

mkdir -p "${DEST}"
cd "${DEST}"

echo "Downloading ${URL}"
curl -L --fail -o "${ARCHIVE}" "${URL}"
tar -xjf "${ARCHIVE}"
rm -f "${ARCHIVE}"

if [[ -d "${EXTRACTED}" ]]; then
  rm -rf "${TARGET}"
  mv "${EXTRACTED}" "${TARGET}"
fi

if [[ ! -f "${ENCODER}" ]]; then
  echo "expected encoder file missing after extract: ${ENCODER}" >&2
  ls -la "${TARGET}" >&2 || true
  exit 1
fi

echo "ASR model ready at ${TARGET}"
