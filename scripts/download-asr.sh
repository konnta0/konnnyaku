#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/models/asr"
RELEASE="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"

mkdir -p "${DEST}"

download_zipformer() {
  local archive="sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2"
  local extracted="sherpa-onnx-streaming-zipformer-en-2023-06-26"
  local target="${DEST}/zipformer-en"
  local encoder="${target}/encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx"

  if [[ -f "${encoder}" ]]; then
    echo "Zipformer already present at ${target}"
    return 0
  fi

  echo "Downloading ${RELEASE}/${archive}"
  curl -L --fail -o "${DEST}/${archive}" "${RELEASE}/${archive}"
  tar -xjf "${DEST}/${archive}" -C "${DEST}"
  rm -f "${DEST}/${archive}"

  if [[ -d "${DEST}/${extracted}" ]]; then
    rm -rf "${target}"
    mv "${DEST}/${extracted}" "${target}"
  fi

  if [[ ! -f "${encoder}" ]]; then
    echo "expected encoder file missing after extract: ${encoder}" >&2
    ls -la "${target}" >&2 || true
    exit 1
  fi

  echo "Zipformer ready at ${target}"
}

download_vad() {
  local target="${DEST}/silero_vad.onnx"
  if [[ -f "${target}" ]]; then
    echo "Silero VAD already present at ${target}"
    return 0
  fi

  echo "Downloading ${RELEASE}/silero_vad.onnx"
  curl -L --fail -o "${target}" "${RELEASE}/silero_vad.onnx"
  echo "Silero VAD ready at ${target}"
}

download_whisper() {
  local archive="sherpa-onnx-whisper-tiny.en.tar.bz2"
  local extracted="sherpa-onnx-whisper-tiny.en"
  local target="${DEST}/whisper-tiny.en"
  local encoder="${target}/tiny.en-encoder.int8.onnx"

  if [[ -f "${encoder}" || -f "${target}/tiny.en-encoder.onnx" ]]; then
    echo "Whisper tiny.en already present at ${target}"
    return 0
  fi

  echo "Downloading ${RELEASE}/${archive}"
  curl -L --fail -o "${DEST}/${archive}" "${RELEASE}/${archive}"
  tar -xjf "${DEST}/${archive}" -C "${DEST}"
  rm -f "${DEST}/${archive}"

  if [[ -d "${DEST}/${extracted}" ]]; then
    rm -rf "${target}"
    mv "${DEST}/${extracted}" "${target}"
  fi

  if [[ ! -f "${encoder}" && ! -f "${target}/tiny.en-encoder.onnx" ]]; then
    echo "expected Whisper encoder missing after extract: ${target}" >&2
    ls -la "${target}" >&2 || true
    exit 1
  fi

  echo "Whisper tiny.en ready at ${target}"
}

download_zipformer
download_vad
download_whisper
