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

download_haas() {
  local target="${DEST}/haas-ja"
  local encoder="${target}/encoder.quantized.onnx"
  local weights="${target}/encoder.data"
  local base="https://huggingface.co/oboroge0/Haas-ja-v1.0-onnx-quantized/resolve/main"

  if [[ -f "${encoder}" && -f "${weights}" && -f "${target}/decoder.onnx" && -f "${target}/joiner.onnx" && -f "${target}/tokens.txt" ]]; then
    echo "Haas Japanese ONNX already present at ${target}"
    return 0
  fi

  mkdir -p "${target}"
  echo "Downloading Haas Japanese ONNX into ${target}"
  curl -L --fail --retry 3 -o "${target}/tokens.txt" "${base}/tokens.txt"
  curl -L --fail --retry 3 -o "${target}/decoder.onnx" "${base}/decoder.onnx"
  curl -L --fail --retry 3 -o "${target}/joiner.onnx" "${base}/joiner.onnx"
  curl -L --fail --retry 3 -o "${encoder}" "${base}/encoder.quantized.onnx"
  curl -L --fail --retry 3 -C - -o "${weights}" "${base}/encoder.data"
  echo "Haas Japanese ONNX ready at ${target}"
}

download_diar() {
  local target="${DEST}/diar"
  local segmentation="${target}/segmentation.onnx"
  local embedding="${target}/embedding.onnx"
  # The release tag really is spelled "recongition".
  local embedding_url="https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
  local archive="sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
  local archive_url="https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/${archive}"

  if [[ -f "${segmentation}" && -f "${embedding}" ]]; then
    echo "Speaker diarization models already present at ${target}"
    return 0
  fi

  mkdir -p "${target}"
  if [[ ! -f "${segmentation}" ]]; then
    echo "Downloading ${archive_url}"
    curl -L --fail --retry 3 -o "${target}/${archive}" "${archive_url}"
    tar -xjf "${target}/${archive}" -C "${target}"
    rm -f "${target}/${archive}"
    local extracted="${target}/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
    if [[ ! -f "${extracted}" ]]; then
      echo "expected segmentation model missing after extract: ${extracted}" >&2
      exit 1
    fi
    mv "${extracted}" "${segmentation}"
    rm -rf "${target}/sherpa-onnx-pyannote-segmentation-3-0"
  fi

  if [[ ! -f "${embedding}" ]]; then
    echo "Downloading ${embedding_url}"
    curl -L --fail --retry 3 -o "${embedding}" "${embedding_url}"
  fi

  echo "Speaker diarization ready at ${target}"
}

download_distil_small() {
  local archive="sherpa-onnx-whisper-distil-small.en.tar.bz2"
  local extracted="sherpa-onnx-whisper-distil-small.en"
  local target="${DEST}/whisper-distil-small.en"
  local encoder="${target}/distil-small.en-encoder.int8.onnx"

  if [[ -f "${encoder}" || -f "${target}/distil-small.en-encoder.onnx" ]]; then
    echo "Distil-small English already present at ${target}"
    return 0
  fi

  echo "Downloading ${RELEASE}/${archive}"
  curl -L --fail --retry 3 -o "${DEST}/${archive}" "${RELEASE}/${archive}"
  tar -xjf "${DEST}/${archive}" -C "${DEST}"
  rm -f "${DEST}/${archive}"

  if [[ -d "${DEST}/${extracted}" ]]; then
    rm -rf "${target}"
    mv "${DEST}/${extracted}" "${target}"
  fi

  if [[ ! -f "${encoder}" && ! -f "${target}/distil-small.en-encoder.onnx" ]]; then
    echo "expected Distil-small encoder missing after extract: ${target}" >&2
    ls -la "${target}" >&2 || true
    exit 1
  fi

  echo "Distil-small English ready at ${target}"
}

download_zipformer
download_vad
download_whisper
download_distil_small
download_haas
download_diar
