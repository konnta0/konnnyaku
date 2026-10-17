# konnnyaku

Fully local English speech → Japanese subtitle PoC for Apple Silicon.

- ASR: sherpa-onnx streaming Zipformer (`sherpa-onnx-streaming-zipformer-en-2023-06-26`)
- Translation: Marian-MT (`staka/fugumt-en-ja`) via ONNX Runtime.
  `Helsinki-NLP/opus-mt-en-jap` was measured first and is too weak on technical speech.
- UI: CLI only. Inference does not use the network after models are downloaded.

## Setup

```bash
# English streaming ASR model (~300 MB)
./scripts/download-asr.sh

# Marian EN→JA ONNX export (one-time; needs Python via uv)
./scripts/export-marian-onnx.sh
```

To re-export a different Marian checkpoint:

```bash
./scripts/export-marian-onnx.sh --model Helsinki-NLP/opus-mt-en-jap --force-export
```

## Commands

```bash
# WAV streaming ASR (partial + final)
cargo run --release -- asr --wav models/asr/zipformer-en/test_wavs/0.wav

# Microphone ASR
cargo run --release -- asr --mic

# Translate a sentence
cargo run --release -- translate --text "We need to update the server."

# Batch-translate finalized utterances
cargo run --release -- translate --file benches/technical_en.txt

# Live captions: English partials, Japanese on endpoint
cargo run --release -- live --mic
```

Endpoint trailing silence defaults to 0.8 s (`--rule2`). Lower it for faster subtitles, raise it to avoid cutting mid-sentence.

Measured on this Apple Silicon machine (release build, not a 60-minute soak):

- ASR RTF ≈ 0.04 on the bundled Zipformer test WAV
- Translation P95 ≈ 220 ms on 20 technical sentences

`staka/fugumt-en-ja` is CC-BY-SA-4.0. The Zipformer ASR package comes from sherpa-onnx releases.

## Layout

```
src/audio.rs         microphone + WAV
src/asr.rs           sherpa-onnx OnlineRecognizer
src/translation.rs   Translator + OnnxMarianTranslator
src/metrics.rs       RTF and translation percentiles
models/              gitignored weights
```
