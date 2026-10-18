# konnnyaku

Fully local English speech → Japanese subtitle PoC for Apple Silicon.

- ASR: sherpa-onnx streaming Zipformer (`sherpa-onnx-streaming-zipformer-en-2023-06-26`)
- Translation: Marian-MT (`staka/fugumt-en-ja`) via ONNX Runtime.
  `Helsinki-NLP/opus-mt-en-jap` was measured first and is too weak on technical speech.
- UI: CLI only. Inference does not use the network after models are downloaded.
- Browser / YouTube: `--app "Google Chrome"` taps that app's output (all tabs). `--loopback` taps every app on the default output.

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

# List running apps that currently have tappable audio
cargo run --release -- apps

# One application (Chrome / Safari / bundle ID)
cargo run --release -- asr --app "Google Chrome"
cargo run --release -- asr --app com.google.Chrome

# System-output ASR (every app on the default speakers)
cargo run --release -- asr --loopback

# Translate a sentence
cargo run --release -- translate --text "We need to update the server."

# Batch-translate finalized utterances
cargo run --release -- translate --file benches/technical_en.txt

# Live captions from the microphone
cargo run --release -- live --mic

# Live captions from one app
cargo run --release -- live --app "Google Chrome"

# Live captions from system output (all apps)
cargo run --release -- live --loopback
```

Endpoint trailing silence defaults to 0.8 s (`--rule2`). Lower it for faster subtitles, raise it to avoid cutting mid-sentence.

`--app` and `--loopback` are macOS-only. `--app` records that application (including helpers/renderers and every tab). `--loopback` records every application on the default output. Keep Built-in Speakers as the output device for `--loopback`: USB interfaces and headsets that also have a microphone are rejected, because cpal would open their physical input instead of a tap.

If you hear playback but captions stay silent, grant **System Audio Recording** (or Screen Recording on older macOS) to Terminal or Cursor in System Settings → Privacy & Security. Start audio in the target app before `--app`; `konnnyaku apps` only lists apps that currently have Core Audio process objects.

Measured on this Apple Silicon machine (release build, not a 60-minute soak):

- ASR RTF ≈ 0.04 on the bundled Zipformer test WAV
- Translation P95 ≈ 220 ms on 20 technical sentences

`staka/fugumt-en-ja` is CC-BY-SA-4.0. The Zipformer ASR package comes from sherpa-onnx releases.

## Layout

```
src/audio.rs         microphone, app tap, system-output loopback, WAV
src/macos_tap.rs     Core Audio process tap (macOS)
src/asr.rs           sherpa-onnx OnlineRecognizer
src/translation.rs   Translator + OnnxMarianTranslator
src/metrics.rs       RTF and translation percentiles
models/              gitignored weights
```
