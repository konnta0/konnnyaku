# konnnyaku

Fully local English speech → Japanese subtitle PoC for Apple Silicon.

- ASR: sherpa-onnx streaming Zipformer (`sherpa-onnx-streaming-zipformer-en-2023-06-26`) with `modified_beam_search`
- Live quality: Silero VAD drops music/BGM, Whisper `tiny.en` rescores each final (punctuation included), junk English/Japanese is not translated
- Translation: Marian-MT (`staka/fugumt-en-ja`) via ONNX Runtime.
  `Helsinki-NLP/opus-mt-en-jap` was measured first and is too weak on technical speech.
- UI: CLI only. Inference does not use the network after models are downloaded.
- Browser / YouTube: `--app "Google Chrome"` taps that app's output (all tabs). `--loopback` taps every app on the default output. Speech and interviews work better than songs.

## Setup

```bash
# Zipformer + Silero VAD + Whisper tiny.en (~420 MB)
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

# Summarize the default microphone (not playback)
cargo run --release -- summary

# Summarize a WAV file
cargo run --release -- summary --wav models/asr/zipformer-en/test_wavs/0.wav
```

Endpoint trailing silence defaults to 0.8 s (`--rule2`). Lower it for faster subtitles, raise it to avoid cutting mid-sentence.

Live finals are VAD-gated, then Whisper-rescored, then translated. `--no-vad` and `--no-whisper` fall back toward the Zipformer-only path. Songs and heavy BGM still confuse ASR; skip those finals instead of emitting `ははは…` / `<unk>`.

`summary` listens to the microphone, or to a WAV file with `--wav`. It does not tap app or system output. After enough distinct speech, it prints up to three Japanese points and replaces that list when the points change. The list is the recent transcript with repeats removed, then translated. It is not a separate abstractive model.

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
src/asr.rs           sherpa-onnx OnlineRecognizer (beam search + utterance PCM)
src/vad.rs           Silero VAD music gate
src/whisper.rs       offline Whisper tiny.en rescorer
src/text.rs          junk English / Japanese filters
src/summary.rs       microphone transcript reduced to a few points
src/translation.rs   Translator + OnnxMarianTranslator
src/metrics.rs       RTF and translation percentiles
models/              gitignored weights
```
