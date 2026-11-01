# konnnyaku

Fully local English speech → Japanese subtitle PoC for Apple Silicon.

- ASR: sherpa-onnx streaming Zipformer (`sherpa-onnx-streaming-zipformer-en-2023-06-26`) with `modified_beam_search`
- Live quality: Silero VAD drops music/BGM, Whisper `tiny.en` rescores each final (punctuation included), junk English/Japanese is not translated
- Translation: Marian-MT (`staka/fugumt-en-ja`) via ONNX Runtime.
  `Helsinki-NLP/opus-mt-en-jap` is too weak on technical speech.
  `facebook/nllb-200-distilled-600M` was slower on this machine (about 0.5–3 s a sentence) and missed technical terms FuguMT keeps, such as pull request and memory dump.
- UI: `live` prints line-by-line captions in the terminal. `summary` opens a separate meeting-note window on macOS.
- Inference does not use the network after models are downloaded.
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

# Meeting note from the microphone, a WAV file, or one app
cargo run --release -- summary
cargo run --release -- summary --app "Google Chrome"
cargo run --release -- summary --wav models/asr/zipformer-en/test_wavs/0.wav
cargo run --release -- summary --json
```

Endpoint trailing silence defaults to 0.8 s (`--rule2`). Lower it for faster subtitles, raise it to avoid cutting mid-sentence.

Live finals are VAD-gated, then rescored, then translated. The default rescorer is Whisper tiny.en. `--whisper-dir` also accepts Whisper distil-small.en or Parakeet TDT 0.6B (`models/asr/parakeet-en`); on the bundled clip they heard the same sentence and did not fix FuguMT. `--no-vad` and `--no-whisper` fall back toward the Zipformer-only path. Songs and heavy BGM still confuse ASR; those finals are skipped instead of emitting `ははは…` / `<unk>`.

`summary` on macOS opens a meeting-note window. It records English and, when you press 要約する, writes `meetings/latest.json` and shows 概要, 議題, 決定, and 次のアクション. `--cli`, `--json`, and `--wav` write that note without a click. Each item keeps both `english` and `japanese`. Another process can read the file. Decisions and actions are lines that say so (`we decided`, `we need to`, `let's`). This is not a separate abstractive model. `live` still translates playback line by line and does not write this note. Japanese speech summarized without translation is not in this build.

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
src/summary.rs       meeting note: transcript, topics, decisions, actions
src/translation.rs   Translator + OnnxMarianTranslator
src/metrics.rs       RTF and translation percentiles
models/              gitignored weights
```
