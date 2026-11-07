# konnnyaku

Fully local English speech → Japanese subtitle PoC for Apple Silicon.

- ASR: sherpa-onnx streaming Zipformer (`sherpa-onnx-streaming-zipformer-en-2023-06-26`) with `modified_beam_search`
- Live quality: Silero VAD drops music/BGM, Whisper `tiny.en` rescores each final (punctuation included), junk English/Japanese is not translated
- Translation: Marian-MT (`staka/fugumt-en-ja`) via ONNX Runtime.
  `Helsinki-NLP/opus-mt-en-jap` is too weak on technical speech.
  `facebook/nllb-200-distilled-600M` was slower on this machine (about 0.5–3 s a sentence) and missed technical terms FuguMT keeps, such as pull request and memory dump.
- UI: on macOS, `live` opens the transcription window. `summary` stays in the terminal and writes `meetings/latest.json`. `--cli` keeps live captions in the terminal.
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

# Live translation window (macOS). `--cli` prints in the terminal instead.
cargo run --release -- live --mic
cargo run --release -- live --app "Google Chrome"
cargo run --release -- live --cli --mic

# Meeting note in the terminal. Writes meetings/latest.json while it listens.
cargo run --release -- summary
cargo run --release -- summary --app "Google Chrome"
cargo run --release -- summary --wav models/asr/zipformer-en/test_wavs/0.wav
cargo run --release -- summary --json
```

Endpoint trailing silence defaults to 0.8 s (`--rule2`). Lower it for faster subtitles, raise it to avoid cutting mid-sentence.

Live finals are VAD-gated, then rescored, then translated. The default rescorer is Whisper tiny.en. `--whisper-dir` also accepts Whisper distil-small.en or Parakeet TDT 0.6B (`models/asr/parakeet-en`); on the bundled clip they heard the same sentence and did not fix FuguMT. `--no-vad` and `--no-whisper` fall back toward the Zipformer-only path. Songs and heavy BGM still confuse ASR; those finals are skipped instead of emitting `ははは…` / `<unk>`.

`live` on macOS opens one transcription window. The header chooses the spoken language and the caption language. The default is English to Japanese, with English on the left and Japanese on the right, without a line between sentences. The same language uses one column and does not translate. Japanese speech uses the Haas ONNX model in `models/asr/haas-ja` (downloaded by `scripts/download-asr.sh`); it rewrites the open line about once a second, then settles at the end of the utterance. Japanese to English shows the Japanese transcript only, because this build has no Japanese-to-English model. English still updates in place while Zipformer is guessing, and the finished Whisper line replaces that sentence. Lines stay until クリア or the window closes. If you have scrolled up, new text does not pull you back to the bottom. `summary` has no window. It records speech in the terminal and writes `meetings/latest.json` as lines arrive. `--json` prints that note on stdout as well. Each item keeps both `english` and `japanese`. Another process can read the file. Decisions and actions are lines that say so (`we decided`, `we need to`, `let's`). This is not a separate abstractive model. Japanese speech summarized without translation is not in this build.

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
