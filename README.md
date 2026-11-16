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

## Install (macOS, Apple Silicon)

Needs a release tarball on GitHub Releases (and, for Homebrew, a formula in [konnta0/homebrew-tap](https://github.com/konnta0/homebrew-tap)). Until the first `v*` release exists, use the development setup below instead.

### Homebrew

```bash
brew install konnta0/tap/konnnyaku
konnnyaku setup
```

### Manual

1. Download `konnnyaku-vX.Y.Z-aarch64-apple-darwin.tar.gz` and its `.sha256` from the [GitHub release](https://github.com/konnta0/konnnyaku/releases).
2. Check the archive: `shasum -a 256 -c konnnyaku-vX.Y.Z-aarch64-apple-darwin.tar.gz.sha256`
3. Unpack and put `konnnyaku` on your `PATH`.
4. Run `konnnyaku setup` once (about 2.5 GB of models).

Nothing else has to be installed. `setup` uses the system `curl` and `tar`, and only keeps files that match a pinned SHA-256.

### After install

```bash
konnnyaku live --mic                          # microphone
konnnyaku live --app "Google Chrome"          # one app's playback (all tabs)
konnnyaku live --loopback                     # default system output
konnnyaku live --mic --app "Google Chrome"    # both
konnnyaku apps                                # apps that currently have audio
konnnyaku captions                            # translated lines as JSON
```

With no capture flags, `live` starts the microphone only. `--app` or `--loopback` starts playback capture and turns the microphone off unless you also pass `--mic`. While the window is open, the header **入力** / **出力** buttons toggle microphone and playback without restarting. There is no separate config file.

Models and captions live in `~/Library/Application Support/konnnyaku`. Set `KONNNYAKU_HOME` to use another directory. In a source checkout that already has `models/asr/zipformer-en`, the working directory is used instead.

System Audio Recording permission (System Settings → Privacy & Security) is granted to the terminal app you start `konnnyaku` from, not to `konnnyaku` itself. If playback is audible but captions stay silent, grant that permission and start audio in the target app before `--app`.

### For AI agents

While `live` runs, each settled line is written to `captions.json` in that directory. Any process can read it with:

```bash
konnnyaku captions
```

```json
{
  "lines": [
    { "primary": "ja", "source": "No, wait.", "translation": "いや、待って。", "text": "いや、待って。" },
    { "primary": "ja", "speaker": 2, "source": "説明がある。", "text": "説明がある。" }
  ],
  "primary": "ja"
}
```

`text` is the line in the display language. `translation` is present only when the speech was in the other language. The file keeps the newest 400 lines, is cleared by the window's クリア button, and is replaced when a new `live` session starts.

## Release (maintainers)

1. Bump `version` in `Cargo.toml` and commit.
2. `git tag vX.Y.Z && git push origin vX.Y.Z`.
3. `.github/workflows/release.yml` checks that the tag matches `Cargo.toml`, runs the tests, builds the arm64 binary, and publishes the tarball and its `.sha256` to the GitHub release.
4. With the `HOMEBREW_TAP_TOKEN` secret set (a fine-grained token that can only write `konnta0/homebrew-tap`), the same workflow opens a PR there from `packaging/homebrew/konnnyaku.rb.in`.

The FuguMT ONNX archives that `setup` downloads are the `models-v1` release. Rebuild them with `scripts/package-marian-models.sh`; if their contents change, publish them under a new tag and update the pins in `src/setup.rs`. Model licenses are listed in `THIRD_PARTY_NOTICES.md`.

## Development setup

```bash
# Same models as `konnnyaku setup`, plus optional Whisper tiny.en and distil-small
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

Live finals are VAD-gated, then recognised, then translated when needed. The default English recognizer is Parakeet TDT 0.6B (`--whisper-dir`); `--no-vad` and `--no-whisper` fall back toward the Zipformer-only path. Songs and heavy BGM still confuse ASR; those finals are skipped instead of emitting `ははは…` / `<unk>`.

`live` on macOS opens one transcription window. The header **表示** chooses Japanese or English. Speech in that language is shown as-is; other speech is translated into it (Marian en↔ja). Japanese speech uses Haas ONNX. Lines stay until クリア or the window closes. If you have scrolled up, new text does not pull you back to the bottom. `summary` stays in the terminal and writes `meetings/latest.json`. `--json` also prints that note on stdout. Decisions and actions are keyword-classified lines (`we decided`, `we need to`, `let's`), not a separate abstractive model.

`--app` and `--loopback` are macOS-only. `--app` records that application (including helpers/renderers and every tab). `--loopback` records every application on the default output. Keep Built-in Speakers as the output device for `--loopback`: USB interfaces and headsets that also have a microphone are rejected, because cpal would open their physical input instead of a tap.

If you hear playback but captions stay silent, grant **System Audio Recording** (or Screen Recording on older macOS) to Terminal or Cursor in System Settings → Privacy & Security. Start audio in the target app before `--app`; `konnnyaku apps` only lists apps that currently have Core Audio process objects.

Measured on this Apple Silicon machine (release build, not a 60-minute soak):

- ASR RTF ≈ 0.04 on the bundled Zipformer test WAV
- Translation P95 ≈ 220 ms on 20 technical sentences

`staka/fugumt-en-ja` is CC-BY-SA-4.0. See `THIRD_PARTY_NOTICES.md` for every model.

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
src/paths.rs         KONNNYAKU_HOME / Application Support / checkout paths
src/setup.rs         `konnnyaku setup`: pinned model downloads
src/caption_log.rs   captions.json for `konnnyaku captions`
models/              gitignored weights
```
