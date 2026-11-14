# Third-party notices

konnnyaku itself is MIT licensed (see `LICENSE`). The release binary does not
contain model weights. `konnnyaku setup` downloads them from the sources below,
pins each file by SHA-256, and stores them under
`~/Library/Application Support/konnnyaku/models`.

## Models

| Use | Model | Source | License |
| --- | --- | --- | --- |
| English streaming ASR | Zipformer `streaming-zipformer-en-2023-06-26` (icefall) | k2-fsa/sherpa-onnx release `asr-models` | Apache-2.0 |
| Voice activity | Silero VAD | k2-fsa/sherpa-onnx release `asr-models` | MIT |
| English recognizer | NVIDIA Parakeet TDT 0.6B v2, int8 | k2-fsa/sherpa-onnx release `asr-models` | CC-BY-4.0 |
| Language detection | OpenAI Whisper tiny, int8 | k2-fsa/sherpa-onnx release `asr-models` | Apache-2.0 (Hugging Face model card) |
| Japanese recognizer | Haas-ja v1.0 ONNX quantized (`oboroge0/Haas-ja-v1.0-onnx-quantized`, base `third-intelligence/Haas-ja-v1.0`) | Hugging Face, pinned revision | Apache-2.0 |
| Speaker turns | pyannote segmentation 3.0 | k2-fsa/sherpa-onnx release `speaker-segmentation-models` | MIT |
| Speaker voices | 3D-Speaker CAM++ zh-en (`iic/speech_campplus_sv_zh_en_16k-common_advanced`) | k2-fsa/sherpa-onnx release `speaker-recongition-models` | Apache-2.0 |
| English to Japanese | FuguMT `staka/fugumt-en-ja`, exported to ONNX | konnnyaku release `models-v1` | CC-BY-SA-4.0 |
| Japanese to English | FuguMT `staka/fugumt-ja-en`, exported to ONNX | konnnyaku release `models-v1` | CC-BY-SA-4.0 |

Attribution for CC-BY-4.0 and CC-BY-SA-4.0 models: the original authors are
NVIDIA (Parakeet) and Satoshi Takahashi (FuguMT). The FuguMT ONNX files are an
adaptation made by exporting the encoder and decoder with `torch.onnx` and
converting the tokenizers; the weights were not retrained. They are shared
under the same CC-BY-SA-4.0 license, with a `NOTICE.txt` inside each archive.
License texts: https://creativecommons.org/licenses/by/4.0/ and
https://creativecommons.org/licenses/by-sa/4.0/.

## Libraries linked into the binary

- sherpa-onnx (Apache-2.0)
- ONNX Runtime (MIT), statically linked through sherpa-onnx and `ort`
- Rust crates listed in `Cargo.lock`, each under its own license (mostly MIT or Apache-2.0)
