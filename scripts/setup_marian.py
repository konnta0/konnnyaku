#!/usr/bin/env python3
"""Download a Marian EN-JA model, export encoder/decoder ONNX, convert tokenizers, and smoke-eval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "models" / "translation" / "marian-en-ja"
DEFAULT_MODEL = "staka/fugumt-en-ja"

TECHNICAL_SENTENCES = [
    "We need to update the server.",
    "The pull request failed continuous integration.",
    "Please share the memory dump from the crash.",
    "The API latency exceeded five hundred milliseconds.",
    "Can you rebase onto the main branch?",
    "The database migration is not idempotent.",
    "We should add a timeout to the HTTP client.",
    "The microphone sample rate is forty eight kilohertz.",
    "Real time factor must stay below zero point five.",
    "Only finalized utterances are translated.",
    "The encoder and decoder are separate ONNX sessions.",
    "Apple Silicon has unified memory.",
    "This is a proof of concept without a user interface.",
    "Streaming speech recognition uses a zipformer transducer.",
    "Drop audio chunks if the queue is full.",
    "Endpoint detection waits for trailing silence.",
    "The model must run fully offline after download.",
    "Japanese subtitles should appear after the utterance ends.",
    "We measured p fifty, p ninety five, and p ninety nine.",
    "Please restart the worker after deploying the patch.",
]


class MarianEncoder(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.encoder = model.get_encoder()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


class MarianDecoder(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.decoder = model.model.decoder
        self.lm_head = model.lm_head
        self.final_logits_bias = model.final_logits_bias

    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=False,
        )
        return self.lm_head(outputs.last_hidden_state) + self.final_logits_bias


def _spm_to_tokenizer(spm_path: Path, out_path: Path) -> None:
    import sentencepiece as spm
    from tokenizers import Tokenizer
    from tokenizers.decoders import Metaspace as MetaspaceDecoder
    from tokenizers.models import Unigram
    from tokenizers.pre_tokenizers import Metaspace

    processor = spm.SentencePieceProcessor(model_file=str(spm_path))
    vocab = [
        (processor.id_to_piece(i), processor.get_score(i))
        for i in range(processor.get_piece_size())
    ]
    tokenizer = Tokenizer(Unigram(vocab, unk_id=processor.unk_id()))
    tokenizer.pre_tokenizer = Metaspace(replacement="▁", prepend_scheme="always")
    tokenizer.decoder = MetaspaceDecoder(replacement="▁", prepend_scheme="always")
    tokenizer.save(str(out_path))


def convert_tokenizers(model_dir: Path, out_dir: Path) -> None:
    from transformers import AutoTokenizer

    _spm_to_tokenizer(model_dir / "source.spm", out_dir / "source_tokenizer.json")
    _spm_to_tokenizer(model_dir / "target.spm", out_dir / "target_tokenizer.json")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=False)
    tokenizer.save_pretrained(str(out_dir))


def write_generation_config(model_dir: Path, out_dir: Path) -> dict:
    raw = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    generation = {
        "decoder_start_token_id": int(raw["decoder_start_token_id"]),
        "eos_token_id": int(raw["eos_token_id"]),
        "pad_token_id": int(raw["pad_token_id"]),
        "decoder_layers": int(raw["decoder_layers"]),
        "decoder_attention_heads": int(raw["decoder_attention_heads"]),
        "d_model": int(raw["d_model"]),
        "max_length": int(raw.get("max_length", 512)),
        "vocab_size": int(raw.get("decoder_vocab_size", raw["vocab_size"])),
    }
    (out_dir / "generation.json").write_text(
        json.dumps(generation, indent=2) + "\n", encoding="utf-8"
    )
    return generation


def export_onnx(model_dir: Path, onnx_dir: Path) -> None:
    from transformers import AutoTokenizer, MarianMTModel

    onnx_dir.mkdir(parents=True, exist_ok=True)
    print("Loading PyTorch Marian model...", flush=True)
    model = MarianMTModel.from_pretrained(str(model_dir))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    encoded = tokenizer("We need to update the server.", return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    start_id = int(model.config.decoder_start_token_id)
    decoder_start = torch.tensor([[start_id, start_id, start_id, start_id]], dtype=torch.long)

    encoder = MarianEncoder(model)
    decoder = MarianDecoder(model)
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        hidden = encoder(input_ids, attention_mask)

    encoder_path = onnx_dir / "encoder_model.onnx"
    decoder_path = onnx_dir / "decoder_model.onnx"

    print(f"Exporting encoder -> {encoder_path}", flush=True)
    _export_onnx(
        encoder,
        (input_ids, attention_mask),
        encoder_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "encoder_sequence"},
            "attention_mask": {0: "batch", 1: "encoder_sequence"},
            "last_hidden_state": {0: "batch", 1: "encoder_sequence"},
        },
    )

    print(f"Exporting decoder -> {decoder_path}", flush=True)
    _export_onnx(
        decoder,
        (decoder_start, hidden, attention_mask),
        decoder_path,
        input_names=["decoder_input_ids", "encoder_hidden_states", "encoder_attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "decoder_input_ids": {0: "batch", 1: "decoder_sequence"},
            "encoder_hidden_states": {0: "batch", 1: "encoder_sequence"},
            "encoder_attention_mask": {0: "batch", 1: "encoder_sequence"},
            "logits": {0: "batch", 1: "decoder_sequence"},
        },
    )


def _export_onnx(model, args, path, input_names, output_names, dynamic_axes) -> None:
    from torch.export import Dim

    batch = Dim("batch")
    encoder_sequence = Dim("encoder_sequence", min=1, max=256)
    decoder_sequence = Dim("decoder_sequence", min=1, max=128)
    dynamic_shapes = None
    if "decoder_input_ids" in input_names:
        dynamic_shapes = {
            "decoder_input_ids": {0: batch, 1: decoder_sequence},
            "encoder_hidden_states": {0: batch, 1: encoder_sequence},
            "encoder_attention_mask": {0: batch, 1: encoder_sequence},
        }
    elif "input_ids" in input_names:
        dynamic_shapes = {
            "input_ids": {0: batch, 1: encoder_sequence},
            "attention_mask": {0: batch, 1: encoder_sequence},
        }

    try:
        torch.onnx.export(
            model,
            args,
            str(path),
            dynamo=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_shapes=dynamic_shapes,
            opset_version=18,
        )
    except (TypeError, ModuleNotFoundError):
        torch.onnx.export(
            model,
            args,
            str(path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17,
        )


def evaluate_transformers(model_dir: Path) -> None:
    from transformers import AutoTokenizer, MarianMTModel

    print("Loading transformers MarianMT for quality smoke test...", flush=True)
    model = MarianMTModel.from_pretrained(str(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    print("--- transformers EN -> JA ---")
    for sentence in TECHNICAL_SENTENCES:
        encoded = tokenizer(sentence, return_tensors="pt", padding=True)
        generated = model.generate(**encoded, max_new_tokens=64, num_beams=1)
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"EN: {sentence}")
        print(f"JA: {text}")
        print()


def evaluate_onnx(onnx_dir: Path) -> None:
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer

    generation = json.loads((onnx_dir / "generation.json").read_text(encoding="utf-8"))
    source = Tokenizer.from_file(str(onnx_dir / "source_tokenizer.json"))
    target = Tokenizer.from_file(str(onnx_dir / "target_tokenizer.json"))
    encoder = ort.InferenceSession(str(onnx_dir / "encoder_model.onnx"), providers=["CPUExecutionProvider"])
    decoder = ort.InferenceSession(str(onnx_dir / "decoder_model.onnx"), providers=["CPUExecutionProvider"])
    eos = int(generation["eos_token_id"])
    start_id = int(generation["decoder_start_token_id"])
    pad = int(generation["pad_token_id"])

    print("--- ONNX greedy EN -> JA ---")
    for sentence in TECHNICAL_SENTENCES[:5]:
        encoded = source.encode(sentence, add_special_tokens=False)
        ids = [int(i) for i in encoded.ids]
        if not ids or ids[-1] != eos:
            ids.append(eos)
        input_ids = np.array([ids], dtype=np.int64)
        mask = np.ones_like(input_ids)
        hidden = encoder.run(None, {"input_ids": input_ids, "attention_mask": mask})[0]
        tokens = [start_id]
        for _ in range(64):
            decoder_ids = np.array([tokens], dtype=np.int64)
            logits = decoder.run(
                None,
                {
                    "decoder_input_ids": decoder_ids,
                    "encoder_hidden_states": hidden,
                    "encoder_attention_mask": mask,
                },
            )[0]
            row = logits[0, -1].copy()
            if 0 <= pad < row.shape[0]:
                row[pad] = np.finfo(row.dtype).min
            next_id = int(np.argmax(row))
            if next_id == eos:
                break
            tokens.append(next_id)
        keep = [t for t in tokens[1:] if t not in (eos, pad)]
        text = target.decode(keep, skip_special_tokens=True).replace("▁", "").replace(" ", "")
        print(f"EN: {sentence}")
        print(f"JA: {text}")
        print()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    hf_dir = args.out_dir / "hf"
    onnx_dir = args.out_dir / "onnx"
    hf_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    print(f"Downloading {args.model} -> {hf_dir}", flush=True)
    snapshot_download(repo_id=args.model, local_dir=str(hf_dir))

    encoder_onnx = onnx_dir / "encoder_model.onnx"
    decoder_onnx = onnx_dir / "decoder_model.onnx"
    if not args.skip_export:
        if encoder_onnx.is_file() and decoder_onnx.is_file() and not args.force_export:
            print("ONNX encoder/decoder already exist; skipping torch.onnx export", flush=True)
        else:
            export_onnx(hf_dir, onnx_dir)
        convert_tokenizers(hf_dir, onnx_dir)
        write_generation_config(hf_dir, onnx_dir)
        print("Export complete. ONNX files:", flush=True)
        for path in sorted(onnx_dir.glob("*.onnx")):
            print(f"  {path.name}")

    if not args.skip_eval:
        try:
            evaluate_transformers(hf_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"skipping transformers eval: {exc}", flush=True)
        evaluate_onnx(onnx_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
