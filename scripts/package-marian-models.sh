#!/usr/bin/env bash
# Package the exported Marian ONNX directories as the models-v1 release assets
# that `konnnyaku setup` downloads. FuguMT is CC-BY-SA-4.0, so each archive
# carries the attribution and license notice.
#
# Usage: scripts/package-marian-models.sh [out-dir]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-${ROOT}/dist/models-v1}"
FILES=(
  encoder_model.onnx
  encoder_model.onnx.data
  decoder_model.onnx
  decoder_model.onnx.data
  source_tokenizer.json
  target_tokenizer.json
  generation.json
)

mkdir -p "${OUT}"

package() {
  local name="$1" hf_id="$2" revision="$3"
  local src="${ROOT}/models/translation/${name}/onnx"
  local stage
  stage="$(mktemp -d)"
  for file in "${FILES[@]}"; do
    if [[ ! -f "${src}/${file}" ]]; then
      echo "missing ${src}/${file}. Export it with scripts/export-marian-onnx.sh first." >&2
      exit 1
    fi
    cp "${src}/${file}" "${stage}/${file}"
  done
  cat > "${stage}/NOTICE.txt" <<EOF
ONNX export of ${hf_id} (revision ${revision}) for konnnyaku.
Original model: https://huggingface.co/${hf_id}
License: Creative Commons Attribution-ShareAlike 4.0 International (CC-BY-SA-4.0)
https://creativecommons.org/licenses/by-sa/4.0/
Changes: encoder and decoder exported to ONNX with torch.onnx (scripts/setup_marian.py),
tokenizers converted to tokenizer.json. Weights were not retrained.
EOF
  COPYFILE_DISABLE=1 tar -czf "${OUT}/${name}-onnx.tar.gz" -C "${stage}" NOTICE.txt "${FILES[@]}"
  rm -rf "${stage}"
  echo "wrote ${OUT}/${name}-onnx.tar.gz"
}

package marian-en-ja staka/fugumt-en-ja fbb91252110cf97d90291eb808629ab0b10f7928
package marian-ja-en staka/fugumt-ja-en f7ce11286e1fb7a8e1f1692ff3ab68c0f9c3aecb

(cd "${OUT}" && shasum -a 256 ./*.tar.gz > SHA256SUMS)
cat "${OUT}/SHA256SUMS"
