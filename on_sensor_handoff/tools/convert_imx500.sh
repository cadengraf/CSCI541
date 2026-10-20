#!/usr/bin/env bash
# Run on the training/conversion computer inside an edge-mdt[pt] environment.
set -euo pipefail

CHECKPOINT="${1:-on_sensor_handoff/artifacts/identity_b0_64.pt}"
TRAIN_OBSERVATIONS="${2:-on_sensor_handoff/data/aicity_train_v1/observations.jsonl}"
OUTPUT_DIR="${3:-on_sensor_handoff/artifacts/imx500_identity_b0_64}"

if ! command -v imxconv-pt >/dev/null 2>&1; then
  echo "imxconv-pt is missing. Install edge-mdt[pt] in the converter environment." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
python -m on_sensor_handoff.tools.quantize_imx500 \
  --checkpoint "$CHECKPOINT" \
  --observations "$TRAIN_OBSERVATIONS" \
  --output "$OUTPUT_DIR/identity_b0_64_mct.onnx"

CONVERTER_ARGS=()
if [[ "${IMX500_NO_INPUT_PERSISTENCY:-0}" == "1" ]]; then
  CONVERTER_ARGS+=(--no-input-persistency)
fi
imxconv-pt -i "$OUTPUT_DIR/identity_b0_64_mct.onnx" \
  -o "$OUTPUT_DIR/converted" "${CONVERTER_ARGS[@]}"

test -f "$OUTPUT_DIR/converted/packerOut.zip"
echo "Converter output: $OUTPUT_DIR/converted/packerOut.zip"
echo "Package that ZIP on a Raspberry Pi with imx500-package."
