#!/usr/bin/env bash
# Export a Gaussian PLY from an ordered image directory.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
INPUT="${INPUT:?set INPUT to an image directory}"
MODEL_NAME="${MODEL_NAME:?set MODEL_NAME to the base Pi3 checkpoint}"
GAUSSIAN_HEAD="${GAUSSIAN_HEAD:?set GAUSSIAN_HEAD to a trained Gaussian-head checkpoint}"
CONFIG="${CONFIG:-$ROOT/configs/LoGeR_star/original_config.yaml}"
OUTPUT="${OUTPUT:-$ROOT/outputs/scene.ply}"
NUM_FRAMES="${NUM_FRAMES:--1}"
DEVICE="${DEVICE:-cuda}"

for path in "$MODEL_NAME" "$GAUSSIAN_HEAD" "$CONFIG"; do [[ -f "$path" ]] || { echo "missing required file: $path" >&2; exit 2; }; done
mkdir -p "$(dirname "$OUTPUT")"
exec "$PYTHON" "$ROOT/infer.py" --input "$INPUT" --output "$OUTPUT" --model_name "$MODEL_NAME" \
  --config "$CONFIG" --gaussian_head "$GAUSSIAN_HEAD" --num_frames "$NUM_FRAMES" \
  --window_size 12 --overlap_size 3 --device "$DEVICE"
