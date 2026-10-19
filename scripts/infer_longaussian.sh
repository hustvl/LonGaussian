#!/usr/bin/env bash
# Full reconstruction of a calibrated DL3DV-style image sequence.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DATASET_ROOT="${INPUT:?set INPUT to a scene containing transforms.json and images_8}"
export OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR to a new output directory}"
export NUM_FRAMES="${NUM_FRAMES:?set NUM_FRAMES to the number of consecutive input frames}"
exec bash "$ROOT/scripts/evaluate_dl3dv140.sh" --inference_scene "$INPUT" "$@"
