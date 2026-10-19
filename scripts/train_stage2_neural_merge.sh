#!/usr/bin/env bash
# Paper-configured stage2b; shares the validated pipeline driver.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python}" "$ROOT/scripts/train_pipeline.py" \
  --prepared "${PREPARED_ROOT:?set PREPARED_ROOT to prepare_training_data.py output}" \
  --weights "${WEIGHTS_ROOT:?set WEIGHTS_ROOT to downloaded external weights}" \
  --output "${TRAIN_OUTPUT:?set TRAIN_OUTPUT to the common stage output root}" \
  --gpus "${GPUS:-0,1,2,3}" --stage stage2b "$@"
