#!/usr/bin/env bash
# Prepared RE10K scenes use the same camera/image interface and metric protocol.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${DATASET_ROOT:?set DATASET_ROOT to the fixed prepared RE10K test root}"
exec bash "$ROOT/scripts/evaluate_dl3dv140.sh" "$@"
