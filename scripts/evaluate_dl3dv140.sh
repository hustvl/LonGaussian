#!/usr/bin/env bash
# Final robust depth filtering/sky evaluation. Dataset must contain the prepared evaluation scenes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
DATASET_ROOT="${DATASET_ROOT:?set DATASET_ROOT to the prepared DL3DV140 root}"
MODEL_NAME="${MODEL_NAME:?set MODEL_NAME to the base Pi3 checkpoint}"
GAUSSIAN_HEAD="${GAUSSIAN_HEAD:?set GAUSSIAN_HEAD to the Stage-II checkpoint}"
CONFIG="${CONFIG:-$ROOT/configs/LoGeR_star/original_config.yaml}"
SKY_MASK_ONNX="${SKY_MASK_ONNX:?set SKY_MASK_ONNX to skyseg.onnx}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/eval_dl3dv140}"
GPU="${GPU:-0}"
NUM_SCENES="${NUM_SCENES:-0}"
NUM_FRAMES="${NUM_FRAMES:-146}"

for path in "$MODEL_NAME" "$GAUSSIAN_HEAD" "$CONFIG" "$SKY_MASK_ONNX"; do [[ -f "$path" ]] || { echo "missing required file: $path" >&2; exit 2; }; done
[[ ! -e "$OUTPUT_DIR" || -z "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]] || {
  echo "refusing non-empty OUTPUT_DIR: $OUTPUT_DIR" >&2; exit 2;
}
mkdir -p "$OUTPUT_DIR"
cd "$ROOT"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export LOGER_DISABLE_TORCH_COMPILE="${LOGER_DISABLE_TORCH_COMPILE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u scripts/evaluate_reconstruction.py \
  --gaussian_head "$GAUSSIAN_HEAD" --model_name "$MODEL_NAME" --config "$CONFIG" \
  --dataset_root "$DATASET_ROOT" --output_dir "$OUTPUT_DIR" --num_scenes "$NUM_SCENES" \
  --num_frames "$NUM_FRAMES" --image_scale 1 --window_size 12 --overlap_size 3 \
  --gaussian_head_batch_size 1 --two_pass_pose --orientation_aware_two_pass \
  --calibrate_local_points_to_k --seed 0 --deterministic --render_chunk 8 \
  --merge_mode neural --track_anchor_mode median_nn --track_anchor_radius 0.15 --track_anchor_beta 6 \
  --alltracker_conf_thr 0.1 --global_complement_voxel_factor 0 --global_complement_voxel_schedule fixed \
  --global_complement_voxel_policy first --global_complement_voxel_min_windows 3 \
  --variant m01_v1_r30 --min_rel_gap 0.01 --visible_ratio_threshold 0.30 \
  --neighborhood_radius 1 --min_valid_neighbors 5 --max_rel_depth_spread 0.15 --alpha_threshold 0.5 \
  --sky_mask_drop --sky_sphere --sky_mask_onnx "$SKY_MASK_ONNX" --sky_mask_score_thr 160 \
  --sky_sphere_radius_factor 50 --sky_sphere_sample_stride 2 --sky_sphere_azimuth_bins 512 \
  --sky_sphere_elevation_bins 256 --sky_sphere_opacity 0.98 --sky_sphere_scale_overlap 1 "$@"
