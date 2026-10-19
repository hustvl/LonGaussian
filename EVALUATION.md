# Evaluation

The main-table protocol evaluates DL3DV-140 and RealEstate10K-180 at 64, 96 and
128 context views, using 9, 13 and 18 held-out views respectively. Images are
280 x 490. The packed prefixes contain 73, 109 and 146 frames; zero-based frame
indices 7, 15, ... are targets.

Use two-pass unposed reconstruction, context-only scene construction,
orientation-aware SE(3) alignment, TTT and SWA, window size 12 and overlap 3.
The final scene includes neural merging, dense complement, the 0.01/0.30
depth filter, sky removal and the far-field shell. Render complete scenes and
compute PSNR, SSIM and LPIPS-VGG.

## Run a setting

Prepare each context root with the commands in [DATASETS.md](DATASETS.md).
Set NUM_FRAMES to 73, 109 or 146 for the corresponding context count:

```bash
DATASET_ROOT=/path/to/eval-dl3dv64 OUTPUT_DIR=/path/to/new_eval/dl3dv/ctx64 \
NUM_FRAMES=73 GPU=0 \
MODEL_NAME=/path/to/weights/LoGeR_star/latest.pt \
GAUSSIAN_HEAD=/path/to/training/stage2b.pt \
ALLTRACKER_CHECKPOINT=/path/to/weights/alltracker.pth \
SKY_MASK_ONNX=/path/to/weights/skyseg.onnx \
bash scripts/evaluate_dl3dv140.sh
```

Use scripts/evaluate_re10k.sh for a prepared RE10K root. Repeat for all three
context lengths. Each root must contain the full fixed scene list, transforms,
images_8 and split.json. Both wrappers write metrics_dl3dv_subset.json; retain
the per-scene and per-view records.

## Run all six main-table settings

After preparing one full root per dataset/context setting, the unified driver
checks every fixed scene ID, 64/96/128 split and required checkpoint before
running. It then checks every target-view metric and writes a six-setting
`main_table_summary.json`. Use `--dry-run` for the input/command preflight and
`--resume-completed` to reuse only settings with complete, validated metrics.

```bash
python scripts/evaluate_main_table.py \
  --dl3dv-64 /path/to/eval-dl3dv64 --dl3dv-96 /path/to/eval-dl3dv96 \
  --dl3dv-128 /path/to/eval-dl3dv128 \
  --re10k-64 /path/to/eval-re10k64 --re10k-96 /path/to/eval-re10k96 \
  --re10k-128 /path/to/eval-re10k128 \
  --weights /path/to/weights --checkpoint /path/to/training/stage2b.pt \
  --output /path/to/new_main_table_eval --gpu 0
```

Path-free reference training selections, checkpoint lineage and six per-view
result files are in [reference/](reference/README.md). These are the reference
retraining results summarized below; no training images or weights are bundled.

## Reference retraining run

The following results identify the publicly configured retraining run's final
Stage-IIb model, SHA256
`d6d7e8b9614e35208f83ee229e6aff066c3295b82e3377ddc6b27362bf989a76`.
They are measured retraining results, distinct from the paper's table values.
All six settings completed with zero failed scenes.

| Dataset | Context | Scenes | Target views | PSNR | SSIM | LPIPS | Gaussians (M) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DL3DV-140 | 64 | 140 | 1260 | 19.5128 | 0.6335 | 0.3923 | 2.2886 |
| DL3DV-140 | 96 | 140 | 1820 | 18.5466 | 0.5826 | 0.4366 | 3.4324 |
| DL3DV-140 | 128 | 140 | 2520 | 17.8420 | 0.5383 | 0.4684 | 4.6586 |
| RealEstate10K-180 | 64 | 180 | 1620 | 19.8471 | 0.6344 | 0.3608 | 0.9089 |
| RealEstate10K-180 | 96 | 180 | 2340 | 18.8996 | 0.5886 | 0.3963 | 1.3552 |
| RealEstate10K-180 | 128 | 180 | 3240 | 18.1406 | 0.5499 | 0.4248 | 1.7624 |

Frame indices, source pixels, checkpoint identity and evaluation settings must
match when comparing runs. Preserve data-download and preprocessing records
alongside newly generated metrics.
