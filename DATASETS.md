# Data preparation from official inputs

Run commands from the release root with Python 3.11. Install `requirements.txt` and `requirements-data.txt`. Dataset access is subject to the official sources: [DL3DV](https://huggingface.co/DL3DV) and [RealEstate10K](https://google.github.io/realestate10k/download.html). Accept DL3DV access terms and run `hf auth login` first. Video extraction also needs ffmpeg and a yt-dlp-supported JavaScript runtime (Deno); supply their paths when not on PATH.

## Download

Set these paths to directories outside the code package:

```bash
export DATA_ROOT=/path/to/datasets
export WEIGHTS_ROOT=/path/to/weights
python scripts/download_assets.py weights --output "$WEIGHTS_ROOT"
python scripts/download_assets.py dl3dv --output "$DATA_ROOT/dl3dv-download"
python scripts/download_assets.py benchmark --output "$DATA_ROOT/dl3dv140"
python scripts/download_assets.py re10k-metadata --output "$DATA_ROOT/re10k-metadata"
```

The downloader resolves Hugging Face branches to fixed commits and records them in `download.json`. Weights are checked against the verified SHA256 values. Dataset downloads can be large; the script runs downloads only when explicitly invoked. The benchmark downloader requests only images_8, camera transforms and scene metadata.

For a small transfer check before requesting the full benchmark, download one
fixed scene to a new directory:

```bash
python scripts/download_assets.py benchmark \
  --benchmark-scene 9641a1ed7963ce5ca734cff3e6ccea3dfa8bcb0b0a3ff78f65d32a080de2d71e \
  --output "$DATA_ROOT/benchmark-download-smoke"
```

Its `download.json` has `complete: false`; this directory is not a substitute
for the 140-scene benchmark. The one-scene mode lists only the selected scene's
files, avoiding a full repository listing before downloading.

If DL3DV arrives as scene ZIP files, extract them with:

```bash
python scripts/download_assets.py unpack-dl3dv \
  --source "$DATA_ROOT/dl3dv-download" --output "$DATA_ROOT/dl3dv"
```

If the downloaded tree already contains extracted scene directories, use that tree as the DL3DV root directly. Extraction rejects unsafe paths and symlinks; successful scene archives have checksum markers and can be skipped on rerun.

## RE10K training videos and cameras

The official metadata archive expands to `re10k-metadata/metadata/RealEstate10K/{train,test}`. Set the path to the directory containing both train and test:

```bash
export RE_METADATA="$DATA_ROOT/re10k-metadata/metadata/RealEstate10K"
python scripts/select_re10k_train_long_metadata.py \
  --metadata-root "$RE_METADATA" --output "$DATA_ROOT/re10k-train-selection.json" \
  --primary-count 150 --reserve-count 450 --min-frame-count 146 --seed 0
python scripts/download_re10k_train_long.py \
  --selection "$DATA_ROOT/re10k-train-selection.json" \
  --output-root "$DATA_ROOT/re10k-train" --target-scenes 150
```

Pass `--ffmpeg-dir /path/to/bin --deno /path/to/deno` if necessary. The downloader extracts a centered 146-frame metadata window, matches video timestamps, rejects duplicate/nonmonotonic matches, center-crops to 490×280 and updates intrinsics. It inverts world-to-camera poses to camera-to-world. Source failures are recorded; train reserves are tried in fixed order. It exits nonzero if fewer than 150 scenes complete. Use `--retry-failed` to retry recorded failures; changing selection or geometry requires a new output directory.

## Prepare the paper-configured training roots

```bash
python scripts/prepare_training_data.py \
  --dl3dv-root "$DATA_ROOT/dl3dv" --re10k-root "$DATA_ROOT/re10k-train" \
  --re10k-metadata "$RE_METADATA" \
  --dl3dv-test-index manifests/dl3dv_eval_shard0.json manifests/dl3dv_eval_shard1.json \
  --output "$DATA_ROOT/prepared" --seed 0 --decode-images
```

This creates four indexed roots:

| Root | Selection |
| --- | --- |
| stage1 | 8,590 eligible DL3DV scenes, excluding benchmark identities |
| stage2a | All Stage-I scenes with at least 146 valid frames |
| stage2b_dl3dv | 118 scenes from the Stage-IIa pool |
| stage2b_re10k | 150 official-training RE10K sequences, excluding official-test IDs and source URLs |

Selection is deterministic: SHA256 ordering of seed, role and scene ID. Stage-I candidates require at least 19 frames; long-sequence pools require at least 146. All referenced images, metadata and finite camera matrices are checked; `--decode-images` also decodes images. If too few scenes qualify, preparation fails instead of duplicating IDs. The script writes relative training indices, scene metadata hashes, selection manifests and a split audit. Scene data are linked without duplicating RGBs.

The deterministic rule above defines the released training split. Stages may share training identities; all training identities are excluded from evaluation.

## Prepare fixed evaluation scenes

Combine the two provided 70-scene DL3DV lists:

```bash
python -c 'import json; from pathlib import Path; a=json.loads(Path("manifests/dl3dv_eval_shard0.json").read_text()); b=json.loads(Path("manifests/dl3dv_eval_shard1.json").read_text()); print(json.dumps(a+b))' > "$DATA_ROOT/dl3dv140-index.json"
python scripts/prepare_dl3dv140_context_eval.py \
  --dataset-root "$DATA_ROOT/dl3dv140" --index "$DATA_ROOT/dl3dv140-index.json" \
  --output-root "$DATA_ROOT/eval-dl3dv128" --context-count 128
```

For RE10K, use the fixed 180-scene list with official-test metadata:

```bash
python scripts/select_re10k_train_long_metadata.py --split test \
  --metadata-root "$RE_METADATA" --scene-list manifests/re10k_eval180.json \
  --output "$DATA_ROOT/re10k-test-selection.json" --min-frame-count 146
python scripts/download_re10k_train_long.py \
  --selection "$DATA_ROOT/re10k-test-selection.json" \
  --output-root "$DATA_ROOT/re10k-test" --target-scenes 180
python scripts/prepare_dl3dv140_context_eval.py \
  --dataset-root "$DATA_ROOT/re10k-test" --index "$DATA_ROOT/re10k-test/test_index.json" \
  --output-root "$DATA_ROOT/eval-re10k128" --context-count 128
```

Unavailable fixed test videos must be reported; they are not replaced silently. Test video reconstruction can differ from previously preprocessed shards due to source video changes and compression; retain timestamp/preprocessing records. If you already have trusted pixelSplat-style RE10K shards, the alternative converter `prepare_re10k_context_eval.py --input-root ... --output-root ... --context-count 128 --num-scenes 180 --scene-list manifests/re10k_eval180.json --min-frame-count 146` is supplied.

Prepared evaluation roots contain `test_index.json`, per-scene `transforms.json`, `images_8` and `split.json`. The 128-context setting packs 146 frames and reserves zero-based indices 7,15,...,143 as the 18 targets. Use 64 or 96 contexts with the same preparation script for 73 or 109 packed frames.

## Smoke preparation

Use `--smoke` on training preparation to create two-scene pools for each source. This uses a limited candidate scan and is explicitly marked as a smoke split. Prepare one scene per evaluation root with `--num-scenes 1`. The training driver rejects mixing smoke datasets with the paper profile. See README for the two-GPU smoke invocation.
