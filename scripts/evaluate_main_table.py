#!/usr/bin/env python3
"""Run and verify all six fixed main-table evaluation settings."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loger.utils.checkpoint import sha256_file
from scripts.prepare_training_data import identity, read_ids

CONTEXTS = (64, 96, 128)
DATASETS = ("dl3dv", "re10k")
EXPECTED_COUNTS = {"dl3dv": 140, "re10k": 180}
EXTERNAL_HASHES = {
    "LoGeR_star/latest.pt": "68a0ae71408eb5605b8409296e7d5bf43c87853595766195257af97c5e60373c",
    "alltracker.pth": "ffd9ebcfb6d206d594b646999a150540f92c049cf9b2bf940facf7123f62aa1d",
    "skyseg.onnx": "ab9c34c64c3d821220a2886a4a06da4642ffa14d5b30e8d5339056a089aa1d39",
}


def packed_split(context_count: int) -> tuple[list[int], list[int]]:
    frames = context_count + context_count // 7
    targets = list(range(7, frames, 8))
    target_set = set(targets)
    return [index for index in range(frames) if index not in target_set], targets


def fixed_ids(dataset: str) -> list[str]:
    if dataset == "dl3dv":
        return read_ids(ROOT / "manifests/dl3dv_eval_shard0.json") + read_ids(
            ROOT / "manifests/dl3dv_eval_shard1.json"
        )
    return read_ids(ROOT / "manifests/re10k_eval180.json")


def validate_root(root: Path, dataset: str, context_count: int, expected_ids: list[str]) -> None:
    ids = read_ids(root / "test_index.json")
    if [identity(item) for item in ids] != [identity(item) for item in expected_ids]:
        raise ValueError(f"{dataset} ctx{context_count}: test_index.json differs from fixed manifest")
    contexts, targets = packed_split(context_count)
    for entry in ids:
        scene = root / entry
        split = json.loads((scene / "split.json").read_text())
        if split.get("ctx_list") != contexts or split.get("tgt_list") != targets:
            raise ValueError(f"{dataset} ctx{context_count}: invalid split for {entry}")
        metadata = json.loads((scene / "transforms.json").read_text())
        if len(metadata.get("frames", [])) != len(contexts) + len(targets):
            raise ValueError(f"{dataset} ctx{context_count}: invalid frame count for {entry}")
        if not (scene / "images_8").is_dir():
            raise ValueError(f"{dataset} ctx{context_count}: missing images_8 for {entry}")


def validate_metrics(path: Path, dataset: str, context_count: int, expected_ids: list[str],
                     *, expected_sources: dict | None = None) -> dict:
    metrics = json.loads(path.read_text())
    if expected_sources is not None:
        for key, expected_path in expected_sources.items():
            if not isinstance(metrics.get(key), str) or Path(metrics[key]).resolve() != expected_path.resolve():
                raise ValueError(f"{path}: {key} differs from requested input")
        required_protocol = {
            "num_frames": context_count + context_count // 7, "image_scale": 1.0,
            "window_size": 12, "overlap_size": 3, "gaussian_head_batch_size": 1,
            "two_pass_pose": True, "orientation_aware_two_pass": True,
            "calibrate_local_points_to_k": True, "merge_mode": "neural",
            "track_dense_complement": True, "dense_accumulation": False,
            "depth_filter_enabled": True, "depth_filter_variant": "m01_v1_r30",
            "min_rel_gap": 0.01, "visible_ratio_threshold": 0.30,
            "max_rel_depth_spread": 0.15, "sky_mask_drop": True,
            "sky_sphere": True, "sky_mask_score_thr": 160,
            "seed": 0, "deterministic": True,
        }
        for key, expected_value in required_protocol.items():
            if metrics.get(key) != expected_value:
                raise ValueError(f"{path}: {key} differs from formal protocol")
    scenes = metrics.get("scenes")
    if not isinstance(scenes, list) or len(scenes) != len(expected_ids):
        raise ValueError(f"{path}: expected {len(expected_ids)} scene results")
    expected_scene_ids = [identity(entry) for entry in expected_ids]
    if [row.get("scene") for row in scenes] != expected_scene_ids:
        raise ValueError(f"{path}: scene identities/order differ from fixed manifest")
    _, target_indices = packed_split(context_count)
    for row in scenes:
        if "error" in row or row.get("tgt_list") != target_indices:
            raise ValueError(f"{path}: failed scene or wrong target split: {row.get('scene')}")
        views = row.get("views")
        if not isinstance(views, list) or [view.get("target_index") for view in views] != target_indices:
            raise ValueError(f"{path}: wrong held-out views: {row.get('scene')}")
        for view in views:
            for key in ("psnr", "ssim", "lpips"):
                if not math.isfinite(float(view[key])):
                    raise ValueError(f"{path}: nonfinite {key}: {row.get('scene')}")
        for key in ("psnr_mean", "ssim_mean", "lpips_mean"):
            if not math.isfinite(float(row[key])):
                raise ValueError(f"{path}: nonfinite {key}: {row.get('scene')}")
    summary = metrics.get("summary", {})
    if summary.get("num_scenes") != len(expected_ids) or summary.get("num_test_views") != len(expected_ids) * len(target_indices):
        raise ValueError(f"{path}: inconsistent summary counts")
    for key in ("psnr_mean", "ssim_mean", "lpips_mean"):
        actual = sum(float(view[key[:-5]]) for row in scenes for view in row["views"]) / summary["num_test_views"]
        if not math.isfinite(float(summary[key])) or not math.isclose(actual, float(summary[key]), abs_tol=1e-6):
            raise ValueError(f"{path}: inconsistent {key}")
    return {"dataset": dataset, "context": context_count, **summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for dataset in DATASETS:
        for context_count in CONTEXTS:
            parser.add_argument(f"--{dataset}-{context_count}", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--dry-run", action="store_true", help="validate inputs and print six commands without GPU work")
    parser.add_argument("--resume-completed", action="store_true", help="reuse only complete, validated setting outputs")
    args = parser.parse_args()

    roots = {(dataset, count): getattr(args, f"{dataset}_{count}").resolve() for dataset in DATASETS for count in CONTEXTS}
    expected = {dataset: fixed_ids(dataset) for dataset in DATASETS}
    for dataset in DATASETS:
        if len(expected[dataset]) != EXPECTED_COUNTS[dataset]:
            raise ValueError(f"{dataset}: fixed manifest has unexpected scene count")
        for count in CONTEXTS:
            validate_root(roots[dataset, count], dataset, count, expected[dataset])

    weights = args.weights.resolve()
    required = [weights / "LoGeR_star/latest.pt", weights / "alltracker.pth", weights / "skyseg.onnx", args.checkpoint, ROOT / "configs/LoGeR_star/original_config.yaml"]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    for name, expected_hash in EXTERNAL_HASHES.items():
        if sha256_file(weights / name) != expected_hash:
            raise ValueError(f"external weight SHA256 differs from WEIGHTS.md: {name}")
    tracker_hash = EXTERNAL_HASHES["alltracker.pth"]
    checkpoint_hash = sha256_file(args.checkpoint)
    output = args.output.resolve()
    if output.exists() and not args.resume_completed and any(output.iterdir()):
        raise FileExistsError(f"refusing nonempty output directory: {output}")
    summaries = {}
    for dataset in DATASETS:
        summaries[dataset] = {}
        for count in CONTEXTS:
            setting_output = output / dataset / f"ctx{count}"
            metrics_path = setting_output / "metrics_dl3dv_subset.json"
            sources = {"gaussian_head": args.checkpoint, "model_name": weights / "LoGeR_star/latest.pt",
                       "dataset_root": roots[dataset, count]}
            if args.resume_completed and metrics_path.is_file():
                summaries[dataset][str(count)] = validate_metrics(
                    metrics_path, dataset, count, expected[dataset], expected_sources=sources)
                continue
            if setting_output.exists() and any(setting_output.iterdir()):
                raise FileExistsError(f"refusing incomplete setting directory: {setting_output}")
            env = dict(os.environ, PYTHON=sys.executable, PYTHONDONTWRITEBYTECODE="1", GPU=args.gpu,
                       DATASET_ROOT=str(roots[dataset, count]), OUTPUT_DIR=str(setting_output),
                       NUM_FRAMES=str(count + count // 7), NUM_SCENES="0",
                       MODEL_NAME=str(weights / "LoGeR_star/latest.pt"), GAUSSIAN_HEAD=str(args.checkpoint.resolve()),
                       ALLTRACKER_CHECKPOINT=str(weights / "alltracker.pth"), ALLTRACKER_SHA256=tracker_hash,
                       SKY_MASK_ONNX=str(weights / "skyseg.onnx"))
            launcher = ROOT / "scripts" / ("evaluate_re10k.sh" if dataset == "re10k" else "evaluate_dl3dv140.sh")
            print(f"{dataset} ctx{count}: {launcher} -> {setting_output}", flush=True)
            if not args.dry_run:
                subprocess.run(["bash", str(launcher)], env=env, check=True)
                summaries[dataset][str(count)] = validate_metrics(
                    metrics_path, dataset, count, expected[dataset], expected_sources=sources)
    if args.dry_run:
        print("All six fixed roots and required files validated; no GPU work or output written.")
        return
    output.mkdir(parents=True, exist_ok=True)
    summary = {"checkpoint_sha256": checkpoint_hash, "settings": summaries}
    (output / "main_table_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Validated six settings: {output / 'main_table_summary.json'}")


if __name__ == "__main__":
    main()
