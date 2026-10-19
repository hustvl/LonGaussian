#!/usr/bin/env python3
"""Export path-free reference split, checkpoint lineage and per-view metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_main_table import CONTEXTS, DATASETS, EXPECTED_COUNTS, fixed_ids, packed_split
from scripts.prepare_training_data import identity, read_ids

ROLES = ("stage1", "stage2a", "stage2b_dl3dv", "stage2b_re10k")
STAGES = ("stage1", "stage2a", "stage2b")
STEPS = {"stage1": 2148, "stage2a": 900, "stage2b": 300}
CONFIG_KEYS = (
    "e11_variant", "depth_filter_variant", "depth_filter_enabled", "sky_mask_drop",
    "sky_sphere", "sky_sphere_color_mode", "sky_mask_score_thr", "min_rel_gap",
    "visible_ratio_threshold", "max_rel_depth_spread", "seed", "deterministic",
    "num_frames", "image_scale", "window_size", "overlap_size",
    "gaussian_head_batch_size", "two_pass_pose", "orientation_aware_two_pass",
    "calibrate_local_points_to_k", "track_merge", "merge_mode",
    "dense_accumulation", "track_dense_complement", "global_complement_voxel_factor",
)


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def training_selection(root: Path) -> dict:
    selections = {}
    for role in ROLES:
        manifest = json.loads((root / role / "manifest.json").read_text())
        ids = read_ids(root / role / "train_index.json")
        rows = manifest["scenes"]
        if len(rows) != len(ids) or [row["id"] for row in rows] != ids:
            raise ValueError(f"{role}: manifest and train_index disagree")
        if role == "stage1" and len(ids) != 8590:
            raise ValueError("stage1 must contain 8,590 scenes")
        if role == "stage2b_dl3dv" and len(ids) != 118:
            raise ValueError("stage2b_dl3dv must contain 118 scenes")
        if role == "stage2b_re10k" and len(ids) != 150:
            raise ValueError("stage2b_re10k must contain 150 scenes")
        selections[role] = {
            "seed": manifest["seed"], "selection_rule": manifest["selection_rule"],
            "count": len(ids), "id_list_sha256": digest(ids),
            "scenes": [{"id": row["id"], "frames": row["frames"],
                        "metadata_sha256": row["metadata_sha256"]} for row in rows],
        }
    dl_test = {identity(item) for item in fixed_ids("dl3dv")}
    re_test = {identity(item) for item in fixed_ids("re10k")}
    if dl_test.intersection(row["id"] for row in selections["stage1"]["scenes"]) or re_test.intersection(row["id"] for row in selections["stage2b_re10k"]["scenes"]):
        raise ValueError("training/evaluation identity overlap")
    return {"format": "longaussian_reference_selection_v1", "selections": selections}


def stage_lineage(reports: dict[str, Path]) -> dict:
    result = {}
    previous = None
    for stage in STAGES:
        report = json.loads(reports[stage].read_text())
        if report.get("stage") != stage or report.get("profile", "paper") != "paper":
            raise ValueError(f"{stage}: wrong stage/profile")
        if int(report.get("steps", -1)) != STEPS[stage]:
            raise ValueError(f"{stage}: wrong optimizer-update count")
        if previous is not None and report.get("parent_sha256") != previous:
            raise ValueError(f"{stage}: broken checkpoint parent chain")
        sha = report["sha256"]
        if len(sha) != 64:
            raise ValueError(f"{stage}: invalid checkpoint SHA256")
        result[stage] = {
            "file": stage + ".pt", "steps": report["steps"], "sha256": sha,
            "parent_sha256": report.get("parent_sha256"),
            "dataset_manifest_sha256": report["dataset_manifests"],
            "external_weight_sha256": report["external_weights"],
        }
        previous = sha
    return {"format": "longaussian_reference_lineage_v1", "stages": result}


def export_metrics(paths: list[Path], dataset: str, count: int) -> dict:
    rows_by_id = {}
    source_hashes = []
    configuration = None
    for path in paths:
        source_hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
        payload = json.loads(path.read_text())
        config = {key: payload[key] for key in CONFIG_KEYS if key in payload}
        if configuration is None:
            configuration = config
        elif config != configuration:
            raise ValueError(f"{dataset} ctx{count}: shard configurations differ")
        for scene in payload["scenes"]:
            scene_id = scene["scene"]
            if scene_id in rows_by_id:
                raise ValueError(f"duplicate result for {scene_id}")
            rows_by_id[scene_id] = scene
    expected = [identity(item) for item in fixed_ids(dataset)]
    if len(expected) != EXPECTED_COUNTS[dataset] or set(rows_by_id) != set(expected):
        raise ValueError(f"{dataset} ctx{count}: result scene set differs from fixed manifest")
    contexts, targets = packed_split(count)
    scenes = []
    for scene_id in expected:
        source = rows_by_id[scene_id]
        if "error" in source or source.get("ctx_list") != contexts or source.get("tgt_list") != targets:
            raise ValueError(f"{dataset} ctx{count}: failed scene or wrong split: {scene_id}")
        views = source["views"]
        if [row["target_index"] for row in views] != targets:
            raise ValueError(f"{dataset} ctx{count}: wrong target views: {scene_id}")
        clean_views = []
        for view in views:
            clean = {key: view[key] for key in ("target_index", "psnr", "ssim", "lpips")}
            if not all(math.isfinite(float(clean[key])) for key in ("psnr", "ssim", "lpips")):
                raise ValueError(f"{dataset} ctx{count}: nonfinite view metric")
            clean_views.append(clean)
        scenes.append({"scene": scene_id, "num_gaussians": source["num_gaussians"],
                       "views": clean_views})
    all_views = [view for scene in scenes for view in scene["views"]]
    summary = {"num_scenes": len(scenes), "num_test_views": len(all_views)}
    for key in ("psnr", "ssim", "lpips"):
        values = [float(view[key]) for view in all_views]
        summary[key + "_mean"] = statistics.fmean(values)
        summary[key + "_std"] = statistics.pstdev(values)
    summary["gaussians_per_scene_mean"] = statistics.fmean(float(row["num_gaussians"]) for row in scenes)
    return {"format": "longaussian_reference_evaluation_v1", "dataset": dataset,
            "context_count": count, "num_frames": count + count // 7,
            "source_metrics_sha256": source_hashes, "configuration": configuration,
            "summary": summary, "scenes": scenes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    for stage in STAGES:
        parser.add_argument(f"--{stage}-report", type=Path, required=True)
    for dataset in DATASETS:
        for count in CONTEXTS:
            parser.add_argument(f"--{dataset}-{count}", nargs="+", type=Path, required=True,
                                help="One full metrics JSON or disjoint shard metrics JSONs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing output: {args.output}")
    selection = training_selection(args.training_root)
    lineage = stage_lineage({stage: getattr(args, stage + "_report") for stage in STAGES})
    evaluations = {(dataset, count): export_metrics(getattr(args, f"{dataset}_{count}"), dataset, count)
                   for dataset in DATASETS for count in CONTEXTS}
    args.output.mkdir(parents=True)
    write_json(args.output / "training_selection.json", selection)
    write_json(args.output / "checkpoint_lineage.json", lineage)
    write_json(args.output / "summary.json", {
        "checkpoint_sha256": lineage["stages"]["stage2b"]["sha256"],
        "settings": {dataset: {str(count): evaluations[dataset, count]["summary"] for count in CONTEXTS}
                     for dataset in DATASETS},
    })
    for (dataset, count), value in evaluations.items():
        write_json(args.output / "evaluations" / f"{dataset}_ctx{count}.json", value)
    print(f"Exported path-free reference artifacts to {args.output}")


if __name__ == "__main__":
    main()
