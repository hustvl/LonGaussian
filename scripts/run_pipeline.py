#!/usr/bin/env python3
"""Validated training -> reconstruction -> DL3DV/RE10K evaluation workflow."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_training_data import read_ids, identity
from loger.utils.checkpoint import sha256_file


def validate_evaluation(root, expected, trained):
    ids = read_ids(root / "test_index.json")
    if len(ids) != expected:
        raise ValueError(f"Expected {expected} evaluation scenes, found {len(ids)}")
    if {identity(x) for x in ids}.intersection(trained):
        raise ValueError("Evaluation identity overlaps a training identity")
    for entry in ids:
        scene = root / entry
        split = json.loads((scene / "split.json").read_text())
        targets = list(range(7, 146, 8))
        contexts = [i for i in range(146) if i not in targets]
        if split["ctx_list"] != contexts or split["tgt_list"] != targets:
            raise ValueError(f"Expected fixed ctx128/tgt18 split for {entry}")
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "weights", "output", "dl3dv-eval", "re10k-eval", "inference-scene"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--profile", choices=["paper","smoke"], default="paper")
    parser.add_argument("--training-output", type=Path,
                        help="Reuse/continue a separately located, hash-verified training run")
    parser.add_argument("--resume-completed", action="store_true")
    args = parser.parse_args()
    smoke = args.profile == "smoke"
    out = args.output.resolve()
    training_out = (args.training_output or (out / "training")).resolve()
    weights = args.weights.resolve()
    trained_dl = set(read_ids(args.prepared / "stage1/train_index.json"))
    trained_re = set(read_ids(args.prepared / "stage2b_re10k/train_index.json"))
    validate_evaluation(args.dl3dv_eval, 1 if smoke else 140, trained_dl)
    validate_evaluation(args.re10k_eval, 1 if smoke else 180, trained_re)
    train = [sys.executable, str(ROOT / "scripts/train_pipeline.py"), "--prepared", str(args.prepared),
             "--weights", str(weights), "--output", str(training_out), "--gpus", args.gpus, "--profile", args.profile]
    if args.resume_completed:
        train.append("--resume-completed")
    subprocess.run(train, check=True)
    env = dict(os.environ, PYTHON=sys.executable, PYTHONDONTWRITEBYTECODE="1", GPU=args.gpus.split(",")[0],
               MODEL_NAME=str(weights / "LoGeR_star/latest.pt"), GAUSSIAN_HEAD=str(training_out / "stage2b.pt"),
               SKY_MASK_ONNX=str(weights / "skyseg.onnx"), ALLTRACKER_CHECKPOINT=str(weights / "alltracker.pth"),
               ALLTRACKER_SHA256=sha256_file(weights / "alltracker.pth"))
    # AllTracker's four-level correlation pyramid is not valid at 0.125 for
    # the released frame sizes.  Half resolution is the smallest smoke-test
    # setting exercised by the complete tracking/merging path.
    extra = ["--image_scale", "0.5"] if smoke else []
    inference_env = dict(env, INPUT=str(args.inference_scene.resolve()), NUM_FRAMES="24", OUTPUT_DIR=str(out / "inference"))
    subprocess.run(["bash",str(ROOT / "scripts/infer_longaussian.sh"),*extra], env=inference_env, check=True)
    info = json.loads((out / "inference/inference.json").read_text())
    if info["num_gaussians"] <= 0 or info["merge_stats"]["merge_mode"] != "neural" or not info["sky_sphere"] or info["depth_filter"] is None:
        raise RuntimeError("Full inference did not complete the expected construction stages")
    summaries = {}
    for name, root, launcher in [("dl3dv",args.dl3dv_eval,"evaluate_dl3dv140.sh"),("re10k",args.re10k_eval,"evaluate_re10k.sh")]:
        env_eval = dict(env, DATASET_ROOT=str(root.resolve()), OUTPUT_DIR=str(out/name), NUM_FRAMES="146", NUM_SCENES="0")
        subprocess.run(["bash",str(ROOT / "scripts" / launcher),*extra],env=env_eval,check=True)
        metrics = json.loads((out/name/"metrics_dl3dv_subset.json").read_text())
        for scene in metrics["scenes"]:
            if "error" in scene or len(scene["views"]) != 18 or not all(math.isfinite(scene[key]) for key in ("psnr_mean","ssim_mean","lpips_mean")):
                raise RuntimeError(f"Invalid {name} evaluation result")
        summaries[name] = metrics["summary"]
    (out / "COMPLETE.json").write_text(json.dumps({"profile":args.profile, "stage2b_sha256":sha256_file(training_out/"stage2b.pt"),
        "inference_gaussians":info["num_gaussians"], "evaluation":summaries},indent=2)+"\n")
    print(f"Complete ({args.profile}): {out / 'COMPLETE.json'}")


if __name__ == "__main__":
    main()
