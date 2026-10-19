#!/usr/bin/env python3
"""Train the paper recipe with validated stage handoffs; no historical checkpoints."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from loger.utils.checkpoint import sha256_file
from scripts.prepare_training_data import read_ids, digest


def build_command(stage, roots, weights, out, gpus, smoke=False, accumulation=1,
                  activation_offload=True):
    world = len(gpus.split(","))
    if accumulation < 1 or len(set(gpus.split(","))) != world:
        raise ValueError("Require positive accumulation and distinct GPU IDs")
    if (not smoke and world * accumulation != 4) or (smoke and world != 2):
        raise ValueError("Paper requires effective batch four (GPUs × accumulation); smoke requires two GPUs")
    steps = {"stage1": 1 if smoke else math.ceil(8590 / 4), "stage2a": 4 if smoke else 900,
             "stage2b": 4 if smoke else 300}[stage]
    data = roots / ("stage2b_dl3dv" if stage == "stage2b" else stage)
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={world}",
               str(ROOT / "run.py"), "--train", "--gaussian", "--dataset", "dl3dv", "--dataset_root", str(data),
               "--model_name", str(weights / "LoGeR_star/latest.pt"), "--config", str(ROOT / "configs/LoGeR_star/original_config.yaml"),
               "--output_folder", str(out / stage), "--require_empty_output", "--train_seed", "0", "--ddp_stripe_scenes",
               "--calibrate_local_points_to_k", "--long_sequence_protocol", "--long_context_weights", "1" if stage == "stage1" else "1,1,1,1",
               "--long_context_counts", "16" if stage == "stage1" else "16,64,96,128",
               "--train_samples", "16" if stage == "stage1" else "128", "--nvs_holdout", "3",
               "--long_train_target_limit", "1" if smoke else "3", "--long_train_context_render_limit", "1" if smoke else "3",
               "--window_size", "12", "--overlap_size", "3", "--image_scale",
               "0.125" if smoke and stage == "stage1" else ("0.5" if smoke else "1"),
               "--grad_clip_norm", "1", "--epochs", "1", "--scenes_per_epoch", str(steps * world * accumulation),
               "--gradient_accumulation_steps", str(accumulation),
               "--train_max_global_steps", str(steps), "--train_gaussian_head_checkpoint", "--train_gaussian_head_batch_size", "1",
               "--train_loss_checkpoint", "--train_render_checkpoint_chunk_size", "1", "--fail_if_zero_updates",
               "--save_every_steps", "1" if smoke else "100", "--save_persistent_every_steps", "1" if smoke else "300"]
    if stage == "stage1":
        command += ["--long_context_modes", "dense", "--long_context_intervals", "1", "--lr", "1e-5",
                    "--aniso_reg_weight", "0.1", "--aniso_threshold", "4", "--offset_reg_weight", "0.01"]
    else:
        # Preserve the complete long-context graph without keeping all merger
        # activations resident alongside the rasterizer's backward buffers.
        if activation_offload:
            command += ["--train_save_activations_on_cpu"]
        command += ["--pretrained_decoder", str(out / ("stage1.pt" if stage == "stage2a" else "stage2a.pt")),
                    "--track_merge_train", "--merge_mode", "neural", "--neural_dense_loss_weight", "0",
                    "--neural_merge_loss_weight", "1", "--long_sparse_context_counts", "16", "--long_sparse_intervals", "8",
                    "--aniso_reg_weight", "0", "--offset_reg_weight", "0"]
        if stage == "stage2a":
            command += ["--neural_freeze_base", "--lr", "1.25e-5"]
        else:
            command += ["--neural_freeze_decoder", "--train_head_lr", "2e-6", "--train_merger_lr", "1.25e-5",
                        "--long_head_update_max_context", "16", "--train_bucket_best_postprocess",
                        "--sky_mask_onnx", str(weights / "skyseg.onnx"),
                        "--balanced_train_root_a", str(roots / "stage2b_dl3dv"),
                        "--balanced_train_root_b", str(roots / "stage2b_re10k"),
                        "--balanced_train_count_a", "2" if smoke else "118", "--balanced_train_count_b", "2" if smoke else "150"]
    return command, steps


def verify_checkpoint(path, steps):
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("global_step") != steps:
        raise RuntimeError(f"Incomplete stage: expected {steps} updates, found {checkpoint.get('global_step')}")
    state = checkpoint.get("model", {})
    for prefix in ("gaussian_decoder.", "gaussian_head.", "neural_track_merger."):
        tensors = [v for k, v in state.items() if k.startswith(prefix)]
        if not tensors or any(not torch.isfinite(v).all() for v in tensors):
            raise RuntimeError(f"Missing/nonfinite module: {prefix}")
    return sha256_file(path)


def validate_roots(roots, smoke):
    audit = json.loads((roots / "split_audit.json").read_text())
    if bool(audit["smoke"]) != smoke:
        raise ValueError("Smoke/paper dataset profile mismatch")
    expected = {"stage1": 2 if smoke else 8590, "stage2b_dl3dv": 2 if smoke else 118, "stage2b_re10k": 2 if smoke else 150}
    manifests = {}
    pools = {}
    for stage in ("stage1", "stage2a", "stage2b_dl3dv", "stage2b_re10k"):
        root = roots / stage
        entries = read_ids(root / "train_index.json")
        if not entries or (stage in expected and len(entries) != expected[stage]):
            raise ValueError(f"Incorrect {stage} count: {len(entries)}")
        manifest = json.loads((root / "manifest.json").read_text())
        if digest(entries) != manifest["scene_list_sha256"]:
            raise ValueError(f"Modified {stage} index")
        by_id = {row["id"]: row for row in manifest["scenes"]}
        excluded = set(audit["excluded_re10k_ids"] if stage == "stage2b_re10k" else audit["excluded_dl3dv_ids"])
        if excluded.intersection(entries):
            raise ValueError(f"Evaluation leakage in {stage}")
        for entry in entries:
            meta = root / entry / "transforms.json"
            if sha256_file(meta) != by_id[entry]["metadata_sha256"]:
                raise ValueError(f"Changed scene metadata: {stage}/{entry}")
        pools[stage] = set(entries)
        manifests[stage] = sha256_file(root / "manifest.json")
    if not pools["stage2a"] <= pools["stage1"] or not pools["stage2b_dl3dv"] <= pools["stage2a"]:
        raise ValueError("DL3DV training subset relation violated")
    return manifests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="Use 2 with --gpus 0,1 to retain the paper's effective batch of four scenes")
    parser.add_argument("--activation-offload", action=argparse.BooleanOptionalAction, default=True,
                        help="Store Stage-II model activations on CPU (default); trades host RAM/transfers for GPU memory without changing the objective")
    parser.add_argument("--profile", choices=["paper", "smoke"], default="paper")
    parser.add_argument("--stage", choices=["all", "stage1", "stage2a", "stage2b"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-completed", action="store_true", help="Only reuse hash-verified completed stages; never auto-resume interrupted stages")
    args = parser.parse_args()
    roots, weights, out = args.prepared.resolve(), args.weights.resolve(), args.output.resolve()
    smoke = args.profile == "smoke"
    manifests = validate_roots(roots, smoke)
    files = [weights / "LoGeR_star/latest.pt", weights / "alltracker.pth", weights / "skyseg.onnx"]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    stages = ["stage1", "stage2a", "stage2b"] if args.stage == "all" else [args.stage]
    commands = [(stage, *build_command(stage, roots, weights, out, args.gpus, smoke,
                                     args.gradient_accumulation_steps,
                                     args.activation_offload)) for stage in stages]
    if args.dry_run:
        print(json.dumps({"profile": args.profile, "commands": commands}, indent=2))
        return
    import torch
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               PYTHONHASHSEED="0", OMP_NUM_THREADS="1", LOGER_DISABLE_TORCH_COMPILE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8",
               ALLTRACKER_CHECKPOINT=str(weights / "alltracker.pth"), ALLTRACKER_SHA256=sha256_file(weights / "alltracker.pth"),
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    # Probe the exact visible-device environment used by torchrun.
    subprocess.run([sys.executable, "-c", "import torch; assert torch.cuda.device_count()==" + str(len(args.gpus.split(','))) +
                    "; x=torch.ones(8,8,device='cuda'); assert torch.isfinite(x@x).all(); torch.cuda.synchronize()"], env=env, check=True)
    weight_hashes = {str(path.relative_to(weights)): sha256_file(path) for path in files}
    out.mkdir(parents=True, exist_ok=True)
    for stage, command, steps in commands:
        final = out / stage / "gaussian_head.pt"
        report_path = out / (stage + ".json")
        parent = None if stage == "stage1" else out / ("stage1.pt" if stage == "stage2a" else "stage2a.pt")
        parent_sha = sha256_file(parent) if parent else weight_hashes["LoGeR_star/latest.pt"]
        signature = digest({"command": command, "manifests": manifests, "weights": weight_hashes, "parent": parent_sha})
        if args.resume_completed and report_path.exists():
            report = json.loads(report_path.read_text())
            if report["signature"] != signature or report["sha256"] != verify_checkpoint(final, steps):
                raise RuntimeError(f"Cannot reuse changed stage: {stage}")
            print(f"Verified completed {stage}", flush=True)
            continue
        if (out / stage).exists() or report_path.exists() or (out / (stage + ".pt")).exists():
            raise FileExistsError(f"Refusing existing stage output: {out / stage}")
        print(f"Starting {stage}: {steps} updates ({args.profile})", flush=True)
        with (out / (stage + ".log")).open("x") as log:
            subprocess.run(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        checksum = verify_checkpoint(final, steps)
        (out / (stage + ".pt")).symlink_to(Path(stage) / "gaussian_head.pt")
        report = {"stage": stage, "profile": args.profile, "steps": steps, "sha256": checksum,
                  "parent_sha256": parent_sha, "dataset_manifests": manifests, "external_weights": weight_hashes,
                  "signature": signature, "command": command}
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Completed {stage}: {checksum}", flush=True)


if __name__ == "__main__":
    main()
