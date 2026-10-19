import os
import glob
import time
import random
from datetime import timedelta

import argparse
import contextlib
import inspect
import math
import tempfile
import shutil
import json
import yaml
import torch
import torch.distributed as dist
import torch.nn as nn
import cv2
from PIL import Image
from torchvision import transforms
from natsort import natsorted
from typing import Dict, List, Optional, Sequence, Tuple

from pathlib import Path
from loger.utils.rotation import mat_to_quat
from loger.utils.geometry import depth_edge
from loger.models.pi3 import Pi3


def _preload_cached_gsplat_extension() -> None:
    """Reuse the already-built py311 extension when ninja is unavailable."""
    import sys

    extension_dir = Path.home() / ".cache/torch_extensions/py311_cu124/gsplat_cuda"
    extension_so = extension_dir / "gsplat_cuda.so"
    if not extension_so.is_file() or "gsplat.csrc" in sys.modules:
        return
    sys.path.insert(0, str(extension_dir))
    try:
        import gsplat_cuda
    except ImportError:
        return
    sys.modules["gsplat.csrc"] = gsplat_cuda


_preload_cached_gsplat_extension()


# ---------------------------------------------------------------------------
#  DDP helpers
# ---------------------------------------------------------------------------

def _is_ddp() -> bool:
    return "LOCAL_RANK" in os.environ


def _rank() -> int:
    if _is_ddp():
        return dist.get_rank()
    return 0


def _log(*args, **kwargs):
    if _rank() == 0:
        print(*args, **kwargs)


def setup_ddp():
    """Initialize DDP process group. Call once at the start of main()."""
    if not _is_ddp():
        return None
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    # C128 dense + merge can exceed the default 30m/10m NCCL timeout when
    # ranks hit unequal scene cost under striping; keep waiting for hours.
    timeout_min = int(os.environ.get("LOGER_DDP_TIMEOUT_MIN", "180"))
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=timeout_min))
    return local_rank


def cleanup_ddp():
    if _is_ddp():
        dist.destroy_process_group()


def _broadcast_object(obj, src: int = 0):
    if not _is_ddp():
        return obj
    obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def _sample_epoch_scenes(train_scenes, scenes_per_epoch: int, epoch: int):
    """Sample the same scene list on every DDP rank.

    When ``scenes_per_epoch`` exceeds the pool size, scenes are tiled with a
    deterministic shuffle so multi-GPU stripe mode can request
    ``STEPS * world_size`` slots from a smaller index.
    """
    if scenes_per_epoch > 0:
        epoch_scenes = None
        if _rank() == 0:
            rng = random.Random(epoch)
            if scenes_per_epoch <= len(train_scenes):
                epoch_scenes = rng.sample(list(train_scenes), scenes_per_epoch)
            else:
                epoch_scenes = []
                while len(epoch_scenes) < scenes_per_epoch:
                    chunk = list(train_scenes)
                    rng.shuffle(chunk)
                    epoch_scenes.extend(chunk)
                epoch_scenes = epoch_scenes[:scenes_per_epoch]
        return _broadcast_object(epoch_scenes)

    shuffled = list(train_scenes)
    if _rank() == 0:
        rng = random.Random(epoch)
        rng.shuffle(shuffled)
    return _broadcast_object(shuffled)


def _sample_balanced_epoch_scenes(
    scene_groups: Sequence[Sequence[str]],
    scenes_per_epoch: int,
    epoch: int,
    global_batch_size: int,
) -> list[str]:
    """Return deterministic 1:1 source-balanced global scene slots.

    Each global batch contains the same number of samples from each source.
    When a source is exhausted, sampling continues from a freshly shuffled copy
    rather than dropping it; unequal source pools therefore remain balanced.
    """
    if len(scene_groups) != 2 or any(not group for group in scene_groups):
        raise ValueError("balanced training requires two non-empty scene groups")
    if scenes_per_epoch <= 0:
        raise ValueError("balanced training requires positive --scenes_per_epoch")
    if global_batch_size <= 0 or global_batch_size % len(scene_groups) != 0:
        raise ValueError(
            "global batch size must be positive and divisible by the number of balanced sources"
        )
    if scenes_per_epoch % global_batch_size != 0:
        raise ValueError(
            "--scenes_per_epoch must be divisible by the global batch size for balanced training"
        )

    epoch_scenes = None
    if _rank() == 0:
        rng = random.Random(epoch)
        pools = [list(group) for group in scene_groups]
        queues: list[list[str]] = [[], []]

        def take(source_idx: int) -> str:
            if not queues[source_idx]:
                queues[source_idx] = list(pools[source_idx])
                rng.shuffle(queues[source_idx])
            return queues[source_idx].pop()

        per_source = global_batch_size // len(scene_groups)
        epoch_scenes = []
        for _ in range(scenes_per_epoch // global_batch_size):
            batch = [
                take(source_idx)
                for source_idx in range(len(scene_groups))
                for _ in range(per_source)
            ]
            rng.shuffle(batch)
            epoch_scenes.extend(batch)
    return _broadcast_object(epoch_scenes)


def _resume_epoch_scenes(epoch_scenes: list, *, scene_cursor: int) -> list:
    """Advance a deterministic global epoch list past consumed scene slots."""
    if scene_cursor < 0 or scene_cursor > len(epoch_scenes):
        raise RuntimeError(
            f"resume scene_cursor={scene_cursor} outside "
            f"epoch scene slots={len(epoch_scenes)}"
        )
    return epoch_scenes[scene_cursor:]


def _ddp_stripe_epoch_scenes(epoch_scenes: list) -> list:
    """Shard scenes across ranks for true data-parallel throughput.

    Rank ``r`` receives ``epoch_scenes[r::world_size]``.  Launchers should set
    ``scenes_per_epoch = desired_optimizer_steps * world_size`` so each rank
    sees ``desired_optimizer_steps`` scenes.  All ranks stay in lockstep for
    DDP gradient sync; each step uses a *different* scene per rank.
    """
    if not _is_ddp():
        return epoch_scenes
    ws = dist.get_world_size()
    rank = dist.get_rank()
    if ws <= 1:
        return epoch_scenes
    striped = list(epoch_scenes[rank::ws])
    # Pad to equal length so every rank runs the same number of iterations.
    max_len = max(len(epoch_scenes[r::ws]) for r in range(ws))
    while len(striped) < max_len:
        striped.append(None)
    return striped


def _ddp_scene_step_valid(split_result, device) -> bool:
    """Return True only if every rank loaded the scene successfully."""
    valid = 1 if split_result is not None else 0
    if _is_ddp():
        flag = torch.tensor([valid], device=device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        valid = int(flag.item())
    return valid == 1


def _ddp_any_true(value: bool, device) -> bool:
    """Synchronize a rank-local failure/skip decision across every rank."""
    if not _is_ddp():
        return bool(value)
    flag = torch.tensor([1 if value else 0], device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def _ddp_noop_step(model, optimizer, *, sync_backward: bool = True):
    """Keep DDP collectives aligned on skipped/padded scenes.

    Default ``sync_backward=True`` so a skip still runs a dummy backward and
    participates in the same reducer collectives as a real step.
    """
    if getattr(optimizer, "scene_accumulation_steps", 1) > 1:
        raise RuntimeError("A scene failed during gradient accumulation; refusing a partial or rebalanced optimizer update")
    optimizer.zero_grad(set_to_none=True)
    if sync_backward and _is_ddp():
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if trainable:
            sum(parameter.sum() * 0.0 for parameter in trainable).backward()
        optimizer.zero_grad(set_to_none=True)


def _load_train_skip_scenes(path: str) -> set[str]:
    """Load canonical scene paths that should be skipped before expensive work."""
    if not path:
        return set()
    with open(path, "r", encoding="utf-8") as handle:
        return {
            os.path.realpath(line.strip())
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        }


def _scene_is_blocked(scene_path: str, blocked_scenes: set[str]) -> bool:
    return os.path.realpath(scene_path) in blocked_scenes


def _grads_are_finite(model) -> bool:
    for param in model.parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            return False
    return True


def _nonfinite_grad_names(model):
    """Return concise diagnostics for a skipped update."""
    bad = []
    for name, param in model.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            bad.append(name)
    return bad


def nonfinite_grad_groups(model) -> dict[str, int]:
    """Count non-finite parameter gradients by Gaussian-head submodule."""
    groups = {}
    for name, param in model.named_parameters():
        if param.grad is None or torch.isfinite(param.grad).all():
            continue
        marker = "gaussian_head."
        if marker in name:
            suffix = name.split(marker, 1)[1]
            group = marker + suffix.split(".", 1)[0]
        elif (
            "neural_track_merger." in name
            or "neural_track_set_merger." in name
            or "neural_track_feature_merger." in name
            or "neural_track_direct_slot_merger." in name
        ):
            group = "neural_track_merger"
        else:
            group = name.split(".", 1)[0]
        groups[group] = groups.get(group, 0) + 1
    return groups


def retain_grad_probes(tensors: dict[str, Optional[torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Retain gradients for selected non-leaf tensors without changing the graph."""
    retained = {}
    for name, tensor in tensors.items():
        if tensor is not None and tensor.requires_grad:
            tensor.retain_grad()
            retained[name] = tensor
    return retained


def grad_probe_diagnostics(tensors: dict[str, torch.Tensor]) -> list[str]:
    """Summarize finite and non-finite values in retained tensor gradients."""
    diagnostics = []
    for name, tensor in tensors.items():
        grad = tensor.grad
        if grad is None:
            diagnostics.append(f"{name}:grad=None")
            continue
        finite = torch.isfinite(grad)
        nan_count = int(torch.isnan(grad).sum().item())
        inf_count = int(torch.isinf(grad).sum().item())
        finite_abs_max = (
            float(grad[finite].abs().max().item()) if finite.any() else float("nan")
        )
        diagnostics.append(
            f"{name}:nan={nan_count},inf={inf_count},finite_abs_max={finite_abs_max:.3e}"
        )
    return diagnostics


# Helper function to check if a path is a video file
def is_video_file(path):
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"]
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in video_extensions

# Helper function to extract frames from video
def extract_frames_from_video(video_path, output_dir, start_frame, end_frame, stride):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame_idx = 0
    saved_frame_count = 0
    image_paths = []

    actual_end_frame = total_frames -1 if end_frame == -1 else end_frame
    if actual_end_frame >= total_frames:
        print(f"Warning: end_frame ({actual_end_frame}) is beyond total frames ({total_frames-1}). Adjusting to last frame.")
        actual_end_frame = total_frames - 1

    print(f"Extracting frames from {video_path}: start={start_frame}, end={actual_end_frame}, stride={stride}")

    while True:
        ret, frame = cap.read()
        if not ret or current_frame_idx > actual_end_frame:
            break

        if current_frame_idx >= start_frame and (current_frame_idx - start_frame) % stride == 0:
            frame_filename = f"frame_{saved_frame_count:06d}.png"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            image_paths.append(frame_path)
            saved_frame_count += 1
        
        current_frame_idx += 1

    cap.release()
    print(f"Successfully extracted {saved_frame_count} frames to {output_dir}.")
    return natsorted(image_paths)

parser = argparse.ArgumentParser(description="Pi3 inference with optional gaussian splatting")
parser.add_argument(
    "--input", type=str, default="data/examples/office", help="Path to input (folder of images or a video file)"
)
parser.add_argument(
    "--input2", type=str, default=None, help="Path to input (folder of images or a video file)"
)
parser.add_argument(
    "--input3", type=str, default=None, help="Path to input (folder of images or a video file)"
)
parser.add_argument(
    "--input4", type=str, default=None, help="Path to input (folder of images or a video file)"
)
parser.add_argument(
    "--input5", type=str, default=None, help="Path to input (folder of images or a video file)"
)
parser.add_argument("--start_frame", type=int, default=0, help="Start frame for video processing")
parser.add_argument("--end_frame", type=int, default=-1, help="End frame for video processing (-1 for last frame)")
parser.add_argument("--stride", type=int, default=1, help="Stride for frame extraction/loading")
parser.add_argument(
    "--output_folder", type=str, default='./results_pi3', help="Path to folder to save inference results"
)
parser.add_argument(
    "--load", type=str, default=None, help="Path to folder or .pt file to load pre-computed inference results from"
)
parser.add_argument("--seq_name", type=str, default=None, help="Name of the sequence for saving results")
parser.add_argument(
    "--model_name",
    type=str,
    default="ckpts/LoGeR_star/latest.pt",
    help="Name of the model to load from Hugging Face Hub or a local path to a checkpoint."
)
parser.add_argument("--config", type=str, default="ckpts/LoGeR_star/original_config.yaml", help="Path to a yaml config file for model initialization.")
parser.add_argument("--resolution", type=list, default=None, help="Target resolution for input images (shorter side).")
parser.add_argument("--window_size", type=int, default=12, help="Window size for non-causal inference (-1 for full sequence).")
parser.add_argument("--overlap_size", type=int, default=3, help="Overlap size for sliding window inference.")
parser.add_argument("--sim3", action="store_true", help="Use sim3 transformation for TTT.")
parser.add_argument("--sim3_scale_mode", type=str, default="median", choices=["median", "trimmed_mean", "median_all", "trimmed_mean_all", "sim3_avg1"], help="Scale estimation mode for Sim3.")
parser.add_argument("--no_se3", action="store_true", help="Disable SE3 window merge alignment (default: on, LoGeR_star config).")
parser.add_argument("--reset_every", type=int, default=None, help="Reset TTT / adapter state every N windows (0 disables).")
parser.add_argument("--output_txt", type=str, default=None, help="Output trajectory txt file path.")
parser.add_argument("--se3", action="store_true", default=None, help="Use se3 transformation for TTT. If omitted, fallback to config value, then False.")
parser.add_argument("--no_ttt", action="store_true", help="Disable TTT.")
parser.add_argument("--no_swa", action="store_true", help="Disable SWA.")
parser.add_argument("--pi3x", action="store_true", help="Use Pi3X model.")
parser.add_argument("--pi3x_metric", action="store_true", default=True, help="Use metric scaling for Pi3X (default: True).")
parser.add_argument("--no_pi3x_metric", action="store_false", dest='pi3x_metric', help="Disable metric scaling for Pi3X.")
parser.add_argument("--warmup", action="store_true", help="Run a warmup inference pass to trigger torch.compile before timing.")
parser.add_argument("--benchmark", action="store_true", help="Run multiple inference passes and report timing statistics.")
parser.add_argument("--load_to_cpu", action="store_true", default=True, help="Keep the preloaded image tensor on CPU (default; avoids GPU OOM on long sequences).")
parser.add_argument("--no_load_to_cpu", action="store_false", dest='load_to_cpu', help="Move the preloaded image tensor onto the GPU before inference.")
parser.add_argument("--gaussian", action="store_true", help="Enable gaussian splatting reconstruction.")
parser.add_argument("--render_gaussian", action="store_true", help="Render gaussian splats to images and embed in saved .pt file.")
parser.add_argument("--save_rendered", action="store_true", help="Also save rendered images as PNG files (requires --output_folder).")
parser.add_argument("--train", action="store_true", help="Train the gaussian head (freezes non-gaussian parameters).")
parser.add_argument("--pretrained_decoder", type=str, default=None, help="Path to old gaussian_head.pt to load decoder weights from.")
parser.add_argument(
    "--train_seed",
    type=int,
    default=0,
    help="Seed Python/NumPy/Torch/CUDA RNGs at train start (also restored from checkpoint when present).",
)
parser.add_argument(
    "--require_empty_output",
    action="store_true",
    help="Fail closed if --output_folder already has files (blocks silent resume into dirty dirs).",
)
parser.add_argument(
    "--ddp_stripe_scenes",
    action="store_true",
    default=True,
    help="When DDP: shard epoch scenes across ranks (rank r gets scenes[r::W]) for throughput. "
         "Set scenes_per_epoch = optimizer_steps * world_size.",
)
parser.add_argument(
    "--no_ddp_stripe_scenes",
    action="store_false",
    dest="ddp_stripe_scenes",
    help="Disable DDP scene striping (legacy lockstep same-scene replicate).",
)
parser.add_argument(
    "--ddp_static_graph",
    action="store_true",
    help="Use DDP static_graph for a fixed train graph; incompatible with changing parameter usage.",
)
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for gaussian head training.")
parser.add_argument(
    "--grad_clip_norm",
    type=float,
    default=0.0,
    help="Clip trainable gradient norm when positive (0 disables clipping).",
)
parser.add_argument("--scale_reg_weight", type=float, default=0.0, help="Scale-to-KNN ratio regularization weight.")
parser.add_argument("--aniso_reg_weight", type=float, default=0.1, help="Anisotropy penalty weight (needle gaussians).")
parser.add_argument("--aniso_threshold", type=float, default=4.0, help="Max axis ratio before anisotropy penalty.")
parser.add_argument(
    "--offset_reg_weight",
    type=float,
    default=0.01,
    help="L2 penalty on per-pixel gaussian offsets (merge-train uses dense bypass).",
)
parser.add_argument(
    "--merge_dense_reg_max",
    type=int,
    default=262144,
    help="Max dense gaussians subsampled for merge-train aniso/offset (0 = no cap).",
)
parser.add_argument("--ssim_weight", type=float, default=0.15,
                    help="Weight for SSIM loss (1-SSIM) in recon branch. 0 to disable.")
parser.add_argument("--train_samples", type=int, default=4, help="Number of frames to render per training step (random subset).")
parser.add_argument("--neighbor_frames", type=int, default=0,
                    help="Recon rendering: ±N neighbor frames whose gaussians to merge per view. 0=current frame only.")
parser.add_argument("--image_scale", type=float, default=1.0, help="Scale factor for input images (e.g. 0.5 = half resolution). Must produce multiples of 14.")
parser.add_argument("--vis_train", action="store_true", help="Save rendered images periodically during training.")
parser.add_argument("--vis_interval", type=int, default=200, help="Step interval between training visualizations (dataset mode) or epoch interval (single-scene).")
parser.add_argument("--dataset", type=str, default=None, help="Dataset name for multi-scene training (e.g., 'dl3dv').")
parser.add_argument(
    "--dataset_root",
    type=str,
    default="data/dl3dv",
    help="Root path for the dataset (pass an explicit path for training or evaluation).",
)
parser.add_argument(
    "--calibrate_local_points_to_k",
    action="store_true",
    help="Calibrate predicted local points with each scene's loaded camera K during training.",
)
parser.add_argument("--scenes_per_epoch", type=int, default=-1, help="Scenes per epoch in dataset mode. -1 = all.")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                    help="Scenes per rank per optimizer update; retain one context bucket across microsteps.")
parser.add_argument(
    "--train_max_global_steps", type=int, default=0,
    help="Stop dataset training at this global optimizer step (0 = no cap).",
)
parser.add_argument(
    "--train_skip_scene_file", default="",
    help="Text file of canonical scene paths to skip synchronously across DDP ranks.",
)
parser.add_argument(
    "--train_skip_scene_min_context", type=int, default=128,
    help="Apply the synchronized scene blocklist only at or above this context count.",
)
parser.add_argument(
    "--train_scene_limit",
    type=int,
    default=0,
    help="Use only the first N indexed training scenes (0 = no limit).",
)
parser.add_argument(
    "--balanced_train_root_a",
    default="",
    help="Prepared source-A root for exact 1:1 balanced training.",
)
parser.add_argument(
    "--balanced_train_root_b",
    default="",
    help="Prepared source-B root for exact 1:1 balanced training.",
)
parser.add_argument(
    "--balanced_train_count_a",
    type=int,
    default=0,
    help="Required number of source-A training scenes for balanced training.",
)
parser.add_argument(
    "--balanced_train_count_b",
    type=int,
    default=0,
    help="Required number of source-B training scenes for balanced training.",
)
parser.add_argument("--frame_gap_max", type=int, default=3,
                    help="Max random gap (0~x skipped frames) between consecutive context frames.")
parser.add_argument("--nvs_holdout", type=int, default=4,
                    help="Number of hold-out NVS target frames sampled from skipped frames.")
parser.add_argument("--nvs_loss_weight", type=float, default=1.0,
                    help="Weight for the NVS (hold-out) loss branch.")
parser.add_argument(
    "--neural_dense_loss_weight", type=float, default=1.0,
    help="Weight of the original dense reconstruction branch in neural merge training.",
)
parser.add_argument(
    "--neural_merge_loss_weight", type=float, default=1.0,
    help="Weight of the merged-cloud reconstruction branch in neural merge training.",
)
parser.add_argument("--direct_slot_merger_lr", type=float, default=1e-4)
parser.add_argument("--direct_slot_base_lr", type=float, default=1e-5)
parser.add_argument("--direct_slot_dense_warmup_steps", type=int, default=500)
parser.add_argument("--direct_slot_dense_warmup_weight", type=float, default=1.0)
parser.add_argument("--direct_slot_dense_weight", type=float, default=0.5)
parser.add_argument(
    "--neural_freeze_base", action="store_true",
    help="In neural merge mode, freeze the existing decoder/head and train only the merger.",
)
parser.add_argument(
    "--neural_freeze_decoder", action="store_true",
    help="In neural merge mode, freeze only the Gaussian decoder; train the head and merger.",
)
parser.add_argument(
    "--diagnose_nonfinite_grads", action="store_true",
    help="Retain and report dense/hybrid Gaussian attribute gradients on skipped updates.",
)
parser.add_argument(
    "--track_merge_train",
    action="store_true",
    help="Train with in-forward AllTracker track merge + dense complement.",
)
parser.add_argument("--merge_min_track_len", type=int, default=2)
parser.add_argument("--merge_min_opacity", type=float, default=0.05)
parser.add_argument("--merge_uv_thresh", type=float, default=3.0)
parser.add_argument("--merge_3d_thresh", type=float, default=0.15)
parser.add_argument("--merge_max_center_variance", type=float, default=0.3)
parser.add_argument(
    "--merge_curriculum_dense_epochs",
    type=int,
    default=0,
    help="Idea-2 curriculum: first N epochs after resume use dense path (no track merge), "
         "even if --track_merge_train is set.",
)
parser.add_argument(
    "--merge_curriculum_soft_epochs",
    type=int,
    default=0,
    help="Idea-2 curriculum: after dense warmup, next M epochs use softer merge filters "
         "(higher max_center_variance, lower min opacity).",
)
parser.add_argument(
    "--merge_soft_max_center_variance",
    type=float,
    default=1.0,
    help="Soft-phase max_center_variance (default 1.0 vs hard 0.3).",
)
parser.add_argument(
    "--merge_soft_min_opacity",
    type=float,
    default=0.01,
    help="Soft-phase min_mean_opacity (default 0.01 vs hard 0.05).",
)
parser.add_argument(
    "--no_track_dense_complement",
    action="store_true",
    help="Disable dense complement during track merge.",
)
parser.add_argument(
    "--alltracker_conf_thr",
    type=float,
    default=0.1,
    help="AllTracker visibility confidence threshold for track merge.",
)
parser.add_argument(
    "--merge_mode",
    choices=["hard", "neural", "neural_set", "neural_feature", "neural_direct_slots", "dsga_v0"],
    default="hard",
    help="Track merge mode: hard, M0 summary neural, M1 observation-set neural, or DSGA v0.",
)
parser.add_argument(
    "--rotation_mode",
    choices=["svd", "argmax", "chordal"],
    default="svd",
    help="Rotation merge: svd (stop-grad), argmax, or chordal (differentiable).",
)
parser.add_argument(
    "--merge_soft_validity_temp",
    type=float,
    default=0.1,
    help="DSGA soft track validity sigmoid temperature.",
)
parser.add_argument(
    "--merge_hybrid_budget_ratio",
    type=float,
    default=0.0,
    help="DSGA cap for hybrid GS relative to dense (0 disables budget pruning).",
)
parser.add_argument("--long_sequence_protocol", action="store_true",
                    help="Sample training steps from explicit context-count buckets.")
parser.add_argument("--long_context_counts", default="16,64,96,128")
parser.add_argument("--long_context_weights", default="25,25,25,25")
parser.add_argument("--long_context_modes", default="",
                    help="Optional per-bucket dense/sparse modes; permits duplicate context counts.")
parser.add_argument("--long_context_intervals", default="",
                    help="Optional per-bucket sampling intervals (dense buckets must use 1).")
parser.add_argument("--long_bucket_max_retries", type=int, default=3,
                    help="Advance past a bucket after this many consecutive failed scenes.")
parser.add_argument("--long_sparse_context_counts", default="16")
parser.add_argument("--long_sparse_intervals", default="8")
parser.add_argument("--long_train_target_limit", type=int, default=3)
parser.add_argument("--train_bucket_best_postprocess", action="store_true")
parser.add_argument("--train_head_lr", type=float, default=0.0)
parser.add_argument("--train_merger_lr", type=float, default=0.0)
parser.add_argument("--save_every_steps", type=int, default=0)
parser.add_argument(
    "--save_persistent_every_steps",
    type=int,
    default=100,
    help=(
        "Keep a numbered step checkpoint at this interval. Intermediate "
        "--save_every_steps checkpoints atomically replace step_latest."
    ),
)
parser.add_argument("--train_save_activations_on_cpu", action="store_true")
parser.add_argument("--train_gaussian_head_checkpoint", action="store_true")
parser.add_argument("--train_gaussian_head_batch_size", type=int, default=0)
parser.add_argument("--train_loss_checkpoint", action="store_true")
parser.add_argument("--train_render_checkpoint_chunk_size", type=int, default=0)
parser.add_argument("--long_train_context_render_limit", type=int, default=3)
parser.add_argument("--long_head_update_max_context", type=int, default=16,
                    help="Update Gaussian head only up to this context bucket; longer buckets update merger only.")
parser.add_argument("--fail_if_zero_updates", action="store_true")
parser.add_argument("--sky_mask_onnx", default="skyseg.onnx")
parser.add_argument("--sky_mask_score_thr", type=float, default=160.0)
parser.add_argument("--sky_sphere_radius_factor", type=float, default=50.0)
parser.add_argument("--sky_sphere_stride", type=int, default=2)
parser.add_argument("--sky_sphere_azimuth_bins", type=int, default=512)
parser.add_argument("--sky_sphere_elevation_bins", type=int, default=256)

# Training defaults (run.sh) — shared with eval scripts for consistent inference.
TRAIN_WINDOW_SIZE = 12
TRAIN_OVERLAP_SIZE = 3
TRAIN_SIM3 = False
TRAIN_SE3 = True


def subsample_dense_reg(
    scales: Optional[torch.Tensor],
    means: Optional[torch.Tensor],
    offsets: Optional[torch.Tensor],
    max_n: int,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Randomly subsample dense reg tensors to cap memory / compute."""
    if scales is None or max_n <= 0 or scales.shape[0] <= max_n:
        return scales, means, offsets
    idx = torch.randperm(scales.shape[0], device=scales.device)[:max_n]
    scales = scales[idx]
    if means is not None:
        means = means[idx]
    if offsets is not None:
        offsets = offsets[idx]
    return scales, means, offsets


def merge_curriculum_phase(
    epoch: int,
    curriculum_start_epoch: int,
    dense_epochs: int,
    soft_epochs: int,
) -> str:
    """Idea-2 curriculum phase for merge-train: dense → soft → hard."""
    if not dense_epochs and not soft_epochs:
        return "hard"
    rel = max(0, int(epoch) - int(curriculum_start_epoch))
    if rel < int(dense_epochs):
        return "dense"
    if rel < int(dense_epochs) + int(soft_epochs):
        return "soft"
    return "hard"


def infer_curriculum_start_epoch(ckpt: dict, ckpt_files: list[str], resume_epoch: int) -> int:
    """Keep curriculum progress stable across resumes, including legacy checkpoints."""
    if "curriculum_start_epoch" in ckpt:
        return int(ckpt["curriculum_start_epoch"])

    # Legacy checkpoints did not persist the curriculum origin. The first epoch
    # checkpoint in the output directory is the bootstrap/resume head, so
    # training began at the following epoch.
    import re

    checkpoint_epochs = []
    for path in ckpt_files:
        match = re.search(r"epoch(\d+)", os.path.basename(path))
        if match:
            checkpoint_epochs.append(int(match.group(1)))
    if checkpoint_epochs:
        return min(checkpoint_epochs) + 1
    return int(resume_epoch)


def sanitize_hybrid_for_render(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Clamp/sanitize hybrid gaussians to keep gsplat rasterization stable."""
    means = torch.nan_to_num(means, nan=0.0, posinf=1e4, neginf=-1e4)
    scales = torch.nan_to_num(scales, nan=1e-3, posinf=1e2, neginf=1e-3).clamp(min=1e-4)
    opacities = torch.nan_to_num(opacities, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    colors = torch.nan_to_num(colors, nan=0.0, posinf=10.0, neginf=-10.0)
    quats = torch.nan_to_num(quats, nan=0.0)
    quats = torch.nn.functional.normalize(quats, dim=-1, eps=1e-8)
    degenerate = quats.abs().sum(dim=-1) < 1e-8
    if degenerate.any():
        quats = quats.clone()
        quats[degenerate, 0] = 1.0
    return means, quats, scales, opacities, colors


def model_sliding_window_kwargs(
    num_frames: int,
    window_size: int = TRAIN_WINDOW_SIZE,
    overlap_size: int = TRAIN_OVERLAP_SIZE,
    sim3: bool = TRAIN_SIM3,
    se3: bool = TRAIN_SE3,
    turn_off_ttt: bool = False,
    turn_off_swa: bool = False,
) -> dict:
    """Sliding-window model.forward() kwargs aligned with training (run.sh)."""
    del num_frames  # Pi3 clamps window_size >= N internally
    if sim3:
        se3 = False
    return {
        "window_size": window_size,
        "overlap_size": overlap_size,
        "sim3": sim3,
        "se3": se3,
        "turn_off_ttt": turn_off_ttt,
        "turn_off_swa": turn_off_swa,
    }


# ---------------------------------------------------------------------------
#  Dataset loading
# ---------------------------------------------------------------------------

def _load_dl3dv_frames_meta(scene_path: str) -> Optional[Tuple[list, dict]]:
    """Load transforms.json frames list and metadata dict."""
    import json

    transforms_path = os.path.join(scene_path, "transforms.json")
    if not os.path.isfile(transforms_path):
        return None
    try:
        with open(transforms_path, "r") as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    frames = meta.get("frames", [])
    if not frames:
        return None
    return frames, meta


def _dl3dv_target_size(
    meta: dict,
    resolution: Optional[List[int]],
    image_scale: float,
) -> Tuple[int, int]:
    """Return (target_w, target_h) snapped to multiples of 14."""
    if resolution is not None:
        return resolution[0], resolution[1]
    native_h, native_w = 270, 480
    target_h = int(native_h * image_scale)
    target_w = int(native_w * image_scale)
    target_h = ((target_h + 13) // 14) * 14
    target_w = ((target_w + 13) // 14) * 14
    return target_w, target_h


def _build_dl3dv_K(meta: dict, target_w: int, target_h: int) -> torch.Tensor:
    """Build camera intrinsics scaled to target resolution."""
    orig_w = meta.get("w", 3840)
    orig_h = meta.get("h", 2160)
    fl_x = meta.get("fl_x", orig_w * 0.7)
    fl_y = meta.get("fl_y", orig_h * 0.7)
    cx = meta.get("cx", orig_w / 2.0)
    cy = meta.get("cy", orig_h / 2.0)
    fx = fl_x * target_w / orig_w
    fy = fl_y * target_h / orig_h
    cx_s = cx * target_w / orig_w
    cy_s = cy * target_h / orig_h
    return torch.tensor(
        [[fx, 0.0, cx_s], [0.0, fy, cy_s], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )


def _load_dl3dv_frame_tensors(
    scene_path: str,
    frame_entries: list,
    target_w: int,
    target_h: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Load images and poses for a list of frame dict entries."""
    from PIL import Image
    from torchvision import transforms as tv_transforms

    to_tensor = tv_transforms.ToTensor()
    images = []
    poses = []
    for frame in frame_entries:
        img_rel = frame["file_path"].replace("images/", "images_8/")
        img_path = os.path.join(scene_path, img_rel)
        if not os.path.isfile(img_path):
            print(f"[training-data] Missing frame: {img_path}", flush=True)
            return None
        try:
            with Image.open(img_path) as handle:
                img = handle.convert("RGB")
            img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            images.append(to_tensor(img))
        except (OSError, IOError, SyntaxError) as exc:
            print(f"[training-data] Cannot decode {img_path}: {exc}", flush=True)
            return None
        try:
            pose = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
        except (ValueError, TypeError):
            return None
        poses.append(pose)
    return torch.stack(images), torch.stack(poses)


def sample_context_holdout_indices(
    num_frames: int,
    num_context: int,
    max_gap: int,
    num_holdout: int,
    max_retries: int = 5,
) -> Optional[Tuple[List[int], List[int]]]:
    """Sample context frames with random gaps and hold-out frames from skipped pool.

    Context: random start, then each next frame is prev + 1 + Uniform(0, max_gap).
    Hold-out: random subset of frames strictly between consecutive context frames.
    """
    if num_frames < num_context:
        return None

    min_span = (num_context - 1) * (max_gap + 1)
    max_start = num_frames - 1 - min_span
    if max_start < 0:
        return None

    require_holdout = num_holdout > 0
    for _ in range(max_retries):
        start = random.randint(0, max_start)
        ctx = [start]
        failed = False
        while len(ctx) < num_context:
            gap = random.randint(0, max_gap)
            nxt = ctx[-1] + 1 + gap
            if nxt >= num_frames:
                failed = True
                break
            ctx.append(nxt)
        if failed:
            continue

        skipped = [i for a, b in zip(ctx, ctx[1:]) for i in range(a + 1, b)]
        if require_holdout:
            if not skipped:
                continue
            holdout = random.sample(skipped, min(num_holdout, len(skipped)))
        else:
            holdout = []
        return ctx, holdout
    return None


def load_dl3dv_scene_indices(
    scene_path: str,
    indices: List[int],
    resolution: Optional[List[int]] = None,
    image_scale: float = 1.0,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Load specific frame indices from a DL3DV scene."""
    meta_result = _load_dl3dv_frames_meta(scene_path)
    if meta_result is None:
        return None
    frames, meta = meta_result
    if any(i < 0 or i >= len(frames) for i in indices):
        return None

    target_w, target_h = _dl3dv_target_size(meta, resolution, image_scale)
    sampled = [frames[i] for i in indices]
    tensors = _load_dl3dv_frame_tensors(scene_path, sampled, target_w, target_h)
    if tensors is None:
        return None
    images, poses = tensors
    K = _build_dl3dv_K(meta, target_w, target_h)
    return images, poses, K


def load_dl3dv_scene_split(
    scene_path: str,
    ctx_indices: List[int],
    holdout_indices: List[int],
    resolution: Optional[List[int]] = None,
    image_scale: float = 1.0,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int], List[int], List[int], List[int]]]:
    """Load context + hold-out frames in temporal order with local/global index maps."""
    all_indices = sorted(set(ctx_indices) | set(holdout_indices))
    result = load_dl3dv_scene_indices(scene_path, all_indices, resolution, image_scale)
    if result is None:
        return None
    images, poses, K = result
    index_to_local = {idx: i for i, idx in enumerate(all_indices)}
    ctx_local = [index_to_local[i] for i in ctx_indices]
    holdout_local = [index_to_local[i] for i in holdout_indices]
    return images, poses, K, ctx_local, holdout_local, ctx_indices, holdout_indices


def render_holdout_views(
    gs_renderer,
    predictions: dict,
    ctx_local: List[int],
    holdout_local: List[int],
    K_gpu: torch.Tensor,
    H: int,
    W: int,
) -> Optional[torch.Tensor]:
    """Render hold-out views with context gaussians. Returns (H, 3, h, w) or None."""
    if not holdout_local:
        return None
    means, quats, scales, opacs, colors, _ = gs_renderer.prepare_gaussians(
        predictions, frame_indices=ctx_local,
    )
    preds = []
    for h in holdout_local:
        out = gs_renderer.render_view(
            means, quats, scales, opacs, colors,
            camera_pose=predictions["camera_poses"][0, h],
            H=H, W=W, K=K_gpu,
        )
        preds.append(out["rgb"].permute(2, 0, 1))
    return torch.stack(preds)


def try_load_dl3dv_scene_split(
    scene_path: str,
    num_context: int,
    max_gap: int,
    num_holdout: int,
    resolution: Optional[List[int]] = None,
    image_scale: float = 1.0,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int], List[int], List[int], List[int]]]:
    """Sample context/hold-out indices and load tensors; retry on sampling failure."""
    meta_result = _load_dl3dv_frames_meta(scene_path)
    if meta_result is None:
        return None
    frames, _ = meta_result
    sampled = sample_context_holdout_indices(
        len(frames), num_context, max_gap, num_holdout,
    )
    if sampled is None:
        # High-context pilot protocol (20 context + held-out target in a
        # 24-frame scene) cannot leave random temporal gaps.  Reuse the
        # locked split when available instead of silently producing zero
        # training updates.
        split_path = os.path.join(scene_path, "split.json")
        if not os.path.isfile(split_path):
            return None
        with open(split_path, "r", encoding="utf-8") as f:
            locked = json.load(f)
        locked_ctx = [int(i) for i in locked.get("ctx_list", [])]
        locked_tgt = [int(i) for i in locked.get("tgt_list", [])]
        if len(locked_ctx) < num_context or (num_holdout > 0 and not locked_tgt):
            return None
        sampled = (
            locked_ctx[:num_context],
            locked_tgt[:min(num_holdout, len(locked_tgt))],
        )
    ctx_indices, holdout_indices = sampled
    if num_holdout > 0 and not holdout_indices:
        return None
    split = load_dl3dv_scene_split(
        scene_path, ctx_indices, holdout_indices, resolution, image_scale,
    )
    if split is None:
        return None
    images, poses, K, ctx_local, holdout_local, _, _ = split
    return images, poses, K, ctx_local, holdout_local, ctx_indices, holdout_indices


def _csv_numbers(value: str, cast):
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def build_weighted_bucket_cycle(bucket_specs, weights):
    """Expand positive rational weights into their exact smallest integer cycle."""
    from fractions import Fraction
    from functools import reduce

    if not bucket_specs or len(bucket_specs) != len(weights):
        raise ValueError("bucket specs/weights must be non-empty and equal length")
    fractions = [Fraction(str(weight)) for weight in weights]
    if any(weight <= 0 for weight in fractions):
        raise ValueError("long context weights must all be positive")
    denominator = math.lcm(*(weight.denominator for weight in fractions))
    counts = [weight.numerator * (denominator // weight.denominator) for weight in fractions]
    divisor = reduce(math.gcd, counts)
    cycle = []
    for spec, count in zip(bucket_specs, counts):
        cycle.extend([spec] * (count // divisor))
    return cycle


class BucketRetryScheduler:
    """Advance on success, but prevent one failing bucket from livelocking training."""

    def __init__(self, cycle, max_retries: int, position: int = 0):
        if not cycle or max_retries < 1:
            raise ValueError("bucket cycle must be non-empty and max_retries positive")
        self.cycle = list(cycle)
        self.max_retries = int(max_retries)
        self.position = int(position) % len(self.cycle)
        self.retries = 0
        self.success_counts = {}
        self.failure_counts = {}

    @property
    def current(self):
        return self.cycle[self.position]

    @staticmethod
    def _key(spec):
        count, mode, interval = spec
        return f"C{count}-{mode}-I{interval}"

    def record_success(self):
        key = self._key(self.current)
        self.success_counts[key] = self.success_counts.get(key, 0) + 1
        self.position = (self.position + 1) % len(self.cycle)
        self.retries = 0

    def record_failure(self):
        failed = self.current
        key = self._key(failed)
        self.failure_counts[key] = self.failure_counts.get(key, 0) + 1
        self.retries += 1
        if self.retries < self.max_retries:
            return False
        # A one-bucket recipe must not loop forever after a failed sample.
        for _ in range(len(self.cycle)):
            self.position = (self.position + 1) % len(self.cycle)
            if self.current != failed:
                break
        self.retries = 0
        return True

    def state_dict(self):
        return {
            "position": self.position,
            "retries": self.retries,
            "success_counts": dict(self.success_counts),
            "failure_counts": dict(self.failure_counts),
        }

    def load_state_dict(self, state):
        self.position = int(state.get("position", self.position)) % len(self.cycle)
        self.retries = int(state.get("retries", 0))
        self.success_counts = dict(state.get("success_counts", {}))
        self.failure_counts = dict(state.get("failure_counts", {}))


def _even_subset(values: Sequence[int], limit: int) -> List[int]:
    values = list(values)
    if limit <= 0 or len(values) <= limit:
        return values
    if limit == 1:
        return [values[len(values) // 2]]
    return [values[round(i * (len(values) - 1) / (limit - 1))] for i in range(limit)]


def select_long_train_targets(
    holdout_local: Sequence[int], limit: int, global_step: int
) -> List[int]:
    """Choose a bounded, deterministic rotating subset of long-sequence targets."""
    values = list(holdout_local)
    if limit <= 0 or len(values) <= limit:
        return values
    offset = int(global_step) % len(values)
    rotated = values[offset:] + values[:offset]
    return _even_subset(rotated, limit)


def select_corresponding_global_indices(
    holdout_local: Sequence[int],
    holdout_global: Sequence[int],
    selected_local: Sequence[int],
) -> List[int]:
    """Map selected local target indices through an explicitly paired list."""
    if len(holdout_local) != len(holdout_global):
        raise ValueError("local/global holdout lists must have equal length")
    mapping = dict(zip(holdout_local, holdout_global))
    missing = [index for index in selected_local if index not in mapping]
    if missing:
        raise ValueError(f"selected local target absent from holdout list: {missing}")
    return [mapping[index] for index in selected_local]


def dense_every8_holdout_min_span(num_context: int) -> int:
    """Minimum raw-frame span for formal dense every-8th holdout.

    Holdouts are taken at local indices ``7, 15, ...`` inside a contiguous
    block (``block[7::8]``).  The old formula ``num_context + ceil(num_context / 7)``
    over-counts by one whenever ``num_context`` is not a multiple of 7, e.g.
    C128 requires 147 frames even though 146 already yields 128 context + 18
    holdout.  Use the tight bound ``span - len(range(7, span, 8)) >= num_context``.
    """
    if num_context < 1:
        raise ValueError(f"num_context must be positive, got {num_context}")
    span = int(num_context)
    while span - len(range(7, span, 8)) < num_context:
        span += 1
    return span


def try_load_dl3dv_bucket_split(
    scene_path: str,
    num_context: int,
    sparse_interval: int,
    num_holdout: int,
    resolution: Optional[List[int]] = None,
    image_scale: float = 1.0,
):
    """Load a formal sparse/dense context bucket with a few distributed targets."""
    meta_result = _load_dl3dv_frames_meta(scene_path)
    if meta_result is None:
        return None
    frames, _ = meta_result
    n = len(frames)
    if sparse_interval > 1:
        span = 1 + (num_context - 1) * sparse_interval
        if n < span:
            return None
        start = random.randint(0, n - span)
        ctx_indices = [start + i * sparse_interval for i in range(num_context)]
        candidates = [x + sparse_interval // 2 for x in ctx_indices[:-1]]
    else:
        # Formal dense evaluation holds out every eighth frame.  Build the
        # same temporal topology, then keep only a distributed target subset
        # to bound differentiable rendering cost.
        span = dense_every8_holdout_min_span(num_context)
        if n < span:
            return None
        start = random.randint(0, n - span)
        block = list(range(start, start + span))
        candidates = block[7::8]
        candidate_set = set(candidates)
        ctx_indices = [i for i in block if i not in candidate_set][:num_context]
    holdout_indices = _even_subset(candidates, num_holdout)
    if len(ctx_indices) != num_context or (num_holdout > 0 and not holdout_indices):
        return None
    split = load_dl3dv_scene_split(
        scene_path, ctx_indices, holdout_indices, resolution, image_scale,
    )
    if split is None:
        # Retry the SAME scene with an intact temporal window. Do not shrink
        # the context, alter its spacing, fill damaged pixels, or advance the
        # bucket/accumulation counter. Every rank still reaches the validity
        # collective below the caller's load operation.
        from loger.utils.training_frames import choose_decodable_start
        ctx_offsets = [index - start for index in ctx_indices]
        holdout_offsets = [index - start for index in holdout_indices]
        recovered_start = choose_decodable_start(
            scene_path, frames, ctx_offsets + holdout_offsets, span, random,
        )
        if recovered_start is not None:
            ctx_indices = [recovered_start + offset for offset in ctx_offsets]
            holdout_indices = [recovered_start + offset for offset in holdout_offsets]
            split = load_dl3dv_scene_split(
                scene_path, ctx_indices, holdout_indices, resolution, image_scale,
            )
            if split is not None:
                print(f"[training-data] Resampled scene={scene_path}: "
                      f"start {start}->{recovered_start}, C{num_context}, interval={sparse_interval}", flush=True)
    if split is None:
        print(f"[training-data] Scene remains unusable: {scene_path}; "
              f"C{num_context}, interval={sparse_interval}; refusing partial update", flush=True)
        return None
    images, poses, K, ctx_local, holdout_local, _, _ = split
    return images, poses, K, ctx_local, holdout_local, ctx_indices, holdout_indices


def packed_merge_owner_groups(
    ctx_list: Sequence[int], num_frames: int, window_size: int, overlap_size: int,
) -> Dict[int, List[int]]:
    from collections import defaultdict
    from loger.merge.window_merge import owner_window_for_frame, sliding_window_ranges

    windows = sliding_window_ranges(num_frames, window_size, overlap_size)
    groups = defaultdict(list)
    for frame in ctx_list:
        owner = owner_window_for_frame(int(frame), windows, overlap_size)
        groups[owner].append(int(frame))
    return dict(groups)


def load_dl3dv_scene(
    scene_path: str,
    num_frames: int,
    resolution: Optional[List[int]] = None,
    image_scale: float = 1.0,
    random_sample: bool = False,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Load a single DL3DV scene. Returns (images, poses, K) or None on failure.

    Args:
        scene_path: Path to scene dir containing transforms.json and images_8/.
        num_frames: Number of frames to sample (random subset).
        resolution: Optional [W, H] target resolution.
        image_scale: Scale factor for native resolution (e.g. 0.5 = half).

    Returns:
        images: (N, 3, H, W) tensor on CPU
        camera_poses: (N, 4, 4) tensor on CPU
        K: (3, 3) camera intrinsics scaled to target resolution
        or None if the scene could not be loaded.
    """
    meta_result = _load_dl3dv_frames_meta(scene_path)
    if meta_result is None:
        return None
    frames, meta = meta_result
    if len(frames) < num_frames:
        return None

    if random_sample:
        indices = random.sample(range(len(frames)), num_frames)
    else:
        indices = list(range(num_frames))

    target_w, target_h = _dl3dv_target_size(meta, resolution, image_scale)
    sampled = [frames[i] for i in indices]
    tensors = _load_dl3dv_frame_tensors(scene_path, sampled, target_w, target_h)
    if tensors is None or tensors[0].shape[0] < num_frames:
        return None
    images, poses = tensors
    K = _build_dl3dv_K(meta, target_w, target_h)
    return images, poses, K


def load_dataset_scene_list(dataset: str, root: str, split: str = "train") -> List[str]:
    """Return a list of absolute scene paths for the given dataset/split."""
    import json
    index_path = os.path.join(root, f"{split}_index.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"Split index not found: {index_path}")
    with open(index_path, "r") as f:
        indices = json.load(f)
    return [os.path.join(root, entry) for entry in indices]


def load_pi3_model(model_name: str, config_path: Optional[str] = None, pi3x: bool = False, pi3x_metric: bool = True, gaussian: bool = False):
    """Initializes the Pi3 model and loads weights."""
    print(f"Initializing Pi3 model...")

    model_kwargs = {}
    if config_path:
        print(f"Loading model configuration from: {config_path}")
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            model_config = config.get('model', {})
            pi3_signature = inspect.signature(Pi3.__init__)
            valid_kwargs = {
                name
                for name, param in pi3_signature.parameters.items()
                if name not in {"self", "args", "kwargs"}
                and param.kind in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
            }

            def _maybe_parse_sequence(value):
                if isinstance(value, str):
                    stripped = value.strip()
                    if stripped.startswith("[") and stripped.endswith("]"):
                        try:
                            parsed = yaml.safe_load(stripped)
                            if isinstance(parsed, (list, tuple)):
                                return list(parsed)
                        except Exception:
                            pass
                return value

            for key in sorted(valid_kwargs):
                if key in model_config:
                    value = model_config[key]
                    if key in {"ttt_insert_after", "attn_insert_after"}:
                        value = _maybe_parse_sequence(value)
                    model_kwargs[key] = value

            print("Model parameters from config:", model_kwargs)
        except Exception as e:
            print(f"Error loading or parsing config file {config_path}: {e}")
            print("Falling back to default model parameters.")
            model_kwargs = {}

    if pi3x:
        model_kwargs['pi3x'] = True
        model_kwargs['pi3x_metric'] = pi3x_metric
        if model_name == "yyfz233/Pi3":
            print("Switching default model to yyfz233/Pi3X because --pi3x is set.")
            model_name = "yyfz233/Pi3X"

    if gaussian:
        model_kwargs['gaussian'] = True


    try:
        # Initialize model with parameters from config
        model = Pi3(**model_kwargs)

        if model_name.startswith("yyfz233/"):
            print("Loading pre-trained weights from Hugging Face Hub...")
            model = model.from_pretrained(model_name, strict=False if pi3x else True, **model_kwargs)
            print("Model loaded successfully from Hugging Face Hub.")
            return model
        
        # Load pre-trained weights
        print(f"Loading pre-trained weights from: {model_name}")
        # Use strict=False to allow for architecture mismatches when loading weights
        # This is useful when the config defines a different architecture than the saved checkpoint
        checkpoint = torch.load(model_name, map_location='cpu', weights_only=False)
        # If the checkpoint is a state_dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Adjust state_dict keys if they are prefixed (e.g., by DDP)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # remove `module.`
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Could not load model. Error: {e}")
        return None
        
    return model


def run_core_inference(
    model_obj: Pi3,
    input_paths: List[str],
    start_frame: int = 0,
    end_frame: int = -1,
    stride: int = 1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    target_resolution: List[int] = [504, 280],
):
    """
    Handles data preparation and runs the core model inference for Pi3.
    """
    model_obj.eval()
    model_obj = model_obj.to(device)
    
    temp_frame_dirs = {}
    input_indices = {}
    all_image_names = []
    
    for i, input_path in enumerate(input_paths):
        input_key = f"input{i+1}"
        if i > 0:
            input_indices[f"cam{i:02d}"] = len(all_image_names)
        
        if is_video_file(input_path):
            temp_dir = tempfile.mkdtemp(prefix=f"pi3_frames_{input_key}_")
            temp_frame_dirs[input_key] = temp_dir
            image_names_current_input = extract_frames_from_video(input_path, temp_dir, start_frame, end_frame, stride)
        elif os.path.isdir(input_path):
            image_names_current_input = natsorted(glob.glob(os.path.join(input_path, "*.png"))+glob.glob(os.path.join(input_path, "*.jpg"))+glob.glob(os.path.join(input_path, "*.jpeg")))
            end_idx = end_frame if end_frame != -1 else None
            image_names_current_input = image_names_current_input[start_frame:end_idx:stride]
        else:
            print(f"Warning: Input path {input_path} is not a valid video file or directory. Skipping.")
            image_names_current_input = []

        if not image_names_current_input:
            if input_key in temp_frame_dirs:
                if os.path.exists(temp_frame_dirs[input_key]): shutil.rmtree(temp_frame_dirs[input_key])
                del temp_frame_dirs[input_key]
            if f"cam{i:02d}" in input_indices: del input_indices[f"cam{i:02d}"]
        else:
            all_image_names.extend(image_names_current_input)

    if not all_image_names:
        print("Error: No images found from any input.")
        return None, [], {}, {}
        
    print(f"Loading images from combined inputs ({len(all_image_names)} images found)...")
    # Use load_images_from_paths to load exactly the images we collected
    images_tensor = load_images_from_paths(all_image_names, Target_W=target_resolution[0], Target_H=target_resolution[1]).to(device)
    print(f"Preprocessed images tensor shape: {images_tensor.shape}")

    print("Running inference...")    
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=dtype):
        raw_model_predictions = model_obj(images_tensor[None]) # Add batch dimension
    
    # Post-process predictions
    raw_model_predictions['images'] = images_tensor[None].permute(0, 1, 3, 4, 2) # B, S, H, W, C
    raw_model_predictions['conf'] = torch.sigmoid(raw_model_predictions['conf'])
    edge = depth_edge(raw_model_predictions['local_points'][..., 2], rtol=0.03)
    raw_model_predictions['conf'][edge] = 0.0
    if 'local_points' in raw_model_predictions:
        del raw_model_predictions['local_points']

    return raw_model_predictions, all_image_names, input_indices, temp_frame_dirs


def _try_load_timestamps_for_images(image_paths, input_rgb_dir: Path):
    """Best-effort timestamp loader.

    Priority:
    1) <parent>/rgb.txt (TUM style: "timestamp rgb/xxxxx.png")
    2) <input_rgb_dir>/timestamps.txt (one timestamp per line)
    3) Fallback to sequential indices starting at 0
    """
    # 1) TUM-format rgb.txt in the parent directory
    if input_rgb_dir.is_file(): # if input is a video file
        return [float(i) for i in range(len(image_paths))]

    rgb_txt_path = input_rgb_dir.parent / "rgb.txt"
    if rgb_txt_path.exists():
        name_to_ts = {}
        with open(rgb_txt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    ts = float(parts[0])
                except ValueError:
                    continue
                img_rel = parts[1]
                # Map by basename for robustness
                name_to_ts[Path(img_rel).name] = ts

        ts_list = []
        for p in image_paths:
            ts_list.append(name_to_ts.get(Path(p).name, None))
        if all(t is not None for t in ts_list) and len(ts_list) == len(image_paths):
            return ts_list
        # If partial or mismatch, fall through to next option

    # 2) timestamps.txt alongside images
    timestamps_txt = input_rgb_dir / "timestamps.txt"
    if timestamps_txt.exists():
        with open(timestamps_txt, "r") as f:
            raw_lines = [l.strip() for l in f.readlines() if l.strip() and not l.strip().startswith("#")]
        # Take as many as needed in order
        ts_list = []
        for i in range(min(len(raw_lines), len(image_paths))):
            try:
                ts_list.append(float(raw_lines[i]))
            except ValueError:
                ts_list.append(float(i))
        # If fewer timestamps than images, pad with indices
        for i in range(len(ts_list), len(image_paths)):
            ts_list.append(float(i))
        return ts_list

    # 3) Fallback: sequential indices as timestamps
    return [float(i) for i in range(len(image_paths))]


def write_trajectory_txt(output_path: Path, timestamps, translations, quaternions):
    """Write trajectory file with lines: ts tx ty tz qx qy qz qw"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for ts, t, q in zip(timestamps, translations, quaternions):
            f.write(
                f"{ts:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n"
            )


def load_images_from_paths(image_paths, PIXEL_LIMIT=255000, Target_W=None, Target_H=None, verbose=True):
    if not image_paths:
        print("No images found or loaded.")
        return torch.empty(0)

    if Target_W is None and Target_H is None:
        with Image.open(image_paths[0]) as first_img:
            W_orig, H_orig = first_img.size
        scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
        W_target, H_target = W_orig * scale, H_orig * scale
        k, m = round(W_target / 14), round(H_target / 14)
        while (k * 14) * (m * 14) > PIXEL_LIMIT:
            if k / m > W_target / H_target: k -= 1
            else: m -= 1
        TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14
    else:
        TARGET_W, TARGET_H = Target_W, Target_H

    if verbose:
        print(f"All images will be resized to a uniform size: ({TARGET_W}, {TARGET_H})")

    out = torch.empty((len(image_paths), 3, TARGET_H, TARGET_W), dtype=torch.float32)
    to_tensor_transform = transforms.ToTensor()

    valid = 0
    for img_path in image_paths:
        try:
            with Image.open(img_path) as img:
                resized = img.convert('RGB').resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
            out[valid].copy_(to_tensor_transform(resized))
            valid += 1
        except Exception as e:
            print(f"Could not load image {img_path}: {e}")

    if valid == 0:
        return torch.empty(0)
    return out[:valid] if valid < len(image_paths) else out

def main():
    args = parser.parse_args()
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if args.gradient_accumulation_steps > 1 and not (
        args.train and args.dataset == "dl3dv" and args.long_sequence_protocol and args.ddp_stripe_scenes
    ):
        raise ValueError("Accumulation requires striped long-sequence dataset training")

    # --- DDP setup ---
    local_rank = setup_ddp()
    if local_rank is not None:
        device = f"cuda:{local_rank}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _log(f"Using device: {device}" + (f" (rank {dist.get_rank()}/{dist.get_world_size()})" if _is_ddp() else ""))

    # Fail fast for local LoGeR checkpoints/configs when files are missing.
    if args.config and not os.path.isfile(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    if not os.path.isfile(args.model_name):
        raise FileNotFoundError(f"Checkpoint file not found: {args.model_name}")

    predictions_dict = None
    temp_frame_dirs = {}
    target_resolution = args.resolution if args.resolution and len(args.resolution) == 2 else None
    
    # Generate seq_name automatically if not provided
    if args.seq_name is None:
        args.seq_name = os.path.basename(os.path.dirname(args.input)) + "_" + os.path.basename(args.input)
        if args.input2:
            args.seq_name += f"_{os.path.basename(os.path.dirname(args.input2))}_{os.path.basename(args.input2)}"
        if args.input3:
            args.seq_name += f"_{os.path.basename(os.path.dirname(args.input3))}_{os.path.basename(args.input3)}"
        if args.input4:
            args.seq_name += f"_{os.path.basename(os.path.dirname(args.input4))}_{os.path.basename(args.input4)}"
        if args.input5:
            args.seq_name += f"_{os.path.basename(os.path.dirname(args.input5))}_{os.path.basename(args.input5)}"
    
    if args.load:
        saved_predictions_path = args.load
        if os.path.isdir(saved_predictions_path):
            if args.seq_name and os.path.exists(os.path.join(saved_predictions_path, f"{args.seq_name}.pt")):
                saved_predictions_path = os.path.join(saved_predictions_path, f"{args.seq_name}.pt")
            else:
                saved_predictions_path = os.path.join(saved_predictions_path, "predictions.pt")
        
        if os.path.exists(saved_predictions_path):
            print(f"Loading pre-computed results from {saved_predictions_path}...")
            try:
                predictions_dict = torch.load(saved_predictions_path, map_location="cpu", weights_only=False)
                print("Successfully loaded pre-computed results.")
                # Convert loaded tensors to numpy format for compatibility
                for key, value in predictions_dict.items():
                    if isinstance(value, torch.Tensor):
                        predictions_dict[key] = value.numpy()
                    elif isinstance(value, dict):  # Handle nested dictionaries (e.g., cam01, cam02)
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, torch.Tensor):
                                predictions_dict[key][subkey] = subvalue.numpy()
            except Exception as e:
                print(f"Error loading {saved_predictions_path}: {e}. Proceeding with inference.")
                predictions_dict = None 
        else:
            print(f"No pre-computed results found at {saved_predictions_path}. Proceeding with inference.")

    if predictions_dict is None:
        if args.train:
            # Seed before constructing randomly initialized Gaussian modules,
            # not only before scene sampling. DDP then broadcasts the same
            # initial parameters regardless of the requested GPU count.
            from loger.util.minimal_train_provenance import seed_all_rng
            seed_all_rng(int(args.train_seed), include_cuda=torch.cuda.is_available())
        model = load_pi3_model(args.model_name, args.config, args.pi3x, args.pi3x_metric, args.gaussian or args.render_gaussian)
        if model is None:
            print("Failed to load model. Exiting.")
            return

        # Move model to device early
        model = model.to(device)

        # --- Load pretrained Gaussian modules from an optional checkpoint ---
        if args.pretrained_decoder and args.train:
            _log(f"Loading pretrained Gaussian modules from {args.pretrained_decoder}")
            from loger.utils.checkpoint import load_gaussian_weights
            load_gaussian_weights(model, args.pretrained_decoder,
                                  require_merger=bool(args.neural_freeze_decoder))

        # --- Training mode setup ---
        if args.train:
            from loger.util.minimal_train_provenance import (
                attach_rng_to_checkpoint,
                build_minimal_provenance,
                require_empty_or_missing_dir,
                restore_rng_from_checkpoint,
                seed_all_rng,
                write_minimal_provenance,
            )

            if args.require_empty_output:
                if not args.output_folder:
                    raise ValueError("--require_empty_output needs --output_folder")
                if _rank() == 0:
                    require_empty_or_missing_dir(args.output_folder)
                if _is_ddp():
                    dist.barrier()
            if args.output_folder:
                os.makedirs(args.output_folder, exist_ok=True)
            if _is_ddp():
                dist.barrier()
            # Explicit seed before any train-time sampling; resume may overwrite below.
            seed_all_rng(int(args.train_seed), include_cuda=torch.cuda.is_available())
            _log(f"train_seed={int(args.train_seed)}")
            if args.output_folder and _rank() == 0:
                try:
                    prov = build_minimal_provenance(
                        resolved_args=vars(args),
                        repo_root=Path(__file__).resolve().parent,
                        key_source_files=[
                            Path(__file__).resolve(),
                            Path(__file__).resolve().parent / "loger" / "util" / "minimal_train_provenance.py",
                        ],
                        dependency_checkpoints={
                            "model_name": args.model_name,
                            "pretrained_decoder": args.pretrained_decoder,
                            "skyseg_onnx": getattr(args, "sky_mask_onnx", None),
                        },
                        dataset_manifest=(
                            Path(args.dataset_root) / "manifest.json"
                            if getattr(args, "dataset_root", None)
                            else None
                        ),
                    )
                    # Record DDP launch shape for forensic audits.
                    prov["ddp"] = {
                        "world_size": dist.get_world_size() if _is_ddp() else 1,
                        "ddp_stripe_scenes": bool(args.ddp_stripe_scenes),
                    }
                    write_minimal_provenance(args.output_folder, prov)
                    _log(f"Wrote training_provenance.json under {args.output_folder}")
                except FileExistsError:
                    _log("training_provenance.json already present; leaving unchanged")
            if _is_ddp():
                dist.barrier()
            model.train()
            # Freeze everything except gaussian_decoder and gaussian_head
            neural_mode = args.merge_mode in ("neural", "neural_set", "neural_feature", "neural_direct_slots")
            active_merger_prefix = (
                "neural_track_direct_slot_merger" if args.merge_mode == "neural_direct_slots"
                else "neural_track_feature_merger" if args.merge_mode == "neural_feature"
                else "neural_track_set_merger" if args.merge_mode == "neural_set"
                else "neural_track_merger"
            )
            for name, param in model.named_parameters():
                direct_slots_mode = args.merge_mode == "neural_direct_slots"
                decoder_trainable = direct_slots_mode or not (neural_mode and (args.neural_freeze_base or args.neural_freeze_decoder))
                head_trainable = direct_slots_mode or not (neural_mode and args.neural_freeze_base)
                m2_projection_trainable = args.merge_mode == "neural_feature" and name.startswith("gaussian_head.merge_feature_projection")
                if (
                    (decoder_trainable and name.startswith("gaussian_decoder"))
                    or (head_trainable and name.startswith("gaussian_head"))
                    or m2_projection_trainable
                    or (neural_mode and name.startswith(active_merger_prefix))
                ):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            _log(f"Training mode: {trainable:,} / {total:,} params ({100*trainable/total:.1f}%)")

            # Wrap with DDP if multi-GPU. find_unused_parameters must be True:
            # long-seq buckets often freeze head (head_update=False) while merger trains,
            # and scene striping can leave different param subsets unused across ranks.
            if _is_ddp():
                ddp_kwargs = {
                    "device_ids": [local_rank],
                    "find_unused_parameters": not args.ddp_static_graph,
                }
                if args.ddp_static_graph:
                    ddp_kwargs["static_graph"] = True
                model = nn.parallel.DistributedDataParallel(model, **ddp_kwargs)
                raw_model = model.module
            else:
                raw_model = model
        else:
            model = model.eval()
            raw_model = model

        # Load images (skip if using dataset mode — scenes are loaded per-step)
        if not (args.dataset and args.train):
            input_paths = [p for p in [args.input, args.input2, args.input3, args.input4, args.input5] if p is not None]

            all_image_names_collected = []

            for i, input_path in enumerate(input_paths):
                if is_video_file(input_path):
                    # Video frames for each input are extracted to its own sub-folder
                    temp_dir = tempfile.mkdtemp(prefix=f"pi3_frames_input{i+1}_")
                    temp_frame_dirs[f"input{i+1}"] = temp_dir
                    current_frames = extract_frames_from_video(input_path, temp_dir, args.start_frame, args.end_frame, args.stride)
                    all_image_names_collected.extend(current_frames)
                elif os.path.isdir(input_path):
                    current_frames = natsorted(glob.glob(os.path.join(input_path, "*.png"))+glob.glob(os.path.join(input_path, "*.jpg"))+glob.glob(os.path.join(input_path, "*.jpeg")))
                    # remove the files that has depth in the name
                    current_frames = [f for f in current_frames if "depth" not in os.path.basename(f).lower()]
                    end_idx = args.end_frame if args.end_frame != -1 else None
                    current_frames = current_frames[args.start_frame:end_idx:args.stride]
                    all_image_names_collected.extend(current_frames)

            if not all_image_names_collected:
                print("No images to process. Exiting.")
                cleanup_ddp()
                return

            print(f"Found {len(all_image_names_collected)} images to process.")
            if target_resolution is not None:
                images_tensor = load_images_from_paths(all_image_names_collected, Target_W=target_resolution[0], Target_H=target_resolution[1])
            else:
                images_tensor = load_images_from_paths(all_image_names_collected)
            if not args.load_to_cpu:
                images_tensor = images_tensor.to(device)

            if images_tensor.numel() == 0:
                print("Error: No images were loaded successfully. Check image paths and formats.")
                cleanup_ddp()
                return

            print("Running inference...")
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
            num_frames = images_tensor.shape[0]
        else:
            num_frames = args.train_samples  # dataset mode: placeholder

        # ----------------------------------------------------------------
        #  Training loop
        # ----------------------------------------------------------------
        if args.train:
            from loger.renderer import GaussianRenderer
            from loger.losses import GaussianLoss

            gs_renderer = GaussianRenderer(sh_degree=2)
            loss_fn = GaussianLoss(mse_weight=1.0, lpips_weight=0.25,
                                    ssim_weight=args.ssim_weight,
                                    scale_reg_weight=args.scale_reg_weight,
                                    scale_target_ratio=0.7,
                                    aniso_reg_weight=args.aniso_reg_weight,
                                    aniso_threshold=args.aniso_threshold,
                                    offset_reg_weight=args.offset_reg_weight,
                                    activation_checkpoint=args.train_loss_checkpoint,
                                    device=device)
            nvs_loss_fn = GaussianLoss(mse_weight=1.0, lpips_weight=0.25,
                                       ssim_weight=args.ssim_weight,
                                       scale_reg_weight=0.0,
                                       aniso_reg_weight=0.0,
                                       offset_reg_weight=0.0,
                                       activation_checkpoint=args.train_loss_checkpoint,
                                       device=device)
            K_gpu = None
            direct_slots_mode = args.merge_mode == "neural_direct_slots"
            if direct_slots_mode:
                merger_params = []
                base_params = []
                for name, parameter in model.named_parameters():
                    if not parameter.requires_grad:
                        continue
                    (merger_params if name.startswith("neural_track_direct_slot_merger") else base_params).append(parameter)
                optimizer = torch.optim.AdamW([
                    {"params": merger_params, "lr": args.direct_slot_merger_lr},
                    {"params": base_params, "lr": args.direct_slot_base_lr},
                ])
                planned_steps = args.epochs * (
                    args.scenes_per_epoch if args.scenes_per_epoch > 0
                    else args.train_scene_limit if args.train_scene_limit > 0 else 1
                )
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lr_lambda=lambda step: 0.1 + 0.9 * (
                        1.0 + math.cos(math.pi * min(step, planned_steps) / max(planned_steps, 1))
                    ) / 2.0,
                )
            else:
                if neural_mode and args.train_head_lr > 0 and args.train_merger_lr > 0:
                    head_params, merger_params = [], []
                    for name, parameter in model.named_parameters():
                        if not parameter.requires_grad:
                            continue
                        (merger_params if name.startswith(active_merger_prefix) else head_params).append(parameter)
                    optimizer = torch.optim.AdamW([
                        {"params": head_params, "lr": args.train_head_lr},
                        {"params": merger_params, "lr": args.train_merger_lr},
                    ])
                    _log(f"Optimizer groups: head={args.train_head_lr:g}, merger={args.train_merger_lr:g}")
                else:
                    optimizer = torch.optim.AdamW(
                        [p for p in model.parameters() if p.requires_grad],
                        lr=args.lr,
                    )
                scheduler = None

            optimizer.scene_accumulation_steps = args.gradient_accumulation_steps
            train_samples = min(args.train_samples, num_frames)

            # --- Dataset training mode ---
            if args.dataset == "dl3dv":
                # Train on the full dl3dv dataset (including 10K).
                all_scenes = load_dataset_scene_list(args.dataset, args.dataset_root, "train")
                if not all_scenes:
                    raise ValueError("Training index is empty; test scenes must never be used for training")
                train_scenes = list(all_scenes)
                if args.train_scene_limit > 0:
                    train_scenes = train_scenes[:args.train_scene_limit]
                    _log(f"  Training scene limit: first {len(train_scenes)} indexed scenes.")
                balanced_train_groups = None
                has_balanced_roots = bool(
                    args.balanced_train_root_a or args.balanced_train_root_b
                )
                if has_balanced_roots:
                    if not (
                        args.balanced_train_root_a
                        and args.balanced_train_root_b
                        and args.balanced_train_count_a > 0
                        and args.balanced_train_count_b > 0
                    ):
                        raise ValueError(
                            "balanced training requires both roots and both positive expected counts"
                        )
                    source_a = load_dataset_scene_list(
                        args.dataset, args.balanced_train_root_a, "train"
                    )
                    source_b = load_dataset_scene_list(
                        args.dataset, args.balanced_train_root_b, "train"
                    )
                    if len(source_a) != args.balanced_train_count_a:
                        raise ValueError(
                            f"balanced source A expected {args.balanced_train_count_a} scenes, "
                            f"found {len(source_a)}"
                        )
                    if len(source_b) != args.balanced_train_count_b:
                        raise ValueError(
                            f"balanced source B expected {args.balanced_train_count_b} scenes, "
                            f"found {len(source_b)}"
                        )
                    overlap = set(source_a).intersection(source_b)
                    if overlap:
                        raise ValueError(
                            f"balanced sources overlap in {len(overlap)} scene path(s)"
                        )
                    balanced_train_groups = (source_a, source_b)
                    train_scenes = source_a + source_b
                    _log(
                        "  Balanced training: "
                        f"source_a={len(source_a)}, source_b={len(source_b)}, "
                        "per-global-batch ratio=1:1"
                    )
                blocked_train_scenes = _load_train_skip_scenes(
                    args.train_skip_scene_file
                )
                if blocked_train_scenes:
                    _log(
                        f"  Synchronized slow-scene skip: {len(blocked_train_scenes)} "
                        f"scene(s), context >= {args.train_skip_scene_min_context}"
                    )
                val_scenes = [s for s in all_scenes if os.sep + "10K" + os.sep in s]
                if not val_scenes:
                    val_scenes = load_dataset_scene_list(args.dataset, args.dataset_root, "test")
                merge_config = None
                merge_config_soft = None
                merge_config_hard = None
                if args.track_merge_train:
                    from loger.merge.merge_config import MergeConfig

                    merge_config_hard = MergeConfig(
                        min_track_len=args.merge_min_track_len,
                        min_mean_opacity=args.merge_min_opacity,
                        uv_thresh=args.merge_uv_thresh,
                        merge_3d_thresh=args.merge_3d_thresh,
                        max_center_variance=args.merge_max_center_variance,
                        merge_mode=args.merge_mode,
                        rotation_mode=args.rotation_mode,
                        soft_validity_temp=args.merge_soft_validity_temp,
                        hybrid_budget_ratio=args.merge_hybrid_budget_ratio,
                    )
                    merge_config_soft = MergeConfig(
                        min_track_len=args.merge_min_track_len,
                        min_mean_opacity=args.merge_soft_min_opacity,
                        uv_thresh=args.merge_uv_thresh,
                        merge_3d_thresh=args.merge_3d_thresh,
                        max_center_variance=args.merge_soft_max_center_variance,
                        merge_mode=args.merge_mode,
                        rotation_mode=args.rotation_mode,
                        soft_validity_temp=args.merge_soft_validity_temp,
                        hybrid_budget_ratio=args.merge_hybrid_budget_ratio,
                    )
                    if args.train_bucket_best_postprocess:
                        for cfg in (merge_config_hard, merge_config_soft):
                            cfg.sky_mask_drop = True
                            cfg.sky_mask_onnx = args.sky_mask_onnx
                            cfg.sky_mask_score_thr = args.sky_mask_score_thr
                            cfg.global_complement_voxel_schedule = "fixed"
                            cfg.global_complement_voxel_policy = "first"
                            cfg.global_complement_voxel_min_windows = 3
                    merge_config = merge_config_hard
                bucket_counts = _csv_numbers(args.long_context_counts, int)
                bucket_weights = _csv_numbers(args.long_context_weights, float)
                if args.long_sequence_protocol:
                    if not bucket_counts or len(bucket_counts) != len(bucket_weights):
                        raise ValueError("long context counts/weights must be non-empty and equal length")
                    sparse_counts = _csv_numbers(args.long_sparse_context_counts, int)
                    sparse_intervals = _csv_numbers(args.long_sparse_intervals, int)
                    if len(sparse_counts) != len(sparse_intervals):
                        raise ValueError("long sparse counts/intervals must have equal length")
                    sparse_by_count = dict(zip(sparse_counts, sparse_intervals))
                    bucket_modes = [x.strip().lower() for x in args.long_context_modes.split(",") if x.strip()]
                    bucket_intervals = _csv_numbers(args.long_context_intervals, int)
                    if bucket_modes or bucket_intervals:
                        if len(bucket_modes) != len(bucket_counts) or len(bucket_intervals) != len(bucket_counts):
                            raise ValueError("long context counts/modes/intervals must have equal length")
                        if any(mode not in {"dense", "sparse"} for mode in bucket_modes):
                            raise ValueError("long context modes must be dense or sparse")
                        if any(interval < 1 for interval in bucket_intervals):
                            raise ValueError("long context intervals must be positive")
                        if any(mode == "dense" and interval != 1 for mode, interval in zip(bucket_modes, bucket_intervals)):
                            raise ValueError("dense long-context buckets must use interval 1")
                    else:
                        bucket_modes = ["sparse" if count in sparse_by_count else "dense" for count in bucket_counts]
                        bucket_intervals = [sparse_by_count.get(count, 1) for count in bucket_counts]
                    bucket_specs = list(zip(bucket_counts, bucket_modes, bucket_intervals))
                    weighted_cycle = build_weighted_bucket_cycle(bucket_specs, bucket_weights)
                    bucket_recipe = ";".join(
                        f"C{count}-{mode}-I{interval}:w{weight:g}"
                        for (count, mode, interval), weight in zip(bucket_specs, bucket_weights)
                    )
                    _log(f"Long buckets: {list(zip(bucket_specs, bucket_weights))}; cycle_len={len(weighted_cycle)}")
                else:
                    sparse_by_count = {}
                sky_masker = None
                if args.train_bucket_best_postprocess:
                    from loger.utils.sky_mask import SkyMasker
                    sky_masker = SkyMasker(args.sky_mask_onnx, args.sky_mask_score_thr)
                print(f"Dataset mode: {len(train_scenes)} training scenes, "
                      f"{train_samples} context frames/step, "
                      f"gap_max={args.frame_gap_max}, holdout={args.nvs_holdout}, "
                      f"nvs_loss_weight={args.nvs_loss_weight}, ssim_weight={args.ssim_weight}, "
                      f"se3={not args.no_se3 and not args.sim3}, sim3={args.sim3}")
                if args.track_merge_train:
                    print(
                        f"Track merge train: in-forward AllTracker, "
                        f"conf_thr={args.alltracker_conf_thr}, dense_complement="
                        f"{not args.no_track_dense_complement}, "
                        f"aniso_reg={args.aniso_reg_weight}, "
                        f"offset_reg={args.offset_reg_weight}, "
                        f"dense_reg_max={args.merge_dense_reg_max}, "
                        f"merge_mode={args.merge_mode}, rotation={args.rotation_mode}, "
                        f"hybrid_budget={args.merge_hybrid_budget_ratio}, "
                        f"curriculum dense_epochs={args.merge_curriculum_dense_epochs}, "
                        f"soft_epochs={args.merge_curriculum_soft_epochs}"
                    )

                # --- Resume from checkpoint if available ---
                global_step = 0
                start_epoch = 0
                curriculum_start_epoch = 0
                resume_bucket_scheduler_state = None
                resume_scene_cursor = 0
                resume_world_size = 1
                if args.output_folder:
                    ckpt_files = sorted(
                        glob.glob(os.path.join(args.output_folder, "gaussian_head_epoch*.pt"))
                        + glob.glob(os.path.join(args.output_folder, "gaussian_head_step*.pt"))
                    )
                    if ckpt_files:
                        latest = ckpt_files[-1]
                        print(f"Resuming from checkpoint: {latest}")
                        # Trusted local training checkpoint includes optimizer/RNG
                        # state. Keep the staging copy on CPU to avoid retaining
                        # an extra Gaussian module copy in limited GPU memory.
                        ckpt = torch.load(latest, map_location="cpu", weights_only=False)
                        # Support both old (flat state dict) and new (dict with metadata) formats
                        if isinstance(ckpt, dict) and "model" in ckpt:
                            model_state = ckpt["model"]
                            if "optimizer" in ckpt:
                                optimizer.load_state_dict(ckpt["optimizer"])
                            start_epoch = ckpt.get("epoch", 0)
                            if "_epoch" in os.path.basename(latest):
                                start_epoch += 1
                            global_step = ckpt.get("global_step", 0)
                            resume_world_size = int(ckpt.get("world_size", 1))
                            resume_accumulation = int(ckpt.get("gradient_accumulation_steps", 1))
                            if resume_accumulation != args.gradient_accumulation_steps:
                                raise ValueError("Cannot resume with changed gradient accumulation")
                            resume_scene_cursor = int(
                                ckpt.get("scene_cursor", global_step * resume_world_size * resume_accumulation)
                            )
                            resume_bucket_scheduler_state = ckpt.get("bucket_scheduler")
                            try:
                                if restore_rng_from_checkpoint(
                                    ckpt, include_cuda=torch.cuda.is_available()
                                ):
                                    print("  Restored RNG state from checkpoint")
                            except ValueError as exc:
                                print(f"  WARNING: could not restore RNG state: {exc}")
                        else:
                            model_state = ckpt  # old format: flat state dict
                            # Infer epoch from filename
                            import re
                            m = re.search(r'epoch(\d+)', latest)
                            if m:
                                start_epoch = int(m.group(1)) + 1
                        curriculum_start_epoch = infer_curriculum_start_epoch(
                            ckpt if isinstance(ckpt, dict) else {},
                            ckpt_files,
                            start_epoch,
                        )
                        if _is_ddp():
                            raw_model.load_state_dict(model_state, strict=False)
                        else:
                            model.load_state_dict(model_state, strict=False)
                        print(f"  Resuming from epoch {start_epoch}, step {global_step}")
                        print(f"  Curriculum origin epoch: {curriculum_start_epoch}")

                bucket_scheduler = None
                if args.long_sequence_protocol:
                    bucket_scheduler = BucketRetryScheduler(
                        weighted_cycle,
                        args.long_bucket_max_retries,
                        position=global_step % len(weighted_cycle),
                    )
                    if resume_bucket_scheduler_state is not None:
                        bucket_scheduler.load_state_dict(resume_bucket_scheduler_state)
                    _log(
                        f"Bucket scheduler: position={bucket_scheduler.position} "
                        f"max_retries={bucket_scheduler.max_retries}"
                    )

                # --- One-shot init stats ---
                _log("  Initial gaussian params (before training):")
                init_scene = random.choice(train_scenes)
                # A full-window 64--128 frame forward defeats the sliding-window
                # protocol and can OOM before the first real long-sequence step.
                init_result = None if args.long_sequence_protocol else load_dl3dv_scene(
                    init_scene, train_samples, target_resolution, args.image_scale
                )
                if args.long_sequence_protocol:
                    _log("           skipped (long-sequence protocol)")
                if init_result is not None:
                    imgs, _, K = init_result
                    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                        init_pred = model(
                            imgs.to(device)[None], gaussian=True,
                            window_size=train_samples, overlap_size=0,
                            sim3=args.sim3,
                            se3=not args.no_se3 and not args.sim3,
                            turn_off_ttt=args.no_ttt,
                            turn_off_swa=args.no_swa,
                        )
                    s = init_pred["gaussian_scales"].float().reshape(-1)
                    o = init_pred["gaussian_opacities"].float().reshape(-1)
                    pts = init_pred["points"].float().reshape(-1, 3)
                    if pts.shape[0] > 1000:
                        pts = pts[torch.randperm(pts.shape[0], device=pts.device)[:1000]]
                    nn_dists = torch.cdist(pts, pts).sort(dim=1).values[:, 1]
                    _log(f"           scale p50={s.quantile(0.5):.4f}  "
                         f"p05={s.quantile(0.05):.4f}  p95={s.quantile(0.95):.4f}")
                    _log(f"           opacity p50={o.quantile(0.5):.4f}  "
                         f"p95={o.quantile(0.95):.4f}")
                    _log(f"           pts nn p50={nn_dists.median():.4f}  "
                         f"p95={nn_dists.quantile(0.95):.4f}")

                for epoch in range(start_epoch, args.epochs):
                    curriculum_phase = "hard"
                    use_merge_this_epoch = bool(args.track_merge_train)
                    active_merge_config = merge_config
                    if args.track_merge_train:
                        curriculum_phase = merge_curriculum_phase(
                            epoch,
                            curriculum_start_epoch,
                            args.merge_curriculum_dense_epochs,
                            args.merge_curriculum_soft_epochs,
                        )
                        if curriculum_phase == "dense":
                            use_merge_this_epoch = False
                            active_merge_config = None
                        elif curriculum_phase == "soft":
                            use_merge_this_epoch = True
                            active_merge_config = merge_config_soft
                        else:
                            use_merge_this_epoch = True
                            active_merge_config = merge_config_hard
                        if _rank() == 0 and (
                            args.merge_curriculum_dense_epochs > 0
                            or args.merge_curriculum_soft_epochs > 0
                        ):
                            _log(
                                f"  curriculum phase={curriculum_phase} "
                                f"(epoch {epoch}, relative "
                                f"{epoch - curriculum_start_epoch})"
                            )

                    if balanced_train_groups is None:
                        epoch_scenes = _sample_epoch_scenes(
                            train_scenes, args.scenes_per_epoch, epoch
                        )
                    else:
                        global_batch_size = (
                            dist.get_world_size() if _is_ddp() else 1
                        ) * args.gradient_accumulation_steps
                        epoch_scenes = _sample_balanced_epoch_scenes(
                            balanced_train_groups,
                            args.scenes_per_epoch,
                            epoch,
                            global_batch_size,
                        )
                    epoch_total_scene_slots = len(epoch_scenes)
                    active_scene_cursor = (
                        resume_scene_cursor
                        if global_step > 0 and epoch == start_epoch
                        else 0
                    )
                    if active_scene_cursor:
                        _log(
                            f"  Resume scene cursor: skipping "
                            f"{active_scene_cursor} consumed global slots "
                            f"(saved world_size={resume_world_size}, "
                            f"current world_size={dist.get_world_size() if _is_ddp() else 1})"
                        )
                    epoch_scenes = _resume_epoch_scenes(
                        epoch_scenes, scene_cursor=active_scene_cursor
                    )
                    if args.ddp_stripe_scenes:
                        epoch_scenes = _ddp_stripe_epoch_scenes(epoch_scenes)
                        if _is_ddp():
                            _log(
                                f"  DDP stripe: world={dist.get_world_size()} "
                                f"rank_scenes={sum(1 for s in epoch_scenes if s is not None)} "
                                f"(microsteps; {args.gradient_accumulation_steps} per update)"
                            )
                    resume_global_step = global_step
                    if len(epoch_scenes) % args.gradient_accumulation_steps:
                        raise ValueError("Scene slots must form complete accumulated updates")
                    target_opt_steps = global_step + len(epoch_scenes) // args.gradient_accumulation_steps
                    if args.train_max_global_steps > 0:
                        target_opt_steps = min(
                            target_opt_steps, args.train_max_global_steps
                        )
                    current_world_size = dist.get_world_size() if _is_ddp() else 1
                    scene_cursor = active_scene_cursor
                    epoch_losses = []
                    epoch_recon_losses = []
                    epoch_nvs_losses = []
                    epoch_gs_stats = []
                    epoch_attempted_steps = 0
                    epoch_nonfinite_losses = 0
                    epoch_nonfinite_grads = 0
                    epoch_successful_steps = 0
                    accumulated_scenes = 0
                    _log(f"  Effective scene batch={current_world_size * args.gradient_accumulation_steps}; "
                         f"accumulation={args.gradient_accumulation_steps}; target updates={target_opt_steps}")

                    for scene_i, scene_path in enumerate(epoch_scenes):
                        if args.long_sequence_protocol and global_step >= target_opt_steps:
                            break
                        # Backward has finished for the previous scene. Do not
                        # retain its clouds/render outputs while constructing
                        # the next long sequence (including accumulation microsteps).
                        # Parameter .grad buffers are deliberately preserved.
                        predictions = dense_predictions = per_window = None
                        means_h = quats_h = scales_h = opacs_h = colors_h = camera_poses = None
                        rendered = nvs_rendered = dense_rendered = None
                        pred_rgb = target_rgb = nvs_pred_rgb = nvs_tgt_rgb = dense_pred_rgb = None
                        loss = recon_loss = nvs_loss = dense_recon_loss = merge_recon_loss = None
                        dense_ctx_scales = dense_ctx_means = dense_offs = dense_offs_t = None
                        dense_scales = dense_means = dense_offsets = None
                        ctx_scales = ctx_means = offs = offs_t = None
                        s = o = s_flat = o_flat = pts = nn_dists = off = None
                        grad_probes = None
                        scene_cursor = min(
                            epoch_total_scene_slots,
                            active_scene_cursor + (scene_i + 1) * current_world_size,
                        )
                        if scene_path is None:
                            # Padding slot for equal-length DDP loops.
                            if bucket_scheduler is not None:
                                bucket_scheduler.record_failure()
                            _ddp_noop_step(model, optimizer)
                            continue
                        bucket_context = train_samples
                        if args.long_sequence_protocol:
                            # Deterministic weighted cycle: unlike independent
                            # draws this guarantees that every configured bucket
                            # is actually attempted in a finite continuation.
                            bucket_context, bucket_mode, bucket_interval = bucket_scheduler.current
                            local_blocked = (
                                bucket_context >= args.train_skip_scene_min_context
                                and _scene_is_blocked(scene_path, blocked_train_scenes)
                            )
                            if _ddp_any_true(local_blocked, device):
                                if local_blocked:
                                    print(
                                        f"[rank {_rank()}] Skip (slow-scene blocklist): "
                                        f"C{bucket_context} {scene_path}"
                                    )
                                if bucket_scheduler.record_failure() and _rank() == 0:
                                    print(
                                        "  bucket retry limit reached; advancing to next bucket"
                                    )
                                _ddp_noop_step(model, optimizer, sync_backward=True)
                                continue
                            split_result = try_load_dl3dv_bucket_split(
                                scene_path, bucket_context,
                                bucket_interval,
                                min(args.nvs_holdout, args.long_train_target_limit),
                                target_resolution, args.image_scale,
                            )
                        else:
                            split_result = try_load_dl3dv_scene_split(
                                scene_path, train_samples, args.frame_gap_max,
                                args.nvs_holdout,
                                target_resolution, args.image_scale,
                            )
                        if not _ddp_scene_step_valid(split_result, device):
                            if bucket_scheduler is not None:
                                bucket_scheduler.record_failure()
                            _ddp_noop_step(model, optimizer)
                            continue
                        (images_step, _, K_step, ctx_local, holdout_local,
                         ctx_global, holdout_global) = split_result
                        epoch_attempted_steps += 1
                        images_gpu = images_step.to(device)
                        K_gpu = K_step.to(device)
                        ctx_local_t = torch.tensor(ctx_local, device=device, dtype=torch.long)
                        holdout_local_t = holdout_local
                        ctx_loss_local = _even_subset(
                            ctx_local, args.long_train_context_render_limit
                        ) if args.long_sequence_protocol else ctx_local
                        ctx_loss_local_t = torch.tensor(
                            ctx_loss_local, device=device, dtype=torch.long
                        )
                        ctx_loss_global = [
                            ctx_global[ctx_local.index(i)] for i in ctx_loss_local
                        ]

                        if args.long_sequence_protocol and neural_mode:
                            head_enabled = (not args.neural_freeze_base
                                            and bucket_context <= args.long_head_update_max_context)
                            for name, parameter in raw_model.named_parameters():
                                if name.startswith("gaussian_head"):
                                    parameter.requires_grad_(head_enabled)
                            _log(
                                f"  train step={global_step + 1} bucket=C{bucket_context} "
                                f"mode={bucket_mode} interval={bucket_interval} "
                                f"head_update={head_enabled} merger_update=True"
                            )

                        if accumulated_scenes == 0:
                            optimizer.zero_grad(set_to_none=True)

                        with torch.cuda.amp.autocast(enabled=False):
                            if args.train_bucket_best_postprocess and active_merge_config is not None:
                                # Short buckets use factor-4 complement dedup
                                # without robust depth filtering; long buckets retain complements
                                # and use strong robust depth filtering below.
                                active_merge_config.global_complement_voxel_factor = (
                                    4.0 if bucket_context <= 32 else 0.0
                                )
                            forward_kw = model_sliding_window_kwargs(
                                images_gpu.shape[0],
                                args.window_size,
                                args.overlap_size,
                                sim3=args.sim3,
                                se3=not args.no_se3,
                                turn_off_ttt=args.no_ttt,
                                turn_off_swa=args.no_swa,
                            )
                            if args.calibrate_local_points_to_k:
                                forward_kw.update(
                                    intrinsics=K_gpu,
                                    calibrate_local_points_to_k=True,
                                )
                            if args.train_gaussian_head_checkpoint:
                                forward_kw["gaussian_head_checkpoint"] = True
                            if args.train_gaussian_head_batch_size > 0:
                                forward_kw["gaussian_head_batch_size"] = args.train_gaussian_head_batch_size
                            merge_stats = None
                            img_h, img_w = images_gpu.shape[2], images_gpu.shape[3]
                            is_neural_merge = bool(
                                use_merge_this_epoch
                                and active_merge_config is not None
                            and active_merge_config.merge_mode in ("neural", "neural_set", "neural_feature", "neural_direct_slots")
                            )
                            direct_dense_only = bool(
                                is_neural_merge
                                and active_merge_config.merge_mode == "neural_direct_slots"
                                and global_step % 2 == 0
                            )
                            dense_predictions = None
                            grad_probes = {}
                            if is_neural_merge and (
                                args.neural_dense_loss_weight > 0
                                or active_merge_config.merge_mode == "neural_direct_slots"
                            ) and not (active_merge_config.merge_mode == "neural_direct_slots" and not direct_dense_only):
                                # Keep the original dense reconstruction branch
                                # in the graph.  This is intentionally a second
                                # forward for the pilot: it prevents the new
                                # merge path from silently replacing the old
                                # training objective.
                                dense_predictions = model(
                                    images_gpu[None], gaussian=True,
                                    no_detach=True, **forward_kw
                                )
                                if args.diagnose_nonfinite_grads:
                                    grad_probes.update(retain_grad_probes({
                                        "dense.means": dense_predictions.get("gaussian_means"),
                                        "dense.scales": dense_predictions.get("gaussian_scales"),
                                        "dense.quats": dense_predictions.get("gaussian_rotations"),
                                        "dense.opacities": dense_predictions.get("gaussian_opacities"),
                                        "dense.sh": dense_predictions.get("gaussian_sh_coeffs"),
                                        "dense.offsets": dense_predictions.get("gaussian_offsets"),
                                    }))

                            if use_merge_this_epoch:
                                from loger.merge.window_merge import is_window_merge_result

                                forward_kw.update(
                                    track_merge=True,
                                    merge_config=active_merge_config,
                                    alltracker_conf_thr=args.alltracker_conf_thr,
                                    track_dense_complement=not args.no_track_dense_complement,
                                    merge_ctx_indices=ctx_local,
                                    no_detach=True,
                                )
                                if args.train_bucket_best_postprocess and bucket_context >= 64:
                                    forward_kw["record_per_window_clouds"] = True
                                if active_merge_config is not None and active_merge_config.merge_mode in ("neural", "neural_set", "neural_feature", "neural_direct_slots"):
                                    forward_kw["neural_merger"] = (
                                        raw_model.neural_track_direct_slot_merger if active_merge_config.merge_mode == "neural_direct_slots"
                                        else raw_model.neural_track_feature_merger if active_merge_config.merge_mode == "neural_feature"
                                        else raw_model.neural_track_set_merger if active_merge_config.merge_mode == "neural_set"
                                        else raw_model.neural_track_merger
                                    )

                            forward_oom = False
                            forward_oom_message = ""
                            try:
                                saved_tensor_ctx = (
                                    torch.autograd.graph.save_on_cpu(pin_memory=False)
                                    if args.train_save_activations_on_cpu
                                    else contextlib.nullcontext()
                                )
                                with saved_tensor_ctx:
                                    if direct_dense_only:
                                        with torch.no_grad():
                                            predictions = model(images_gpu[None], gaussian=True, **forward_kw)
                                    else:
                                        predictions = model(images_gpu[None], gaussian=True, **forward_kw)
                            except RuntimeError as exc:
                                if (
                                    use_merge_this_epoch
                                    and torch.cuda.is_available()
                                    and "out of memory" in str(exc).lower()
                                ):
                                    forward_oom = True
                                    forward_oom_message = str(exc)
                                    predictions = None
                                    torch.cuda.empty_cache()
                                else:
                                    raise
                            if _is_ddp():
                                oom_flag = torch.tensor(
                                    [1 if forward_oom else 0],
                                    device=device,
                                    dtype=torch.int32,
                                )
                                dist.all_reduce(oom_flag, op=dist.ReduceOp.MAX)
                                forward_oom = bool(oom_flag.item())
                            if forward_oom:
                                if _rank() == 0:
                                    print(f"Skip (OOM): {scene_path}; {forward_oom_message}")
                                _ddp_noop_step(model, optimizer, sync_backward=True)
                                if bucket_scheduler.record_failure() and _rank() == 0:
                                    print("  bucket retry limit reached; advancing to next bucket")
                                continue

                            if use_merge_this_epoch:
                                if not is_window_merge_result(predictions):
                                    raise TypeError(
                                        "track_merge_train expected WindowMergeResult"
                                    )
                                postprocess_oom = False
                                postprocess_oom_message = ""
                                if args.train_bucket_best_postprocess:
                                    try:
                                        if bucket_context >= 64:
                                            from loger.merge.robust_foreign_depth_filter import (
                                                robust_foreign_depth_filter,
                                            )
                                            per_window = list(predictions.per_window_results or [])
                                            if not per_window:
                                                raise RuntimeError("strong robust depth filtering requires per-window clouds")
                                            predictions, depth_filter_stats = robust_foreign_depth_filter(
                                                predictions, per_window, gs_renderer,
                                                packed_merge_owner_groups(
                                                    ctx_local, images_gpu.shape[0],
                                                    args.window_size, args.overlap_size,
                                                ),
                                                K_gpu, img_h, img_w,
                                                rel_margin=0.01, min_votes=1,
                                                visible_ratio_threshold=0.30,
                                                decisive_ratio_threshold=0.30,
                                                min_rel_gap=0.01, alpha_threshold=0.5,
                                                neighborhood_radius=1, min_valid_neighbors=5,
                                                max_rel_depth_spread=0.15, render_chunk=8,
                                                allow_empty=True, collect_diag=False,
                                            )
                                        from loger.merge.sky_sphere_prototype import append_fixed_sky_sphere
                                        predictions = append_fixed_sky_sphere(
                                            predictions, images_gpu, ctx_local, K_gpu, sky_masker,
                                            radius_factor=args.sky_sphere_radius_factor,
                                            sample_stride=args.sky_sphere_stride,
                                            azimuth_bins=args.sky_sphere_azimuth_bins,
                                            elevation_bins=args.sky_sphere_elevation_bins,
                                            opacity=0.98,
                                        )
                                    except RuntimeError as exc:
                                        if torch.cuda.is_available() and "out of memory" in str(exc).lower():
                                            postprocess_oom = True
                                            postprocess_oom_message = str(exc)
                                            torch.cuda.empty_cache()
                                        else:
                                            raise
                                if _ddp_any_true(postprocess_oom, device):
                                    if _rank() == 0:
                                        print(
                                            f"Skip (postprocess OOM on at least one rank): "
                                            f"{scene_path}; {postprocess_oom_message}"
                                        )
                                    del predictions
                                    torch.cuda.empty_cache()
                                    _ddp_noop_step(model, optimizer, sync_backward=True)
                                    if bucket_scheduler.record_failure() and _rank() == 0:
                                        print("  bucket retry limit reached; advancing to next bucket")
                                    continue
                                merge_stats = predictions.stats
                                means_h, quats_h, scales_h, opacs_h, colors_h, camera_poses = (
                                    gs_renderer.prepare_hybrid(predictions)
                                )
                                means_h, quats_h, scales_h, opacs_h, colors_h = sanitize_hybrid_for_render(
                                    means_h, quats_h, scales_h, opacs_h, colors_h
                                )
                                if args.diagnose_nonfinite_grads:
                                    grad_probes.update(retain_grad_probes({
                                        "hybrid.means": means_h,
                                        "hybrid.scales": scales_h,
                                        "hybrid.quats": quats_h,
                                        "hybrid.opacities": opacs_h,
                                        "hybrid.colors": colors_h,
                                    }))
                                if _ddp_any_true(means_h.shape[0] == 0, device):
                                    if _rank() == 0:
                                        print(
                                            f"Skip (no hybrid gaussians on at least one rank): "
                                            f"{scene_path}"
                                        )
                                    _ddp_noop_step(model, optimizer, sync_backward=True)
                                    bucket_scheduler.record_failure()
                                    continue
                                rendered = gs_renderer.render_views_from_tensors(
                                    means_h,
                                    quats_h,
                                    scales_h,
                                    opacs_h,
                                    colors_h,
                                    camera_poses,
                                    ctx_loss_local,
                                    img_h,
                                    img_w,
                                    K_gpu,
                                    checkpoint_chunk_size=args.train_render_checkpoint_chunk_size,
                                )
                            else:
                                # Match render views to the context-loss subset
                                # (long_train_context_render_limit), not full ctx.
                                rendered = gs_renderer.render_sequence(
                                    predictions, target_indices=ctx_loss_local,
                                    neighbor_frames=args.neighbor_frames, K=K_gpu,
                                )

                            pred_rgb = rendered["rgb"].permute(0, 3, 1, 2)
                            target_rgb = images_gpu[ctx_loss_local_t]
                            if is_neural_merge:
                                dense_recon_loss = torch.zeros((), device=device)
                                if dense_predictions is not None:
                                    dense_rendered = gs_renderer.render_sequence(
                                        dense_predictions, target_indices=ctx_loss_local,
                                        neighbor_frames=args.neighbor_frames, K=K_gpu,
                                    )
                                    dense_pred_rgb = dense_rendered["rgb"].permute(0, 3, 1, 2)
                                    dense_ctx_scales = dense_predictions["gaussian_scales"].float()[:, ctx_loss_local].permute(0, 1, 3, 4, 2).reshape(-1, 3)
                                    dense_ctx_means = dense_predictions["points"].float()[:, ctx_loss_local].reshape(-1, 3)
                                    dense_offs = dense_predictions.get("gaussian_offsets")
                                    dense_offs_t = None
                                    if dense_offs is not None:
                                        dense_offs_t = dense_offs.float()[:, ctx_loss_local].permute(0, 1, 3, 4, 2).reshape(-1, 3)
                                    dense_recon_loss, _ = loss_fn(
                                        dense_pred_rgb, target_rgb,
                                        scales=dense_ctx_scales,
                                        means=dense_ctx_means,
                                        offsets=dense_offs_t,
                                    )
                                merge_recon_loss, merge_recon_logs = nvs_loss_fn(
                                    pred_rgb, target_rgb,
                                )
                                dense_weight = args.neural_dense_loss_weight
                                if active_merge_config is not None and active_merge_config.merge_mode == "neural_direct_slots":
                                    dense_weight = (
                                        args.direct_slot_dense_warmup_weight
                                        if global_step < args.direct_slot_dense_warmup_steps
                                        else args.direct_slot_dense_weight
                                    )
                                recon_loss = dense_weight * dense_recon_loss if direct_dense_only else (
                                    dense_weight * dense_recon_loss
                                    + args.neural_merge_loss_weight * merge_recon_loss
                                )
                                recon_logs = {
                                    "total": recon_loss.detach(),
                                    "dense": dense_recon_loss.detach(),
                                    "merged": merge_recon_loss.detach(),
                                }
                            elif use_merge_this_epoch:
                                dense_scales, dense_means, dense_offsets = subsample_dense_reg(
                                    predictions.dense_scales,
                                    predictions.dense_means,
                                    predictions.dense_offsets,
                                    args.merge_dense_reg_max,
                                )
                                recon_loss, recon_logs = loss_fn(
                                    pred_rgb,
                                    target_rgb,
                                    scales=dense_scales,
                                    means=dense_means,
                                    offsets=dense_offsets,
                                )
                            else:
                                ctx_scales = predictions["gaussian_scales"].float()[:, ctx_loss_local].permute(0, 1, 3, 4, 2).reshape(-1, 3)
                                ctx_means = predictions["points"].float()[:, ctx_loss_local].reshape(-1, 3)
                                offs = predictions.get("gaussian_offsets")
                                offs_t = None
                                if offs is not None:
                                    offs_t = offs.float()[:, ctx_loss_local].permute(0, 1, 3, 4, 2).reshape(-1, 3)
                                recon_loss, recon_logs = loss_fn(
                                    pred_rgb, target_rgb,
                                    scales=ctx_scales,
                                    means=ctx_means,
                                    offsets=offs_t,
                                )

                            nvs_loss = torch.tensor(0.0, device=device)
                            nvs_logs = {"total": nvs_loss.detach()}
                            nvs_pred_rgb = None
                            if holdout_local_t and not direct_dense_only:
                                if use_merge_this_epoch:
                                    nvs_rendered = gs_renderer.render_views_from_tensors(
                                        means_h,
                                        quats_h,
                                        scales_h,
                                        opacs_h,
                                        colors_h,
                                        camera_poses,
                                        holdout_local_t,
                                        img_h,
                                        img_w,
                                        K_gpu,
                                        checkpoint_chunk_size=args.train_render_checkpoint_chunk_size,
                                    )
                                    nvs_pred_rgb = nvs_rendered["rgb"].permute(0, 3, 1, 2)
                                else:
                                    nvs_pred_rgb = render_holdout_views(
                                        gs_renderer, predictions, ctx_local, holdout_local_t,
                                        K_gpu, img_h, img_w,
                                    )
                                if args.nvs_loss_weight > 0:
                                    nvs_tgt_rgb = images_gpu[holdout_local_t]
                                    nvs_loss, nvs_logs = nvs_loss_fn(nvs_pred_rgb, nvs_tgt_rgb)

                            loss = recon_loss + args.nvs_loss_weight * nvs_loss
                            logs = {
                                "total": loss.detach(),
                                "recon": recon_logs["total"],
                                "nvs": nvs_logs["total"],
                            }
                            if is_neural_merge:
                                logs["dense_recon"] = dense_recon_loss.detach()
                                logs["merge_recon"] = merge_recon_loss.detach()
                            loss_bad = not bool(torch.isfinite(loss).item())
                            if _is_ddp():
                                loss_flag = torch.tensor(
                                    [1 if loss_bad else 0],
                                    device=device,
                                    dtype=torch.int32,
                                )
                                dist.all_reduce(loss_flag, op=dist.ReduceOp.MAX)
                                loss_bad = bool(loss_flag.item())
                            if loss_bad:
                                epoch_nonfinite_losses += 1
                                if _rank() == 0:
                                    print(f"Skip (non-finite loss): {scene_path}")
                                _ddp_noop_step(model, optimizer, sync_backward=True)
                                bucket_scheduler.record_failure()
                                continue

                            if merge_stats is not None and scene_i % 10 == 0 and _rank() == 0:
                                _log(
                                    f"  merge stats: hybrid={merge_stats.get('num_gaussians_hybrid', 0)} "
                                    f"compact={merge_stats.get('num_gaussians_compact', 0)} "
                                    f"complement={merge_stats.get('num_gaussians_complement', 0)}"
                                )

                        (loss / args.gradient_accumulation_steps).backward()
                        trainable = [p for p in model.parameters() if p.requires_grad]
                        grad_bad = not _grads_are_finite(model)
                        if _is_ddp():
                            grad_flag = torch.tensor(
                                [1 if grad_bad else 0], device=device, dtype=torch.int32
                            )
                            dist.all_reduce(grad_flag, op=dist.ReduceOp.MAX)
                            grad_bad = bool(grad_flag.item())
                        if grad_bad:
                            epoch_nonfinite_grads += 1
                            if _rank() == 0:
                                bad_names = _nonfinite_grad_names(model)
                                print(
                                    f"Skip (non-finite grad): {scene_path}"
                                    f"; params={bad_names[:8]}"
                                )
                                if args.diagnose_nonfinite_grads:
                                    print(
                                        "  non-finite parameter groups: "
                                        f"{nonfinite_grad_groups(model)}"
                                    )
                                    for diagnostic in grad_probe_diagnostics(grad_probes):
                                        print(f"  grad probe {diagnostic}")
                            _ddp_noop_step(model, optimizer, sync_backward=False)
                            bucket_scheduler.record_failure()
                            continue
                        epoch_losses.append(logs['total'].item())
                        epoch_recon_losses.append(logs['recon'].item())
                        if holdout_local_t and args.nvs_loss_weight > 0:
                            epoch_nvs_losses.append(logs['nvs'].item())
                        accumulated_scenes += 1
                        if accumulated_scenes < args.gradient_accumulation_steps:
                            continue
                        accumulated_scenes = 0
                        if trainable and args.grad_clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(
                                trainable, max_norm=args.grad_clip_norm
                            )
                        if trainable:
                            optimizer.step()
                            if scheduler is not None:
                                scheduler.step()
                        epoch_successful_steps += 1
                        global_step += 1
                        bucket_scheduler.record_success()
                        if global_step == 1 or global_step % 10 == 0:
                            _log(f"  Optimizer update {global_step}/{target_opt_steps}: "
                                 f"loss={sum(epoch_losses[-args.gradient_accumulation_steps:]) / args.gradient_accumulation_steps:.6f}")

                        if (args.output_folder and args.save_every_steps > 0
                                and global_step % args.save_every_steps == 0
                                and _rank() == 0):
                            step_state = raw_model.state_dict() if _is_ddp() else model.state_dict()
                            step_state = {
                                k: v for k, v in step_state.items()
                                if k.startswith("gaussian_decoder")
                                or k.startswith("gaussian_head")
                                or k.startswith("neural_track_merger")
                                or k.startswith("neural_track_set_merger")
                                or k.startswith("neural_track_feature_merger")
                                or k.startswith("neural_track_direct_slot_merger")
                            }
                            checkpoint = {
                                "model": step_state,
                                "optimizer": optimizer.state_dict(),
                                "epoch": epoch,
                                "global_step": global_step,
                                "scene_cursor": scene_cursor,
                                "world_size": current_world_size,
                                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                                "bucket_recipe": bucket_recipe,
                                "bucket_scheduler": bucket_scheduler.state_dict(),
                                "train_seed": int(args.train_seed),
                            }
                            checkpoint = attach_rng_to_checkpoint(
                                checkpoint,
                                include_cuda=torch.cuda.is_available(),
                            )
                            latest_path = os.path.join(
                                args.output_folder, "gaussian_head_step_latest.pt"
                            )
                            fd, temporary_path = tempfile.mkstemp(
                                prefix=".gaussian_head_step_",
                                suffix=".tmp",
                                dir=args.output_folder,
                            )
                            os.close(fd)
                            try:
                                torch.save(checkpoint, temporary_path)
                                os.replace(temporary_path, latest_path)
                            finally:
                                if os.path.exists(temporary_path):
                                    os.unlink(temporary_path)

                            persistent_interval = args.save_persistent_every_steps
                            if (persistent_interval > 0
                                    and global_step % persistent_interval == 0):
                                persistent_path = os.path.join(
                                    args.output_folder,
                                    f"gaussian_head_step{global_step:07d}.pt",
                                )
                                link_path = persistent_path + ".tmp"
                                try:
                                    if os.path.exists(link_path):
                                        os.unlink(link_path)
                                    os.link(latest_path, link_path)
                                    os.replace(link_path, persistent_path)
                                finally:
                                    if os.path.exists(link_path):
                                        os.unlink(link_path)
                        if args.long_sequence_protocol and torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        # Track gaussian parameter stats (every 10th scene)
                        if scene_i % 10 == 0 and not use_merge_this_epoch:
                            s = predictions["gaussian_scales"].float()  # (1, T, 3, H, W)
                            o = predictions["gaussian_opacities"].float()
                            # Use median and percentiles (robust to outliers from exp)
                            s_flat = s.reshape(-1)
                            o_flat = o.reshape(-1)
                            gs_stats = {
                                "scale_p50": s_flat.quantile(0.50).item(),
                                "scale_p95": s_flat.quantile(0.95).item(),
                                "scale_p05": s_flat.quantile(0.05).item(),
                                "opacity_p50": o_flat.quantile(0.50).item(),
                                "opacity_p95": o_flat.quantile(0.95).item(),
                            }
                            # Point cloud nearest-neighbor distances (sample 1000 points)
                            pts = predictions["points"].float().reshape(-1, 3)  # (M, 3)
                            if pts.shape[0] > 1000:
                                idx = torch.randperm(pts.shape[0], device=pts.device)[:1000]
                                pts = pts[idx]
                            nn_dists = torch.cdist(pts, pts).sort(dim=1).values[:, 1]  # exclude self
                            gs_stats["pts_nn_p50"] = nn_dists.median().item()
                            gs_stats["pts_nn_p95"] = nn_dists.quantile(0.95).item()

                            if "gaussian_offsets" in predictions and predictions["gaussian_offsets"] is not None:
                                off = predictions["gaussian_offsets"].float().reshape(-1, 3).norm(dim=1)
                                gs_stats["offset_p50"] = off.quantile(0.50).item()
                            epoch_gs_stats.append(gs_stats)

                        # Progress (every 100 scenes for ~8000-scene epochs)
                        if (scene_i + 1) % 100 == 0:
                            nvs_str = f" nvs {logs['nvs'].item():.4f}" if holdout_local_t and args.nvs_loss_weight > 0 else ""
                            _log(f"  epoch {epoch:4d} [{scene_i+1:4d}/{len(epoch_scenes)}] | "
                                 f"loss {logs['total'].item():.4f} | "
                                 f"recon {logs['recon'].item():.4f}{nvs_str}")

                        # --- Visualization (step-based) ---
                        if (args.vis_train and args.output_folder
                                and global_step % args.vis_interval == 0
                                and _rank() == 0):
                            vis_dir = os.path.join(args.output_folder, "train_vis", f"step_{global_step:06d}")
                            recon_dir = os.path.join(vis_dir, "recon")
                            nvs_dir = os.path.join(vis_dir, "nvs")
                            os.makedirs(recon_dir, exist_ok=True)
                            from torchvision.utils import save_image
                            rgb_nhwc = rendered["rgb"].detach().cpu().float()
                            for t, global_idx in enumerate(ctx_loss_global):
                                img = rgb_nhwc[t].permute(2, 0, 1)
                                save_image(img, os.path.join(recon_dir, f"frame_{global_idx:04d}_pred.png"))
                                gt = target_rgb[t].cpu().float()
                                save_image(gt, os.path.join(recon_dir, f"frame_{global_idx:04d}_gt.png"))
                            if nvs_pred_rgb is not None:
                                os.makedirs(nvs_dir, exist_ok=True)
                                for t, global_idx in enumerate(holdout_global):
                                    img = nvs_pred_rgb[t].detach().cpu().float()
                                    save_image(img, os.path.join(nvs_dir, f"frame_{global_idx:04d}_pred.png"))
                                    gt = images_gpu[holdout_local_t[t]].cpu().float()
                                    save_image(gt, os.path.join(nvs_dir, f"frame_{global_idx:04d}_gt.png"))

                    avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
                    avg_recon = sum(epoch_recon_losses) / max(len(epoch_recon_losses), 1)
                    avg_nvs = sum(epoch_nvs_losses) / max(len(epoch_nvs_losses), 1) if epoch_nvs_losses else 0.0
                    _log(f"  epoch {epoch:4d}/{args.epochs} | "
                         f"steps {global_step} | "
                         f"avg_loss {avg_loss:.4f} | "
                         f"avg_recon {avg_recon:.4f} | "
                         f"avg_nvs {avg_nvs:.4f}")
                    skip_rate = (
                        (epoch_nonfinite_losses + epoch_nonfinite_grads)
                        / max(epoch_attempted_steps, 1)
                    )
                    _log(
                        f"         updates {epoch_successful_steps}/{epoch_attempted_steps} | "
                        f"nonfinite_loss {epoch_nonfinite_losses} | "
                        f"nonfinite_grad {epoch_nonfinite_grads} | "
                        f"nonfinite_skip_rate {skip_rate:.2%}"
                    )
                    if args.fail_if_zero_updates and epoch_successful_steps == 0:
                        raise RuntimeError("training epoch completed with zero optimizer updates")
                    if args.fail_if_zero_updates and global_step < target_opt_steps:
                        raise RuntimeError(f"Incomplete training: {global_step}/{target_opt_steps} optimizer steps")
                    if epoch_gs_stats:
                        def _avg(key):
                            vals = [s[key] for s in epoch_gs_stats if key in s]
                            return sum(vals) / max(len(vals), 1)
                        _log(f"         gaussian params (median over {len(epoch_gs_stats)} samples):")
                        _log(f"           scale   p50={_avg('scale_p50'):.4f}  "
                             f"p05={_avg('scale_p05'):.4f}  p95={_avg('scale_p95'):.4f}")
                        _log(f"           opacity p50={_avg('opacity_p50'):.4f}  "
                             f"p95={_avg('opacity_p95'):.4f}")
                        _log(f"           pts nn  p50={_avg('pts_nn_p50'):.4f}  "
                             f"p95={_avg('pts_nn_p95'):.4f}")
                        if "offset_p50" in epoch_gs_stats[0]:
                            _log(f"           offset  p50={_avg('offset_p50'):.6f}")

                    # Save checkpoint each epoch
                    if args.output_folder and _rank() == 0:
                        model_state = raw_model.state_dict() if _is_ddp() else model.state_dict()
                        gaussian_state = {k: v for k, v in model_state.items()
                                          if k.startswith("gaussian_decoder")
                                          or k.startswith("gaussian_head")
                                          or k.startswith("neural_track_merger")
                                          or k.startswith("neural_track_set_merger")
                                          or k.startswith("neural_track_feature_merger")
                                          or k.startswith("neural_track_direct_slot_merger")}
                        ckpt = {
                            "model": gaussian_state,
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch,
                            "global_step": global_step,
                            "scene_cursor": scene_cursor,
                            "world_size": current_world_size,
                            "gradient_accumulation_steps": args.gradient_accumulation_steps,
                            "curriculum_start_epoch": curriculum_start_epoch,
                            "train_seed": int(args.train_seed),
                        }
                        ckpt = attach_rng_to_checkpoint(
                            ckpt, include_cuda=torch.cuda.is_available()
                        )
                        ckpt_path = os.path.join(args.output_folder, f"gaussian_head_epoch{epoch:04d}.pt")
                        torch.save(ckpt, ckpt_path)
                    if _is_ddp():
                        dist.barrier()

                # Save final weights
                if args.output_folder and _rank() == 0:
                    model_state = raw_model.state_dict() if _is_ddp() else model.state_dict()
                    gaussian_state = {k: v for k, v in model_state.items()
                                      if k.startswith("gaussian_decoder")
                                      or k.startswith("gaussian_head")
                                      or k.startswith("neural_track_merger")
                                      or k.startswith("neural_track_set_merger")
                                      or k.startswith("neural_track_feature_merger")
                                      or k.startswith("neural_track_direct_slot_merger")}
                    torch.save({
                                   "model": gaussian_state,
                                   "epoch": epoch,
                                   "global_step": global_step,
                                   "world_size": current_world_size,
                                   "gradient_accumulation_steps": args.gradient_accumulation_steps,
                                   "curriculum_start_epoch": curriculum_start_epoch,
                               },
                               os.path.join(args.output_folder, "gaussian_head.pt"))
                    print(f"Saved gaussian head weights to {args.output_folder}")

                # Sync all ranks before post-training tasks
                if _is_ddp():
                    torch.cuda.synchronize()
                    dist.barrier()

                # Final render on a validation scene (all ranks, save only on rank 0)
                if args.render_gaussian and val_scenes:
                    val_scene = val_scenes[0]  # deterministic — same scene across ranks
                    split_result = try_load_dl3dv_scene_split(
                        val_scene, train_samples, args.frame_gap_max, args.nvs_holdout,
                        target_resolution, args.image_scale,
                    )
                    if split_result is not None:
                        (images_test, _, val_K, ctx_local, holdout_local,
                         ctx_global, holdout_global) = split_result
                        K_gpu = val_K.to(device)
                        images_gpu = images_test.to(device)
                        img_h, img_w = images_gpu.shape[2], images_gpu.shape[3]
                        with torch.no_grad():
                            predictions = model(
                                images_gpu[None], gaussian=True,
                                **model_sliding_window_kwargs(
                                    images_gpu.shape[0],
                                    min(args.window_size, images_gpu.shape[0]),
                                    args.overlap_size,
                                    sim3=args.sim3,
                                    se3=not args.no_se3,
                                    turn_off_ttt=args.no_ttt,
                                    turn_off_swa=args.no_swa,
                                ),
                            )
                            rendered = gs_renderer.render_sequence(
                                predictions, target_indices=ctx_local,
                                neighbor_frames=args.neighbor_frames, K=K_gpu,
                            )
                            nvs_pred_rgb = render_holdout_views(
                                gs_renderer, predictions, ctx_local, holdout_local,
                                K_gpu, img_h, img_w,
                            )
                        if args.save_rendered and args.output_folder and _rank() == 0:
                            from torchvision.utils import save_image
                            recon_dir = os.path.join(args.output_folder, "rendered_val", "recon")
                            nvs_dir = os.path.join(args.output_folder, "rendered_val", "nvs")
                            os.makedirs(recon_dir, exist_ok=True)
                            rgb_nhwc = rendered['rgb'].cpu().float()
                            ctx_local_t = torch.tensor(ctx_local, device=device, dtype=torch.long)
                            gt_ctx = images_gpu[ctx_local_t]
                            for t, global_idx in enumerate(ctx_global):
                                img = rgb_nhwc[t].permute(2, 0, 1)
                                save_image(img, os.path.join(recon_dir, f"frame_{global_idx:04d}_pred.png"))
                                save_image(gt_ctx[t].cpu().float(),
                                           os.path.join(recon_dir, f"frame_{global_idx:04d}_gt.png"))
                            if nvs_pred_rgb is not None:
                                os.makedirs(nvs_dir, exist_ok=True)
                                for t, global_idx in enumerate(holdout_global):
                                    save_image(
                                        nvs_pred_rgb[t].cpu().float(),
                                        os.path.join(nvs_dir, f"frame_{global_idx:04d}_pred.png"),
                                    )
                                    save_image(
                                        images_gpu[holdout_local[t]].cpu().float(),
                                        os.path.join(nvs_dir, f"frame_{global_idx:04d}_gt.png"),
                                    )
                            _log(f"  Saved {len(ctx_global)} recon + {len(holdout_global)} nvs "
                                 f"pred/gt pairs under rendered_val/")

                if _is_ddp():
                    dist.barrier()
                cleanup_ddp()
                return  # dataset training done

            # --- Single-scene training mode ---
            images_gpu = images_tensor.to(device)
            print(f"Starting training: {args.epochs} epochs, lr={args.lr}, "
                  f"frames={num_frames}, train_samples={train_samples}")

            for epoch in range(args.epochs):
                optimizer.zero_grad()

                # Random subset of frames for this step
                indices = torch.randperm(num_frames)[:train_samples].sort().values
                images_subset = images_gpu[indices]  # (K, 3, H, W)

                # Forward: predict gaussians (no autocast — gsplat needs fp32)
                with torch.cuda.amp.autocast(enabled=False):
                    predictions = model(
                        images_subset[None], gaussian=True,
                        **model_sliding_window_kwargs(
                            train_samples,
                            min(args.window_size, train_samples),
                            min(args.overlap_size, max(train_samples // 10, 1)),
                            sim3=args.sim3,
                            se3=not args.no_se3,
                            turn_off_ttt=args.no_ttt,
                            turn_off_swa=args.no_swa,
                        ),
                    )

                    # Render only the sampled frames
                    rendered = gs_renderer.render_sequence(predictions, neighbor_frames=args.neighbor_frames, K=K_gpu)

                    # Loss: predicted vs original images
                    pred_rgb = rendered["rgb"].permute(0, 3, 1, 2)       # (K, 3, H, W)
                    target_rgb = images_subset                            # (K, 3, H, W)

                    loss, logs = loss_fn(pred_rgb, target_rgb,
                                          scales=predictions["gaussian_scales"].float().permute(0, 1, 3, 4, 2).reshape(-1, 3),
                                          means=predictions["points"].float().reshape(-1, 3))

                loss.backward()
                optimizer.step()

                # --- Training visualization ---
                if args.vis_train and args.output_folder and (
                    epoch % args.vis_interval == 0 or epoch == args.epochs - 1
                ):
                    vis_dir = os.path.join(args.output_folder, "train_vis", f"epoch_{epoch:04d}")
                    os.makedirs(vis_dir, exist_ok=True)
                    from torchvision.utils import save_image
                    rgb_nhwc = rendered["rgb"].detach().cpu().float()
                    for t, idx in enumerate(indices.tolist()):
                        # Save rendered
                        img = rgb_nhwc[t].permute(2, 0, 1)
                        save_image(img, os.path.join(vis_dir, f"frame_{idx:06d}_pred.png"))
                        # Save ground truth
                        gt = target_rgb[t].cpu().float()
                        save_image(gt, os.path.join(vis_dir, f"frame_{idx:06d}_gt.png"))

                if epoch % 10 == 0 or epoch == args.epochs - 1:
                    ssim_str = ""
                    if "ssim" in logs:
                        ssim_str = f" | ssim {logs['ssim'].item():.4f}"
                    print(f"  epoch {epoch:4d}/{args.epochs} | "
                          f"loss {logs['total'].item():.4f} | "
                          f"mse {logs['mse'].item():.4f} | "
                          f"lpips {logs['lpips'].item():.4f}{ssim_str}")

            # Save trained weights
            if args.output_folder:
                os.makedirs(args.output_folder, exist_ok=True)
                ckpt_path = os.path.join(args.output_folder, "gaussian_head.pt")
                gaussian_state = {k: v for k, v in model.state_dict().items()
                                  if k.startswith("gaussian_decoder")
                                  or k.startswith("gaussian_head")
                                  or k.startswith("neural_track_merger")
                                  or k.startswith("neural_track_set_merger")
                                  or k.startswith("neural_track_feature_merger")
                                  or k.startswith("neural_track_direct_slot_merger")}
                torch.save(gaussian_state, ckpt_path)
                print(f"Saved gaussian head weights to {ckpt_path}")

            # Run a final render + save if requested
            if args.render_gaussian:
                print("Final render after training...")
                with torch.no_grad():
                    predictions = model(
                        images_gpu[None], gaussian=True,
                        **model_sliding_window_kwargs(
                            num_frames,
                            min(args.window_size, num_frames),
                            min(args.overlap_size, max(num_frames // 10, 1)),
                            sim3=args.sim3,
                            se3=not args.no_se3,
                            turn_off_ttt=args.no_ttt,
                            turn_off_swa=args.no_swa,
                        ),
                    )
                    rendered = gs_renderer.render_sequence(predictions, neighbor_frames=args.neighbor_frames, K=K_gpu)
                    # Store for the save path below
                    predictions['rendered_rgb'] = rendered['rgb'].unsqueeze(0)
                    predictions['rendered_depth'] = rendered['depth'].unsqueeze(0)

                if args.save_rendered and args.output_folder:
                    rendered_dir = os.path.join(args.output_folder, "rendered")
                    os.makedirs(rendered_dir, exist_ok=True)
                    from torchvision.utils import save_image
                    rgb_nhwc = rendered['rgb'].cpu().float()
                    for t in range(rgb_nhwc.shape[0]):
                        img = rgb_nhwc[t].permute(2, 0, 1)
                        save_image(img, os.path.join(rendered_dir, f"frame_{t:06d}.png"))
                    print(f"  Saved {rgb_nhwc.shape[0]} rendered frames to {rendered_dir}")

            return  # training done — skip inference path below
        
        forward_kwargs = {}
        if args.gaussian or args.render_gaussian:
            forward_kwargs['gaussian'] = True
        if args.config:
            try:
                with open(args.config, 'r') as f:
                    config = yaml.safe_load(f)

                training_settings = config.get('training_settings', {})
                model_settings = config.get('model', {})
                se3_from_config = model_settings.get('se3', config.get('se3', False))
                se3_value = args.se3 if args.se3 is not None else bool(se3_from_config)
                forward_kwargs.update({
                    'window_size': args.window_size if args.window_size is not None else training_settings.get('window_size', -1),
                    'overlap_size': args.overlap_size if args.overlap_size is not None else training_settings.get('overlap_size', 0),
                    'reset_every': args.reset_every if args.reset_every is not None else training_settings.get('reset_every', 0),
                    'num_iterations': config.get('num_iterations', 1), # Or from training_settings
                    'sim3': config.get('sim3', False) or args.sim3,
                    'sim3_scale_mode': args.sim3_scale_mode,
                    'se3': se3_value,
                    'turn_off_ttt': args.no_ttt,
                    'turn_off_swa': args.no_swa,
                })
                print(f"Forward pass kwargs from config: {forward_kwargs}")

            except Exception as e:
                print(f"Could not read config for forward pass arguments: {e}")
        elif args.window_size or args.overlap_size or args.sim3 or args.reset_every is not None:
            forward_kwargs.update({
                'window_size': args.window_size,
                'overlap_size': args.overlap_size,
                'window_size': args.window_size,
                'overlap_size': args.overlap_size,
                'sim3': args.sim3,
                'pi3x': args.pi3x,
                'pi3x_metric': args.pi3x_metric,
                'se3': bool(args.se3) if args.se3 is not None else False,
                'sim3_scale_mode': args.sim3_scale_mode,
                'reset_every': args.reset_every if args.reset_every is not None else 0
            })

        # Warmup run to trigger torch.compile (first run has compilation overhead)
        if args.warmup or args.benchmark:
            print("Running warmup inference (to trigger torch.compile)...")
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=dtype):
                _ = model(images_tensor[None], **forward_kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print("Warmup complete.")

        # Benchmark mode: run multiple times and report statistics
        if args.benchmark:
            num_runs = 3
            print(f"\nRunning benchmark with {num_runs} inference passes...")
            inference_times = []
            for run_idx in range(num_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_start = time.time()
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=dtype):
                    raw_model_predictions = model(images_tensor[None], **forward_kwargs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_end = time.time()
                inference_times.append(t_end - t_start)
                print(f"  Run {run_idx + 1}/{num_runs}: {t_end - t_start:.3f}s")
            
            avg_time = sum(inference_times) / len(inference_times)
            min_time = min(inference_times)
            max_time = max(inference_times)
            std_time = (sum((t - avg_time) ** 2 for t in inference_times) / len(inference_times)) ** 0.5
            
            print(f"\n{'='*50}")
            print(f"Benchmark Results ({num_runs} runs):")
            print(f"  Total frames: {num_frames}")
            print(f"  Avg inference time: {avg_time:.3f}s (std: {std_time:.3f}s)")
            print(f"  Min/Max: {min_time:.3f}s / {max_time:.3f}s")
            print(f"  Avg FPS: {num_frames / avg_time:.2f}")
            print(f"  Avg time per frame: {(avg_time / num_frames) * 1000:.2f} ms")
            print(f"{'='*50}\n")
            inference_time = avg_time
        else:
            # Single timed inference
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_start_time = time.time()
            
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=dtype):
                raw_model_predictions = model(images_tensor[None], **forward_kwargs) # Add batch dimension

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_end_time = time.time()
            
            # Calculate and display timing
            inference_time = inference_end_time - inference_start_time
            fps = num_frames / inference_time
            ms_per_frame = (inference_time / num_frames) * 1000
            print(f"\n{'='*50}")
            print(f"Inference Timing Results:")
            print(f"  Total frames: {num_frames}")
            print(f"  Inference time: {inference_time:.3f} seconds")
            print(f"  FPS: {fps:.2f}")
            print(f"  Time per frame: {ms_per_frame:.2f} ms")
            if not args.warmup:
                print(f"  (Note: First run includes torch.compile overhead. Use --warmup for accurate timing)")
            print(f"{'='*50}\n")

        # Post-process predictions
        # Using permute to get (B, S, H, W, C) for easier numpy conversion later
        raw_model_predictions['images'] = images_tensor[None].permute(0, 1, 3, 4, 2)
        raw_model_predictions['conf'] = torch.sigmoid(raw_model_predictions['conf'])
        # Edge mask on depth can be noisy, optional
        # edge = depth_edge(raw_model_predictions['local_points'][..., 2], rtol=0.03)
        # raw_model_predictions['conf'][edge] = 0.0
        if 'local_points' in raw_model_predictions:
            del raw_model_predictions['local_points']

        # --- Gaussian rendering ---
        if args.render_gaussian and 'gaussian_scales' in raw_model_predictions:
            print("Rendering gaussian splats to images...")
            from loger.renderer import GaussianRenderer
            gs_renderer = GaussianRenderer(sh_degree=2)
            try:
                with torch.no_grad():
                    # Sync CUDA to flush any pending errors from inference
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    # CPU round-trip to fully detach tensors from model's CUDA state
                    gs_pred = {
                        'points': raw_model_predictions['points'].cpu().cuda(),
                        'gaussian_scales': raw_model_predictions['gaussian_scales'].cpu().cuda(),
                        'gaussian_rotations': raw_model_predictions['gaussian_rotations'].cpu().cuda(),
                        'gaussian_opacities': raw_model_predictions['gaussian_opacities'].cpu().cuda(),
                        'gaussian_sh_coeffs': raw_model_predictions['gaussian_sh_coeffs'].cpu().cuda(),
                        'camera_poses': raw_model_predictions['camera_poses'].cpu().cuda(),
                    }
                    rendered = gs_renderer.render_sequence(gs_pred, neighbor_frames=args.neighbor_frames)
                # Store rendered images in the predictions
                raw_model_predictions['rendered_rgb'] = rendered['rgb'].unsqueeze(0)    # (1, T, H, W, 3)
                raw_model_predictions['rendered_depth'] = rendered['depth'].unsqueeze(0)
                print(f"  Rendered {rendered['rgb'].shape[0]} frames.")

                # Save rendered images to disk if requested
                if args.save_rendered and args.output_folder:
                    rendered_dir = os.path.join(args.output_folder, "rendered")
                    os.makedirs(rendered_dir, exist_ok=True)
                    from torchvision.utils import save_image
                    rgb_nhwc = rendered['rgb'].cpu().float()  # (T, H, W, 3)
                    for t in range(rgb_nhwc.shape[0]):
                        img = rgb_nhwc[t].permute(2, 0, 1)  # (3, H, W)
                        path = os.path.join(rendered_dir, f"frame_{t:06d}.png")
                        save_image(img, path)
                    print(f"  Saved {rendered['rgb'].shape[0]} rendered frames to {rendered_dir}")
            except Exception as e:
                import traceback
                print(f"  Gaussian rendering failed: {e}")
                traceback.print_exc()

        # Convert all tensors to numpy and remove batch dimension
        # Filter out non-tensor values (e.g., window_ttt_losses which is a list)
        predictions_dict = {k: v.squeeze(0).cpu().float().numpy() 
                           for k, v in raw_model_predictions.items() 
                           if v is not None and torch.is_tensor(v)}

        if args.output_folder:
            os.makedirs(args.output_folder, exist_ok=True)
            seq_name_to_use = f"{args.seq_name}_{str(args.start_frame)}_{str(args.end_frame)}_{str(args.stride)}"
            
            # Count number of inputs processed
            input_paths = [p for p in [args.input, args.input2, args.input3, args.input4, args.input5] if p is not None]
            num_inputs_processed = len(input_paths)
            if num_inputs_processed > 1:
                seq_name_to_use += f"_x{num_inputs_processed}"
            
            output_filename = f"{seq_name_to_use}.pt" if seq_name_to_use else "predictions.pt"
            output_path = os.path.join(args.output_folder, output_filename)
            print(f"Saving inference results to {output_path}...")
            try:
                # Save the numpy dict. For consistency, can convert back to tensors for saving.
                torch.save({k: torch.from_numpy(v) for k,v in predictions_dict.items()}, output_path)
                print("Successfully saved inference results.")
            except Exception as e:
                print(f"Error saving results to {output_path}: {e}")

    if args.output_txt and predictions_dict is not None and "camera_poses" in predictions_dict:
        print(f"Saving trajectory to {args.output_txt}...")
        try:
            # 1) Prepare timestamps
            # We use the first input path to try finding timestamps
            input_path_for_ts = Path(args.input)
            # If we have multiple inputs, we might need to be careful, but usually we evaluate on the first sequence or combined.
            # Here we use all_image_names_collected which corresponds to the inference frames.
            # Note: all_image_names_collected might be temp paths if we copied them.
            # If we copied them, we lost the original path connection for timestamp lookup if we rely on temp dir.
            # However, _try_load_timestamps_for_images uses input_rgb_dir to find rgb.txt.
            # If we pass the original input path as input_rgb_dir, it might work if filenames match.
            # But filenames in temp dir are frame_xxxxxx.png.
            # So we should probably use the original filenames if possible, or just fallback to indices if using temp dir.
            
            # If we used temp dir (which we did for combined inputs), filenames are frame_000000.png.
            # This breaks mapping to rgb.txt which uses original filenames.
            # So for now, if we used temp dir, we might have to fallback to indices unless we tracked original names.
            # In run_core_inference or main, we didn't track original names in a way that maps easily back for rgb.txt lookup 
            # unless we parse them.
            
            # However, if args.input is a directory and we are just processing it (and maybe others), 
            # and if we want to evaluate, we usually care about the timestamps of the frames we processed.
            
            # Let's try to use indices as timestamps if we can't easily map back, 
            # OR if the user provided a single input folder, we can try to be smarter.
            
            timestamps = None
            if len(input_paths) == 1 and os.path.isdir(args.input) and not is_video_file(args.input):
                 # If single folder input, we can try to load timestamps using the original filenames
                 # Re-glob to get original paths
                 # import glob # Already imported globally
                 # from natsort import natsorted # Already imported globally
                 current_frames = natsorted(glob.glob(os.path.join(args.input, "*.png"))+glob.glob(os.path.join(args.input, "*.jpg"))+glob.glob(os.path.join(args.input, "*.jpeg")))
                 current_frames = [f for f in current_frames if "depth" not in os.path.basename(f).lower()]
                 end_idx = args.end_frame if args.end_frame != -1 else None
                 current_frames = current_frames[args.start_frame:end_idx:args.stride]
                 
                 timestamps = _try_load_timestamps_for_images(current_frames, Path(args.input))
            else:
                 # Fallback to indices
                 timestamps = [float(i) for i in range(len(all_image_names_collected))]

            # 2) Extract poses
            # predictions_dict['camera_poses'] is (N, 4, 4) numpy array
            camera_poses = torch.from_numpy(predictions_dict['camera_poses'])
            
            # Pi3 outputs Twc (Camera to World) directly
            Twc = camera_poses
            Rwc = Twc[..., :3, :3]
            twc = Twc[..., :3, 3]
            
            qwc = mat_to_quat(Rwc) # XYZW
            
            # 3) Write
            # Ensure lengths match
            S = min(len(timestamps), twc.shape[0], qwc.shape[0])
            write_trajectory_txt(Path(args.output_txt), timestamps[:S], twc[:S].tolist(), qwc[:S].tolist())
            print(f"Successfully saved trajectory to {args.output_txt}")
        except Exception as e:
            print(f"Error saving trajectory to {args.output_txt}: {e}")

    if predictions_dict is None:
        print("Error: Predictions are not available. Exiting.")
        for temp_dir_path in temp_frame_dirs.values():
            if os.path.exists(temp_dir_path): shutil.rmtree(temp_dir_path)
        cleanup_ddp()
        return

    # Clean up temporary directories
    for temp_dir_path in temp_frame_dirs.values():
        if os.path.exists(temp_dir_path):
            print(f"Cleaning up temporary directory: {temp_dir_path}")
            shutil.rmtree(temp_dir_path)

    print("Done.")
    cleanup_ddp()

if __name__ == "__main__":
    main()
