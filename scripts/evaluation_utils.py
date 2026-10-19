#!/usr/bin/env python3
"""Shared pose alignment, fixed scene selection and image metrics."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


def configure_reproducibility(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def fit_se3(
    source_centers: torch.Tensor, target_centers: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    if source_centers.shape != target_centers.shape or source_centers.ndim != 2:
        raise ValueError("SE3 fit expects equally shaped (N,3) camera-center tensors")
    if source_centers.shape[0] < 3:
        raise ValueError("Two-pass pose alignment needs at least three context views")
    src = source_centers.float()
    dst = target_centers.float()
    src_mean, dst_mean = src.mean(dim=0), dst.mean(dim=0)
    covariance = (src - src_mean).transpose(0, 1) @ (dst - dst_mean)
    u, _, vh = torch.linalg.svd(covariance)
    rot = vh.transpose(0, 1) @ u.transpose(0, 1)
    if torch.linalg.det(rot) < 0:
        vh = vh.clone()
        vh[-1] *= -1
        rot = vh.transpose(0, 1) @ u.transpose(0, 1)
    trans = dst_mean - rot @ src_mean
    return rot, trans


def transform_pose(pose: torch.Tensor, rot: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    out = pose.clone()
    out[:3, :3] = rot @ pose[:3, :3]
    out[:3, 3] = rot @ pose[:3, 3] + trans
    return out


def resolve_scene_path(entry: str, dataset_root: Path) -> Path:
    scene_path = Path(entry)
    if not scene_path.is_absolute():
        scene_path = dataset_root / entry
    if (scene_path / "transforms.json").is_file():
        return scene_path
    ns_path = scene_path / "nerfstudio"
    if (ns_path / "transforms.json").is_file():
        return ns_path
    raise FileNotFoundError(f"missing transforms.json under {scene_path}")


def load_scene_list(
    dataset_root: Path, num_scenes: int, scene_offset: int
) -> List[Path]:
    from run import load_dataset_scene_list

    scenes = [resolve_scene_path(p, dataset_root) for p in load_dataset_scene_list(
        "dl3dv", str(dataset_root), "test"
    )]
    if num_scenes > 0:
        scenes = scenes[scene_offset : scene_offset + num_scenes]
    elif scene_offset > 0:
        scenes = scenes[scene_offset:]
    if not scenes:
        raise SystemExit("no scenes selected")
    return scenes


def compute_view_metrics(
    pred: torch.Tensor, gt: torch.Tensor, lpips_fn, win: int
) -> Dict[str, float]:
    from skimage.metrics import structural_similarity as ssim_fn

    pred = pred.clamp(0, 1)
    gt = gt.clamp(0, 1)
    mse = (pred - gt).pow(2).mean().item()
    psnr = float(20 * np.log10(1.0) - 10 * np.log10(max(mse, 1e-10)))
    pred_np = pred.permute(1, 2, 0).detach().cpu().numpy()
    gt_np = gt.permute(1, 2, 0).detach().cpu().numpy()
    ssim = float(
        ssim_fn(pred_np, gt_np, channel_axis=2, data_range=1.0, win_size=win)
    )
    with torch.no_grad():
        lpips = float(
            lpips_fn(
                pred.detach().cpu().unsqueeze(0),
                gt.detach().cpu().unsqueeze(0),
                normalize=True,
            ).item()
        )
    return {"psnr": psnr, "ssim": ssim, "lpips": lpips}


def prepare_merge_inputs(
    images: torch.Tensor,
    ctx_list: Sequence[int],
    two_pass_pose: bool,
) -> Tuple[torch.Tensor, List[int], bool]:
    """Formal two-pass: pass-1 uses packed frames; pass-2 merge is context-only."""
    if not two_pass_pose:
        return images, [int(i) for i in ctx_list], False
    ctx_idx = torch.tensor(list(ctx_list), device=images.device, dtype=torch.long)
    merge_images = images.index_select(0, ctx_idx)
    return merge_images, list(range(len(ctx_list))), True


def ctx_merge_owner_groups(
    ctx_list: Sequence[int],
    window_size: int,
    overlap_size: int,
) -> Dict[int, List[int]]:
    """Owner windows on the context-only merge topology (local ctx indices)."""
    from loger.merge.window_merge import owner_window_for_frame, sliding_window_ranges

    windows = sliding_window_ranges(len(ctx_list), window_size, overlap_size)
    groups: Dict[int, List[int]] = defaultdict(list)
    for local_i, global_f in enumerate(ctx_list):
        owner = owner_window_for_frame(local_i, windows, overlap_size)
        groups[owner].append(int(global_f))
    return dict(groups)


def packed_merge_owner_groups(
    ctx_list: Sequence[int],
    num_frames: int,
    window_size: int,
    overlap_size: int,
) -> Dict[int, List[int]]:
    from loger.merge.window_merge import owner_window_for_frame, sliding_window_ranges

    windows = sliding_window_ranges(num_frames, window_size, overlap_size)
    groups: Dict[int, List[int]] = defaultdict(list)
    for frame in ctx_list:
        owner = owner_window_for_frame(int(frame), windows, overlap_size)
        groups[owner].append(int(frame))
    return dict(groups)


def align_two_pass_poses(
    merged,
    poses_pass1: torch.Tensor,
    ctx_list: Sequence[int],
    *,
    pass2_ctx_isolated: bool = False,
    orientation_aware: bool = False,
):
    poses_pass2 = merged.camera_poses
    if poses_pass2.dim() == 4:
        poses_pass2 = poses_pass2[0]
    ctx_idx = torch.tensor(list(ctx_list), device=poses_pass1.device, dtype=torch.long)
    if pass2_ctx_isolated:
        if poses_pass2.shape[0] != len(ctx_list):
            raise ValueError(
                "ctx-isolated pass2 expected "
                f"{len(ctx_list)} poses, got {poses_pass2.shape[0]}"
            )
        src_centers = poses_pass1[ctx_idx, :3, 3]
        dst_centers = poses_pass2[:, :3, 3]
    else:
        src_centers = poses_pass1[ctx_idx, :3, 3]
        dst_centers = poses_pass2[ctx_idx, :3, 3]
    if orientation_aware:
        from loger.utils.two_pass_pose import fit_orientation_aware_se3

        source_ctx_poses = poses_pass1.index_select(0, ctx_idx)
        target_ctx_poses = poses_pass2 if pass2_ctx_isolated else poses_pass2.index_select(0, ctx_idx)
        rot, trans = fit_orientation_aware_se3(source_ctx_poses, target_ctx_poses)
    else:
        rot, trans = fit_se3(src_centers, dst_centers)
    aligned = torch.stack(
        [transform_pose(poses_pass1[i], rot, trans) for i in range(poses_pass1.shape[0])],
        dim=0,
    )
    merged.camera_poses = aligned.unsqueeze(0)
