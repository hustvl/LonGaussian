"""Apply Sim(3)/SE(3) alignment transforms to Gaussian prediction tensors."""

from __future__ import annotations

from typing import Optional

import torch

from loger.utils.rotation import mat_to_quat, quat_to_mat, standardize_quaternion


def _wxyz_to_xyzw(q_wxyz: torch.Tensor) -> torch.Tensor:
    return q_wxyz[..., [1, 2, 3, 0]]


def _xyzw_to_wxyz(q_xyzw: torch.Tensor) -> torch.Tensor:
    return q_xyzw[..., [3, 0, 1, 2]]


def transform_spatial_vectors(
    vectors: torch.Tensor,
    rot: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Transform 3-vectors stored as (B, N, 3, H, W)."""
    v = vectors.permute(0, 1, 3, 4, 2)
    rotated = torch.einsum("bij, bnhwj -> bnhwi", rot, v)
    if scale is not None:
        rotated = rotated * scale.view(scale.shape[0], 1, 1, 1, 1)
    return rotated.permute(0, 1, 4, 2, 3)


def transform_spatial_points(
    points: torch.Tensor,
    rot: torch.Tensor,
    trans: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Transform 3D points stored as (B, N, H, W, 3)."""
    rotated = torch.einsum("bij, bnhwj -> bnhwi", rot, points)
    if scale is not None:
        rotated = rotated * scale.view(scale.shape[0], 1, 1, 1, 1)
    return rotated + trans.view(trans.shape[0], 1, 1, 1, 3)


def transform_gaussian_rotations(
    rotations: torch.Tensor,
    rot_align: torch.Tensor,
) -> torch.Tensor:
    """Compose a global alignment rotation with per-pixel Gaussian orientations.

    Args:
        rotations: (B, N, 4, H, W) quaternions in gsplat wxyz convention.
        rot_align: (B, 3, 3) alignment rotation applied to the whole window.

    Returns:
        Transformed rotations with the same shape and wxyz convention.
    """
    batch_size, num_frames, _, height, width = rotations.shape
    num_pixels = num_frames * height * width

    rots_wxyz = rotations.permute(0, 1, 3, 4, 2).reshape(batch_size, num_pixels, 4)
    rots_xyzw = _wxyz_to_xyzw(rots_wxyz)
    rot_local = quat_to_mat(rots_xyzw)

    rot_align_exp = rot_align.unsqueeze(1).expand(-1, num_pixels, -1, -1)
    rot_global = torch.matmul(rot_align_exp, rot_local)

    quats_xyzw = mat_to_quat(rot_global.reshape(-1, 3, 3)).reshape(batch_size, num_pixels, 4)
    quats_xyzw = standardize_quaternion(quats_xyzw)
    quats_wxyz = _xyzw_to_wxyz(quats_xyzw)

    return quats_wxyz.reshape(batch_size, num_frames, height, width, 4).permute(0, 1, 4, 2, 3)


def apply_sim3_to_gaussian_predictions(
    pred: dict,
    rot: torch.Tensor,
    trans: torch.Tensor,
    scale: torch.Tensor,
    allow_scale: bool,
    aligned_points: Optional[torch.Tensor] = None,
) -> dict:
    """Return Sim(3)/SE(3)-aligned copies of Gaussian tensors from one window prediction.

    SH coefficients and opacities are returned unchanged: gsplat evaluates SH in each
    Gaussian's local frame, so updating orientations is sufficient under rigid alignment.
    """
    aligned: dict = {}
    batch_size = rot.shape[0]
    scale_factor = scale if allow_scale else None

    offsets = pred.get("gaussian_offsets")
    if offsets is not None:
        aligned_offsets = transform_spatial_vectors(offsets, rot, scale_factor)
        aligned["gaussian_offsets"] = aligned_offsets
        if aligned_points is not None:
            aligned["gaussian_means"] = aligned_points + aligned_offsets.permute(0, 1, 3, 4, 2)
    else:
        means = pred.get("gaussian_means")
        if means is not None:
            aligned["gaussian_means"] = transform_spatial_points(
                means, rot, trans, scale_factor
            )
        elif aligned_points is not None:
            aligned["gaussian_means"] = aligned_points

    rotations = pred.get("gaussian_rotations")
    if rotations is not None:
        aligned["gaussian_rotations"] = transform_gaussian_rotations(rotations, rot)

    scales = pred.get("gaussian_scales")
    if scales is not None:
        if allow_scale:
            aligned["gaussian_scales"] = scales * scale.view(batch_size, 1, 1, 1, 1)
        else:
            aligned["gaussian_scales"] = scales

    for key in ("gaussian_opacities", "gaussian_sh_coeffs"):
        if key in pred:
            aligned[key] = pred[key]

    return aligned
