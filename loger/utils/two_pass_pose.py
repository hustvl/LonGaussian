"""Pose alignment helpers for two-pass context/target inference."""

from __future__ import annotations

import torch


def fit_orientation_aware_se3(
    source_poses: torch.Tensor,
    target_poses: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit the SE(3) mapping source c2w poses into the target world frame.

    Rotation is the chordal SO(3) mean of ``R_target @ R_source.T``.  This
    remains constrained when camera centers are collinear.  Translation is
    then the least-squares mean residual between aligned camera centers.
    """
    if source_poses.shape != target_poses.shape:
        raise ValueError("source and target poses must have identical shapes")
    if source_poses.ndim != 3 or source_poses.shape[-2:] != (4, 4):
        raise ValueError(
            f"poses must have shape [N,4,4], got {tuple(source_poses.shape)}"
        )
    if source_poses.shape[0] < 1:
        raise ValueError("at least one pose is required")
    if not torch.isfinite(source_poses).all() or not torch.isfinite(target_poses).all():
        raise ValueError("poses contain non-finite values")

    source = source_poses.float()
    target = target_poses.float()
    relative_rotations = (
        target[:, :3, :3] @ source[:, :3, :3].transpose(-1, -2)
    )
    u, _, vh = torch.linalg.svd(relative_rotations.sum(dim=0))
    correction = torch.eye(3, device=u.device, dtype=u.dtype)
    correction[-1, -1] = torch.sign(torch.linalg.det(u @ vh))
    rotation = u @ correction @ vh

    source_centers = source[:, :3, 3]
    target_centers = target[:, :3, 3]
    translation = (
        target_centers - torch.einsum("ij,nj->ni", rotation, source_centers)
    ).mean(dim=0)
    return rotation, translation
