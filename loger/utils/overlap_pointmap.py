"""Robust residual SE(3) fitting between two overlapping pointmaps.

This is deliberately a small geometry-only primitive.  Semantic dynamic masks
are not available from Pi3, so moving objects are rejected through the robust
residual pass rather than being mislabeled as a semantic segmentation result.
"""

from __future__ import annotations

import torch


def depth_edge_mask(points: torch.Tensor, max_relative_gradient: float) -> torch.Tensor:
    """Pixels away from discontinuities, using a scale-invariant depth gradient."""
    depth = points[..., 2].abs().clamp_min(1e-5)
    grad_x = torch.zeros_like(depth)
    grad_y = torch.zeros_like(depth)
    grad_x[..., :, 1:] = (depth[..., :, 1:] - depth[..., :, :-1]).abs()
    grad_y[..., 1:, :] = (depth[..., 1:, :] - depth[..., :-1, :]).abs()
    rel = (grad_x + grad_y) / depth
    return rel <= max_relative_gradient


def weighted_kabsch(source: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return rigid transform mapping source points to target points."""
    if source.ndim != 2 or source.shape != target.shape or source.shape[-1] != 3:
        raise ValueError("source and target must have shape (N, 3)")
    weights = weights.reshape(-1).float().clamp_min(0)
    if len(source) < 3 or not bool((weights > 0).any()):
        raise ValueError("at least three positive-weight correspondences are required")
    total = weights.sum().clamp_min(1e-8)
    src_center = (source.float() * weights[:, None]).sum(0) / total
    tgt_center = (target.float() * weights[:, None]).sum(0) / total
    covariance = ((source.float() - src_center) * weights[:, None]).transpose(0, 1) @ (target.float() - tgt_center)
    u, _, vh = torch.linalg.svd(covariance)
    rotation = vh.transpose(0, 1) @ u.transpose(0, 1)
    if torch.det(rotation) < 0:
        vh = vh.clone()
        vh[-1] *= -1
        rotation = vh.transpose(0, 1) @ u.transpose(0, 1)
    translation = tgt_center - rotation @ src_center
    return rotation, translation


def transform_points(points: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    return points @ rotation.transpose(-1, -2) + translation


def robust_pointmap_se3(
    source: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    *,
    max_iterations: int = 3,
    huber_multiplier: float = 2.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit SE(3), iteratively downweighting geometric (e.g. dynamic) outliers."""
    active_weights = weights.float().clone()
    rotation = torch.eye(3, device=source.device, dtype=torch.float32)
    translation = torch.zeros(3, device=source.device, dtype=torch.float32)
    for _ in range(max_iterations):
        # Start from identity.  In the diagnostic this is the pose-stitch
        # initialization, so static overlap points are already close whereas
        # moving/incorrect correspondences are not.  Reweight *before* the
        # first Kabsch fit; a vanilla first fit can be dragged far away by a
        # coherent moving foreground.
        residual = torch.linalg.vector_norm(transform_points(source, rotation, translation) - target, dim=-1)
        scale = residual.median().clamp_min(1e-5)
        cutoff = huber_multiplier * scale
        # Tukey's bounded influence is intentional here: the desired
        # diagnostic must be able to exclude a coherent moving foreground,
        # not merely reduce its leverage.
        ratio = residual / cutoff.clamp_min(1e-5)
        robust = torch.where(ratio < 1.0, (1.0 - ratio.square()).square(), torch.zeros_like(ratio))
        active_weights = weights * robust
        rotation, translation = weighted_kabsch(source, target, active_weights)
    return rotation, translation, active_weights
