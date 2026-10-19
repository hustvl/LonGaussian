"""Vectorized helpers for track-guided merge."""

from __future__ import annotations

import torch


def batch_gather_gaussians(
    pred_slice: dict,
    effective: torch.Tensor,
    trajs_uv: torch.Tensor,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather per-track Gaussian attributes for all frames at once."""
    t_len, num_tracks, _ = trajs_uv.shape
    device = effective.device
    t_idx = torch.arange(t_len, device=device)[:, None].expand(t_len, num_tracks)
    cols = trajs_uv[..., 0].round().long().clamp(0, width - 1)
    rows = trajs_uv[..., 1].round().long().clamp(0, height - 1)

    centers = effective[0, t_idx, rows, cols]

    scales = pred_slice["gaussian_scales"][0].permute(0, 2, 3, 1)[t_idx, rows, cols]
    quats = pred_slice["gaussian_rotations"][0].permute(0, 2, 3, 1)[t_idx, rows, cols]
    opacities = pred_slice["gaussian_opacities"][0, :, 0][t_idx, rows, cols]
    sh = pred_slice["gaussian_sh_coeffs"][0].permute(0, 2, 3, 1)[t_idx, rows, cols]
    return centers, quats, scales, opacities, sh


def batch_gather_feature_context(
    pred_slice: dict, trajs_uv: torch.Tensor, height: int, width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather M2 projected features and camera-relative observation context."""
    t_len, num_tracks, _ = trajs_uv.shape
    device = trajs_uv.device
    t_idx = torch.arange(t_len, device=device)[:, None].expand(t_len, num_tracks)
    cols = trajs_uv[..., 0].round().long().clamp(0, width - 1)
    rows = trajs_uv[..., 1].round().long().clamp(0, height - 1)
    features = pred_slice["gaussian_merge_features"][0].permute(0, 2, 3, 1)[t_idx, rows, cols]
    local_points = pred_slice["local_points"][0, t_idx, rows, cols]
    rays = torch.nn.functional.normalize(local_points, dim=-1)
    depths = local_points[..., 2].abs().clamp_min(1e-5)
    centers = pred_slice["gaussian_means"][0, t_idx, rows, cols]
    camera_centers = pred_slice["camera_poses"][0, :, :3, 3][t_idx]
    views = torch.nn.functional.normalize(centers - camera_centers, dim=-1)
    confidence = pred_slice["conf"][0, t_idx, rows, cols]
    if confidence.ndim == 3:
        confidence = confidence.squeeze(-1)
    return features, rays, views, depths, confidence


def batch_gather_direct_slot_context(
    pred_slice: dict, trajs_uv: torch.Tensor, height: int, width: int,
) -> tuple[torch.Tensor, ...]:
    """Gather raw pre-head features and source observations for M3 slots."""
    t_len, num_tracks, _ = trajs_uv.shape
    device = trajs_uv.device
    t_idx = torch.arange(t_len, device=device)[:, None].expand(t_len, num_tracks)
    cols = trajs_uv[..., 0].round().long().clamp(0, width - 1)
    rows = trajs_uv[..., 1].round().long().clamp(0, height - 1)
    features = pred_slice["gaussian_prehead_features"][0].permute(0, 2, 3, 1)[t_idx, rows, cols]
    rgb = pred_slice["gaussian_source_rgb"][0].permute(0, 2, 3, 1)[t_idx, rows, cols]
    source_points = pred_slice["points"][0, t_idx, rows, cols]
    local_points = pred_slice["local_points"][0, t_idx, rows, cols]
    rays = torch.nn.functional.normalize(local_points, dim=-1)
    depths = local_points[..., 2].abs().clamp_min(1e-5)
    camera_centers = pred_slice["camera_poses"][0, :, :3, 3][t_idx]
    views = torch.nn.functional.normalize(source_points - camera_centers, dim=-1)
    confidence = pred_slice["conf"][0, t_idx, rows, cols]
    if confidence.ndim == 3:
        confidence = confidence.squeeze(-1)
    footprints = depths / max(float(height), float(width)) / 0.7
    return features, rgb, source_points, local_points, rays, views, depths, footprints, confidence


def mask_pixel_ownership(
    weights: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Keep only the highest-weight track per (frame, pixel)."""
    t_len, num_tracks = weights.shape
    device = weights.device
    pixel_id = (
        torch.arange(t_len, device=device)[:, None] * height * width
        + rows * width
        + cols
    )
    flat_pid = pixel_id.reshape(-1)
    flat_w = weights.reshape(-1)
    flat_track = torch.arange(num_tracks, device=device).repeat(t_len)

    order = flat_w.argsort(descending=True)
    owned = torch.full((t_len * height * width,), -1, dtype=torch.long, device=device)
    owned_w = torch.zeros_like(owned, dtype=weights.dtype)
    pid_sorted = flat_pid[order]
    track_sorted = flat_track[order]
    w_sorted = flat_w[order]
    better = w_sorted > owned_w[pid_sorted]
    owned[pid_sorted[better]] = track_sorted[better]
    owned_w[pid_sorted[better]] = w_sorted[better]

    owner = owned[pixel_id.reshape(-1)].reshape(t_len, num_tracks)
    return (owner == torch.arange(num_tracks, device=device)[None, :]) & (weights > 0)


def filter_track_vis_by_query_radius(
    points: torch.Tensor,
    vis: torch.Tensor,
    *,
    radius: float,
    query_frame_idx: int = 0,
) -> torch.Tensor:
    """Drop track observations farther than ``radius`` from the query-frame point."""
    if radius <= 0:
        return vis
    query = points[query_frame_idx]  # [M, 3]
    dist = (points - query.unsqueeze(0)).norm(dim=-1)
    in_radius = dist <= radius
    return vis & in_radius & vis[query_frame_idx].unsqueeze(0)


def estimate_median_1nn(
    points: torch.Tensor,
    *,
    num_samples: int = 5000,
    generator: torch.Generator | None = None,
) -> float:
    """Median nearest-neighbor distance on a point cloud (flattened [N,3])."""
    pts = points.reshape(-1, 3)
    n = pts.shape[0]
    if n < 2:
        return 0.0
    k = min(int(num_samples), n)
    idx = torch.randperm(n, generator=generator, device=pts.device)[:k]
    sample = pts[idx]
    d = torch.cdist(sample, sample)
    d.fill_diagonal_(float("inf"))
    nn = d.min(dim=1).values
    return float(nn.median().item())


def resolve_track_anchor_radius(
    config: "MergeConfig",
    dense_points: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> tuple[float, float | None]:
    """Return ``(radius, median_1nn_or_None)`` for the query-anchor gate."""
    from loger.merge.merge_config import MergeConfig as _MergeConfig  # noqa: F401

    mode = getattr(config, "track_anchor_mode", "fixed")
    if mode == "fixed":
        return float(config.track_anchor_radius), None
    if mode != "median_nn":
        raise ValueError(f"unsupported track_anchor_mode: {mode}")
    med_nn = estimate_median_1nn(
        dense_points,
        num_samples=int(config.track_anchor_nn_samples),
        generator=generator,
    )
    if med_nn <= 0:
        return float(config.track_anchor_radius), None
    radius = float(config.track_anchor_beta) * med_nn
    radius = max(float(config.track_anchor_r_min), min(float(config.track_anchor_r_max), radius))
    return radius, med_nn
