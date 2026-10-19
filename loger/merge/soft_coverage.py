"""Soft coverage blending for DSGA v0 hybrid merge."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

if TYPE_CHECKING:
    from loger.merge.window_merge import WindowMergeTensors


def mark_soft_track_coverage(
    soft_coverage: torch.Tensor,
    global_frames: Sequence[int],
    trajs_uv: torch.Tensor,
    vis: torch.Tensor,
    track_ids: torch.Tensor,
    merged_opacity: torch.Tensor,
    height: int,
    width: int,
) -> None:
    """Accumulate per-pixel soft coverage from merged tracks (in-place)."""
    if track_ids.numel() == 0:
        return

    vis_m = vis[:, track_ids]
    uv_m = trajs_uv[:, track_ids]
    # DSGA already folds track_validity into merged_opacity. Applying it again
    # underestimates coverage and unnecessarily duplicates dense complements.
    opa_m = merged_opacity[track_ids]
    rows = uv_m[..., 1].round().long().clamp(0, height - 1)
    cols = uv_m[..., 0].round().long().clamp(0, width - 1)

    visible = vis_m.nonzero(as_tuple=False)
    if visible.numel() == 0:
        return

    local_t = visible[:, 0]
    track_k = visible[:, 1]
    if len(global_frames) == soft_coverage.shape[0]:
        frame_idx = local_t.to(device=soft_coverage.device, dtype=torch.long)
    else:
        frame_idx = torch.tensor(
            [int(global_frames[t]) for t in local_t.tolist()],
            dtype=torch.long,
            device=soft_coverage.device,
        )
    row_idx = rows[local_t, track_k].to(soft_coverage.device)
    col_idx = cols[local_t, track_k].to(soft_coverage.device)
    contrib = opa_m[track_k].to(soft_coverage.dtype)
    soft_coverage[frame_idx, row_idx, col_idx] = torch.maximum(
        soft_coverage[frame_idx, row_idx, col_idx],
        contrib,
    )


def extract_soft_complement_tensors(
    pred_slice: dict,
    soft_coverage: torch.Tensor,
    *,
    min_weight: float = 1e-4,
    max_gaussians: int | None = None,
) -> "WindowMergeTensors":
    """Extract dense complement, optionally keeping top effective opacities."""
    from loger.merge.window_merge import WindowMergeTensors
    device = pred_slice["points"].device
    sh_dim = int(pred_slice["gaussian_sh_coeffs"].shape[2])
    t_len = pred_slice["points"].shape[1]
    if t_len == 0:
        return WindowMergeTensors.empty(device, sh_dim=sh_dim)

    means_list = []
    quats_list = []
    scales_list = []
    opacs_list = []
    sh_list = []

    for local_t in range(t_len):
        one_minus = (1.0 - soft_coverage[local_t].clamp(0.0, 1.0)).clamp(min=0.0)
        mask = one_minus > min_weight
        if not bool(mask.any()):
            continue
        pts = pred_slice["points"][0, local_t]
        if pred_slice.get("gaussian_offsets") is not None:
            pts = pts + pred_slice["gaussian_offsets"][0, local_t].permute(1, 2, 0)
        scales = pred_slice["gaussian_scales"][0, local_t].permute(1, 2, 0)
        quats = pred_slice["gaussian_rotations"][0, local_t].permute(1, 2, 0)
        opacs = pred_slice["gaussian_opacities"][0, local_t, 0]
        sh = pred_slice["gaussian_sh_coeffs"][0, local_t].permute(1, 2, 0)

        w = one_minus[mask]
        means_list.append(pts[mask].reshape(-1, 3))
        scales_list.append(scales[mask].reshape(-1, 3))
        quats_list.append(quats[mask].reshape(-1, 4))
        opacs_list.append((opacs[mask] * w).reshape(-1))
        sh_list.append(sh[mask].reshape(-1, sh_dim))

    if not means_list:
        return WindowMergeTensors.empty(device, sh_dim=sh_dim)

    means = torch.cat(means_list, dim=0)
    quats = torch.cat(quats_list, dim=0)
    scales = torch.cat(scales_list, dim=0)
    opacities = torch.cat(opacs_list, dim=0)
    sh_flat = torch.cat(sh_list, dim=0)
    if max_gaussians is not None and means.shape[0] > max_gaussians:
        if max_gaussians <= 0:
            return WindowMergeTensors.empty(device, sh_dim=sh_dim)
        keep = torch.topk(
            opacities.detach(), k=int(max_gaussians), sorted=False
        ).indices
        means = means[keep]
        quats = quats[keep]
        scales = scales[keep]
        opacities = opacities[keep]
        sh_flat = sh_flat[keep]

    return WindowMergeTensors(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        sh_flat=sh_flat,
    )
