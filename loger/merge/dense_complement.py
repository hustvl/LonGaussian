"""Dense Gaussian complement for hybrid track-merge rendering."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch

from loger.merge.compact_gaussians import FinalCompactGaussians
from loger.renderer.gaussian_renderer import GaussianRenderer


def init_covered_mask(pred: dict) -> torch.Tensor:
    """Create a per-frame pixel coverage mask on CPU."""
    _, num_frames, height, width = pred["points"].shape[:4]
    return torch.zeros(num_frames, height, width, dtype=torch.bool, device="cpu")


def mark_merged_track_coverage(
    covered: torch.Tensor,
    global_frames: Sequence[int],
    trajs_uv: torch.Tensor,
    vis: torch.Tensor,
    merge_ids: torch.Tensor,
    height: int,
    width: int,
) -> None:
    """Mark pixels consumed by successfully merged tracks (in-place)."""
    if merge_ids.numel() == 0:
        return

    vis_m = vis[:, merge_ids]
    uv_m = trajs_uv[:, merge_ids]
    rows = uv_m[..., 1].round().long().clamp(0, height - 1)
    cols = uv_m[..., 0].round().long().clamp(0, width - 1)

    visible = vis_m.nonzero(as_tuple=False)
    if visible.numel() == 0:
        return

    local_t = visible[:, 0]
    track_k = visible[:, 1]
    if len(global_frames) == covered.shape[0]:
        frame_idx = local_t.to(device=covered.device, dtype=torch.long)
    else:
        frame_idx = torch.tensor(
            [int(global_frames[t]) for t in local_t.tolist()],
            dtype=torch.long,
            device=covered.device,
        )
    row_idx = rows[local_t, track_k].to(covered.device)
    col_idx = cols[local_t, track_k].to(covered.device)
    covered[frame_idx, row_idx, col_idx] = True


def complement_pixel_mask(
    covered: torch.Tensor,
    ctx_indices: Sequence[int],
) -> torch.Tensor:
    """Return (len(ctx), H, W) mask of dense pixels not covered by merged tracks."""
    ctx_idx = torch.tensor(list(ctx_indices), dtype=torch.long, device=covered.device)
    return ~covered[ctx_idx]


def build_hybrid_render_tensors(
    final_compact: FinalCompactGaussians,
    pred: dict,
    ctx_indices: Sequence[int],
    renderer: GaussianRenderer,
    *,
    conf_gate: bool = False,
    conf_thr: float = 0.0,
    conf_gate_mode: str = "hard",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concatenate compact merged Gaussians with dense complement for rendering."""
    device = pred["points"].device
    means_c, quats_c, scales_c, opacs_c, colors_c = final_compact.as_render_tensors(
        sh_degree=renderer.sh_degree,
        opacity_threshold=renderer.opacity_threshold,
    )
    means_c = means_c.to(device)
    quats_c = quats_c.to(device)
    scales_c = scales_c.to(device)
    opacs_c = opacs_c.to(device)
    colors_c = colors_c.to(device)
    num_compact = int(means_c.shape[0])

    covered = final_compact.covered_pixels
    if covered is not None:
        covered = covered.to(device)
    if covered is None:
        pixel_mask = None
        num_complement = 0
    else:
        pixel_mask = complement_pixel_mask(covered, ctx_indices)
        if conf_gate and pred.get("conf") is not None:
            conf = pred["conf"]
            # conf: (B,T,H,W,?) → (len(ctx), H, W)
            conf_ctx = conf[0, list(ctx_indices)]
            if conf_ctx.ndim == 4 and conf_ctx.shape[-1] == 1:
                conf_ctx = conf_ctx[..., 0]
            conf_s = torch.sigmoid(conf_ctx.float())
            if str(conf_gate_mode) == "hard":
                pixel_mask = pixel_mask & (conf_s > float(conf_thr))
        means_d, quats_d, scales_d, opacs_d, colors_d, _ = renderer.prepare_gaussians(
            pred,
            frame_indices=list(ctx_indices),
            pixel_include_mask=pixel_mask,
        )
        # soft_opacity path for this helper: scale after prepare is awkward
        # because prepare_gaussians already flattened; hard is the primary A1 path.
        num_complement = int(means_d.shape[0])
        if num_complement > 0:
            means_c = torch.cat([means_c, means_d], dim=0)
            quats_c = torch.cat([quats_c, quats_d], dim=0)
            scales_c = torch.cat([scales_c, scales_d], dim=0)
            opacs_c = torch.cat([opacs_c, opacs_d], dim=0)
            colors_c = torch.cat([colors_c, colors_d], dim=0)

    num_covered_pixels = int((~pixel_mask).sum().item()) if pixel_mask is not None else 0
    num_complement_pixels = int(pixel_mask.sum().item()) if pixel_mask is not None else 0
    final_compact.stats.update(
        {
            "num_gaussians_compact": num_compact,
            "num_gaussians_complement": num_complement,
            "num_gaussians_hybrid": int(means_c.shape[0]),
            "num_pixels_covered_by_merge": num_covered_pixels,
            "num_pixels_complement": num_complement_pixels,
            "complement_conf_gate": bool(conf_gate),
            "complement_conf_thr": float(conf_thr),
            "complement_conf_gate_mode": str(conf_gate_mode),
        }
    )
    return means_c, quats_c, scales_c, opacs_c, colors_c
