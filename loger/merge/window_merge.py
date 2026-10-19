"""Within-window track merge and global cloud assembly."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from loger.merge.dense_complement import mark_merged_track_coverage
from loger.merge.merge_config import MergeConfig
from loger.merge.merge_utils import (
    batch_merge_tracks,
    batch_merge_tracks_dsga,
    batch_merge_tracks_medoid,
    batch_merge_tracks_nested,
    batch_merge_tracks_tracklet_split,
    batch_merge_tracks_flatk,
)
from loger.merge.soft_coverage import (
    extract_soft_complement_tensors,
    mark_soft_track_coverage,
)
from loger.merge.vectorized_merge import batch_gather_feature_context, batch_gather_gaussians, mask_pixel_ownership
from loger.tracking.window_chunks import _contribute_range


GAUSSIAN_PRED_KEYS = (
    "points",
    "local_points",
    "conf",
    "gaussian_scales",
    "gaussian_rotations",
    "gaussian_opacities",
    "gaussian_sh_coeffs",
    "gaussian_offsets",
    "gaussian_means",
    "gaussian_merge_features",
    "gaussian_prehead_features",
    "gaussian_source_rgb",
)


def slice_predictions(pred: dict, local_frame_indices: Sequence[int]) -> dict:
    idx = torch.tensor(list(local_frame_indices), device=pred["points"].device, dtype=torch.long)
    sliced: dict = {}
    for key in GAUSSIAN_PRED_KEYS:
        if key in pred and pred[key] is not None:
            sliced[key] = pred[key][:, idx]
    if "camera_poses" in pred and pred["camera_poses"] is not None:
        sliced["camera_poses"] = pred["camera_poses"][:, idx]
    return sliced


def global_contribute_range(
    window_idx: int,
    num_windows: int,
    window_start: int,
    window_end: int,
    overlap_size: int,
) -> Tuple[int, int]:
    """Return global [start, end) frame indices this window contributes."""
    local_start, local_end = _contribute_range(
        window_idx, num_windows, 0, window_end - window_start, overlap_size
    )
    return window_start + local_start, window_start + local_end


def sliding_window_ranges(
    num_frames: int,
    window_size: int,
    overlap_size: int,
) -> List[Tuple[int, int]]:
    """Return ``[(start, end), ...]`` sliding windows over ``num_frames``."""
    if window_size <= 0 or window_size >= num_frames:
        return [(0, num_frames)]
    windows: List[Tuple[int, int]] = []
    step = max(window_size - overlap_size, 1)
    for start_idx in range(0, num_frames, step):
        end_idx = min(start_idx + window_size, num_frames)
        if end_idx - start_idx >= overlap_size or (end_idx == num_frames and start_idx < num_frames):
            windows.append((start_idx, end_idx))
        if end_idx == num_frames:
            break
    return windows


def owner_window_for_frame(
    frame_idx: int,
    windows: Sequence[Tuple[int, int]],
    overlap_size: int,
) -> int:
    """Pick the window that owns ``frame_idx`` for per-window diagnostics."""
    num_windows = len(windows)
    for window_idx, (start_idx, end_idx) in enumerate(windows):
        contrib_start, contrib_end = global_contribute_range(
            window_idx, num_windows, start_idx, end_idx, overlap_size
        )
        if contrib_start <= frame_idx < contrib_end:
            return window_idx
    for window_idx, (start_idx, end_idx) in enumerate(windows):
        if start_idx <= frame_idx < end_idx:
            return window_idx
    return max(num_windows - 1, 0)


def stack_window_hybrid_tensors(
    compact: WindowMergeTensors,
    complement: WindowMergeTensors,
) -> WindowMergeTensors:
    """Concatenate one window's compact + complement tensors."""
    if compact.num_gaussians == 0:
        return complement
    if complement.num_gaussians == 0:
        return compact
    device = compact.means.device
    if complement.means.device != device:
        complement = WindowMergeTensors(
            means=complement.means.to(device),
            quats=complement.quats.to(device),
            scales=complement.scales.to(device),
            opacities=complement.opacities.to(device),
            sh_flat=complement.sh_flat.to(device),
            conf_scores=(
                None
                if complement.conf_scores is None
                else complement.conf_scores.to(device)
            ),
        )
    return WindowMergeTensors(
        means=torch.cat([compact.means, complement.means], dim=0),
        quats=torch.cat([compact.quats, complement.quats], dim=0),
        scales=torch.cat([compact.scales, complement.scales], dim=0),
        opacities=torch.cat([compact.opacities, complement.opacities], dim=0),
        sh_flat=torch.cat([compact.sh_flat, complement.sh_flat], dim=0),
    )


def _effective_centers(pred_slice: dict) -> torch.Tensor:
    points = pred_slice["points"]
    offsets = pred_slice.get("gaussian_offsets")
    if offsets is not None:
        return points + offsets.permute(0, 1, 3, 4, 2)
    if pred_slice.get("gaussian_means") is not None:
        return pred_slice["gaussian_means"]
    return points


@dataclass
class WindowMergeTensors:
    means: torch.Tensor
    quats: torch.Tensor
    scales: torch.Tensor
    opacities: torch.Tensor
    sh_flat: torch.Tensor
    base_compact_count: Optional[int] = None
    tracklet_split_count: int = 0
    flatk_k2_count: int = 0
    flatk_k4_count: int = 0
    flatk_trace_digest: Optional[str] = None
    nested_k2_count: int = 0
    nested_k4_count: int = 0
    nested_trace_digest: Optional[str] = None
    # Optional per-Gaussian sigmoid(conf) for complement (E12 vis / diagnostics).
    conf_scores: Optional[torch.Tensor] = None
    budget_selection_stats: Optional[Dict[str, int | str]] = None

    @property
    def num_gaussians(self) -> int:
        return int(self.means.shape[0])

    @classmethod
    def empty(cls, device: torch.device, sh_dim: int = 27) -> "WindowMergeTensors":
        return cls(
            means=torch.empty(0, 3, device=device),
            quats=torch.empty(0, 4, device=device),
            scales=torch.empty(0, 3, device=device),
            opacities=torch.empty(0, device=device),
            sh_flat=torch.empty(0, sh_dim, device=device),
            conf_scores=torch.empty(0, device=device),
        )


def merge_within_window(
    pred_slice: dict,
    tracks: dict,
    config: MergeConfig,
    covered: torch.Tensor,
    global_frame_ids: Sequence[int],
    *,
    track_rd_records: Optional[List[dict]] = None,
    selector_trace_records: Optional[List[dict]] = None,
    window_idx: int = -1,
    neural_merger: Optional[torch.nn.Module] = None,
    suppressed: Optional[torch.Tensor] = None,
) -> WindowMergeTensors:
    """Merge dense Gaussians for ctx frames in one window.

    ``suppressed`` is an optional ``(T, H, W)`` mask of pixels already refined
    into the streaming map (E8); those observations are hidden from the tracker
    so they neither anchor nor feed a compact Gaussian.
    """
    trajs_uv_np = tracks["trajs_uv"]
    vis_np = tracks["vis"]
    height, width = pred_slice["points"].shape[2:4]
    device = pred_slice["points"].device
    sh_dim = int(pred_slice["gaussian_sh_coeffs"].shape[2])
    effective = _effective_centers(pred_slice)

    trajs_uv = torch.from_numpy(trajs_uv_np).to(dtype=effective.dtype, device=device)
    vis = torch.from_numpy(vis_np).to(device=device)

    centers, quats, scales, opacities, sh = batch_gather_gaussians(
        pred_slice, effective, trajs_uv, height, width
    )
    rows = trajs_uv[..., 1].round().long().clamp(0, height - 1)
    cols = trajs_uv[..., 0].round().long().clamp(0, width - 1)
    if suppressed is not None:
        frames = torch.arange(rows.shape[0], device=device)[:, None].expand_as(rows)
        vis = vis & ~suppressed.to(device)[frames, rows, cols]
    owned = mask_pixel_ownership(opacities * vis.to(opacities.dtype), rows, cols, height, width)
    vis = vis & owned
    if config.merge_mode in ("neural", "neural_set", "neural_feature", "neural_direct_slots"):
        if neural_merger is None:
            raise ValueError("neural merge modes require a merger module")
        (
            base_mean, base_quat, base_scale, base_opacity, base_sh,
            _merge_weight, _source_count, merge_mask, _skipped_mask,
        ) = batch_merge_tracks(
            centers, quats, scales, opacities, sh, vis,
            config.min_track_len, config.min_visible_frames, config.min_mean_opacity,
            config.max_center_variance, config.scale_variance_boost,
        )
        if config.merge_mode == "neural_direct_slots":
            from loger.merge.vectorized_merge import batch_gather_direct_slot_context

            direct_inputs = batch_gather_direct_slot_context(pred_slice, trajs_uv, height, width)
            neural_mean, neural_quat, neural_scale, neural_opacity, neural_sh = neural_merger(*direct_inputs, vis)
            num_tracks = merge_mask.numel()
            num_slots = neural_mean.shape[0] // num_tracks
            if num_slots * num_tracks != neural_mean.shape[0]:
                raise ValueError("direct slot merger output must contain an equal number of slots per track")
            compact_mean = neural_mean.reshape(num_tracks, num_slots, 3)[merge_mask].reshape(-1, 3)
            compact_quat = neural_quat.reshape(num_tracks, num_slots, 4)[merge_mask].reshape(-1, 4)
            compact_scale = neural_scale.reshape(num_tracks, num_slots, 3)[merge_mask].reshape(-1, 3)
            compact_opacity = neural_opacity.reshape(num_tracks, num_slots)[merge_mask].reshape(-1)
            compact_sh = neural_sh.reshape(num_tracks, num_slots, sh.shape[-1])[merge_mask].reshape(-1, sh.shape[-1])
        elif config.merge_mode == "neural_feature":
            features, rays, views, depths, confidence = batch_gather_feature_context(
                pred_slice, trajs_uv, height, width
            )
            neural_mean, neural_quat, neural_scale, neural_opacity, neural_sh = neural_merger(
                centers, quats, scales, opacities, sh, features, rays, views, depths, confidence, vis,
                base_mean, base_quat, base_scale, base_opacity, base_sh,
            )
        elif config.merge_mode == "neural_set":
            neural_mean, neural_quat, neural_scale, neural_opacity, neural_sh = neural_merger(
                centers, quats, scales, opacities, sh, vis,
                base_mean, base_quat, base_scale, base_opacity, base_sh,
            )
        else:
            neural_mean, neural_quat, neural_scale, neural_opacity, neural_sh = neural_merger(
                centers, scales, opacities, sh, vis,
                base_mean, base_quat, base_scale, base_opacity, base_sh,
            )
        if config.merge_mode != "neural_direct_slots":
            compact_mean = neural_mean[merge_mask]
            compact_quat = neural_quat[merge_mask]
            compact_scale = neural_scale[merge_mask]
            compact_opacity = neural_opacity[merge_mask]
            compact_sh = neural_sh[merge_mask]
        split_mask = torch.zeros_like(merge_mask)
        flatk_trace_digest = None
        nested_trace_digest = None
    elif config.merge_mode == "hard_flatk":
        (
            compact_mean, compact_quat, compact_scale, compact_opacity, compact_sh,
            merge_mask, requested_k, actual_k, _score, assignment_trace,
        ) = batch_merge_tracks_flatk(
            centers, quats, scales, opacities, sh, vis,
            config.min_track_len, config.min_visible_frames, config.min_mean_opacity,
            config.max_center_variance, config.scale_variance_boost,
            fraction_k4=config.flatk_fraction_k4,
            fraction_k2=config.flatk_fraction_k2,
        )
        split_mask = actual_k > 1
        flatk_trace_digest = hashlib.sha256(
            torch.stack((requested_k.to(torch.int8), actual_k.to(torch.int8))).detach().cpu().contiguous().numpy().tobytes()
            + assignment_trace.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        nested_trace_digest = None
    elif config.merge_mode == "hard_nested":
        (
            compact_mean, compact_quat, compact_scale, compact_opacity, compact_sh,
            merge_mask, requested_k, actual_k, _score, assignment_trace,
        ) = batch_merge_tracks_nested(
            centers, quats, scales, opacities, sh, vis,
            config.min_track_len, config.min_visible_frames, config.min_mean_opacity,
            config.max_center_variance, config.scale_variance_boost,
            fraction_k4=config.nested_fraction_k4,
            fraction_k2=config.nested_fraction_k2,
        )
        split_mask = actual_k > 1
        nested_trace_digest = hashlib.sha256(
            torch.stack((requested_k.to(torch.int8), actual_k.to(torch.int8))).detach().cpu().contiguous().numpy().tobytes()
            + assignment_trace.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        flatk_trace_digest = None
    elif config.merge_mode in ("hard_tracklet", "hard_appearance"):
        (
            compact_mean, compact_quat, compact_scale, compact_opacity, compact_sh,
            merge_mask, split_mask,
        ) = batch_merge_tracks_tracklet_split(
            centers, quats, scales, opacities, sh, vis,
            config.min_track_len, config.min_visible_frames, config.min_mean_opacity,
            config.max_center_variance, config.scale_variance_boost,
            mad_multiplier=config.tracklet_mad_multiplier,
            min_jump=config.tracklet_min_jump,
            split_signal="appearance" if config.merge_mode == "hard_appearance" else "geometry",
        )
        flatk_trace_digest = None
        nested_trace_digest = None
    else:
        (
            merged_mean, merged_quat, merged_scale, merged_opacity, merged_sh,
            _merge_weight, _source_count, merge_mask, _skipped_mask,
        ) = (batch_merge_tracks_medoid if config.merge_mode == "hard_medoid" else batch_merge_tracks)(
        centers,
        quats,
        scales,
        opacities,
        sh,
        vis,
        config.min_track_len,
        config.min_visible_frames,
        config.min_mean_opacity,
        config.max_center_variance,
        config.scale_variance_boost,
        )
        compact_mean = merged_mean[merge_mask]
        compact_quat = merged_quat[merge_mask]
        compact_scale = merged_scale[merge_mask]
        compact_opacity = merged_opacity[merge_mask]
        compact_sh = merged_sh[merge_mask]
        split_mask = torch.zeros_like(merge_mask)
        flatk_trace_digest = None
        nested_trace_digest = None

    merge_ids = merge_mask.nonzero(as_tuple=True)[0]
    if track_rd_records is not None:
        from loger.merge.track_rd import build_track_window_record

        track_rd_records.append(
            build_track_window_record(
                window_idx=window_idx,
                global_frame_ids=global_frame_ids,
                centers=centers,
                quats=quats,
                scales=scales,
                opacities=opacities,
                sh=sh,
                trajs_uv=trajs_uv,
                visible=vis,
                owned=owned,
                k1_reference={
                    "track_ids": merge_ids,
                    "means": compact_mean,
                    "quats": compact_quat,
                    "scales": compact_scale,
                    "opacities": compact_opacity,
                    "sh": compact_sh,
                },
            )
        )
    if merge_ids.numel() > 0:
        mark_merged_track_coverage(
            covered,
            list(global_frame_ids),
            trajs_uv,
            vis,
            merge_ids,
            height,
            width,
        )
    if selector_trace_records is not None:
        from loger.merge.track_rd import build_selector_window_record

        selector_trace_records.append(
            build_selector_window_record(
                window_idx=window_idx,
                global_frame_ids=global_frame_ids,
                merge_track_ids=merge_ids,
                selected_rows=rows[:, merge_ids],
                selected_cols=cols[:, merge_ids],
                selected_visible=vis[:, merge_ids],
                covered=covered,
            )
        )
    if merge_ids.numel() > 0:
        return WindowMergeTensors(
            means=compact_mean,
            quats=compact_quat,
            scales=compact_scale,
            opacities=compact_opacity,
            sh_flat=compact_sh,
            base_compact_count=int(merge_ids.numel()),
            tracklet_split_count=int(split_mask.sum().item()),
            flatk_k2_count=int((actual_k == 2).sum().item()) if config.merge_mode == "hard_flatk" else 0,
            flatk_k4_count=int((actual_k == 4).sum().item()) if config.merge_mode == "hard_flatk" else 0,
            flatk_trace_digest=flatk_trace_digest,
            nested_k2_count=int((actual_k == 2).sum().item()) if config.merge_mode == "hard_nested" else 0,
            nested_k4_count=int((actual_k == 4).sum().item()) if config.merge_mode == "hard_nested" else 0,
            nested_trace_digest=nested_trace_digest,
        )
    return WindowMergeTensors.empty(device, sh_dim=sh_dim)


def merge_within_window_dsga(
    pred_slice: dict,
    tracks: dict,
    config: MergeConfig,
    soft_coverage: torch.Tensor,
    global_frame_ids: Sequence[int],
    *,
    min_opacity: float = 1e-4,
) -> WindowMergeTensors:
    """DSGA v0: soft track validity + soft per-pixel coverage."""
    trajs_uv_np = tracks["trajs_uv"]
    vis_np = tracks["vis"]
    height, width = pred_slice["points"].shape[2:4]
    device = pred_slice["points"].device
    sh_dim = int(pred_slice["gaussian_sh_coeffs"].shape[2])
    effective = _effective_centers(pred_slice)

    trajs_uv = torch.from_numpy(trajs_uv_np).to(dtype=effective.dtype, device=device)
    vis = torch.from_numpy(vis_np).to(device=device)

    centers, quats, scales, opacities, sh = batch_gather_gaussians(
        pred_slice, effective, trajs_uv, height, width
    )
    rows = trajs_uv[..., 1].round().long().clamp(0, height - 1)
    cols = trajs_uv[..., 0].round().long().clamp(0, width - 1)
    owned = mask_pixel_ownership(opacities * vis.to(opacities.dtype), rows, cols, height, width)
    vis = vis & owned

    (
        merged_mean,
        merged_quat,
        merged_scale,
        merged_opacity,
        merged_sh,
        _merge_weight,
        _source_count,
        track_validity,
    ) = batch_merge_tracks_dsga(
        centers,
        quats,
        scales,
        opacities,
        sh,
        vis,
        min_track_len=config.min_track_len,
        min_visible_frames=config.min_visible_frames,
        min_mean_opacity=config.min_mean_opacity,
        max_center_variance=config.max_center_variance,
        rotation_mode=config.rotation_mode,
        scale_variance_boost=config.scale_variance_boost,
        soft_validity_temp=config.soft_validity_temp,
    )

    keep = merged_opacity > min_opacity
    if not bool(keep.any()):
        return WindowMergeTensors.empty(device, sh_dim=sh_dim)

    track_ids = keep.nonzero(as_tuple=True)[0]
    mark_soft_track_coverage(
        soft_coverage,
        list(global_frame_ids),
        trajs_uv,
        vis,
        track_ids,
        merged_opacity,
        height,
        width,
    )
    return WindowMergeTensors(
        means=merged_mean[keep],
        quats=merged_quat[keep],
        scales=merged_scale[keep],
        opacities=merged_opacity[keep],
        sh_flat=merged_sh[keep],
    )


def pred_low_conf_mask(pred_slice: dict, thr: float) -> Optional[torch.Tensor]:
    """Return ``(T, H, W)`` bool mask where ``sigmoid(conf) <= thr``.

    Returns ``None`` when ``pred_slice`` has no conf map.
    """
    conf_map = pred_slice.get("conf")
    if conf_map is None:
        return None
    conf_t = conf_map[0]
    if conf_t.ndim == 4 and conf_t.shape[-1] == 1:
        conf_t = conf_t[..., 0]
    elif conf_t.ndim != 3:
        raise ValueError(f"unexpected conf shape {tuple(conf_map.shape)}")
    return torch.sigmoid(conf_t.float()) <= float(thr)


def extract_complement_tensors(
    pred_slice: dict,
    covered: torch.Tensor,
    max_gaussians: Optional[int] = None,
    *,
    budget_selection: str = "topk",
    budget_explore_fraction: float = 0.0,
    budget_sampling_alpha: float = 1.0,
    drop_lowest_renderable: Optional[int] = None,
    render_opacity_threshold: float = 0.01,
    conf_gate: bool = False,
    conf_thr: float = 0.0,
    conf_gate_mode: str = "hard",
) -> WindowMergeTensors:
    """Extract dense complement Gaussians for uncovered pixels (pred_slice already contribution ctx).

    When ``conf_gate`` is set, applies E12-A1 source-side confidence filtering:
      - ``hard``: drop pixels with sigmoid(conf) <= thr
      - ``soft_opacity``: keep pixels but multiply opacity by sigmoid(conf)

    Always attaches ``conf_scores`` (sigmoid conf per kept Gaussian) when ``pred["conf"]``
    is present, so ungated merges can still visualize which points a hard gate would drop.
    """
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
    conf_list = []
    conf_map = pred_slice.get("conf")
    gate_on = bool(conf_gate) and conf_map is not None
    mode = str(conf_gate_mode)
    thr = float(conf_thr)

    for local_t in range(t_len):
        uncovered = ~covered[local_t]
        conf_s = None
        if conf_map is not None:
            conf_t = conf_map[0, local_t]
            if conf_t.ndim == 3 and conf_t.shape[-1] == 1:
                conf_t = conf_t[..., 0]
            conf_s = torch.sigmoid(conf_t.float())
            if gate_on and mode == "hard":
                uncovered = uncovered & (conf_s > thr)
        if not bool(uncovered.any()):
            continue
        pts = pred_slice["points"][0, local_t]
        if pred_slice.get("gaussian_offsets") is not None:
            pts = pts + pred_slice["gaussian_offsets"][0, local_t].permute(1, 2, 0)
        scales = pred_slice["gaussian_scales"][0, local_t].permute(1, 2, 0)
        quats = pred_slice["gaussian_rotations"][0, local_t].permute(1, 2, 0)
        opacs = pred_slice["gaussian_opacities"][0, local_t, 0]
        sh = pred_slice["gaussian_sh_coeffs"][0, local_t].permute(1, 2, 0)

        means_list.append(pts[uncovered].reshape(-1, 3))
        scales_list.append(scales[uncovered].reshape(-1, 3))
        quats_list.append(quats[uncovered].reshape(-1, 4))
        opac_sel = opacs[uncovered].reshape(-1)
        if gate_on and mode == "soft_opacity" and conf_s is not None:
            opac_sel = opac_sel * conf_s[uncovered].reshape(-1)
        opacs_list.append(opac_sel)
        sh_list.append(sh[uncovered].reshape(-1, sh_dim))
        if conf_s is not None:
            conf_list.append(conf_s[uncovered].reshape(-1))

    if not means_list:
        if drop_lowest_renderable is not None and drop_lowest_renderable > 0:
            raise RuntimeError(
                "R102 budget infeasible: tracklet splits have no residuals to pay for them"
            )
        return WindowMergeTensors.empty(device, sh_dim=sh_dim)

    means = torch.cat(means_list, dim=0)
    quats = torch.cat(quats_list, dim=0)
    scales = torch.cat(scales_list, dim=0)
    opacities = torch.cat(opacs_list, dim=0)
    sh_flat = torch.cat(sh_list, dim=0)
    conf_scores = torch.cat(conf_list, dim=0) if conf_list else None
    budget_stats = None
    if max_gaussians is not None and means.shape[0] > max_gaussians:
        if max_gaussians <= 0:
            empty = WindowMergeTensors.empty(device, sh_dim=sh_dim)
            empty.budget_selection_stats = {
                "mode": str(budget_selection), "topk_kept": 0, "explore_kept": 0
            }
            return empty
        budget = int(max_gaussians)
        mode = str(budget_selection)
        if mode == "topk":
            keep = torch.topk(opacities.detach(), k=budget, sorted=False).indices
            topk_kept, explore_kept = budget, 0
        elif mode == "mixed_opacity":
            explore_kept = min(
                budget, max(0, round(budget * float(budget_explore_fraction)))
            )
            topk_kept = budget - explore_kept
            topk = (
                torch.topk(opacities.detach(), k=topk_kept, sorted=False).indices
                if topk_kept
                else torch.empty(0, dtype=torch.long, device=device)
            )
            available = torch.ones(opacities.shape[0], dtype=torch.bool, device=device)
            available[topk] = False
            pool = available.nonzero(as_tuple=True)[0]
            weights = opacities.detach()[pool].clamp_min(0).pow(
                float(budget_sampling_alpha)
            )
            if not bool((weights > 0).any()):
                weights = torch.ones_like(weights)
            explore = pool[torch.multinomial(weights, explore_kept, replacement=False)]
            keep = torch.cat([topk, explore])
        else:
            raise ValueError(f"unsupported hybrid budget selection: {mode}")
        means, quats, scales, opacities, sh_flat = (
            means[keep], quats[keep], scales[keep], opacities[keep], sh_flat[keep]
        )
        if conf_scores is not None:
            conf_scores = conf_scores[keep]
        budget_stats = {
            "mode": mode,
            "topk_kept": int(topk_kept),
            "explore_kept": int(explore_kept),
        }
    if drop_lowest_renderable is not None and drop_lowest_renderable > 0:
        renderable = (opacities > render_opacity_threshold).nonzero(as_tuple=True)[0]
        if renderable.numel() < drop_lowest_renderable:
            raise RuntimeError(
                "R102 budget infeasible: fewer renderable residuals than admitted tracklet splits"
            )
        drop_local = torch.topk(
            opacities[renderable].detach(), k=int(drop_lowest_renderable), largest=False
        ).indices
        keep_mask = torch.ones(opacities.shape[0], dtype=torch.bool, device=opacities.device)
        keep_mask[renderable[drop_local]] = False
        means, quats, scales, opacities, sh_flat = (
            means[keep_mask], quats[keep_mask], scales[keep_mask],
            opacities[keep_mask], sh_flat[keep_mask],
        )
        if conf_scores is not None:
            conf_scores = conf_scores[keep_mask]
    return WindowMergeTensors(
        means, quats, scales, opacities, sh_flat,
        conf_scores=conf_scores,
        budget_selection_stats=budget_stats,
    )


def _preferred_cat_device(chunks: Sequence[torch.Tensor]) -> torch.device:
    """Prefer CUDA when any chunk is on CUDA (CPU-spilled mix at finalize)."""
    for tensor in reversed(list(chunks)):
        if tensor.is_cuda:
            return tensor.device
    return chunks[0].device if chunks else torch.device("cpu")


def _cat_tensor_chunks(
    means: List[torch.Tensor],
    quats: List[torch.Tensor],
    scales: List[torch.Tensor],
    opacities: List[torch.Tensor],
    sh_flat: List[torch.Tensor],
) -> WindowMergeTensors:
    if not means:
        device = torch.device("cpu")
        return WindowMergeTensors.empty(device)
    device = _preferred_cat_device(means)
    return WindowMergeTensors(
        means=torch.cat([t.to(device, non_blocking=False) for t in means], dim=0),
        quats=torch.cat([t.to(device, non_blocking=False) for t in quats], dim=0),
        scales=torch.cat([t.to(device, non_blocking=False) for t in scales], dim=0),
        opacities=torch.cat([t.to(device, non_blocking=False) for t in opacities], dim=0),
        sh_flat=torch.cat([t.to(device, non_blocking=False) for t in sh_flat], dim=0),
    )


def window_merge_tensors_to_device(
    tensors: WindowMergeTensors, device: torch.device
) -> WindowMergeTensors:
    """Move a window cloud to ``device`` (used for CPU spill / reload)."""
    conf = tensors.conf_scores
    return WindowMergeTensors(
        means=tensors.means.to(device, non_blocking=False),
        quats=tensors.quats.to(device, non_blocking=False),
        scales=tensors.scales.to(device, non_blocking=False),
        opacities=tensors.opacities.to(device, non_blocking=False),
        sh_flat=tensors.sh_flat.to(device, non_blocking=False),
        base_compact_count=tensors.base_compact_count,
        tracklet_split_count=tensors.tracklet_split_count,
        flatk_k2_count=tensors.flatk_k2_count,
        flatk_k4_count=tensors.flatk_k4_count,
        flatk_trace_digest=tensors.flatk_trace_digest,
        nested_k2_count=tensors.nested_k2_count,
        nested_k4_count=tensors.nested_k4_count,
        nested_trace_digest=tensors.nested_trace_digest,
        conf_scores=None if conf is None else conf.to(device, non_blocking=False),
    )


@dataclass
class GlobalCloudAccumulator:
    """Concatenate per-window compact + complement into one global cloud."""

    compact_means: List[torch.Tensor] = field(default_factory=list)
    compact_quats: List[torch.Tensor] = field(default_factory=list)
    compact_scales: List[torch.Tensor] = field(default_factory=list)
    compact_opacities: List[torch.Tensor] = field(default_factory=list)
    compact_sh_flat: List[torch.Tensor] = field(default_factory=list)
    complement_means: List[torch.Tensor] = field(default_factory=list)
    complement_quats: List[torch.Tensor] = field(default_factory=list)
    complement_scales: List[torch.Tensor] = field(default_factory=list)
    complement_opacities: List[torch.Tensor] = field(default_factory=list)
    complement_sh_flat: List[torch.Tensor] = field(default_factory=list)
    compact_window_ids: List[torch.Tensor] = field(default_factory=list)
    complement_window_ids: List[torch.Tensor] = field(default_factory=list)
    complement_conf_scores: List[torch.Tensor] = field(default_factory=list)
    num_compact: int = 0
    num_complement: int = 0
    num_tracklet_splits: int = 0
    num_tracklet_base_compact: int = 0
    num_flatk_k2: int = 0
    num_flatk_k4: int = 0
    flatk_trace_digests: List[str] = field(default_factory=list)
    num_nested_k2: int = 0
    num_nested_k4: int = 0
    nested_trace_digests: List[str] = field(default_factory=list)

    def append(
        self,
        tensors: WindowMergeTensors,
        *,
        is_complement: bool = False,
        window_idx: int = -1,
    ) -> None:
        if tensors.num_gaussians == 0:
            return
        window_ids = torch.full(
            (tensors.num_gaussians,),
            int(window_idx),
            dtype=torch.int32,
            device=tensors.means.device,
        )
        if is_complement:
            self.complement_means.append(tensors.means)
            self.complement_quats.append(tensors.quats)
            self.complement_scales.append(tensors.scales)
            self.complement_opacities.append(tensors.opacities)
            self.complement_sh_flat.append(tensors.sh_flat)
            self.complement_window_ids.append(window_ids)
            if tensors.conf_scores is not None and tensors.conf_scores.numel() == tensors.num_gaussians:
                self.complement_conf_scores.append(tensors.conf_scores)
            else:
                self.complement_conf_scores.append(
                    torch.full(
                        (tensors.num_gaussians,),
                        float("nan"),
                        device=tensors.means.device,
                        dtype=tensors.means.dtype,
                    )
                )
            self.num_complement += tensors.num_gaussians
        else:
            self.compact_window_ids.append(window_ids)
            self.compact_means.append(tensors.means)
            self.compact_quats.append(tensors.quats)
            self.compact_scales.append(tensors.scales)
            self.compact_opacities.append(tensors.opacities)
            self.compact_sh_flat.append(tensors.sh_flat)
            self.num_compact += tensors.num_gaussians
            self.num_tracklet_splits += tensors.tracklet_split_count
            self.num_tracklet_base_compact += (
                tensors.base_compact_count
                if tensors.base_compact_count is not None else tensors.num_gaussians
            )
            self.num_flatk_k2 += tensors.flatk_k2_count
            self.num_flatk_k4 += tensors.flatk_k4_count
            if tensors.flatk_trace_digest is not None:
                self.flatk_trace_digests.append(tensors.flatk_trace_digest)
            self.num_nested_k2 += tensors.nested_k2_count
            self.num_nested_k4 += tensors.nested_k4_count
            if tensors.nested_trace_digest is not None:
                self.nested_trace_digests.append(tensors.nested_trace_digest)

    def stacked_compact(self) -> WindowMergeTensors:
        return _cat_tensor_chunks(
            self.compact_means,
            self.compact_quats,
            self.compact_scales,
            self.compact_opacities,
            self.compact_sh_flat,
        )

    def stacked_complement(self) -> WindowMergeTensors:
        out = _cat_tensor_chunks(
            self.complement_means,
            self.complement_quats,
            self.complement_scales,
            self.complement_opacities,
            self.complement_sh_flat,
        )
        conf = self.stacked_complement_conf_scores()
        if conf.numel() == out.num_gaussians:
            out.conf_scores = conf
        return out

    def _empty_id_device(self) -> torch.device:
        """Prefer a live stream device so empty ids match non-empty siblings."""
        for chunks in (self.compact_means, self.complement_means):
            if chunks:
                return _preferred_cat_device(chunks)
        return torch.device("cpu")

    def stacked_compact_window_ids(self) -> torch.Tensor:
        if not self.compact_window_ids:
            return torch.empty(0, dtype=torch.int32, device=self._empty_id_device())
        device = _preferred_cat_device(self.compact_window_ids)
        return torch.cat(
            [t.to(device, non_blocking=False) for t in self.compact_window_ids], dim=0
        )

    def stacked_complement_window_ids(self) -> torch.Tensor:
        if not self.complement_window_ids:
            return torch.empty(0, dtype=torch.int32, device=self._empty_id_device())
        device = _preferred_cat_device(self.complement_window_ids)
        return torch.cat(
            [t.to(device, non_blocking=False) for t in self.complement_window_ids], dim=0
        )

    def stacked_complement_conf_scores(self) -> torch.Tensor:
        if not self.complement_conf_scores:
            return torch.empty(0, device=self._empty_id_device())
        device = _preferred_cat_device(self.complement_conf_scores)
        return torch.cat(
            [t.to(device, non_blocking=False) for t in self.complement_conf_scores], dim=0
        )

    @staticmethod
    def _spill_list_to_cpu(chunks: List[torch.Tensor], keep_last: int) -> int:
        """Move all but the last ``keep_last`` chunks to CPU. Returns spilled count."""
        if keep_last < 0:
            keep_last = 0
        n = len(chunks)
        spill_end = max(n - keep_last, 0)
        moved = 0
        for i in range(spill_end):
            tensor = chunks[i]
            if tensor.device.type != "cpu":
                chunks[i] = tensor.to("cpu", non_blocking=False)
                moved += 1
        return moved

    def spill_old_chunks_to_cpu(self, keep_last_windows: int) -> Dict[str, int]:
        """Keep only the newest ``keep_last_windows`` append-chunks on GPU.

        Each append is one window's compact or complement block, so this roughly
        bounds GPU-resident merge clouds to ~``keep_last_windows`` windows.
        """
        return {
            "compact_chunks_moved": self._spill_list_to_cpu(
                self.compact_means, keep_last_windows
            )
            + self._spill_list_to_cpu(self.compact_quats, keep_last_windows)
            + self._spill_list_to_cpu(self.compact_scales, keep_last_windows)
            + self._spill_list_to_cpu(self.compact_opacities, keep_last_windows)
            + self._spill_list_to_cpu(self.compact_sh_flat, keep_last_windows)
            + self._spill_list_to_cpu(self.compact_window_ids, keep_last_windows),
            "complement_chunks_moved": self._spill_list_to_cpu(
                self.complement_means, keep_last_windows
            )
            + self._spill_list_to_cpu(self.complement_quats, keep_last_windows)
            + self._spill_list_to_cpu(self.complement_scales, keep_last_windows)
            + self._spill_list_to_cpu(self.complement_opacities, keep_last_windows)
            + self._spill_list_to_cpu(self.complement_sh_flat, keep_last_windows)
            + self._spill_list_to_cpu(self.complement_window_ids, keep_last_windows)
            + self._spill_list_to_cpu(self.complement_conf_scores, keep_last_windows),
            "compact_chunks": len(self.compact_means),
            "complement_chunks": len(self.complement_means),
            "keep_last_windows": int(keep_last_windows),
        }

    def stacked(self) -> WindowMergeTensors:
        return stack_window_hybrid_tensors(self.stacked_compact(), self.stacked_complement())

    def ema_update_attributes(
        self,
        *,
        streams: torch.Tensor,
        chunks: torch.Tensor,
        local_indices: torch.Tensor,
        incoming_opacities: torch.Tensor,
        incoming_sh_flat: torch.Tensor,
        alpha: float,
    ) -> int:
        """EMA opacity and SH-DC of existing points, grouped by target.

        This deliberately leaves mean, scale, and rotation unchanged.  Grouping
        prevents a window with many redundant observations from receiving a
        larger update simply because it contains more duplicate samples.
        """
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        n = int(local_indices.numel())
        if not (streams.shape == chunks.shape == local_indices.shape == incoming_opacities.shape):
            raise ValueError("target locations and incoming opacities must have matching shape")
        if incoming_sh_flat.ndim != 2 or incoming_sh_flat.shape[0] != n or incoming_sh_flat.shape[1] < 3:
            raise ValueError("incoming_sh_flat must have shape (N, >=3)")
        if n == 0:
            return 0
        if bool((streams < 0).any()) or bool((chunks < 0).any()) or bool((local_indices < 0).any()):
            raise ValueError("attribute updates require valid accumulator locations")
        # Compact/complement chunk indices are separately addressed, so encode
        # the small categorical tuple into a unique row key on-device.
        key = (streams.to(torch.long) << 48) + (chunks.to(torch.long) << 24) + local_indices.to(torch.long)
        unique, inverse = torch.unique(key, sorted=True, return_inverse=True)
        count = torch.bincount(inverse, minlength=unique.numel()).to(incoming_opacities.dtype)
        opacity_sum = torch.zeros(unique.numel(), device=incoming_opacities.device, dtype=incoming_opacities.dtype)
        opacity_sum.index_add_(0, inverse, incoming_opacities.detach())
        dc_sum = torch.zeros(unique.numel(), 3, device=incoming_sh_flat.device, dtype=incoming_sh_flat.dtype)
        dc_sum.index_add_(0, inverse, incoming_sh_flat[:, :3].detach())
        mean_opacity = opacity_sum / count.clamp_min(1)
        mean_dc = dc_sum / count[:, None].clamp_min(1)
        first = torch.full((unique.numel(),), n, dtype=torch.long, device=unique.device)
        first.scatter_reduce_(0, inverse, torch.arange(n, device=unique.device), reduce="amin", include_self=True)
        with torch.no_grad():
            for stream in (0, 1):
                select = streams[first] == stream
                if not bool(select.any()):
                    continue
                source_opacity = self.compact_opacities if stream == 0 else self.complement_opacities
                source_sh = self.compact_sh_flat if stream == 0 else self.complement_sh_flat
                group_chunks = chunks[first][select]
                group_local = local_indices[first][select]
                group_opacity = mean_opacity[select]
                group_dc = mean_dc[select]
                for chunk_idx in torch.unique(group_chunks).tolist():
                    in_chunk = group_chunks == int(chunk_idx)
                    target = group_local[in_chunk]
                    source_opacity[int(chunk_idx)][target] = torch.lerp(
                        source_opacity[int(chunk_idx)][target], group_opacity[in_chunk], float(alpha)
                    )
                    source_sh[int(chunk_idx)][target, :3] = torch.lerp(
                        source_sh[int(chunk_idx)][target, :3], group_dc[in_chunk], float(alpha)
                    )
        return int(unique.numel())

    def deduplicate_complement_keep_one(
        self,
        edge: float,
        *,
        policy: str = "max_opacity",
    ) -> Dict[str, object]:
        """In-place voxel keep-one on accumulated complement Gaussians."""
        from loger.merge.global_voxel_dedup import select_voxel_keep_one

        complement = self.stacked_complement()
        n_in = complement.num_gaussians
        if n_in == 0 or edge <= 0:
            return {
                "edge": float(edge),
                "policy": policy,
                "input": n_in,
                "output": n_in,
                "pruned": 0,
                "keep_ratio": 1.0 if n_in == 0 else 1.0,
            }

        keep = select_voxel_keep_one(
            complement.means,
            complement.opacities,
            edge=float(edge),
            policy=policy,  # type: ignore[arg-type]
        )
        deduped = WindowMergeTensors(
            means=complement.means[keep],
            quats=complement.quats[keep],
            scales=complement.scales[keep],
            opacities=complement.opacities[keep],
            sh_flat=complement.sh_flat[keep],
        )
        window_ids = self.stacked_complement_window_ids()
        if window_ids.numel() == n_in:
            self.complement_window_ids = [window_ids[keep.to(window_ids.device)]]
        conf_scores = self.stacked_complement_conf_scores()
        if conf_scores.numel() == n_in:
            self.complement_conf_scores = [conf_scores[keep.to(conf_scores.device)]]
            deduped.conf_scores = self.complement_conf_scores[0]
        else:
            self.complement_conf_scores = []
        self.complement_means = [deduped.means]
        self.complement_quats = [deduped.quats]
        self.complement_scales = [deduped.scales]
        self.complement_opacities = [deduped.opacities]
        self.complement_sh_flat = [deduped.sh_flat]
        n_out = deduped.num_gaussians
        self.num_complement = n_out
        return {
            "edge": float(edge),
            "policy": policy,
            "input": n_in,
            "output": n_out,
            "pruned": n_in - n_out,
            "keep_ratio": float(n_out) / float(max(n_in, 1)),
        }


def merge_result_from_tensors(
    tensors: WindowMergeTensors,
    camera_poses: torch.Tensor,
    stats: Optional[Dict[str, object]] = None,
) -> WindowMergeResult:
    """Wrap ``WindowMergeTensors`` as a ``WindowMergeResult`` for rendering."""
    return WindowMergeResult(
        means=tensors.means,
        quats=tensors.quats,
        scales=tensors.scales,
        opacities=tensors.opacities,
        sh_flat=tensors.sh_flat,
        camera_poses=camera_poses,
        stats=dict(stats or {}),
    )


def apply_global_complement_voxel_dedup(
    accumulator: GlobalCloudAccumulator,
    *,
    factor: float,
    policy: str = "first",
    min_windows: int = 0,
    num_windows: int = 0,
    min_complement: int = 0,
) -> Dict[str, object]:
    """Deploy-faithful complement voxel dedup using first-window compact median-NN."""
    from loger.merge.vectorized_merge import estimate_median_1nn

    complement = accumulator.stacked_complement()
    if factor <= 0:
        return {"applied": False, "reason": "factor_disabled"}
    if num_windows < min_windows:
        return {"applied": False, "reason": "min_windows"}
    if complement.num_gaussians < min_complement:
        return {"applied": False, "reason": "min_complement"}

    compact = accumulator.stacked_compact()
    if compact.num_gaussians == 0:
        return {"applied": False, "reason": "no_compact"}

    first_compact = WindowMergeTensors(
        means=accumulator.compact_means[0],
        quats=accumulator.compact_quats[0],
        scales=accumulator.compact_scales[0],
        opacities=accumulator.compact_opacities[0],
        sh_flat=accumulator.compact_sh_flat[0],
    )
    med_nn = estimate_median_1nn(first_compact.means)
    edge = float(factor) * float(med_nn)
    stats = accumulator.deduplicate_complement_keep_one(edge, policy=policy)
    stats["applied"] = True
    stats["factor"] = float(factor)
    stats["median_nn"] = float(med_nn)
    return stats


@dataclass
class PerWindowCloudRecord:
    """One window's local hybrid cloud before global accumulation."""

    window_idx: int
    start_idx: int
    end_idx: int
    contrib_start: int
    contrib_end: int
    compact: WindowMergeTensors
    complement: WindowMergeTensors


@dataclass
class WindowMergeResult:
    """Output of Pi3.forward(track_merge=True)."""

    means: torch.Tensor
    quats: torch.Tensor
    scales: torch.Tensor
    opacities: torch.Tensor
    sh_flat: torch.Tensor
    camera_poses: torch.Tensor
    stats: Dict[str, object] = field(default_factory=dict)
    # Dense bypass for merge-train geometric regs (contrib ∩ ctx frames only).
    dense_scales: Optional[torch.Tensor] = None
    dense_means: Optional[torch.Tensor] = None
    dense_offsets: Optional[torch.Tensor] = None
    gate_margin_loss: Optional[torch.Tensor] = None
    per_window_results: Optional[List["WindowMergeResult"]] = None
    per_window_compact: Optional[List["WindowMergeResult"]] = None
    per_window_complement: Optional[List["WindowMergeResult"]] = None
    global_compact: Optional[WindowMergeTensors] = None
    global_complement: Optional[WindowMergeTensors] = None
    global_compact_window_ids: Optional[torch.Tensor] = None
    global_complement_window_ids: Optional[torch.Tensor] = None
    global_complement_conf: Optional[torch.Tensor] = None

    @property
    def num_gaussians(self) -> int:
        return int(self.means.shape[0])


def is_window_merge_result(obj: object) -> bool:
    return isinstance(obj, WindowMergeResult)


def assemble_camera_poses(pose_by_frame: Dict[int, torch.Tensor], num_frames: int) -> torch.Tensor:
    """Build (1, N, 4, 4) camera pose tensor from contribution slices."""
    if not pose_by_frame:
        raise ValueError("No camera poses collected for window merge")
    device = next(iter(pose_by_frame.values())).device
    dtype = next(iter(pose_by_frame.values())).dtype
    out = torch.zeros(1, num_frames, 4, 4, device=device, dtype=dtype)
    out[0, :, :, :] = torch.eye(4, device=device, dtype=dtype)
    for frame_idx, pose in pose_by_frame.items():
        out[0, frame_idx] = pose
    return out
