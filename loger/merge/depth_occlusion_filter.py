"""Frame-dependent occlusion filtering for global compact Gaussians.

E6 showed that static point-cloud properties (multi-window voxel support, scale)
cannot separate harmful Gaussians from useful ones, because the damage is
*frame-dependent*: a drifted window cloud stays internally consistent yet renders
in front of the correct surface for frames owned by other windows.

This module scores each Gaussian against the per-frame owner-window reference
depth. A Gaussian that repeatedly projects well in front of the reference surface
is an occluder; one that repeatedly lands on it is real geometry.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

from loger.merge.window_merge import WindowMergeTensors


class OcclusionVoteAccumulator:
    """Accumulate per-Gaussian occluder / agreement votes across frames."""

    def __init__(self, num_gaussians: int, device: torch.device) -> None:
        self.occluder = torch.zeros(num_gaussians, device=device, dtype=torch.int32)
        self.agree = torch.zeros(num_gaussians, device=device, dtype=torch.int32)
        self.visible = torch.zeros(num_gaussians, device=device, dtype=torch.int32)

    def add_frame(
        self,
        means: torch.Tensor,
        camera_pose: torch.Tensor,
        K: torch.Tensor,
        ref_depth: torch.Tensor,
        ref_alpha: torch.Tensor,
        *,
        rel_margin: float,
        alpha_threshold: float,
        near_plane: float = 1e-4,
        candidate_mask: Optional[torch.Tensor] = None,
        min_rel_gap: Optional[float] = None,
    ) -> Optional[torch.Tensor]:
        """Vote using one frame's owner-window reference depth map.

        ``ref_depth`` and ``ref_alpha`` are ``(H, W)`` maps rendered from the
        window that owns this frame; ``camera_pose`` is its ``(4, 4)``
        camera-to-world matrix.

        If ``candidate_mask`` is set, only those Gaussians receive votes
        (foreign-only filters). If ``min_rel_gap`` is set, occluder votes
        require ``(depth_ref - z) / depth_ref >= min_rel_gap`` instead of
        the weaker ``rel_margin`` front threshold.

        Returns per-Gaussian relative gap ``(depth_ref - z) / depth_ref`` for
        valid candidates (NaN elsewhere) when useful for diagnostics.
        """
        height, width = ref_depth.shape[-2:]
        viewmat = torch.inverse(camera_pose.float())
        points_cam = means @ viewmat[:3, :3].T + viewmat[:3, 3]
        z = points_cam[:, 2]

        uv = points_cam @ K.float().T
        in_front = z > near_plane
        safe_z = torch.where(in_front, z, torch.ones_like(z))
        u = torch.round(uv[:, 0] / safe_z).long()
        v = torch.round(uv[:, 1] / safe_z).long()

        inside = in_front & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        flat_index = (v.clamp(0, height - 1) * width + u.clamp(0, width - 1))
        depth_ref = ref_depth.reshape(-1)[flat_index]
        alpha_ref = ref_alpha.reshape(-1)[flat_index]

        valid = inside & (alpha_ref > alpha_threshold) & (depth_ref > near_plane)
        if candidate_mask is not None:
            valid = valid & candidate_mask.to(device=valid.device, dtype=torch.bool)
        self.visible += valid.to(torch.int32)

        margin = depth_ref * float(rel_margin)
        rel_gap = (depth_ref - z) / depth_ref.clamp(min=near_plane)
        if min_rel_gap is None:
            occluder = valid & (z < depth_ref - margin)
        else:
            occluder = valid & (rel_gap >= float(min_rel_gap))
        self.occluder += occluder.to(torch.int32)
        self.agree += (valid & ((z - depth_ref).abs() <= margin)).to(torch.int32)

        gap_out = torch.full_like(rel_gap, float("nan"))
        gap_out = torch.where(valid, rel_gap, gap_out)
        return gap_out

    def occluder_ratio(self) -> torch.Tensor:
        """Fraction of decisive observations that saw the Gaussian as an occluder."""
        decisive = (self.occluder + self.agree).clamp(min=1)
        return self.occluder.to(torch.float32) / decisive.to(torch.float32)

    def keep_mask(self, *, min_votes: int, ratio_threshold: float) -> torch.Tensor:
        """Drop Gaussians with enough occluder votes and a high occluder ratio."""
        drop = (self.occluder >= int(min_votes)) & (
            self.occluder_ratio() >= float(ratio_threshold)
        )
        return ~drop

    def summary(self) -> Dict[str, float]:
        ratio = self.occluder_ratio()
        seen = self.visible > 0
        return {
            "mean_visible_frames": float(self.visible.to(torch.float32).mean().item()),
            "frac_never_visible": float((~seen).to(torch.float32).mean().item()),
            "mean_occluder_votes": float(self.occluder.to(torch.float32).mean().item()),
            "mean_agree_votes": float(self.agree.to(torch.float32).mean().item()),
            "mean_occluder_ratio": float(ratio[seen].mean().item()) if bool(seen.any()) else float("nan"),
            "frac_ratio_ge_0.5": float((ratio >= 0.5).to(torch.float32).mean().item()),
            "frac_ratio_ge_0.8": float((ratio >= 0.8).to(torch.float32).mean().item()),
        }


def apply_keep_mask(
    tensors: WindowMergeTensors,
    keep: torch.Tensor,
) -> WindowMergeTensors:
    return WindowMergeTensors(
        means=tensors.means[keep],
        quats=tensors.quats[keep],
        scales=tensors.scales[keep],
        opacities=tensors.opacities[keep],
        sh_flat=tensors.sh_flat[keep],
    )


def filter_by_occlusion_votes(
    tensors: WindowMergeTensors,
    accumulator: OcclusionVoteAccumulator,
    *,
    min_votes: int,
    ratio_threshold: float,
    extra_stats: Optional[Dict[str, object]] = None,
    allow_empty: bool = False,
) -> Tuple[WindowMergeTensors, Dict[str, object]]:
    """Remove voted occluders, returning the surviving cloud plus statistics."""
    n_in = tensors.num_gaussians
    keep = accumulator.keep_mask(min_votes=min_votes, ratio_threshold=ratio_threshold)
    kept = int(keep.sum().item())
    stats: Dict[str, object] = {
        "min_votes": int(min_votes),
        "ratio_threshold": float(ratio_threshold),
        "input": n_in,
        "output": kept,
        "removed": n_in - kept,
        "keep_ratio": float(kept) / float(max(n_in, 1)),
        "removed_mean_scale": _masked_mean(tensors.scales.mean(dim=-1), ~keep),
        "kept_mean_scale": _masked_mean(tensors.scales.mean(dim=-1), keep),
        "removed_mean_opacity": _masked_mean(tensors.opacities, ~keep),
        "kept_mean_opacity": _masked_mean(tensors.opacities, keep),
    }
    if extra_stats:
        stats.update(extra_stats)
    if kept == 0 and not allow_empty:
        raise RuntimeError(
            f"occlusion filter removed every Gaussian "
            f"(min_votes={min_votes}, ratio_threshold={ratio_threshold})"
        )
    return apply_keep_mask(tensors, keep), stats


def removed_window_breakdown(
    keep: torch.Tensor,
    window_ids: torch.Tensor,
    owner_windows: Sequence[int],
) -> Dict[str, float]:
    """Share of removals coming from windows that own at least one frame."""
    removed = ~keep
    if not bool(removed.any()):
        return {"removed_fraction": 0.0}
    owner_set = torch.tensor(sorted(set(int(w) for w in owner_windows)), device=keep.device)
    is_owner_window = torch.isin(window_ids.to(keep.device), owner_set)
    return {
        "removed_fraction": float(removed.to(torch.float32).mean().item()),
        "removed_from_owner_windows": float(
            (removed & is_owner_window).sum().item() / max(int(removed.sum().item()), 1)
        ),
    }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return float("nan")
    return float(values[mask].to(torch.float32).mean().item())


def _owner_frame_groups(
    ctx_list: Sequence[int],
    vote_frames: Sequence[int],
    window_size: int,
    overlap_size: int,
    merge_context_count: Optional[int],
) -> Tuple[Dict[int, List[int]], Optional[Dict[int, int]]]:
    from loger.merge.window_merge import owner_window_for_frame, sliding_window_ranges

    window_frames = (
        int(merge_context_count)
        if merge_context_count is not None
        else len(ctx_list)
    )
    windows = sliding_window_ranges(window_frames, window_size, overlap_size)
    ctx_to_local = (
        {int(g): i for i, g in enumerate(ctx_list)}
        if merge_context_count is not None
        else None
    )
    groups: Dict[int, List[int]] = {}
    for frame in vote_frames:
        if ctx_to_local is not None:
            owner = owner_window_for_frame(
                ctx_to_local[int(frame)], windows, overlap_size
            )
        else:
            owner = owner_window_for_frame(int(frame), windows, overlap_size)
        groups.setdefault(owner, []).append(int(frame))
    return groups, ctx_to_local


def _accumulate_owner_depth_votes(
    target: WindowMergeTensors,
    accumulator: OcclusionVoteAccumulator,
    renderer,
    per_window_by_idx: Dict[int, object],
    owner_groups: Dict[int, List[int]],
    camera_poses: torch.Tensor,
    K: torch.Tensor,
    height: int,
    width: int,
    ctx_to_local: Optional[Dict[int, int]],
    *,
    rel_margin: float,
    alpha_threshold: float,
    render_chunk: int,
) -> None:
    with torch.no_grad():
        for owner, frames in owner_groups.items():
            local = per_window_by_idx.get(owner)
            if local is None or local.num_gaussians == 0:
                continue
            for start in range(0, len(frames), int(render_chunk)):
                chunk = frames[start : start + int(render_chunk)]
                render_indices = (
                    [ctx_to_local[int(f)] for f in chunk]
                    if ctx_to_local is not None
                    else chunk
                )
                out = renderer.render_hybrid(
                    local, target_indices=render_indices, K=K, H=height, W=width
                )
                depth = out["depth"].squeeze(-1)
                alpha = out["alpha"].squeeze(-1)
                for offset, frame in enumerate(chunk):
                    accumulator.add_frame(
                        target.means,
                        camera_poses[frame],
                        K,
                        depth[offset],
                        alpha[offset],
                        rel_margin=rel_margin,
                        alpha_threshold=alpha_threshold,
                    )


def _rebuild_merge_with_filtered_branch(
    merge_result,
    *,
    compact: WindowMergeTensors,
    complement: Optional[WindowMergeTensors],
    filter_stats_key: str,
    filter_stats: Dict[str, object],
    compact_keep: Optional[torch.Tensor] = None,
    complement_keep: Optional[torch.Tensor] = None,
):
    from loger.merge.window_merge import merge_result_from_tensors, stack_window_hybrid_tensors

    if complement is None or complement.num_gaussians == 0:
        hybrid = compact
    else:
        hybrid = stack_window_hybrid_tensors(compact, complement)
    extra_stats = dict(merge_result.stats)
    extra_stats[filter_stats_key] = filter_stats
    result = merge_result_from_tensors(hybrid, merge_result.camera_poses, stats=extra_stats)
    result.global_compact = compact
    result.global_complement = complement
    if compact_keep is not None:
        compact_ids = merge_result.global_compact_window_ids
        if compact_ids is not None and int(compact_ids.numel()) == int(compact_keep.numel()):
            result.global_compact_window_ids = compact_ids[compact_keep]
        else:
            result.global_compact_window_ids = compact_ids
    else:
        result.global_compact_window_ids = merge_result.global_compact_window_ids
    if complement_keep is not None:
        complement_ids = merge_result.global_complement_window_ids
        if (
            complement_ids is not None
            and int(complement_ids.numel()) == int(complement_keep.numel())
        ):
            result.global_complement_window_ids = complement_ids[complement_keep]
        else:
            result.global_complement_window_ids = complement_ids
    else:
        result.global_complement_window_ids = merge_result.global_complement_window_ids
    result.per_window_results = merge_result.per_window_results
    result.per_window_compact = merge_result.per_window_compact
    result.per_window_complement = merge_result.per_window_complement
    result.dense_scales = merge_result.dense_scales
    result.dense_means = merge_result.dense_means
    result.dense_offsets = merge_result.dense_offsets
    filter_stats["applied"] = True
    return result


def filter_complement_by_owner_compact_depth(
    merge_result,
    renderer,
    images: torch.Tensor,
    K: torch.Tensor,
    ctx_list: Sequence[int],
    *,
    window_size: int,
    overlap_size: int,
    rel_margin: float = 0.05,
    min_votes: int = 2,
    ratio_threshold: float = 0.65,
    alpha_threshold: float = 0.5,
    render_chunk: int = 4,
    max_vote_frames: int = 8,
    merge_context_count: Optional[int] = None,
    variant: str = "complement_m05_v2_r65",
) -> Tuple[object, Dict[str, object]]:
    """Optional complement filter: drop global complement Gaussians that occlude owner compact.

    Reference depth/alpha comes from each frame's owner-window **compact-only**
    cloud, not the hybrid, so complement is judged against track-merged geometry
    instead of against itself.
    """
    compact = merge_result.global_compact
    complement = merge_result.global_complement
    per_window_compact = merge_result.per_window_compact
    if complement is None or complement.num_gaussians == 0:
        return merge_result, {"applied": False, "reason": "empty_complement"}
    if compact is None:
        return merge_result, {"applied": False, "reason": "missing_compact"}
    if not per_window_compact:
        return merge_result, {"applied": False, "reason": "missing_per_window_compact"}

    device = complement.means.device
    height, width = images.shape[-2:]
    vote_frames = list(ctx_list)
    if len(vote_frames) > int(max_vote_frames):
        stride = max(len(vote_frames) // int(max_vote_frames), 1)
        vote_frames = vote_frames[::stride][: int(max_vote_frames)]

    owner_groups, ctx_to_local = _owner_frame_groups(
        ctx_list,
        vote_frames,
        window_size,
        overlap_size,
        merge_context_count,
    )
    by_idx = {int(w.stats["window_idx"]): w for w in per_window_compact}
    poses = merge_result.camera_poses
    if poses.dim() == 4:
        poses = poses[0]

    accumulator = OcclusionVoteAccumulator(complement.num_gaussians, device)
    _accumulate_owner_depth_votes(
        complement,
        accumulator,
        renderer,
        by_idx,
        owner_groups,
        poses,
        K,
        height,
        width,
        ctx_to_local,
        rel_margin=rel_margin,
        alpha_threshold=alpha_threshold,
        render_chunk=render_chunk,
    )
    filtered_complement, stats = filter_by_occlusion_votes(
        complement,
        accumulator,
        min_votes=min_votes,
        ratio_threshold=ratio_threshold,
        allow_empty=True,
        extra_stats={
            "variant": variant,
            "rel_margin": float(rel_margin),
            "vote_frames": len(vote_frames),
            "reference": "owner_window_compact",
        },
    )
    complement_keep = accumulator.keep_mask(
        min_votes=min_votes, ratio_threshold=ratio_threshold
    )
    result = _rebuild_merge_with_filtered_branch(
        merge_result,
        compact=compact,
        complement=filtered_complement,
        filter_stats_key="complement_complement_occlusion_filter",
        filter_stats=stats,
        complement_keep=complement_keep,
    )
    return result, stats


def filter_merge_result_by_owner_depth(
    merge_result,
    renderer,
    images: torch.Tensor,
    K: torch.Tensor,
    ctx_list: Sequence[int],
    *,
    window_size: int,
    overlap_size: int,
    rel_margin: float = 0.05,
    min_votes: int = 2,
    ratio_threshold: float = 0.65,
    alpha_threshold: float = 0.5,
    render_chunk: int = 4,
    max_vote_frames: int = 8,
    merge_context_count: Optional[int] = None,
) -> Tuple[object, Dict[str, object]]:
    """Apply Compact depth filter (m05_v2_r65) occlusion votes to the merged compact cloud.

    Votes are collected on a subset of context frames using each frame's
    owner-window hybrid as the reference depth. Complement is left untouched.
    Returns ``(merge_result, stats)``; the input is returned unchanged when
    the owner-window records are missing.
    """
    compact = merge_result.global_compact
    complement = merge_result.global_complement
    per_window = merge_result.per_window_results
    if compact is None or compact.num_gaussians == 0 or not per_window:
        return merge_result, {"applied": False, "reason": "missing_owner_or_compact"}

    device = compact.means.device
    height, width = images.shape[-2:]
    vote_frames = list(ctx_list)
    if len(vote_frames) > int(max_vote_frames):
        stride = max(len(vote_frames) // int(max_vote_frames), 1)
        vote_frames = vote_frames[::stride][: int(max_vote_frames)]

    owner_groups, ctx_to_local = _owner_frame_groups(
        ctx_list,
        vote_frames,
        window_size,
        overlap_size,
        merge_context_count,
    )
    by_idx = {int(w.stats["window_idx"]): w for w in per_window}
    poses = merge_result.camera_poses
    if poses.dim() == 4:
        poses = poses[0]
    accumulator = OcclusionVoteAccumulator(compact.num_gaussians, device)
    _accumulate_owner_depth_votes(
        compact,
        accumulator,
        renderer,
        by_idx,
        owner_groups,
        poses,
        K,
        height,
        width,
        ctx_to_local,
        rel_margin=rel_margin,
        alpha_threshold=alpha_threshold,
        render_chunk=render_chunk,
    )

    filtered, stats = filter_by_occlusion_votes(
        compact,
        accumulator,
        min_votes=min_votes,
        ratio_threshold=ratio_threshold,
        extra_stats={
            "variant": "m05_v2_r65",
            "rel_margin": float(rel_margin),
            "vote_frames": len(vote_frames),
            "reference": "owner_window_hybrid",
        },
    )
    compact_keep = accumulator.keep_mask(
        min_votes=min_votes, ratio_threshold=ratio_threshold
    )
    result = _rebuild_merge_with_filtered_branch(
        merge_result,
        compact=filtered,
        complement=complement,
        filter_stats_key="depth_occlusion_filter",
        filter_stats=stats,
        compact_keep=compact_keep,
    )
    return result, stats
