"""Foreign-only bidirectional depth filter with robust occlusion votes.

Changes vs the previous filter:
  - count behind observations
  - drop uses occluder/visible AND occluder/decisive
  - 3x3 median depth + relative-spread boundary rejection
  - still never votes same-window (owner) points
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from loger.merge.depth_occlusion_filter import apply_keep_mask
from loger.merge.foreign_depth_filter import GAP_BINS, _hybrid_and_window_ids
from loger.merge.robust_occlusion_votes import RobustOcclusionVoteAccumulator
from loger.merge.window_merge import (
    WindowMergeResult,
    WindowMergeTensors,
    merge_result_from_tensors,
    window_merge_tensors_to_device,
)


def _window_merge_tensor_view(
    tensors: WindowMergeTensors, start: int, end: int
) -> WindowMergeTensors:
    """Slice Gaussian rows without allocating another copy of their storage."""
    return WindowMergeTensors(
        means=tensors.means[start:end],
        quats=tensors.quats[start:end],
        scales=tensors.scales[start:end],
        opacities=tensors.opacities[start:end],
        sh_flat=tensors.sh_flat[start:end],
        conf_scores=(
            None if tensors.conf_scores is None else tensors.conf_scores[start:end]
        ),
    )


def _split_filtered_hybrid_views(
    filtered_hybrid: WindowMergeTensors,
    compact_count: int,
    complement_count: int,
) -> Tuple[WindowMergeTensors, WindowMergeTensors]:
    """Recover compact/complement partitions as views of a filtered hybrid cloud."""
    split = int(compact_count)
    end = split + int(complement_count)
    if end != filtered_hybrid.num_gaussians:
        raise ValueError(
            "filtered partition sizes do not match hybrid: "
            f"{compact_count}+{complement_count}!={filtered_hybrid.num_gaussians}"
        )
    return (
        _window_merge_tensor_view(filtered_hybrid, 0, split),
        _window_merge_tensor_view(filtered_hybrid, split, end),
    )


def _window_result_to_device(
    window: WindowMergeResult, device: torch.device
) -> WindowMergeResult:
    """Move a per-window hybrid cloud (not camera poses) to ``device``."""
    moved = window_merge_tensors_to_device(
        WindowMergeTensors(
            means=window.means,
            quats=window.quats,
            scales=window.scales,
            opacities=window.opacities,
            sh_flat=window.sh_flat,
        ),
        device,
    )
    out = merge_result_from_tensors(moved, window.camera_poses, stats=dict(window.stats or {}))
    return out


def accumulate_robust_votes(
    merge_result: WindowMergeResult,
    per_window_hybrid: Sequence[WindowMergeResult],
    renderer,
    owner_groups: Dict[int, List[int]],
    K: torch.Tensor,
    H: int,
    W: int,
    *,
    rel_margin: float,
    min_rel_gap: float = 0.10,
    alpha_threshold: float = 0.5,
    neighborhood_radius: int = 1,
    min_valid_neighbors: int = 5,
    max_rel_depth_spread: float = 0.15,
    render_chunk: int = 8,
    collect_diag: bool = False,
    render_index_map: Optional[Dict[int, int]] = None,
    stream_hybrid: bool = False,
    gaussian_chunk_size: int = 500_000,
    max_gpu_windows: int = 1,
) -> Tuple[RobustOcclusionVoteAccumulator, object, torch.Tensor, int, int, Dict[str, object]]:
    if not per_window_hybrid:
        raise ValueError("per_window_hybrid is empty; enable record_per_window_clouds")

    hybrid, window_ids, n_compact, n_complement = _hybrid_and_window_ids(merge_result)
    n = hybrid.num_gaussians
    poses = merge_result.camera_poses
    if poses.dim() == 4:
        poses = poses[0]

    # Streaming mode: keep the full hybrid on CPU; vote counters + one (or a few)
    # owner-window clouds live on CUDA.
    if stream_hybrid:
        vote_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hybrid_cpu = window_merge_tensors_to_device(hybrid, torch.device("cpu"))
        window_ids_cpu = window_ids.to(device="cpu", dtype=torch.long)
        means_cpu = hybrid_cpu.means
        acc = RobustOcclusionVoteAccumulator(n, vote_device)
        K_vote = K.to(vote_device)
        poses_vote = poses.to(vote_device)
        per_window = {int(w.stats["window_idx"]): w for w in per_window_hybrid}
        gpu_cache: Dict[int, WindowMergeResult] = {}
        cache_order: List[int] = []
        max_gpu_windows = max(int(max_gpu_windows), 1)

        def _acquire_window(owner: int) -> WindowMergeResult:
            if owner in gpu_cache:
                return gpu_cache[owner]
            while len(cache_order) >= max_gpu_windows:
                old = cache_order.pop(0)
                old_win = gpu_cache.pop(old)
                # Drop GPU residency; original CPU record remains in per_window.
                del old_win
                if vote_device.type == "cuda":
                    torch.cuda.empty_cache()
            src = per_window[owner]
            moved = _window_result_to_device(src, vote_device)
            # Ensure poses used by the renderer live with the gaussians.
            if moved.camera_poses.device != vote_device:
                moved.camera_poses = moved.camera_poses.to(vote_device)
            gpu_cache[owner] = moved
            cache_order.append(owner)
            return moved

        gap_bin_counts = {label: 0 for _, _, label in GAP_BINS}
        vote_pairs_occluder: Dict[Tuple[int, int], int] = defaultdict(int)
        max_window = max(per_window.keys()) if per_window else 0
        late_cutoff = max(max_window - 2, 0)
        chunk = max(int(gaussian_chunk_size), 1)

        with torch.no_grad():
            for owner, frames in owner_groups.items():
                if int(owner) not in per_window:
                    continue
                local = _acquire_window(int(owner))
                if local.num_gaussians == 0:
                    continue
                for start in range(0, len(frames), int(render_chunk)):
                    frame_chunk = frames[start : start + int(render_chunk)]
                    render_indices = (
                        [render_index_map[int(f)] for f in frame_chunk]
                        if render_index_map is not None
                        else list(frame_chunk)
                    )
                    out = renderer.render_hybrid(
                        local, target_indices=render_indices, K=K_vote, H=H, W=W
                    )
                    depth = out["depth"].squeeze(-1)
                    alpha = out["alpha"].squeeze(-1)
                    for offset, frame in enumerate(frame_chunk):
                        pose_g = poses_vote[int(frame)]
                        depth_g = depth[offset]
                        alpha_g = alpha[offset]
                        for gs in range(0, n, chunk):
                            ge = min(gs + chunk, n)
                            means_g = means_cpu[gs:ge].to(vote_device, non_blocking=False)
                            foreign = window_ids_cpu[gs:ge] != int(owner)
                            foreign_g = foreign.to(vote_device)
                            child = RobustOcclusionVoteAccumulator(ge - gs, vote_device)
                            gap = child.add_frame(
                                means_g,
                                pose_g,
                                K_vote,
                                depth_g,
                                alpha_g,
                                rel_margin=float(rel_margin),
                                alpha_threshold=float(alpha_threshold),
                                candidate_mask=foreign_g,
                                min_rel_gap=float(min_rel_gap),
                                neighborhood_radius=int(neighborhood_radius),
                                min_valid_neighbors=int(min_valid_neighbors),
                                max_rel_depth_spread=float(max_rel_depth_spread),
                                gaussian_chunk_size=0,
                            )
                            for field in (
                                "occluder",
                                "agree",
                                "behind",
                                "visible",
                                "rejected_boundary",
                                "projected",
                            ):
                                getattr(acc, field)[gs:ge].add_(getattr(child, field))
                            if collect_diag and gap is not None:
                                valid = torch.isfinite(gap)
                                occ = valid & (gap >= float(min_rel_gap))
                                if bool(occ.any()):
                                    src = window_ids_cpu[gs:ge][occ.cpu()]
                                    for s in src.tolist():
                                        vote_pairs_occluder[(int(owner), int(s))] += 1
                                    gvals = gap[occ].detach().cpu()
                                    for lo, hi, label in GAP_BINS:
                                        gap_bin_counts[label] += int(
                                            ((gvals >= lo) & (gvals < hi)).sum().item()
                                        )
                            del means_g, foreign_g, child
                        if vote_device.type == "cuda":
                            torch.cuda.empty_cache()

        meta: Dict[str, object] = {
            "rel_margin": float(rel_margin),
            "min_rel_gap": float(min_rel_gap),
            "alpha_threshold": float(alpha_threshold),
            "neighborhood_radius": int(neighborhood_radius),
            "min_valid_neighbors": int(min_valid_neighbors),
            "max_rel_depth_spread": float(max_rel_depth_spread),
            "max_window": int(max_window),
            "late_cutoff": int(late_cutoff),
            "gap_bin_counts": gap_bin_counts,
            "vote_pairs_occluder": dict(vote_pairs_occluder),
            "collect_diag": bool(collect_diag),
            "stream_hybrid": True,
            "gaussian_chunk_size": int(chunk),
            "max_gpu_windows": int(max_gpu_windows),
        }
        # Keep hybrid on CPU for apply_keep; move vote masks accordingly later.
        return acc, hybrid_cpu, window_ids_cpu, n_compact, n_complement, meta

    device = hybrid.means.device
    window_ids = window_ids.to(device=device, dtype=torch.long)
    acc = RobustOcclusionVoteAccumulator(n, device)

    per_window = {int(w.stats["window_idx"]): w for w in per_window_hybrid}

    gap_bin_counts = {label: 0 for _, _, label in GAP_BINS}
    vote_pairs_occluder: Dict[Tuple[int, int], int] = defaultdict(int)
    max_window = max(per_window.keys()) if per_window else 0
    late_cutoff = max(max_window - 2, 0)

    with torch.no_grad():
        for owner, frames in owner_groups.items():
            local = per_window.get(int(owner))
            if local is None or local.num_gaussians == 0:
                continue
            foreign = window_ids != int(owner)
            for start in range(0, len(frames), int(render_chunk)):
                chunk = frames[start : start + int(render_chunk)]
                render_indices = (
                    [render_index_map[int(f)] for f in chunk]
                    if render_index_map is not None
                    else list(chunk)
                )
                out = renderer.render_hybrid(
                    local, target_indices=render_indices, K=K, H=H, W=W
                )
                depth = out["depth"].squeeze(-1)
                alpha = out["alpha"].squeeze(-1)
                for offset, frame in enumerate(chunk):
                    gap = acc.add_frame(
                        hybrid.means,
                        poses[int(frame)],
                        K,
                        depth[offset],
                        alpha[offset],
                        rel_margin=float(rel_margin),
                        alpha_threshold=float(alpha_threshold),
                        candidate_mask=foreign,
                        min_rel_gap=float(min_rel_gap),
                        neighborhood_radius=int(neighborhood_radius),
                        min_valid_neighbors=int(min_valid_neighbors),
                        max_rel_depth_spread=float(max_rel_depth_spread),
                        gaussian_chunk_size=int(gaussian_chunk_size),
                    )
                    if collect_diag and gap is not None:
                        valid = torch.isfinite(gap)
                        occ = valid & (gap >= float(min_rel_gap))
                        if bool(occ.any()):
                            src = window_ids[occ]
                            for s in src.tolist():
                                vote_pairs_occluder[(int(owner), int(s))] += 1
                            gvals = gap[occ].detach().cpu()
                            for lo, hi, label in GAP_BINS:
                                gap_bin_counts[label] += int(
                                    ((gvals >= lo) & (gvals < hi)).sum().item()
                                )

    meta = {
        "rel_margin": float(rel_margin),
        "min_rel_gap": float(min_rel_gap),
        "alpha_threshold": float(alpha_threshold),
        "neighborhood_radius": int(neighborhood_radius),
        "min_valid_neighbors": int(min_valid_neighbors),
        "max_rel_depth_spread": float(max_rel_depth_spread),
        "max_window": int(max_window),
        "late_cutoff": int(late_cutoff),
        "gap_bin_counts": gap_bin_counts,
        "vote_pairs_occluder": dict(vote_pairs_occluder),
        "collect_diag": bool(collect_diag),
        "stream_hybrid": False,
    }
    return acc, hybrid, window_ids, n_compact, n_complement, meta


def apply_robust_keep(
    merge_result: WindowMergeResult,
    acc: RobustOcclusionVoteAccumulator,
    hybrid,
    window_ids: torch.Tensor,
    n_compact: int,
    n_complement: int,
    meta: Dict[str, object],
    *,
    min_votes: int,
    visible_ratio_threshold: float,
    decisive_ratio_threshold: float,
    allow_empty: bool = True,
    edge_min_boundary: int = 0,
    edge_boundary_ratio: float = 1.0,
    edge_min_votes: int = 0,
    edge_visible_ratio: float = 0.0,
    edge_decisive_ratio: float = 0.0,
    edge_max_agree: Optional[int] = None,
) -> Tuple[WindowMergeResult, Dict[str, object]]:
    hard_keep = acc.keep_mask(
        min_votes=int(min_votes),
        visible_ratio_threshold=float(visible_ratio_threshold),
        decisive_ratio_threshold=float(decisive_ratio_threshold),
        edge_min_boundary=0,
    )
    keep = acc.keep_mask(
        min_votes=int(min_votes),
        visible_ratio_threshold=float(visible_ratio_threshold),
        decisive_ratio_threshold=float(decisive_ratio_threshold),
        edge_min_boundary=int(edge_min_boundary),
        edge_boundary_ratio=float(edge_boundary_ratio),
        edge_min_votes=int(edge_min_votes),
        edge_visible_ratio=float(edge_visible_ratio),
        edge_decisive_ratio=float(edge_decisive_ratio),
        edge_max_agree=edge_max_agree,
    )
    # Vote counters may live on CUDA while the hybrid cloud was streamed on CPU.
    keep = keep.to(device=hybrid.means.device)
    hard_keep = hard_keep.to(device=hybrid.means.device)
    extra_edge = (~keep) & hard_keep
    n = hybrid.num_gaussians
    removed = ~keep
    kept_n = int(keep.sum().item())
    max_window = int(meta["max_window"])
    late_cutoff = int(meta["late_cutoff"])
    stats: Dict[str, object] = {
        "policy": "robust_foreign_bidirectional",
        "rel_margin": meta["rel_margin"],
        "min_rel_gap": meta["min_rel_gap"],
        "min_votes": int(min_votes),
        "visible_ratio_threshold": float(visible_ratio_threshold),
        "decisive_ratio_threshold": float(decisive_ratio_threshold),
        "alpha_threshold": meta["alpha_threshold"],
        "neighborhood_radius": meta["neighborhood_radius"],
        "min_valid_neighbors": meta["min_valid_neighbors"],
        "max_rel_depth_spread": meta["max_rel_depth_spread"],
        "edge_min_boundary": int(edge_min_boundary),
        "edge_boundary_ratio": float(edge_boundary_ratio),
        "edge_min_votes": int(edge_min_votes),
        "edge_visible_ratio": float(edge_visible_ratio),
        "edge_decisive_ratio": float(edge_decisive_ratio),
        "edge_max_agree": None if edge_max_agree is None else int(edge_max_agree),
        "input": n,
        "output": kept_n,
        "removed": n - kept_n,
        "removed_by_edge_only": int(extra_edge.sum().item()),
        "keep_ratio": float(kept_n) / float(max(n, 1)),
        "vote_summary": acc.summary(),
        "allow_empty": bool(allow_empty),
    }
    if kept_n == 0 and not allow_empty:
        raise RuntimeError("robust foreign depth filter removed every Gaussian")

    if bool(removed.any()):
        rem_ids = window_ids[removed]
        src_hist = torch.bincount(rem_ids, minlength=max_window + 1).tolist()
        stats["removed_by_source_window"] = {
            str(i): int(c) for i, c in enumerate(src_hist) if c > 0
        }
        stats["removed_from_late_windows"] = int((rem_ids >= late_cutoff).sum().item())
        stats["removed_from_late_windows_frac"] = float(
            (rem_ids >= late_cutoff).to(torch.float32).mean().item()
        )
        stats["removed_compact"] = int(removed[:n_compact].sum().item())
        stats["removed_complement"] = int(removed[n_compact:].sum().item())
    else:
        stats["removed_by_source_window"] = {}
        stats["removed_from_late_windows"] = 0
        stats["removed_from_late_windows_frac"] = 0.0
        stats["removed_compact"] = 0
        stats["removed_complement"] = 0

    if meta.get("collect_diag"):
        vote_pairs_occluder = meta["vote_pairs_occluder"]
        gap_bin_counts = meta["gap_bin_counts"]
        stats["gap_bin_occluder_counts"] = gap_bin_counts
        stats["occluder_vote_pairs_owner_source"] = {
            f"{o}->{s}": c for (o, s), c in sorted(vote_pairs_occluder.items())
        }
        early_on_late = 0
        late_on_early = 0
        for (o, s), c in vote_pairs_occluder.items():
            if s >= late_cutoff and o < late_cutoff:
                early_on_late += c
            if s < late_cutoff and o >= late_cutoff:
                late_on_early += c
        stats["early_owner_votes_on_late_source"] = early_on_late
        stats["late_owner_votes_on_early_source"] = late_on_early
        stats["late_window_cutoff"] = int(late_cutoff)
        stats["max_window"] = int(max_window)
        total_pair = early_on_late + late_on_early
        stats["vote_flow_symmetry"] = (
            min(early_on_late, late_on_early) / max(total_pair, 1)
            if total_pair
            else 0.0
        )

    compact_keep = keep[:n_compact]
    complement_keep = keep[n_compact : n_compact + n_complement]
    filtered_hybrid = apply_keep_mask(hybrid, keep)
    compact_output = int(compact_keep.sum().item())
    complement_output = int(complement_keep.sum().item())
    # ``hybrid`` is compact followed by complement. Boolean indexing preserves
    # that order, so the two partitions can share the one filtered allocation.
    # Previously all three clouds were copied, causing a large transient CUDA
    # memory peak for dense scenes.
    filtered_compact, filtered_complement = _split_filtered_hybrid_views(
        filtered_hybrid, compact_output, complement_output
    )
    stats["compact_input"] = n_compact
    stats["compact_output"] = compact_output
    stats["complement_input"] = n_complement
    stats["complement_output"] = complement_output

    result = merge_result_from_tensors(
        filtered_hybrid,
        merge_result.camera_poses,
        stats={**dict(merge_result.stats or {}), "robust_foreign_depth_filter": stats},
    )
    result.global_compact = filtered_compact
    result.global_complement = filtered_complement
    if merge_result.global_compact_window_ids is not None:
        result.global_compact_window_ids = merge_result.global_compact_window_ids[
            compact_keep.to(merge_result.global_compact_window_ids.device)
        ]
    if merge_result.global_complement_window_ids is not None:
        result.global_complement_window_ids = merge_result.global_complement_window_ids[
            complement_keep.to(merge_result.global_complement_window_ids.device)
        ]
    result.per_window_results = merge_result.per_window_results
    result.per_window_compact = merge_result.per_window_compact
    result.per_window_complement = merge_result.per_window_complement
    return result, stats


def robust_foreign_depth_filter(
    merge_result: WindowMergeResult,
    per_window_hybrid: Sequence[WindowMergeResult],
    renderer,
    owner_groups: Dict[int, List[int]],
    K: torch.Tensor,
    H: int,
    W: int,
    *,
    rel_margin: float,
    min_votes: int,
    visible_ratio_threshold: float,
    decisive_ratio_threshold: float,
    min_rel_gap: float = 0.10,
    alpha_threshold: float = 0.5,
    neighborhood_radius: int = 1,
    min_valid_neighbors: int = 5,
    max_rel_depth_spread: float = 0.15,
    render_chunk: int = 8,
    allow_empty: bool = True,
    collect_diag: bool = False,
    render_index_map: Optional[Dict[int, int]] = None,
    edge_min_boundary: int = 0,
    edge_boundary_ratio: float = 1.0,
    edge_min_votes: int = 0,
    edge_visible_ratio: float = 0.0,
    edge_decisive_ratio: float = 0.0,
    edge_max_agree: Optional[int] = None,
    stream_hybrid: bool = False,
    gaussian_chunk_size: int = 500_000,
    max_gpu_windows: int = 1,
) -> Tuple[WindowMergeResult, Dict[str, object]]:
    acc, hybrid, window_ids, n_compact, n_complement, meta = accumulate_robust_votes(
        merge_result,
        per_window_hybrid,
        renderer,
        owner_groups,
        K,
        H,
        W,
        rel_margin=rel_margin,
        min_rel_gap=min_rel_gap,
        alpha_threshold=alpha_threshold,
        neighborhood_radius=neighborhood_radius,
        min_valid_neighbors=min_valid_neighbors,
        max_rel_depth_spread=max_rel_depth_spread,
        render_chunk=render_chunk,
        collect_diag=collect_diag,
        render_index_map=render_index_map,
        stream_hybrid=stream_hybrid,
        gaussian_chunk_size=gaussian_chunk_size,
        max_gpu_windows=max_gpu_windows,
    )
    return apply_robust_keep(
        merge_result,
        acc,
        hybrid,
        window_ids,
        n_compact,
        n_complement,
        meta,
        min_votes=min_votes,
        visible_ratio_threshold=visible_ratio_threshold,
        decisive_ratio_threshold=decisive_ratio_threshold,
        allow_empty=allow_empty,
        edge_min_boundary=edge_min_boundary,
        edge_boundary_ratio=edge_boundary_ratio,
        edge_min_votes=edge_min_votes,
        edge_visible_ratio=edge_visible_ratio,
        edge_decisive_ratio=edge_decisive_ratio,
        edge_max_agree=edge_max_agree,
    )
