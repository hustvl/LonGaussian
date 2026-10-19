"""Foreign-only owner-depth occlusion filter (bidirectional).

For each context frame f owned by window o:
  - Reference = owner-window hybrid depth
  - Candidates = global hybrid points with source_window != o
  - Same-window points never receive votes on their own frames

A point is dropped only if it accumulates enough foreign-view occluder
votes (optionally requiring a large relative depth gap).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from loger.merge.depth_occlusion_filter import OcclusionVoteAccumulator, apply_keep_mask
from loger.merge.window_merge import (
    WindowMergeResult,
    merge_result_from_tensors,
    stack_window_hybrid_tensors,
)

GAP_BINS: Sequence[Tuple[float, float, str]] = (
    (0.00, 0.02, "[0,.02)"),
    (0.02, 0.05, "[.02,.05)"),
    (0.05, 0.10, "[.05,.10)"),
    (0.10, 0.25, "[.10,.25)"),
    (0.25, 0.50, "[.25,.50)"),
    (0.50, 1.01, "[.50,+)"),
)


def _hybrid_and_window_ids(merge_result: WindowMergeResult):
    compact = merge_result.global_compact
    complement = merge_result.global_complement
    if compact is None or complement is None:
        raise ValueError("merge_result missing global_compact/global_complement")
    compact_ids = merge_result.global_compact_window_ids
    complement_ids = merge_result.global_complement_window_ids
    if compact_ids is None or complement_ids is None:
        raise ValueError("merge_result missing global_*_window_ids")
    hybrid = stack_window_hybrid_tensors(compact, complement)
    # Empty complement (no_track_dense_complement) yields CPU empty ids; align
    # to the hybrid/compact device before cat.
    device = hybrid.means.device if hybrid.num_gaussians else compact.means.device
    window_ids = torch.cat(
        [
            compact_ids.to(device=device, dtype=torch.long),
            complement_ids.to(device=device, dtype=torch.long),
        ],
        dim=0,
    )
    return hybrid, window_ids, int(compact.num_gaussians), int(complement.num_gaussians)


def foreign_hybrid_depth_filter(
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
    ratio_threshold: float,
    min_rel_gap: float = 0.10,
    alpha_threshold: float = 0.5,
    render_chunk: int = 8,
    allow_empty: bool = True,
    collect_diag: bool = False,
    render_index_map: Optional[Dict[int, int]] = None,
) -> Tuple[WindowMergeResult, Dict[str, object]]:
    """Filter global hybrid using foreign-only owner-depth votes.

    ``owner_groups`` maps owner window id -> global frame ids used for GT / global
    poses. When pass-2 merge is context-isolated, pass ``render_index_map`` so
    owner-window renders use local indices into each per-window pose tensor.
    """
    if not per_window_hybrid:
        raise ValueError("per_window_hybrid is empty; enable record_per_window_clouds")

    hybrid, window_ids, n_compact, n_complement = _hybrid_and_window_ids(merge_result)
    device = hybrid.means.device
    window_ids = window_ids.to(device=device, dtype=torch.long)
    n = hybrid.num_gaussians
    acc = OcclusionVoteAccumulator(n, device)

    per_window = {int(w.stats["window_idx"]): w for w in per_window_hybrid}
    poses = merge_result.camera_poses
    if poses.dim() == 4:
        poses = poses[0]

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

    keep = acc.keep_mask(min_votes=int(min_votes), ratio_threshold=float(ratio_threshold))
    removed = ~keep
    kept_n = int(keep.sum().item())
    stats: Dict[str, object] = {
        "rel_margin": float(rel_margin),
        "min_rel_gap": float(min_rel_gap),
        "min_votes": int(min_votes),
        "ratio_threshold": float(ratio_threshold),
        "alpha_threshold": float(alpha_threshold),
        "input": n,
        "output": kept_n,
        "removed": n - kept_n,
        "keep_ratio": float(kept_n) / float(max(n, 1)),
        "vote_summary": acc.summary(),
        "policy": "foreign_only_bidirectional",
        "allow_empty": bool(allow_empty),
    }
    if kept_n == 0 and not allow_empty:
        raise RuntimeError("foreign depth filter removed every Gaussian")

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

    if collect_diag:
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

    filtered_hybrid = apply_keep_mask(hybrid, keep)
    compact_keep = keep[:n_compact]
    complement_keep = keep[n_compact : n_compact + n_complement]
    filtered_compact = apply_keep_mask(merge_result.global_compact, compact_keep)
    filtered_complement = apply_keep_mask(merge_result.global_complement, complement_keep)
    stats["compact_input"] = n_compact
    stats["compact_output"] = int(compact_keep.sum().item())
    stats["complement_input"] = n_complement
    stats["complement_output"] = int(complement_keep.sum().item())

    result = merge_result_from_tensors(
        filtered_hybrid,
        merge_result.camera_poses,
        stats={**dict(merge_result.stats or {}), "foreign_depth_filter": stats},
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
    return result, stats
