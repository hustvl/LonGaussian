"""Weighted Gaussian merge utilities."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import torch

from loger.utils.rotation import mat_to_quat, quat_to_mat, standardize_quaternion


def _wxyz_to_xyzw(q_wxyz: torch.Tensor) -> torch.Tensor:
    return q_wxyz[..., [1, 2, 3, 0]]


def _xyzw_to_wxyz(q_xyzw: torch.Tensor) -> torch.Tensor:
    return q_xyzw[..., [3, 0, 1, 2]]


def combine_opacity_or(opacities: Sequence[torch.Tensor]) -> torch.Tensor:
    """OR-combine opacities: alpha = 1 - prod(1 - alpha_i)."""
    one_minus = torch.stack([1.0 - o.clamp(0.0, 1.0) for o in opacities])
    return 1.0 - one_minus.prod(dim=0)


def weighted_mean(values: Sequence[torch.Tensor], weights: Sequence[torch.Tensor]) -> torch.Tensor:
    w = torch.stack(weights)
    v = torch.stack(values)
    total = w.sum().clamp(min=1e-8)
    return (v * w.unsqueeze(-1)).sum(dim=0) / total


def weighted_mean_scalar(values: Sequence[torch.Tensor], weights: Sequence[torch.Tensor]) -> torch.Tensor:
    w = torch.stack(weights)
    v = torch.stack(values)
    total = w.sum().clamp(min=1e-8)
    return (v * w).sum() / total


def weighted_quat_mean(
    quats_wxyz: Sequence[torch.Tensor],
    weights: Sequence[torch.Tensor],
    *,
    stop_grad: bool = True,
) -> torch.Tensor:
    """Weighted rotation merge via matrix average + SVD.

    By default ``stop_grad=True`` (option B): SVD runs in the forward pass for
    rendering geometry, but quaternions/weights are detached so NaN-prone SVD
    backward does not corrupt Gaussian-head training. Position / scale / SH /
    opacity merge paths remain differentiable.
    """
    if stop_grad:
        quats_wxyz = [q.detach() for q in quats_wxyz]
        weights = [w.detach() for w in weights]
    mats = []
    for q, w in zip(quats_wxyz, weights):
        q_xyzw = _wxyz_to_xyzw(q.unsqueeze(0))
        mat = quat_to_mat(q_xyzw)[0]
        mats.append(mat * w)
    rot = torch.stack(mats).sum(dim=0)
    u, _, vt = torch.linalg.svd(rot)
    rot_ortho = u @ vt
    if torch.det(rot_ortho) < 0:
        u = u.clone()
        u[:, -1] *= -1
        rot_ortho = u @ vt
    q_xyzw = mat_to_quat(rot_ortho.unsqueeze(0))[0]
    q_xyzw = standardize_quaternion(q_xyzw.unsqueeze(0))[0]
    return _xyzw_to_wxyz(q_xyzw)


def batch_weighted_quat_mean(
    quats_wxyz: torch.Tensor,
    weights: torch.Tensor,
    *,
    stop_grad: bool = True,
) -> torch.Tensor:
    """Weighted rotation merge for many tracks at once.

    Args:
        quats_wxyz: (T, M, 4) quaternions in gsplat wxyz layout.
        weights: (T, M) non-negative weights.
        stop_grad: If True (default, option B), detach inputs before SVD so
            rotation merge is forward-only. Set False only for differentiable
            experiments (known unstable with autograd through SVD).

    Returns:
        (M, 4) merged quaternions.
    """
    if stop_grad:
        quats_wxyz = quats_wxyz.detach()
        weights = weights.detach()
    t_len, num_tracks = weights.shape
    w_sum = weights.sum(dim=0, keepdim=True).transpose(0, 1).clamp(min=1e-8)  # (M, 1)
    quats_xyzw = _wxyz_to_xyzw(quats_wxyz.reshape(-1, 4)).reshape(t_len, num_tracks, 4)
    mats = quat_to_mat(quats_xyzw.reshape(-1, 4)).reshape(t_len, num_tracks, 3, 3)
    rot = (mats * weights[..., None, None]).sum(dim=0) / w_sum[..., None]
    u, _, vt = torch.linalg.svd(rot)
    rot_ortho = u @ vt
    det = torch.linalg.det(rot_ortho)
    if torch.any(det < 0):
        u = u.clone()
        u[det < 0, :, -1] *= -1
        rot_ortho = u @ vt
    quats_xyzw_out = mat_to_quat(rot_ortho.reshape(-1, 3, 3)).reshape(num_tracks, 4)
    quats_xyzw_out = standardize_quaternion(quats_xyzw_out)
    return _xyzw_to_wxyz(quats_xyzw_out)


def batch_weighted_quat_chordal(
    quats_wxyz: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Differentiable weighted quaternion mean (chordal L2 + normalize)."""
    t_len, num_tracks = weights.shape
    w_sum = weights.sum(dim=0, keepdim=True).transpose(0, 1).clamp(min=1e-8)
    ref = quats_wxyz[0]
    dots = (quats_wxyz * ref.unsqueeze(0)).sum(dim=-1)
    aligned = torch.where(dots[..., None] < 0, -quats_wxyz, quats_wxyz)
    out = (aligned * weights[..., None]).sum(dim=0) / w_sum
    return torch.nn.functional.normalize(out, dim=-1, eps=1e-8)


def batch_weighted_quat_argmax(
    quats_wxyz: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Pick quaternion from highest-weight visible frame per track."""
    idx = weights.argmax(dim=0)
    num_tracks = quats_wxyz.shape[1]
    track_idx = torch.arange(num_tracks, device=quats_wxyz.device)
    return quats_wxyz[idx, track_idx]


def batch_merge_rotation(
    quats_wxyz: torch.Tensor,
    weights: torch.Tensor,
    rotation_mode: str = "svd",
    *,
    stop_grad: bool = True,
) -> torch.Tensor:
    if rotation_mode == "argmax":
        return batch_weighted_quat_argmax(quats_wxyz, weights)
    if rotation_mode == "chordal":
        return batch_weighted_quat_chordal(quats_wxyz, weights)
    return batch_weighted_quat_mean(quats_wxyz, weights, stop_grad=stop_grad)


def soft_track_validity(
    visible_count: torch.Tensor,
    w_sum: torch.Tensor,
    mean_opacity: torch.Tensor,
    spread: torch.Tensor,
    *,
    min_track_len: int,
    min_visible_frames: int,
    min_mean_opacity: float,
    max_center_variance: float,
    temp: float = 0.1,
) -> torch.Tensor:
    """Continuous [0,1] track acceptance weight for DSGA v0."""
    len_ok = (
        (visible_count >= float(min_track_len))
        & (visible_count >= float(min_visible_frames))
    ).to(dtype=spread.dtype)
    opa_ok = torch.sigmoid((mean_opacity - min_mean_opacity) / max(temp, 1e-6))
    spread_ok = torch.sigmoid((max_center_variance - spread) / max(temp, 1e-6))
    w_ok = (w_sum > 0).to(dtype=spread.dtype)
    return len_ok * opa_ok * spread_ok * w_ok


def batch_merge_tracks_dsga(
    centers: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    sh: torch.Tensor,
    vis: torch.Tensor,
    *,
    min_track_len: int,
    min_visible_frames: int,
    min_mean_opacity: float,
    max_center_variance: float,
    rotation_mode: str = "chordal",
    scale_variance_boost: float = 1.0,
    soft_validity_temp: float = 0.1,
) -> tuple[torch.Tensor, ...]:
    """DSGA v0: soft track validity + differentiable rotation merge."""
    vis_f = vis.to(dtype=centers.dtype)
    visible_count = vis_f.sum(dim=0)
    w = opacities * vis_f
    w_sum = w.sum(dim=0).clamp(min=1e-8)

    merged_mean = (centers * w[..., None]).sum(dim=0) / w_sum[:, None]
    merged_scale = (scales * w[..., None]).sum(dim=0) / w_sum[:, None]
    merged_sh = (sh * w[..., None]).sum(dim=0) / w_sum[:, None]
    merged_quat = batch_merge_rotation(quats, w, rotation_mode)

    one_minus = torch.where(
        vis,
        1.0 - opacities.clamp(0.0, 1.0),
        torch.ones_like(opacities),
    )
    merged_opacity = 1.0 - one_minus.prod(dim=0)

    count = visible_count.clamp(min=1.0)
    mean_for_spread = (centers * vis_f[..., None]).sum(dim=0) / count[:, None]
    center_variance = (
        ((centers - mean_for_spread.unsqueeze(0)) ** 2 * vis_f[..., None])
        .sum(dim=0)
        .div(count[:, None])
    )
    # norm(sqrt(var_xyz)) == sqrt(sum(var_xyz)), but the former has an
    # infinite derivative for every zero-variance axis. Static/partially
    # visible tracks hit that case routinely and produce 0 * inf -> NaN.
    spread = torch.sqrt(center_variance.sum(dim=-1) + 1e-8)
    mean_opacity = (opacities * vis_f).sum(dim=0) / count

    track_validity = soft_track_validity(
        visible_count,
        w_sum,
        mean_opacity,
        spread,
        min_track_len=min_track_len,
        min_visible_frames=min_visible_frames,
        min_mean_opacity=min_mean_opacity,
        max_center_variance=max_center_variance,
        temp=soft_validity_temp,
    )
    merged_opacity = merged_opacity * track_validity

    if scale_variance_boost != 1.0:
        boost = visible_count > 2.0
        merged_scale = torch.where(
            boost[:, None],
            merged_scale * scale_variance_boost,
            merged_scale,
        )

    return (
        merged_mean,
        merged_quat,
        merged_scale,
        merged_opacity,
        merged_sh,
        w_sum,
        visible_count,
        track_validity,
    )


def batch_merge_tracks(
    centers: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    sh: torch.Tensor,
    vis: torch.Tensor,
    min_track_len: int,
    min_visible_frames: int,
    min_mean_opacity: float,
    max_center_variance: float,
    scale_variance_boost: float = 1.0,
) -> tuple[torch.Tensor, ...]:
    """Vectorized per-track merge over (T, M, ...) gathered attributes.

    Returns merged tensors and boolean masks:
        merged_mean, merged_quat, merged_scale, merged_opacity, merged_sh,
        merge_weight, source_count, merge_mask, skipped_mask
    """
    vis_f = vis.to(dtype=centers.dtype)
    visible_count = vis_f.sum(dim=0)
    w = opacities * vis_f
    w_sum = w.sum(dim=0).clamp(min=1e-8)

    merged_mean = (centers * w[..., None]).sum(dim=0) / w_sum[:, None]
    merged_scale = (scales * w[..., None]).sum(dim=0) / w_sum[:, None]
    merged_sh = (sh * w[..., None]).sum(dim=0) / w_sum[:, None]
    merged_quat = batch_weighted_quat_mean(quats, w)

    one_minus = torch.where(
        vis,
        1.0 - opacities.clamp(0.0, 1.0),
        torch.ones_like(opacities),
    )
    merged_opacity = 1.0 - one_minus.prod(dim=0)

    count = visible_count.clamp(min=1.0)
    mean_for_spread = (centers * vis_f[..., None]).sum(dim=0) / count[:, None]
    spread = (
        ((centers - mean_for_spread.unsqueeze(0)) ** 2 * vis_f[..., None])
        .sum(dim=0)
        .div(count[:, None])
        .sqrt()
        .norm(dim=-1)
    )

    mean_opacity = (opacities * vis_f).sum(dim=0) / count
    valid = (
        (visible_count >= float(min_track_len))
        & (visible_count >= float(min_visible_frames))
        & (w_sum > 0)
        & (mean_opacity >= min_mean_opacity)
    )
    merge_mask = valid & (spread <= max_center_variance)
    skipped_mask = valid & (spread > max_center_variance)

    if scale_variance_boost != 1.0:
        boost = visible_count > 2.0
        merged_scale = torch.where(
            boost[:, None],
            merged_scale * scale_variance_boost,
            merged_scale,
        )

    merge_weight = w_sum
    source_count = visible_count
    return (
        merged_mean,
        merged_quat,
        merged_scale,
        merged_opacity,
        merged_sh,
        merge_weight,
        source_count,
        merge_mask,
        skipped_mask,
    )


def batch_keep_raw_track_gaussians(
    centers: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    sh: torch.Tensor,
    vis: torch.Tensor,
    min_track_len: int,
    min_visible_frames: int,
    min_mean_opacity: float,
    max_center_variance: float,
    keep_k: int,
    scale_variance_boost: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, ...]:
    """Keep up to ``keep_k`` raw visible sources per hard-valid track.

    Validity / merge masks match :func:`batch_merge_tracks`. Compact Gaussians
    are unfused copies of randomly sampled visible observations (no averaging,
    no OR-opacity). Coverage should still mark the full merged track so the
    dense complement matches the hard-merge ownership pattern.
    """
    if keep_k < 1:
        raise ValueError(f"keep_k must be >= 1, got {keep_k}")

    (
        _merged_mean,
        _merged_quat,
        _merged_scale,
        _merged_opacity,
        _merged_sh,
        merge_weight,
        source_count,
        merge_mask,
        skipped_mask,
    ) = batch_merge_tracks(
        centers,
        quats,
        scales,
        opacities,
        sh,
        vis,
        min_track_len,
        min_visible_frames,
        min_mean_opacity,
        max_center_variance,
        scale_variance_boost,
    )

    t_len, num_tracks = vis.shape
    device = centers.device
    k_eff = min(int(keep_k), int(t_len))
    scores = torch.rand(
        (t_len, num_tracks),
        device=device,
        dtype=centers.dtype,
        generator=generator,
    )
    scores = scores.masked_fill(~vis, float("-inf"))
    scores = scores.masked_fill(~merge_mask.unsqueeze(0), float("-inf"))
    top_vals, top_idx = torch.topk(scores, k=k_eff, dim=0, largest=True, sorted=True)
    pick = torch.isfinite(top_vals)
    track_idx = torch.arange(num_tracks, device=device).unsqueeze(0).expand(k_eff, num_tracks)

    if not bool(pick.any()):
        sh_dim = int(sh.shape[-1])
        empty_mean = centers.new_zeros((0, 3))
        empty_quat = quats.new_zeros((0, 4))
        empty_scale = scales.new_zeros((0, 3))
        empty_opacity = opacities.new_zeros((0,))
        empty_sh = sh.new_zeros((0, sh_dim))
        kept_per_track = torch.zeros(num_tracks, device=device, dtype=torch.long)
        return (
            empty_mean,
            empty_quat,
            empty_scale,
            empty_opacity,
            empty_sh,
            merge_weight,
            source_count,
            merge_mask,
            skipped_mask,
            kept_per_track,
        )

    compact_mean = centers[top_idx, track_idx][pick]
    compact_quat = quats[top_idx, track_idx][pick]
    compact_scale = scales[top_idx, track_idx][pick]
    compact_opacity = opacities[top_idx, track_idx][pick]
    compact_sh = sh[top_idx, track_idx][pick]
    kept_per_track = pick.sum(dim=0).to(dtype=torch.long)
    return (
        compact_mean,
        compact_quat,
        compact_scale,
        compact_opacity,
        compact_sh,
        merge_weight,
        source_count,
        merge_mask,
        skipped_mask,
        kept_per_track,
    )


def batch_merge_tracks_medoid(
    centers: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    sh: torch.Tensor,
    vis: torch.Tensor,
    min_track_len: int,
    min_visible_frames: int,
    min_mean_opacity: float,
    max_center_variance: float,
    scale_variance_boost: float = 1.0,
) -> tuple[torch.Tensor, ...]:
    """Hard merge with a weighted 3D medoid geometry/attribute representative.

    This is a strict K=1 control: validity, merge masks, source counts and
    opacity OR-combination are delegated to :func:`batch_merge_tracks`.
    For every track, the representative is the visible source Gaussian that
    minimizes the opacity-weighted sum of squared 3D distances to the other
    visible sources.  Its complete non-opacity attributes are copied, while
    opacity remains the hard merge's OR-combined value.
    """
    out = batch_merge_tracks(
        centers, quats, scales, opacities, sh, vis,
        min_track_len, min_visible_frames, min_mean_opacity,
        max_center_variance, scale_variance_boost,
    )
    (
        _merged_mean, _merged_quat, _merged_scale, merged_opacity, _merged_sh,
        merge_weight, source_count, merge_mask, skipped_mask,
    ) = out
    weights = opacities * vis.to(dtype=centers.dtype)
    # Looping over the (short) temporal axis avoids materializing T*T*M*3.
    costs = torch.stack([
        (
            torch.where(
                vis,
                ((centers[candidate] - centers) ** 2).sum(dim=-1),
                torch.zeros_like(weights),
            )
            * weights
        ).sum(dim=0)
        for candidate in range(centers.shape[0])
    ])
    costs = costs.masked_fill(~vis, float("inf"))
    source_idx = costs.argmin(dim=0)
    track_idx = torch.arange(centers.shape[1], device=centers.device)
    return (
        centers[source_idx, track_idx],
        quats[source_idx, track_idx],
        scales[source_idx, track_idx],
        merged_opacity,
        sh[source_idx, track_idx],
        merge_weight,
        source_count,
        merge_mask,
        skipped_mask,
    )


def batch_merge_tracks_tracklet_split(
    centers: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    sh: torch.Tensor,
    vis: torch.Tensor,
    min_track_len: int,
    min_visible_frames: int,
    min_mean_opacity: float,
    max_center_variance: float,
    scale_variance_boost: float = 1.0,
    *,
    mad_multiplier: float = 3.0,
    min_jump: float = 0.15,
    split_signal: str = "geometry",
) -> tuple[torch.Tensor, ...]:
    """Split robust 3D change-point tracks into two hard-merged tracklets.

    R102 is deliberately a grouping diagnostic, rather than a general
    variable-K allocator: each accepted original track contributes either one
    hard K=1 prototype or two contiguous K=1 tracklets. A split is admitted
    only at the largest adjacent visible 3D jump exceeding both a MAD threshold
    and ``min_jump``, with two visible observations on each side. If either
    child is not hard-valid, the original track is retained unchanged.

    Returns compact tensors, the original hard merge mask, and a per-track
    split mask. Coverage must be marked from the original mask because the two
    children partition precisely the same visible observations.
    """
    base = batch_merge_tracks(
        centers, quats, scales, opacities, sh, vis,
        min_track_len, min_visible_frames, min_mean_opacity,
        max_center_variance, scale_variance_boost,
    )
    base_mean, base_quat, base_scale, base_opacity, base_sh = base[:5]
    base_merge_mask = base[7]
    t_len, num_tracks = vis.shape
    if t_len < 4 or not bool(base_merge_mask.any()):
        return (
            base_mean[base_merge_mask], base_quat[base_merge_mask],
            base_scale[base_merge_mask], base_opacity[base_merge_mask],
            base_sh[base_merge_mask], base_merge_mask,
            torch.zeros(num_tracks, dtype=torch.bool, device=vis.device),
        )

    if split_signal == "geometry":
        jumps = torch.linalg.vector_norm(centers[1:] - centers[:-1], dim=-1)
    elif split_signal == "appearance":
        jumps = torch.linalg.vector_norm(sh[1:] - sh[:-1], dim=-1)
    else:
        raise ValueError(f"unknown tracklet split signal: {split_signal}")
    adjacent = vis[1:] & vis[:-1]
    def _masked_lower_median(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Deterministic median over the short temporal axis.

        CUDA ``nanmedian`` has no deterministic implementation. Sorting at
        most 11 temporal observations and gathering the lower median preserves
        the robust change-point rule under the locked evaluator.
        """
        sorted_values = values.masked_fill(~mask, float("inf")).sort(dim=0).values
        count = mask.sum(dim=0)
        index = ((count - 1).clamp(min=0) // 2).unsqueeze(0)
        median_value = sorted_values.gather(0, index).squeeze(0)
        return torch.where(count > 0, median_value, torch.zeros_like(median_value))

    median = _masked_lower_median(jumps, adjacent)
    mad = _masked_lower_median((jumps - median.unsqueeze(0)).abs(), adjacent)
    threshold = torch.maximum(
        torch.full_like(median, float(min_jump)), median + float(mad_multiplier) * mad
    )
    candidate = adjacent & (jumps > threshold.unsqueeze(0))

    frame_ids = torch.arange(t_len - 1, device=vis.device).unsqueeze(1)
    # A jump at index j separates frame j from frame j+1.
    left_count = (vis.unsqueeze(0) & (torch.arange(t_len, device=vis.device).view(1, -1, 1) <= frame_ids[:, None, :])).sum(dim=1)
    right_count = vis.sum(dim=0, keepdim=True) - left_count
    candidate &= (left_count >= min_track_len) & (right_count >= min_track_len)
    candidate &= base_merge_mask.unsqueeze(0)
    score = torch.where(candidate, jumps, torch.full_like(jumps, -float("inf")))
    split_at = score.argmax(dim=0)
    proposed_split = candidate.any(dim=0)

    timeline = torch.arange(t_len, device=vis.device).unsqueeze(1)
    left_vis = vis & (timeline <= split_at.unsqueeze(0))
    right_vis = vis & ~left_vis
    left = batch_merge_tracks(
        centers, quats, scales, opacities, sh, left_vis,
        min_track_len, min_visible_frames, min_mean_opacity,
        max_center_variance, scale_variance_boost,
    )
    right = batch_merge_tracks(
        centers, quats, scales, opacities, sh, right_vis,
        min_track_len, min_visible_frames, min_mean_opacity,
        max_center_variance, scale_variance_boost,
    )
    split_mask = proposed_split & left[7] & right[7]
    unsplit_mask = base_merge_mask & ~split_mask

    def _collect(index: int) -> torch.Tensor:
        return torch.cat((base[index][unsplit_mask], left[index][split_mask], right[index][split_mask]))

    return (
        _collect(0), _collect(1), _collect(2), _collect(3), _collect(4),
        base_merge_mask, split_mask,
    )


def batch_merge_tracks_flatk(
    centers: torch.Tensor, quats: torch.Tensor, scales: torch.Tensor,
    opacities: torch.Tensor, sh: torch.Tensor, vis: torch.Tensor,
    min_track_len: int, min_visible_frames: int, min_mean_opacity: float,
    max_center_variance: float, scale_variance_boost: float = 1.0, *,
    fraction_k4: float = 0.15, fraction_k2: float = 0.35,
) -> tuple[torch.Tensor, ...]:
    """R103 deterministic entropy-routed flat K={1,2,4} hard prototypes.

    Tracks are ranked by a dimensionless dispersion-entropy proxy. Eligible high-score
    tracks are split by three deterministic weighted Lloyd iterations; a child
    that fails the existing hard validity test makes its parent fall back to K=1.
    """
    base = batch_merge_tracks(centers, quats, scales, opacities, sh, vis,
                              min_track_len, min_visible_frames, min_mean_opacity,
                              max_center_variance, scale_variance_boost)
    base_mask = base[7]
    visible_count = vis.sum(dim=0)
    w = opacities * vis.to(centers.dtype)
    wsum = w.sum(dim=0).clamp(min=1e-8)
    mean = (centers * w[..., None]).sum(dim=0) / wsum[:, None]
    center_var = (((centers - mean.unsqueeze(0)) ** 2).sum(dim=-1) * w).sum(dim=0) / wsum
    sh_mean = (sh * w[..., None]).sum(dim=0) / wsum[:, None]
    sh_var = (((sh - sh_mean.unsqueeze(0)) ** 2).mean(dim=-1) * w).sum(dim=0) / wsum
    sh_energy = ((sh ** 2).mean(dim=-1) * w).sum(dim=0) / wsum
    # Both terms are dimensionless: center dispersion relative to the locked
    # hard acceptance radius and appearance variance relative to SH energy.
    score = torch.log1p(center_var / max(max_center_variance ** 2, 1e-8)) + torch.log1p(sh_var / (sh_energy + 1e-8))
    requested_k = torch.ones_like(visible_count, dtype=torch.long)

    def _top_fraction(mask: torch.Tensor, fraction: float, k: int) -> None:
        ids = mask.nonzero(as_tuple=True)[0]
        take = min(int(ids.numel()), int(round(float(ids.numel()) * fraction)))
        if take:
            ranked = torch.argsort(score[ids], descending=True, stable=True)
            chosen = ids[ranked[:take]]
            requested_k[chosen] = k

    eligible4 = base_mask & (visible_count >= 8)
    _top_fraction(eligible4, fraction_k4, 4)
    eligible2 = base_mask & (visible_count >= 4) & (requested_k == 1)
    _top_fraction(eligible2, fraction_k2, 2)
    actual_k = torch.ones_like(requested_k)
    assignment_trace = torch.full_like(vis, -1, dtype=torch.int8)

    output = [[base[i][base_mask]] for i in range(5)]
    for k in (2, 4):
        ids = (requested_k == k).nonzero(as_tuple=True)[0]
        if ids.numel() == 0:
            continue
        c, q, s, o, h, v = (x[:, ids] for x in (centers, quats, scales, opacities, sh, vis))
        # Farthest-point deterministic initialization from the earliest visible source.
        first = v.float().argmax(dim=0)
        tid = torch.arange(ids.numel(), device=centers.device)
        prototypes = [c[first, tid]]
        for _ in range(1, k):
            dist = torch.stack([((c - p.unsqueeze(0)) ** 2).sum(dim=-1) for p in prototypes]).amin(dim=0)
            dist = dist.masked_fill(~v, -float("inf"))
            nxt = dist.argmax(dim=0)
            prototypes.append(c[nxt, tid])
        proto = torch.stack(prototypes)
        assign = torch.zeros_like(v, dtype=torch.long)
        for _ in range(3):
            dist = ((c.unsqueeze(0) - proto.unsqueeze(1)) ** 2).sum(dim=-1)
            assign = dist.argmin(dim=0)
            new_proto = []
            for j in range(k):
                member = v & (assign == j)
                mw = o * member.to(o.dtype)
                denom = mw.sum(dim=0).clamp(min=1e-8)
                new_proto.append((c * mw[..., None]).sum(dim=0) / denom[:, None])
            proto = torch.stack(new_proto)
        child_out = []
        child_valid = []
        for j in range(k):
            child = batch_merge_tracks(c, q, s, o, h, v & (assign == j),
                                       min_track_len, min_visible_frames, min_mean_opacity,
                                       max_center_variance, scale_variance_boost)
            child_out.append(child)
            child_valid.append(child[7])
        valid = torch.stack(child_valid).all(dim=0)
        # Remove base copies for valid multi-K parents, then append their children.
        for i in range(5):
            output[i][0] = output[i][0]  # retained base collection is filtered below
        actual_k[ids[valid]] = k
        assignment_trace[:, ids[valid]] = assign[:, valid].to(torch.int8)
        # Store children; final base K=1 is rebuilt after actual_k is known.
        for i in range(5):
            output[i].append(torch.cat([child_out[j][i][valid] for j in range(k)]))
    base_keep = base_mask & (actual_k == 1)
    final = [[base[i][base_keep], *output[i][1:]] for i in range(5)]
    return (*[torch.cat(parts) for parts in final], base_mask, requested_k, actual_k, score, assignment_trace)


def _weighted_reconstruction_loss(
    centers: torch.Tensor,
    sh: torch.Tensor,
    opacities: torch.Tensor,
    mask: torch.Tensor,
    *,
    max_center_variance: float,
) -> torch.Tensor:
    """Dimensionless per-track source-attribute reconstruction loss."""
    w = opacities * mask.to(centers.dtype)
    wsum = w.sum(dim=0).clamp(min=1e-8)
    mean = (centers * w[..., None]).sum(dim=0) / wsum[:, None]
    center_loss = (((centers - mean.unsqueeze(0)) ** 2).sum(dim=-1) * w).sum(dim=0) / wsum
    sh_mean = (sh * w[..., None]).sum(dim=0) / wsum[:, None]
    sh_loss = (((sh - sh_mean.unsqueeze(0)) ** 2).mean(dim=-1) * w).sum(dim=0) / wsum
    sh_energy = ((sh ** 2).mean(dim=-1) * w).sum(dim=0) / wsum
    return center_loss / max(max_center_variance ** 2, 1e-8) + sh_loss / (sh_energy + 1e-8)


def _weighted_reconstruction_components(
    centers: torch.Tensor,
    sh: torch.Tensor,
    opacities: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    w = opacities * mask.to(centers.dtype)
    wsum = w.sum(dim=0).clamp(min=1e-8)
    mean = (centers * w[..., None]).sum(dim=0) / wsum[:, None]
    center_sse = (((centers - mean.unsqueeze(0)) ** 2).sum(dim=-1) * w).sum(dim=0)
    sh_mean = (sh * w[..., None]).sum(dim=0) / wsum[:, None]
    sh_sse = (((sh - sh_mean.unsqueeze(0)) ** 2).mean(dim=-1) * w).sum(dim=0)
    sh_energy = ((sh ** 2).mean(dim=-1) * w).sum(dim=0).clamp(min=1e-8)
    return center_sse, sh_sse, sh_energy, wsum


def _normalized_sse(
    center_sse: torch.Tensor,
    sh_sse: torch.Tensor,
    sh_energy: torch.Tensor,
    weight: torch.Tensor,
    *,
    max_center_variance: float,
) -> torch.Tensor:
    return center_sse / (max(max_center_variance ** 2, 1e-8) * weight.clamp(min=1e-8)) + sh_sse / sh_energy


def _prototype_set_loss(
    centers: torch.Tensor,
    sh: torch.Tensor,
    opacities: torch.Tensor,
    mask: torch.Tensor,
    prototype_centers: Sequence[torch.Tensor],
    prototype_sh: Sequence[torch.Tensor],
    *,
    max_center_variance: float,
) -> torch.Tensor:
    """Nearest-prototype source distortion for the exact emitted prefix set."""
    w = opacities * mask.to(centers.dtype)
    weight = w.sum(dim=0).clamp(min=1e-8)
    sh_energy = ((sh ** 2).mean(dim=-1) * w).sum(dim=0).clamp(min=1e-8)
    pc = torch.stack(prototype_centers, dim=0)
    ps = torch.stack(prototype_sh, dim=0)
    center_dist = ((centers.unsqueeze(0) - pc.unsqueeze(1)) ** 2).sum(dim=-1)
    sh_dist = ((sh.unsqueeze(0) - ps.unsqueeze(1)) ** 2).mean(dim=-1)
    loss = center_dist / max(max_center_variance ** 2, 1e-8) + sh_dist / sh_energy.view(1, 1, -1)
    nearest = loss.amin(dim=0)
    return (nearest * w).sum(dim=0) / weight


def _deterministic_binary_assign(
    centers: torch.Tensor,
    opacities: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Deterministic weighted 2-means assignment over the short time axis."""
    first = mask.float().argmax(dim=0)
    track_idx = torch.arange(mask.shape[1], device=centers.device)
    p0 = centers[first, track_idx]
    d0 = ((centers - p0.unsqueeze(0)) ** 2).sum(dim=-1).masked_fill(~mask, -float("inf"))
    second = d0.argmax(dim=0)
    proto = torch.stack((p0, centers[second, track_idx]))
    assign = torch.zeros_like(mask, dtype=torch.long)
    for _ in range(3):
        dist = ((centers.unsqueeze(0) - proto.unsqueeze(1)) ** 2).sum(dim=-1)
        assign = dist.argmin(dim=0)
        new_proto = []
        for j in range(2):
            member = mask & (assign == j)
            mw = opacities * member.to(opacities.dtype)
            denom = mw.sum(dim=0).clamp(min=1e-8)
            new_proto.append((centers * mw[..., None]).sum(dim=0) / denom[:, None])
        proto = torch.stack(new_proto)
    return assign


def batch_merge_tracks_nested(
    centers: torch.Tensor, quats: torch.Tensor, scales: torch.Tensor,
    opacities: torch.Tensor, sh: torch.Tensor, vis: torch.Tensor,
    min_track_len: int, min_visible_frames: int, min_mean_opacity: float,
    max_center_variance: float, scale_variance_boost: float = 1.0, *,
    fraction_k4: float = 0.15, fraction_k2: float = 0.35,
) -> tuple[torch.Tensor, ...]:
    """R104 nested residual K={1,2,4} allocation from source attributes.

    The first split is ranked by binary reconstruction-loss reduction from the
    K=1 parent. K=4 is then ranked by the conditional marginal reduction after
    the accepted K=2 split, so the K=4 set is nested inside the K=2 set.
    """
    base = batch_merge_tracks(centers, quats, scales, opacities, sh, vis,
                              min_track_len, min_visible_frames, min_mean_opacity,
                              max_center_variance, scale_variance_boost)
    base_mask = base[7]
    visible_count = vis.sum(dim=0)
    requested_k = torch.zeros_like(visible_count, dtype=torch.long)
    requested_k[base_mask] = 1
    actual_k = torch.zeros_like(requested_k)
    actual_k[base_mask] = 1
    assignment_trace = torch.full_like(vis, -1, dtype=torch.int8)
    parent_loss = _prototype_set_loss(
        centers, sh, opacities, vis, [base[0]], [base[4]],
        max_center_variance=max_center_variance,
    )

    first_assign = _deterministic_binary_assign(centers, opacities, vis)
    first_children = []
    first_valid = []
    prefix2_loss_by_child = []
    for j in range(2):
        child_mask = vis & (first_assign == j)
        child = batch_merge_tracks(
            centers, quats, scales, opacities, sh, child_mask,
            min_track_len, min_visible_frames, min_mean_opacity,
            max_center_variance, scale_variance_boost,
        )
        first_children.append(child)
        first_valid.append(child[7])
        prefix2_loss_by_child.append(
            _prototype_set_loss(
                centers, sh, opacities, vis, [base[0], child[0]], [base[4], child[4]],
                max_center_variance=max_center_variance,
            )
        )
    first_valid_all = torch.stack(first_valid).all(dim=0)
    prefix2_losses = torch.stack(prefix2_loss_by_child)
    residual_branch = prefix2_losses.argmin(dim=0)
    prefix2_loss = prefix2_losses.amin(dim=0)
    gain_k2 = torch.where(base_mask & first_valid_all, parent_loss - prefix2_loss, torch.zeros_like(parent_loss))

    eligible2 = base_mask & first_valid_all & (visible_count >= 4) & (gain_k2 > 0)
    ids2 = eligible2.nonzero(as_tuple=True)[0]
    take2 = min(int(ids2.numel()), int(round(float(ids2.numel()) * fraction_k2)))
    selected2 = torch.zeros_like(base_mask)
    if take2:
        ranked2 = torch.argsort(gain_k2[ids2], descending=True, stable=True)
        selected2[ids2[ranked2[:take2]]] = True
    requested_k[selected2] = 2

    gain_k4 = torch.zeros_like(parent_loss)
    second_assign_by_child = []
    second_children_by_child = []
    second_valid_by_child = []
    for j in range(2):
        child_mask = vis & (first_assign == j)
        second_assign = _deterministic_binary_assign(centers, opacities, child_mask)
        second_assign_by_child.append(second_assign)
        grandchildren = []
        grand_valid = []
        for g in range(2):
            grand_mask = child_mask & (second_assign == g)
            grand = batch_merge_tracks(
                centers, quats, scales, opacities, sh, grand_mask,
                min_track_len, min_visible_frames, min_mean_opacity,
                max_center_variance, scale_variance_boost,
            )
            grandchildren.append(grand)
            grand_valid.append(grand[7])
        second_children_by_child.append(grandchildren)
        second_valid_by_child.append(torch.stack(grand_valid).all(dim=0))
        prefix4_loss = _prototype_set_loss(
            centers,
            sh,
            opacities,
            vis,
            [base[0], first_children[j][0], grandchildren[0][0], grandchildren[1][0]],
            [base[4], first_children[j][4], grandchildren[0][4], grandchildren[1][4]],
            max_center_variance=max_center_variance,
        )
        gain_k4 = torch.where(residual_branch == j, prefix2_loss - prefix4_loss, gain_k4)

    second_valid_all = torch.where(residual_branch == 0, second_valid_by_child[0], second_valid_by_child[1])
    eligible4 = selected2 & second_valid_all & (visible_count >= 8) & (gain_k4 > 0)
    ids4 = eligible4.nonzero(as_tuple=True)[0]
    take4 = min(int(ids4.numel()), int(round(float(ids2.numel()) * fraction_k4)))
    selected4 = torch.zeros_like(base_mask)
    if take4:
        ranked4 = torch.argsort(gain_k4[ids4], descending=True, stable=True)
        selected4[ids4[ranked4[:take4]]] = True
    requested_k[selected4] = 4
    actual_k[selected2 & ~selected4] = 2
    actual_k[selected4] = 4

    parts = [[base[i][base_mask]] for i in range(5)]
    valid2 = actual_k == 2
    if bool(valid2.any()):
        prefix_assign = torch.where(
            first_assign == residual_branch.unsqueeze(0),
            torch.ones_like(first_assign),
            torch.zeros_like(first_assign),
        )
        prefix_assign = torch.where(vis, prefix_assign, torch.full_like(prefix_assign, -1))
        assignment_trace[:, valid2] = prefix_assign[:, valid2].to(torch.int8)
        for i in range(5):
            residual_parts = []
            for j in range(2):
                keep = valid2 & (residual_branch == j)
                if bool(keep.any()):
                    residual_parts.append(first_children[j][i][keep])
            if residual_parts:
                parts[i].append(torch.cat(residual_parts))
    valid4 = actual_k == 4
    if bool(valid4.any()):
        nested_assign = first_assign.clone()
        nested_assign = torch.where(
            residual_branch.unsqueeze(0) == 0,
            torch.where(first_assign == 0, second_assign_by_child[0] + 1, torch.zeros_like(first_assign)),
            torch.where(first_assign == 1, second_assign_by_child[1] + 1, torch.zeros_like(first_assign)),
        )
        nested_assign = torch.where(vis, nested_assign, torch.full_like(nested_assign, -1))
        assignment_trace[:, valid4] = nested_assign[:, valid4].to(torch.int8)
        for i in range(5):
            residual_parts = []
            for j in range(2):
                keep = valid4 & (residual_branch == j)
                if bool(keep.any()):
                    residual_parts.append(first_children[j][i][keep])
                    residual_parts.extend(second_children_by_child[j][g][i][keep] for g in range(2))
            if residual_parts:
                parts[i].append(torch.cat(residual_parts))

    return (*[torch.cat(p) for p in parts], base_mask, requested_k, actual_k, gain_k2 + gain_k4, assignment_trace)


def merge_source_gaussians(
    means: Sequence[torch.Tensor],
    quats: Sequence[torch.Tensor],
    scales: Sequence[torch.Tensor],
    opacities: Sequence[torch.Tensor],
    sh_flats: Sequence[torch.Tensor],
    vis_flags: Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge per-pixel source Gaussians within one track."""
    weights = [opacities[i] * float(vis_flags[i]) for i in range(len(means))]
    total_w = sum(weights)
    if total_w <= 0:
        raise ValueError("Track has zero total merge weight")

    merged_mean = weighted_mean(means, weights)
    merged_scale = weighted_mean(scales, weights)
    merged_sh = weighted_mean(sh_flats, weights)
    merged_quat = weighted_quat_mean(quats, weights)
    merged_opacity = combine_opacity_or(opacities)
    merge_weight = torch.tensor(float(total_w), device=means[0].device, dtype=means[0].dtype)
    source_count = torch.tensor(float(len(means)), device=means[0].device, dtype=means[0].dtype)
    return merged_mean, merged_quat, merged_scale, merged_opacity, merged_sh, merge_weight, source_count


def merge_compact_entries(
    entries: Iterable,
) -> "CompactGaussianEntry":
    from loger.merge.compact_gaussians import CompactGaussianEntry

    items = list(entries)
    if len(items) == 1:
        e = items[0]
        return CompactGaussianEntry(
            mean=e.mean,
            quat=e.quat,
            scale=e.scale,
            opacity=e.opacity,
            sh_flat=e.sh_flat,
            merge_weight=e.merge_weight,
            source_count=e.source_count,
            track_meta=items[0].track_meta,
        )

    weights = [e.merge_weight for e in items]
    merged_mean = weighted_mean([e.mean for e in items], weights)
    merged_scale = weighted_mean([e.scale for e in items], weights)
    merged_sh = weighted_mean([e.sh_flat for e in items], weights)
    merged_quat = weighted_quat_mean([e.quat for e in items], weights)
    merged_opacity = combine_opacity_or([e.opacity for e in items])
    merge_weight = torch.stack(weights).sum()
    source_count = torch.stack([e.source_count for e in items]).sum()
    track_meta = items[0].track_meta.copy()
    track_meta["stitched_from"] = len(items)
    return CompactGaussianEntry(
        mean=merged_mean,
        quat=merged_quat,
        scale=merged_scale,
        opacity=merged_opacity,
        sh_flat=merged_sh,
        merge_weight=merge_weight,
        source_count=source_count,
        track_meta=track_meta,
    )
