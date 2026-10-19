"""Robust SE(3) estimation from corresponding sliding-window overlap poses."""

from __future__ import annotations

import torch


def _project_to_so3(matrix: torch.Tensor) -> torch.Tensor:
    """Closest proper rotation under the Frobenius norm."""
    u, _, vh = torch.linalg.svd(matrix.float())
    rot = u @ vh
    det = torch.det(rot)
    fix = torch.ones((*det.shape, 3), dtype=rot.dtype, device=rot.device)
    fix[..., 2] = torch.where(det < 0, -1.0, 1.0)
    return u @ torch.diag_embed(fix) @ vh


def estimate_overlap_se3(
    prev_poses: torch.Tensor,
    curr_poses: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate the transform mapping ``curr`` coordinates into ``prev``.

    Both inputs are c2w poses of the same overlap frames with shape
    ``(B, O, 4, 4)``.  A single pose is algebraically sufficient, but is
    fragile when that prediction is an outlier.  We use all overlap poses:
    select a rotation medoid, Huber-downweight rotation outliers, then take a
    weighted chordal rotation mean and a coordinate-wise median translation.
    The latter is robust to a translational outlier without assuming metric
    scale.
    """
    if prev_poses.ndim != 4 or curr_poses.ndim != 4:
        raise ValueError("overlap poses must have shape (B, O, 4, 4)")
    if prev_poses.shape != curr_poses.shape or prev_poses.shape[1] < 1:
        raise ValueError("overlap pose tensors must match and contain at least one frame")

    dtype = prev_poses.dtype
    prev_rot = prev_poses[..., :3, :3]
    curr_rot = curr_poses[..., :3, :3]
    candidate_rot = prev_rot @ curr_rot.transpose(-1, -2)  # (B,O,3,3)
    candidate_trans = prev_poses[..., :3, 3] - (
        candidate_rot @ curr_poses[..., :3, 3].unsqueeze(-1)
    ).squeeze(-1)

    if candidate_rot.shape[1] == 1:
        return candidate_rot[:, 0].to(dtype), candidate_trans[:, 0].to(dtype)

    # Rotation medoid supplies an outlier-resistant reference.  Frobenius
    # distance is chordal distance on SO(3) for rotation matrices.
    pairwise = torch.linalg.vector_norm(
        candidate_rot.unsqueeze(2) - candidate_rot.unsqueeze(1), dim=(-2, -1)
    )
    medoid_idx = pairwise.sum(dim=-1).argmin(dim=-1)  # (B,)
    batch = torch.arange(candidate_rot.shape[0], device=candidate_rot.device)
    medoid_rot = candidate_rot[batch, medoid_idx]
    residual = torch.linalg.vector_norm(candidate_rot - medoid_rot[:, None], dim=(-2, -1))
    median_residual = residual.median(dim=1, keepdim=True).values
    # The tiny floor makes exactly agreeing overlap frames decisive; otherwise
    # the robust scale follows their observed median residual.
    floor = torch.full_like(median_residual, 1e-4)
    delta = torch.maximum(2.0 * median_residual, floor)
    weights = torch.where(residual > 0, (delta / residual).clamp(max=1.0), torch.ones_like(residual))

    chordal = (weights[..., None, None] * candidate_rot).sum(dim=1)
    rot = _project_to_so3(chordal).to(dtype)
    # Component-wise median remains stable under one bad overlap frame and
    # does not introduce a scale-dependent translation threshold.
    trans = candidate_trans.median(dim=1).values.to(dtype)
    return rot, trans
