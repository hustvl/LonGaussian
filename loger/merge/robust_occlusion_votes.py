"""robust depth filtering robust occlusion votes: behind counts, dual ratios, 3x3 depth.

Does not change earlier occlusion filters ``OcclusionVoteAccumulator`` behaviour.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch


def _sample_neighborhood(
    image: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    radius: int,
) -> torch.Tensor:
    """Gather a (2r+1)^2 neighborhood for each (u, v). Returns (N, K)."""
    height, width = image.shape[-2:]
    offsets = [
        (du, dv)
        for dv in range(-radius, radius + 1)
        for du in range(-radius, radius + 1)
    ]
    samples = []
    for du, dv in offsets:
        uu = (u + du).clamp(0, width - 1)
        vv = (v + dv).clamp(0, height - 1)
        samples.append(image[vv, uu])
    return torch.stack(samples, dim=1)


class RobustOcclusionVoteAccumulator:
    """Foreign-view occlusion votes with behind / boundary rejection.

    Visible votes only come from reliable reference pixels:
      alpha above threshold, enough 3x3 neighbors, low relative depth spread.

    Drop rule (hard, all must hold):
      occluder >= min_votes
      occluder / visible >= visible_ratio_threshold
      occluder / (occluder + agree) >= decisive_ratio_threshold

    Optional edge path: points that often land on owner depth
    discontinuities can be dropped with *relaxed* occluder thresholds.
    Boundary hits are a prior, not a substitute for front evidence unless
    ``edge_min_votes`` is set to 0.
    """

    def __init__(self, num_gaussians: int, device: torch.device) -> None:
        self.occluder = torch.zeros(num_gaussians, device=device, dtype=torch.int32)
        self.agree = torch.zeros(num_gaussians, device=device, dtype=torch.int32)
        self.behind = torch.zeros(num_gaussians, device=device, dtype=torch.int32)
        self.visible = torch.zeros(num_gaussians, device=device, dtype=torch.int32)
        self.rejected_boundary = torch.zeros(
            num_gaussians, device=device, dtype=torch.int32
        )
        self.projected = torch.zeros(num_gaussians, device=device, dtype=torch.int32)

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
        neighborhood_radius: int = 1,
        min_valid_neighbors: int = 5,
        max_rel_depth_spread: float = 0.15,
        gaussian_chunk_size: int = 500_000,
    ) -> Optional[torch.Tensor]:
        # Large dense-complement scenes can contain 5--8M Gaussians.  The
        # neighborhood tensors scale as N x (2r+1)^2 and otherwise add several
        # hundred MiB at once on top of the merged cloud.  Process independent
        # Gaussian rows in exact chunks; counters and returned gaps are
        # concatenated in original order, so the filter result is unchanged.
        if gaussian_chunk_size > 0 and means.shape[0] > gaussian_chunk_size:
            fields = (
                "occluder", "agree", "behind", "visible",
                "rejected_boundary", "projected",
            )
            gaps = []
            for start in range(0, means.shape[0], gaussian_chunk_size):
                end = min(start + gaussian_chunk_size, means.shape[0])
                child = RobustOcclusionVoteAccumulator(end - start, means.device)
                gap = child.add_frame(
                    means[start:end], camera_pose, K, ref_depth, ref_alpha,
                    rel_margin=rel_margin,
                    alpha_threshold=alpha_threshold,
                    near_plane=near_plane,
                    candidate_mask=(
                        None if candidate_mask is None else candidate_mask[start:end]
                    ),
                    min_rel_gap=min_rel_gap,
                    neighborhood_radius=neighborhood_radius,
                    min_valid_neighbors=min_valid_neighbors,
                    max_rel_depth_spread=max_rel_depth_spread,
                    gaussian_chunk_size=0,
                )
                for field in fields:
                    getattr(self, field)[start:end].add_(getattr(child, field))
                if gap is not None:
                    gaps.append(gap)
            return torch.cat(gaps, dim=0) if gaps else None

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
        if candidate_mask is not None:
            inside = inside & candidate_mask.to(device=inside.device, dtype=torch.bool)
        self.projected += inside.to(torch.int32)

        neigh_d = _sample_neighborhood(ref_depth, u, v, neighborhood_radius)
        neigh_a = _sample_neighborhood(ref_alpha, u, v, neighborhood_radius)
        neighbor_ok = (neigh_a > alpha_threshold) & (neigh_d > near_plane)
        n_ok = neighbor_ok.to(torch.int32).sum(dim=1)
        # Median of valid neighbors: invalids sorted to +inf, pick middle valid.
        filled_hi = torch.where(neighbor_ok, neigh_d, neigh_d.new_tensor(1e12))
        sorted_d, _ = torch.sort(filled_hi, dim=1)
        mid = ((n_ok - 1).clamp(min=0) // 2).clamp(max=sorted_d.shape[1] - 1)
        depth_med = sorted_d.gather(1, mid.unsqueeze(1)).squeeze(1)

        filled_lo = torch.where(neighbor_ok, neigh_d, neigh_d.new_tensor(-1e12))
        d_max = filled_lo.amax(dim=1)
        d_min = filled_hi.amin(dim=1)
        spread = (d_max - d_min) / depth_med.clamp(min=near_plane)
        enough = n_ok >= int(min_valid_neighbors)
        smooth = spread <= float(max_rel_depth_spread)
        reliable = inside & enough & smooth & (depth_med > near_plane)
        rejected = inside & (~reliable)
        self.rejected_boundary += rejected.to(torch.int32)
        self.visible += reliable.to(torch.int32)

        margin = depth_med * float(rel_margin)
        rel_gap = (depth_med - z) / depth_med.clamp(min=near_plane)
        if min_rel_gap is None:
            occluder = reliable & (z < depth_med - margin)
        else:
            occluder = reliable & (rel_gap >= float(min_rel_gap))
        agree = reliable & ((z - depth_med).abs() <= margin)
        behind = reliable & (z > depth_med + margin)
        self.occluder += occluder.to(torch.int32)
        self.agree += agree.to(torch.int32)
        self.behind += behind.to(torch.int32)

        gap_out = torch.full_like(rel_gap, float("nan"))
        return torch.where(reliable, rel_gap, gap_out)

    def occluder_decisive_ratio(self) -> torch.Tensor:
        decisive = (self.occluder + self.agree).clamp(min=1)
        return self.occluder.to(torch.float32) / decisive.to(torch.float32)

    def occluder_visible_ratio(self) -> torch.Tensor:
        vis = self.visible.clamp(min=1)
        return self.occluder.to(torch.float32) / vis.to(torch.float32)

    def boundary_ratio(self) -> torch.Tensor:
        proj = self.projected.clamp(min=1)
        return self.rejected_boundary.to(torch.float32) / proj.to(torch.float32)

    def keep_mask(
        self,
        *,
        min_votes: int,
        visible_ratio_threshold: float,
        decisive_ratio_threshold: float,
        min_visible: Optional[int] = None,
        edge_min_boundary: int = 0,
        edge_boundary_ratio: float = 1.0,
        edge_min_votes: int = 0,
        edge_visible_ratio: float = 0.0,
        edge_decisive_ratio: float = 0.0,
        edge_max_agree: Optional[int] = None,
    ) -> torch.Tensor:
        vis_floor = int(min_votes if min_visible is None else min_visible)
        drop = (
            (self.occluder >= int(min_votes))
            & (self.visible >= vis_floor)
            & (self.occluder_visible_ratio() >= float(visible_ratio_threshold))
            & (self.occluder_decisive_ratio() >= float(decisive_ratio_threshold))
        )
        if int(edge_min_boundary) > 0:
            edge = (
                (self.rejected_boundary >= int(edge_min_boundary))
                & (self.boundary_ratio() >= float(edge_boundary_ratio))
                & (self.occluder >= int(edge_min_votes))
                & (self.occluder_visible_ratio() >= float(edge_visible_ratio))
                & (self.occluder_decisive_ratio() >= float(edge_decisive_ratio))
            )
            if edge_max_agree is not None:
                edge = edge & (self.agree <= int(edge_max_agree))
            drop = drop | edge
        return ~drop

    def summary(self) -> Dict[str, float]:
        seen = self.visible > 0
        vis_r = self.occluder_visible_ratio()
        dec_r = self.occluder_decisive_ratio()
        return {
            "mean_visible_frames": float(self.visible.to(torch.float32).mean().item()),
            "mean_behind_votes": float(self.behind.to(torch.float32).mean().item()),
            "mean_agree_votes": float(self.agree.to(torch.float32).mean().item()),
            "mean_occluder_votes": float(self.occluder.to(torch.float32).mean().item()),
            "mean_rejected_boundary": float(
                self.rejected_boundary.to(torch.float32).mean().item()
            ),
            "mean_boundary_ratio": float(self.boundary_ratio().mean().item()),
            "frac_never_visible": float((~seen).to(torch.float32).mean().item()),
            "mean_occluder_visible_ratio": (
                float(vis_r[seen].mean().item()) if bool(seen.any()) else float("nan")
            ),
            "mean_occluder_decisive_ratio": (
                float(dec_r[seen].mean().item()) if bool(seen.any()) else float("nan")
            ),
        }
