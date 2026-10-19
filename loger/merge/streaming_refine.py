"""E8: streaming update-or-spawn for per-pixel Gaussians.

Feed-forward per-pixel prediction spawns a Gaussian for every context pixel of
every window. Over a long sequence the same physical surface is re-spawned by
many windows; once window alignment drifts, those near-duplicates render as
ghosting rather than as extra detail (E0/E4).

This module keeps a persistent voxel-slot map of already-committed Gaussians.
For each incoming pixel it decides:

  - ``refine``: the surface is already represented and the new observation is
    depth-consistent with it -> apply a learned delta to the owning slots and
    drop the pixel before track merge / complement extraction
  - ``spawn``: no depth-consistent slot exists -> let the pixel flow through the
    existing merge pipeline unchanged

The depth-consistency test reuses the depth-occlusion criterion (relative depth margin
against a reference surface), applied at association time instead of as a
post-hoc occlusion filter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from loger.merge.window_merge import WindowMergeTensors

_COORD_LIMIT = 1 << 20
_HASH_PRIMES = (73856093, 19349663, 83492791)


@dataclass
class StreamingRefineConfig:
    """Knobs for the update-or-spawn decision."""

    enabled: bool = False
    voxel_factor: float = 4.0
    slots_per_voxel: int = 2
    depth_rel_margin: float = 0.05
    gate_threshold: float = 0.7
    max_delta_translation: float = 0.5
    delta_hidden_dim: int = 64
    detach_map: bool = True

    def __post_init__(self) -> None:
        if self.voxel_factor <= 0:
            raise ValueError("voxel_factor must be positive")
        if self.slots_per_voxel < 1:
            raise ValueError("slots_per_voxel must be >= 1")
        if not 0.0 < self.depth_rel_margin < 1.0:
            raise ValueError("depth_rel_margin must lie in (0, 1)")
        if not 0.0 <= self.gate_threshold <= 1.0:
            raise ValueError("gate_threshold must lie in [0, 1]")


def voxel_keys(points: torch.Tensor, edge: float) -> torch.Tensor:
    """Quantize world points to int64 voxel keys.

    Coordinates are packed when they fit the addressable range and spatially
    hashed otherwise, so an unbounded scene never silently aliases into the
    packed range.
    """
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"points must have shape (N,3), got {tuple(points.shape)}")
    if not edge > 0:
        raise ValueError("edge must be positive")
    coords = torch.floor(points.detach() / float(edge)).to(torch.int64)
    if int(coords.numel()) == 0:
        return torch.empty(0, dtype=torch.int64, device=points.device)
    if bool((coords.abs() < _COORD_LIMIT).all()):
        shifted = coords + _COORD_LIMIT
        return (
            shifted[:, 0]
            + (shifted[:, 1] << 21)
            + (shifted[:, 2] << 42)
        )
    keys = coords[:, 0] * _HASH_PRIMES[0]
    keys = keys ^ (coords[:, 1] * _HASH_PRIMES[1])
    keys = keys ^ (coords[:, 2] * _HASH_PRIMES[2])
    return keys


class GlobalVoxelMap:
    """Persistent voxel-slot map of committed Gaussians for one scene.

    Entries carry a stable id so that a frame observed twice (sliding-window
    overlap) refines the same slot instead of spawning a duplicate.
    """

    def __init__(self, edge: float, slots_per_voxel: int, device: torch.device) -> None:
        if not edge > 0:
            raise ValueError("edge must be positive")
        self.edge = float(edge)
        self.slots_per_voxel = int(slots_per_voxel)
        self.device = device
        self._means = torch.empty(0, 3, device=device)
        self._opacities = torch.empty(0, device=device)
        self._sh_dc = torch.empty(0, 3, device=device)
        self._delta_means = torch.empty(0, 3, device=device)
        self._delta_opacities = torch.empty(0, device=device)
        self._delta_sh_dc = torch.empty(0, 3, device=device)
        self._entry_keys = torch.empty(0, dtype=torch.int64, device=device)
        self._slot_keys = torch.empty(0, dtype=torch.int64, device=device)
        self._slot_entries = torch.empty(
            0, self.slots_per_voxel, dtype=torch.int64, device=device
        )
        self._ingest_log: List[Tuple[str, int]] = []
        self.num_refine_hits = 0
        self.num_spawned = 0

    @property
    def num_entries(self) -> int:
        return int(self._means.shape[0])

    def means(self) -> torch.Tensor:
        """Committed centers including accumulated deltas."""
        return self._means + self._delta_means

    def ingest(self, tensors: WindowMergeTensors, stream: str = "compact") -> None:
        """Commit newly spawned Gaussians and rebuild the slot table.

        ``stream`` labels which accumulator list the chunk was appended to, so
        deltas can later be spliced back onto the matching chunk.
        """
        count = tensors.num_gaussians
        if count == 0:
            return
        self._ingest_log.append((stream, count))
        means = tensors.means.detach().reshape(-1, 3)
        opacities = tensors.opacities.detach().reshape(-1)
        sh_dc = tensors.sh_flat.detach().reshape(count, -1)[:, :3]

        self._means = torch.cat([self._means, means], dim=0)
        self._opacities = torch.cat([self._opacities, opacities], dim=0)
        self._sh_dc = torch.cat([self._sh_dc, sh_dc], dim=0)
        zeros3 = torch.zeros_like(means)
        self._delta_means = torch.cat([self._delta_means, zeros3], dim=0)
        self._delta_opacities = torch.cat(
            [self._delta_opacities, torch.zeros_like(opacities)], dim=0
        )
        self._delta_sh_dc = torch.cat([self._delta_sh_dc, torch.zeros_like(sh_dc)], dim=0)
        self._entry_keys = torch.cat(
            [self._entry_keys, voxel_keys(means, self.edge)], dim=0
        )
        self.num_spawned += count
        self._rebuild_slots()

    def _rebuild_slots(self) -> None:
        keys = self._entry_keys
        if int(keys.numel()) == 0:
            return
        unique_keys, inverse = torch.unique(keys, sorted=True, return_inverse=True)
        num_voxels = int(unique_keys.numel())
        entries = torch.full(
            (num_voxels, self.slots_per_voxel),
            -1,
            dtype=torch.int64,
            device=keys.device,
        )
        indices = torch.arange(keys.numel(), device=keys.device, dtype=torch.int64)
        # Oldest entries win the slots, so a voxel keeps a stable identity as the
        # sequence grows instead of being rewritten by every later window.
        for slot in range(self.slots_per_voxel):
            taken = entries[:, :slot]
            available = indices
            if slot > 0:
                is_taken = (indices[:, None] == taken[inverse]).any(dim=1)
                available = torch.where(is_taken, torch.full_like(indices, keys.numel()), indices)
            best = torch.full((num_voxels,), keys.numel(), dtype=torch.int64, device=keys.device)
            best.scatter_reduce_(0, inverse, available, reduce="amin", include_self=True)
            entries[:, slot] = torch.where(best == keys.numel(), -1, best)
        self._slot_keys = unique_keys
        self._slot_entries = entries

    def query(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return per-point candidate entry indices and their validity mask."""
        num_points = int(points.shape[0])
        empty_idx = torch.zeros(
            num_points, self.slots_per_voxel, dtype=torch.int64, device=points.device
        )
        empty_valid = torch.zeros(
            num_points, self.slots_per_voxel, dtype=torch.bool, device=points.device
        )
        if self.num_entries == 0 or num_points == 0:
            return empty_idx, empty_valid

        keys = voxel_keys(points, self.edge)
        position = torch.searchsorted(self._slot_keys, keys)
        position = position.clamp(max=int(self._slot_keys.numel()) - 1)
        hit = self._slot_keys[position] == keys
        candidates = self._slot_entries[position]
        valid = hit[:, None] & (candidates >= 0)
        return candidates.clamp(min=0), valid

    def gather(self, indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        flat = indices.reshape(-1)
        return {
            "means": (self._means + self._delta_means)[flat].reshape(*indices.shape, 3),
            "opacities": (self._opacities + self._delta_opacities)[flat].reshape(indices.shape),
            "sh_dc": (self._sh_dc + self._delta_sh_dc)[flat].reshape(*indices.shape, 3),
        }

    def scatter_delta(
        self,
        indices: torch.Tensor,
        weights: torch.Tensor,
        delta_mean: torch.Tensor,
        delta_opacity: torch.Tensor,
        delta_sh_dc: torch.Tensor,
    ) -> None:
        """Accumulate weighted deltas onto the addressed slots."""
        flat = indices.reshape(-1)
        weight = weights.reshape(-1, 1)
        self._delta_means = self._delta_means.index_add(
            0, flat, delta_mean.reshape(-1, 3) * weight
        )
        self._delta_opacities = self._delta_opacities.index_add(
            0, flat, delta_opacity.reshape(-1) * weights.reshape(-1)
        )
        self._delta_sh_dc = self._delta_sh_dc.index_add(
            0, flat, delta_sh_dc.reshape(-1, 3) * weight
        )
        self.num_refine_hits += int(weights.reshape(-1).gt(0).sum().item())

    def chunk_deltas(self, stream: str) -> List[Dict[str, torch.Tensor]]:
        """Per-ingest-chunk deltas for one stream, in ingest order."""
        chunks: List[Dict[str, torch.Tensor]] = []
        offset = 0
        for tag, count in self._ingest_log:
            if tag == stream:
                chunks.append(
                    {
                        "means": self._delta_means[offset : offset + count],
                        "opacities": self._delta_opacities[offset : offset + count],
                        "sh_dc": self._delta_sh_dc[offset : offset + count],
                    }
                )
            offset += count
        return chunks

    def apply_to(self, tensors: WindowMergeTensors) -> WindowMergeTensors:
        """Re-attach accumulated deltas to a cloud holding every ingested entry.

        The caller guarantees ``tensors`` was built by concatenating exactly the
        entries this map ingested, in ingest order.
        """
        count = tensors.num_gaussians
        if count != self.num_entries:
            raise ValueError(
                f"cloud has {count} Gaussians but map holds {self.num_entries} entries"
            )
        if count == 0:
            return tensors
        return apply_chunk_delta(
            tensors, self._delta_means, self._delta_opacities, self._delta_sh_dc
        )

    def stats(self) -> Dict[str, float]:
        return {
            "map_entries": self.num_entries,
            "map_voxels": int(self._slot_keys.numel()),
            "num_spawned": self.num_spawned,
            "num_refine_hits": self.num_refine_hits,
            "mean_delta_translation": float(
                self._delta_means.detach().norm(dim=-1).mean().item()
            )
            if self.num_entries
            else 0.0,
        }


def apply_chunk_delta(
    tensors: WindowMergeTensors,
    delta_means: torch.Tensor,
    delta_opacities: torch.Tensor,
    delta_sh_dc: torch.Tensor,
) -> WindowMergeTensors:
    """Add a refine delta onto one committed chunk."""
    if delta_means.shape[0] != tensors.num_gaussians:
        raise ValueError(
            f"delta covers {delta_means.shape[0]} entries but chunk has "
            f"{tensors.num_gaussians}"
        )
    sh_flat = torch.cat(
        [tensors.sh_flat[:, :3] + delta_sh_dc, tensors.sh_flat[:, 3:]], dim=1
    )
    return WindowMergeTensors(
        means=tensors.means + delta_means,
        quats=tensors.quats,
        scales=tensors.scales,
        opacities=(tensors.opacities + delta_opacities).clamp(0.0, 1.0),
        sh_flat=sh_flat,
    )


def depth_consistency(
    query_points: torch.Tensor,
    candidate_points: torch.Tensor,
    camera_pose: torch.Tensor,
    rel_margin: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """depth-occlusion-style relative depth agreement in one camera frame.

    Returns the signed relative depth difference and a boolean agreement mask.
    A candidate in front of the query beyond the margin is an occluder, one
    behind it is a different surface; only agreement admits a refine.
    """
    viewmat = torch.inverse(camera_pose.float())
    rotation = viewmat[:3, :3]
    translation = viewmat[:3, 3]
    z_query = (query_points @ rotation.T + translation)[..., 2]
    z_cand = (candidate_points @ rotation.T + translation)[..., 2]
    safe_z = z_query.clamp(min=1e-4)
    rel_diff = (z_cand - z_query.unsqueeze(-1)) / safe_z.unsqueeze(-1)
    agree = rel_diff.abs() <= float(rel_margin)
    return rel_diff, agree


class RefineGate(nn.Module):
    """Score whether an incoming pixel is already explained by the map."""

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        rel_distance: torch.Tensor,
        rel_depth_diff: torch.Tensor,
        candidate_opacity: torch.Tensor,
        query_opacity: torch.Tensor,
        color_distance: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.stack(
            [
                rel_distance,
                rel_depth_diff,
                rel_depth_diff.abs(),
                candidate_opacity,
                query_opacity.unsqueeze(-1).expand_as(candidate_opacity),
                color_distance,
            ],
            dim=-1,
        )
        logits = self.net(features).squeeze(-1)
        return logits.masked_fill(~valid, float("-inf"))


class GaussianDeltaHead(nn.Module):
    """Predict a bounded correction for an addressed map slot."""

    def __init__(self, hidden_dim: int = 64, max_translation: float = 0.5) -> None:
        super().__init__()
        self.max_translation = float(max_translation)
        self.net = nn.Sequential(
            nn.Linear(12, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 7),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        offset: torch.Tensor,
        rel_depth_diff: torch.Tensor,
        candidate_opacity: torch.Tensor,
        query_opacity: torch.Tensor,
        candidate_sh_dc: torch.Tensor,
        query_sh_dc: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = torch.cat(
            [
                offset,
                rel_depth_diff.unsqueeze(-1),
                candidate_opacity.unsqueeze(-1),
                query_opacity.unsqueeze(-1),
                candidate_sh_dc,
                query_sh_dc,
            ],
            dim=-1,
        )
        out = self.net(features)
        delta_mean = torch.tanh(out[..., :3]) * self.max_translation
        delta_opacity = torch.tanh(out[..., 3]) * 0.5
        delta_sh_dc = torch.tanh(out[..., 4:7])
        return delta_mean, delta_opacity, delta_sh_dc


class StreamingRefineModule(nn.Module):
    """Gate + delta head; the only trainable part of the E8 arm."""

    def __init__(self, config: StreamingRefineConfig) -> None:
        super().__init__()
        self.config = config
        self.gate = RefineGate(config.delta_hidden_dim)
        self.delta_head = GaussianDeltaHead(
            config.delta_hidden_dim, config.max_delta_translation
        )
        self._gate_losses: List[torch.Tensor] = []

    def reset_episode(self) -> None:
        """Clear per-forward aux state (called once per scene)."""
        self._gate_losses = []

    def gate_aux_loss(self) -> Optional[torch.Tensor]:
        """Mean BCE of refine-vs-spawn targets accumulated this forward."""
        if not self._gate_losses:
            return None
        return torch.stack(self._gate_losses).mean()

    def decide(
        self,
        world_map: GlobalVoxelMap,
        query_means: torch.Tensor,
        query_opacities: torch.Tensor,
        query_sh_dc: torch.Tensor,
        camera_pose: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Decide refine vs spawn for one frame's pixels and apply the deltas.

        Returns the boolean refine mask, the per-pixel gate probability, and
        diagnostics.
        """
        num_points = int(query_means.shape[0])
        no_refine = torch.zeros(num_points, dtype=torch.bool, device=query_means.device)
        zero_prob = torch.zeros(num_points, device=query_means.device)
        if num_points == 0 or world_map.num_entries == 0:
            return no_refine, zero_prob, {"refine_fraction": 0.0, "candidate_fraction": 0.0}

        indices, valid = world_map.query(query_means)
        if not bool(valid.any()):
            return no_refine, zero_prob, {"refine_fraction": 0.0, "candidate_fraction": 0.0}

        candidates = world_map.gather(indices)
        offset = candidates["means"] - query_means.unsqueeze(1)
        rel_distance = offset.norm(dim=-1) / world_map.edge
        rel_depth_diff, depth_agree = depth_consistency(
            query_means, candidates["means"], camera_pose, self.config.depth_rel_margin
        )
        color_distance = (candidates["sh_dc"] - query_sh_dc.unsqueeze(1)).norm(dim=-1)

        admissible = valid & depth_agree
        logits = self.gate(
            rel_distance,
            rel_depth_diff,
            candidates["opacities"],
            query_opacities,
            color_distance,
            admissible,
        )
        best_logit, best_slot = logits.max(dim=1)
        gate_prob = torch.sigmoid(best_logit)
        gate_prob = torch.where(
            torch.isfinite(best_logit), gate_prob, torch.zeros_like(gate_prob)
        )
        refine = gate_prob > self.config.gate_threshold

        has_candidate = valid.any(dim=1)
        residual = rel_depth_diff.abs().masked_fill(~admissible, 1e6).amin(dim=1)
        target = (admissible.any(dim=1) & (residual <= self.config.depth_rel_margin)).to(
            gate_prob.dtype
        )
        if bool(has_candidate.any()):
            self._gate_losses.append(
                F.binary_cross_entropy(gate_prob[has_candidate], target[has_candidate].detach())
            )

        if bool(refine.any()):
            rows = refine.nonzero(as_tuple=True)[0]
            slots = best_slot[rows]
            target = indices[rows, slots]
            delta_mean, delta_opacity, delta_sh_dc = self.delta_head(
                offset[rows, slots],
                rel_depth_diff[rows, slots],
                candidates["opacities"][rows, slots],
                query_opacities[rows],
                candidates["sh_dc"][rows, slots],
                query_sh_dc[rows],
            )
            world_map.scatter_delta(
                target, gate_prob[rows], delta_mean, delta_opacity, delta_sh_dc
            )

        diagnostics = {
            "refine_fraction": float(refine.to(torch.float32).mean().item()),
            "candidate_fraction": float(admissible.any(dim=1).to(torch.float32).mean().item()),
            "mean_gate_prob": float(gate_prob.mean().item()),
        }
        return refine, gate_prob, diagnostics


def dense_pixel_attributes(
    pred_slice: dict, local_t: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten one frame's dense per-pixel Gaussian centers, opacity and SH DC."""
    points = pred_slice["points"][0, local_t]
    offsets = pred_slice.get("gaussian_offsets")
    if offsets is not None:
        points = points + offsets[0, local_t].permute(1, 2, 0)
    height, width = points.shape[0], points.shape[1]
    opacities = pred_slice["gaussian_opacities"][0, local_t, 0]
    sh = pred_slice["gaussian_sh_coeffs"][0, local_t].permute(1, 2, 0)
    return (
        points.reshape(height * width, 3),
        opacities.reshape(height * width),
        sh.reshape(height * width, -1)[:, :3],
    )
