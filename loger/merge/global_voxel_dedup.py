"""Scene-global deterministic voxel keep-one for complement Gaussians."""

from __future__ import annotations

import math
from typing import Literal

import torch

VoxelKeepPolicy = Literal["max_opacity", "first"]


class OnlineFirstWinsVoxelDeduper:
    """Window-causal, first-wins voxel occupancy used for the E8 premise test.

    A call to :meth:`keep_mask` only consults Gaussians committed by *earlier*
    windows.  The caller therefore evaluates compact and complement together,
    then commits their retained union.  This deliberately does not collapse
    multiple samples made inside the current window.
    """

    def __init__(self, *, edge: float, device: torch.device) -> None:
        if not edge > 0:
            raise ValueError("edge must be positive")
        self.edge = float(edge)
        self.device = device
        self._occupied = torch.empty(0, dtype=torch.int64, device=device)
        self._representative_means: torch.Tensor | None = None
        self._representative_scales: torch.Tensor | None = None
        self._representative_dc: torch.Tensor | None = None
        self._representative_opacities: torch.Tensor | None = None
        # Location of each voxel representative in the persistent accumulator.
        # ``-1`` denotes a representative committed without an accumulator
        # location (used by small unit tests / occupancy-only callers).
        self._representative_stream = torch.empty(0, dtype=torch.int8, device=device)
        self._representative_chunk = torch.empty(0, dtype=torch.long, device=device)
        self._representative_local = torch.empty(0, dtype=torch.long, device=device)
        self.num_candidates = 0
        self.num_voxel_matches = 0
        self.num_geometry_matches = 0
        self.num_appearance_matches = 0
        self.num_rejected = 0
        self.num_committed = 0

    @property
    def num_voxels(self) -> int:
        return int(self._occupied.numel())

    def keep_mask(
        self,
        means: torch.Tensor,
        *,
        scales: torch.Tensor | None = None,
        sh_flat: torch.Tensor | None = None,
        geometry_factor: float = 0.0,
        dc_l2_threshold: float = -1.0,
    ) -> torch.Tensor:
        """Keep candidates not consistently associated with an earlier point.

        With both gates disabled this is the original occupancy-only ablation.
        A positive geometry factor requires center distance to be small relative
        to the two Gaussian scale norms; a non-negative DC threshold then also
        requires SH-DC appearance agreement.
        """
        if means.ndim != 2 or means.shape[-1] != 3:
            raise ValueError(f"means must have shape (N,3), got {tuple(means.shape)}")
        keep, _ = self.associate(
            means, scales=scales, sh_flat=sh_flat,
            geometry_factor=geometry_factor, dc_l2_threshold=dc_l2_threshold,
        )
        return keep

    def associate(
        self,
        means: torch.Tensor,
        *,
        scales: torch.Tensor | None = None,
        sh_flat: torch.Tensor | None = None,
        geometry_factor: float = 0.0,
        dc_l2_threshold: float = -1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the keep mask and matched earlier-representative slots.

        A slot of ``-1`` means that the candidate must spawn.  The explicit
        association is what lets E8 update an existing Gaussian without ever
        moving its geometry or letting same-window points affect each other.
        """
        if means.ndim != 2 or means.shape[-1] != 3:
            raise ValueError(f"means must have shape (N,3), got {tuple(means.shape)}")
        n = int(means.shape[0])
        self.num_candidates += n
        slots = torch.full((n,), -1, dtype=torch.long, device=means.device)
        if n == 0 or self.num_voxels == 0:
            return torch.ones(n, dtype=torch.bool, device=means.device), slots
        keys = _stable_voxel_keys(means, edge=self.edge)
        position = torch.searchsorted(self._occupied, keys)
        position = position.clamp(max=self.num_voxels - 1)
        matched = self._occupied[position] == keys
        self.num_voxel_matches += int(matched.sum().item())
        if geometry_factor > 0:
            if scales is None or self._representative_means is None or self._representative_scales is None:
                raise ValueError("geometry gate requires scales and representative attributes")
            dist = (means - self._representative_means[position]).norm(dim=-1)
            radius = scales.norm(dim=-1) + self._representative_scales[position].norm(dim=-1)
            geometry = dist <= float(geometry_factor) * radius.clamp_min(1e-8)
            matched &= geometry
            self.num_geometry_matches += int((geometry & (self._occupied[position] == keys)).sum().item())
        if dc_l2_threshold >= 0:
            if sh_flat is None or self._representative_dc is None:
                raise ValueError("appearance gate requires SH coefficients and representative attributes")
            dc = sh_flat[:, :3]
            appearance = (dc - self._representative_dc[position]).norm(dim=-1) <= float(dc_l2_threshold)
            matched &= appearance
            self.num_appearance_matches += int(appearance.sum().item())
        self.num_rejected += int(matched.sum().item())
        slots[matched] = position[matched]
        return ~matched, slots

    def representative_locations(
        self, slots: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Resolve matched representative slots to accumulator locations."""
        if slots.ndim != 1 or bool((slots < 0).any()):
            raise ValueError("slots must be a 1-D tensor of matched representative indices")
        return (
            self._representative_stream[slots],
            self._representative_chunk[slots],
            self._representative_local[slots],
        )

    def representative_features(self, slots: torch.Tensor) -> dict[str, torch.Tensor]:
        """Detached attributes for bounded B1 teacher-candidate records."""
        if slots.ndim != 1 or bool((slots < 0).any()):
            raise ValueError("slots must be matched representative indices")
        if self._representative_means is None or self._representative_opacities is None:
            raise RuntimeError("representative attributes are unavailable before commit")
        return {
            "old_means": self._representative_means[slots].detach(),
            "old_scales": self._representative_scales[slots].detach(),
            "old_dc": self._representative_dc[slots].detach(),
            "old_opacities": self._representative_opacities[slots].detach(),
        }

    def commit(
        self,
        means: torch.Tensor,
        *,
        scales: torch.Tensor | None = None,
        sh_flat: torch.Tensor | None = None,
        opacities: torch.Tensor | None = None,
        source_stream: torch.Tensor | None = None,
        source_chunk: torch.Tensor | None = None,
        source_local: torch.Tensor | None = None,
    ) -> None:
        """Make retained points visible to later windows."""
        if int(means.shape[0]) == 0:
            return
        keys = _stable_voxel_keys(means, edge=self.edge)
        new_keys, inverse = torch.unique(keys, sorted=True, return_inverse=True)
        first = torch.full((new_keys.numel(),), int(keys.numel()), dtype=torch.long, device=keys.device)
        first.scatter_reduce_(0, inverse, torch.arange(keys.numel(), device=keys.device), reduce="amin", include_self=True)
        # A strict-gate miss may still share a voxel with an existing point.
        # It remains in the render cloud, but it must not displace the first
        # causal representative used for future associations.
        if self.num_voxels:
            pos = torch.searchsorted(self._occupied, new_keys).clamp(max=self.num_voxels - 1)
            novel = self._occupied[pos] != new_keys
            new_keys, first = new_keys[novel], first[novel]
        if int(new_keys.numel()) == 0:
            self.num_committed += int(means.shape[0])
            return
        new_means = means[first].detach()
        new_scales = (scales[first] if scales is not None else torch.zeros_like(new_means)).detach()
        new_dc = (sh_flat[first, :3] if sh_flat is not None else torch.zeros_like(new_means)).detach()
        new_opacities = (opacities[first] if opacities is not None else torch.zeros(new_means.shape[0], device=means.device, dtype=means.dtype)).detach()
        if source_stream is None:
            source_stream = torch.full((means.shape[0],), -1, dtype=torch.int8, device=means.device)
            source_chunk = torch.full((means.shape[0],), -1, dtype=torch.long, device=means.device)
            source_local = torch.full((means.shape[0],), -1, dtype=torch.long, device=means.device)
        if source_chunk is None or source_local is None:
            raise ValueError("source_stream, source_chunk and source_local must be provided together")
        new_stream, new_chunk, new_local = source_stream[first], source_chunk[first], source_local[first]
        if self._representative_means is None:
            self._occupied, self._representative_means = new_keys, new_means
            self._representative_scales, self._representative_dc = new_scales, new_dc
            self._representative_opacities = new_opacities
            self._representative_stream, self._representative_chunk = new_stream, new_chunk
            self._representative_local = new_local
        else:
            all_keys = torch.cat((self._occupied, new_keys))
            order = all_keys.argsort()
            self._occupied = all_keys[order]
            self._representative_means = torch.cat((self._representative_means, new_means))[order]
            self._representative_scales = torch.cat((self._representative_scales, new_scales))[order]
            self._representative_dc = torch.cat((self._representative_dc, new_dc))[order]
            self._representative_opacities = torch.cat((self._representative_opacities, new_opacities))[order]
            self._representative_stream = torch.cat((self._representative_stream, new_stream))[order]
            self._representative_chunk = torch.cat((self._representative_chunk, new_chunk))[order]
            self._representative_local = torch.cat((self._representative_local, new_local))[order]
        self.num_committed += int(means.shape[0])

    def stats(self) -> dict[str, object]:
        return {
            "applied": True,
            "edge": self.edge,
            "occupied_voxels": self.num_voxels,
            "candidates": self.num_candidates,
            "voxel_matches": self.num_voxel_matches,
            "geometry_matches": self.num_geometry_matches,
            "appearance_matches": self.num_appearance_matches,
            "rejected": self.num_rejected,
            "committed": self.num_committed,
            "reject_ratio": float(self.num_rejected) / float(max(self.num_candidates, 1)),
        }


def _stable_voxel_keys(means: torch.Tensor, *, edge: float) -> torch.Tensor:
    """Translation-stable int64 voxel keys for a persistent scene map."""
    if not bool(torch.isfinite(means).all()):
        raise ValueError("means contain non-finite values")
    coords = torch.floor(means.detach() / float(edge)).to(torch.int64)
    limit = 1 << 20
    if bool((coords.abs() < limit).all()):
        shifted = coords + limit
        return shifted[:, 0] + (shifted[:, 1] << 21) + (shifted[:, 2] << 42)
    return (
        coords[:, 0] * 73856093
        ^ coords[:, 1] * 19349663
        ^ coords[:, 2] * 83492791
    )


def context_count_from_num_frames(num_frames: int) -> int:
    """Map formal protocol total frames → context count."""
    # Formal protocol: C + C//7 frames (every-8 targets). ctx32 → 36 frames.
    mapping = {9: 8, 18: 16, 36: 32, 73: 64, 146: 128}
    if int(num_frames) in mapping:
        return mapping[int(num_frames)]
    # Fallback: every-8 targets → targets ≈ floor((N-1)/8)+something; prefer exact.
    raise ValueError(f"unsupported formal num_frames={num_frames}")


def resolve_global_complement_voxel_factor(
    *,
    schedule: str,
    context_count: int,
    fixed_factor: float = 0.0,
    locked_long_factor: float = 4.0,
) -> float:
    """Resolve eval-time voxel factor from a length schedule.

    Schedules (U2-STATE deterministic arms):
    - ``fixed``: use ``fixed_factor`` as-is
    - ``threshold64``: 0 if ctx<64 else ``locked_long_factor``
    - ``log_b``: ``min(locked, max(0, 2*log2(ctx/8)))`` → 8:0, 16:2, 64+:4
    """
    name = (schedule or "fixed").strip().lower()
    ctx = int(context_count)
    locked = float(locked_long_factor)
    if name in ("fixed", "none", ""):
        return float(fixed_factor)
    if name == "threshold64":
        return locked if ctx >= 64 else 0.0
    if name == "log_b":
        if ctx <= 0:
            return 0.0
        return float(min(locked, max(0.0, 2.0 * math.log2(ctx / 8.0))))
    raise ValueError(f"unsupported global_complement_voxel_schedule: {schedule}")


def _voxel_inverse(means: torch.Tensor, *, edge: float) -> tuple[torch.Tensor, int]:
    if means.ndim != 2 or means.shape[-1] != 3:
        raise ValueError(f"means must have shape (N,3), got {tuple(means.shape)}")
    if not edge > 0:
        raise ValueError("edge must be positive")
    n = int(means.shape[0])
    if n == 0:
        empty = torch.empty(0, dtype=torch.long, device=means.device)
        return empty, 0
    if not bool(torch.isfinite(means).all()):
        raise ValueError("means contain non-finite values")

    coords = torch.floor(means.detach() / float(edge)).to(torch.int64)
    lo = coords.amin(dim=0)
    shifted = coords - lo
    dims = shifted.amax(dim=0) + 1
    max_int64 = torch.iinfo(torch.int64).max
    dx, dy, dz = (int(value.item()) for value in dims)
    if dx <= 0 or dy <= 0 or dz <= 0 or dx > max_int64 // dy or dx * dy > max_int64 // dz:
        raise ValueError("voxel grid cannot be packed into int64")
    keys = shifted[:, 0] + dx * (shifted[:, 1] + dy * shifted[:, 2])
    _, inverse = torch.unique(keys, sorted=True, return_inverse=True)
    num_voxels = int(inverse.max().item()) + 1
    return inverse, num_voxels


def select_voxel_keep_one(
    means: torch.Tensor,
    opacities: torch.Tensor,
    *,
    edge: float,
    policy: VoxelKeepPolicy = "max_opacity",
) -> torch.Tensor:
    """Return one stable index per occupied world-space voxel.

    Policies:
    - ``max_opacity``: highest opacity; opacity ties keep lowest index
    - ``first``: persistent / age preference — always keep lowest index (earlier append)
    """
    n = int(means.shape[0])
    if n == 0:
        return torch.empty(0, dtype=torch.long, device=means.device)
    if opacities.ndim != 1 or opacities.shape[0] != n:
        raise ValueError("opacities must have shape (N,)")
    if not bool(torch.isfinite(opacities).all()):
        raise ValueError("opacities contain non-finite values")

    inverse, num_voxels = _voxel_inverse(means, edge=edge)
    indices = torch.arange(n, device=means.device, dtype=torch.long)
    if policy == "first":
        keep = torch.full((num_voxels,), n, device=means.device, dtype=torch.long)
        keep.scatter_reduce_(0, inverse, indices, reduce="amin", include_self=True)
    elif policy == "max_opacity":
        opacity = opacities.detach()
        max_by_voxel = torch.full(
            (num_voxels,),
            -torch.inf,
            device=opacity.device,
            dtype=opacity.dtype,
        )
        max_by_voxel.scatter_reduce_(0, inverse, opacity, reduce="amax", include_self=True)
        candidate = opacity == max_by_voxel[inverse]
        sentinel = torch.full_like(indices, n)
        candidate_indices = torch.where(candidate, indices, sentinel)
        keep = torch.full((num_voxels,), n, device=means.device, dtype=torch.long)
        keep.scatter_reduce_(0, inverse, candidate_indices, reduce="amin", include_self=True)
    else:
        raise ValueError(f"unsupported voxel keep policy: {policy}")
    if bool((keep == n).any()):
        raise RuntimeError("failed to select a representative for an occupied voxel")
    return keep
