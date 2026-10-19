"""Compact Gaussian data structures for track-guided merge."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


@dataclass
class CompactGaussianEntry:
    """One merged (or singleton) compact Gaussian."""

    mean: torch.Tensor
    quat: torch.Tensor
    scale: torch.Tensor
    opacity: torch.Tensor
    sh_flat: torch.Tensor
    merge_weight: torch.Tensor
    source_count: torch.Tensor
    track_meta: Dict[str, Any]


@dataclass
class CompactGaussianChunk:
    """Per-chunk compact Gaussians after intra-chunk merge."""

    means: torch.Tensor
    quats: torch.Tensor
    scales: torch.Tensor
    opacities: torch.Tensor
    sh_flat: torch.Tensor
    merge_weights: torch.Tensor
    source_counts: torch.Tensor
    global_frame_span: Tuple[int, int]
    track_meta: List[Dict[str, Any]]
    chunk_idx: int = 0

    @classmethod
    def from_entries(
        cls,
        entries: Sequence[CompactGaussianEntry],
        global_frame_span: Tuple[int, int],
        chunk_idx: int = 0,
    ) -> "CompactGaussianChunk":
        if not entries:
            device = torch.device("cpu")
            empty_f = torch.empty(0, device=device)
            empty_3 = torch.empty(0, 3, device=device)
            empty_4 = torch.empty(0, 4, device=device)
            return cls(
                means=empty_3,
                quats=empty_4,
                scales=empty_3,
                opacities=empty_f,
                sh_flat=torch.empty(0, 0, device=device),
                merge_weights=empty_f,
                source_counts=empty_f,
                global_frame_span=global_frame_span,
                track_meta=[],
                chunk_idx=chunk_idx,
            )

        return cls(
            means=torch.stack([e.mean for e in entries]),
            quats=torch.stack([e.quat for e in entries]),
            scales=torch.stack([e.scale for e in entries]),
            opacities=torch.stack([e.opacity for e in entries]),
            sh_flat=torch.stack([e.sh_flat for e in entries]),
            merge_weights=torch.stack([e.merge_weight for e in entries]),
            source_counts=torch.stack([e.source_count for e in entries]),
            global_frame_span=global_frame_span,
            track_meta=[e.track_meta for e in entries],
            chunk_idx=chunk_idx,
        )

    @classmethod
    def from_stacked(
        cls,
        means: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        opacities: torch.Tensor,
        sh_flat: torch.Tensor,
        merge_weights: torch.Tensor,
        source_counts: torch.Tensor,
        track_meta: List[Dict[str, Any]],
        global_frame_span: Tuple[int, int],
        chunk_idx: int = 0,
    ) -> "CompactGaussianChunk":
        return cls(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            sh_flat=sh_flat,
            merge_weights=merge_weights,
            source_counts=source_counts,
            global_frame_span=global_frame_span,
            track_meta=track_meta,
            chunk_idx=chunk_idx,
        )

    def __len__(self) -> int:
        return int(self.means.shape[0])

    def entry(self, idx: int) -> CompactGaussianEntry:
        return CompactGaussianEntry(
            mean=self.means[idx],
            quat=self.quats[idx],
            scale=self.scales[idx],
            opacity=self.opacities[idx],
            sh_flat=self.sh_flat[idx],
            merge_weight=self.merge_weights[idx],
            source_count=self.source_counts[idx],
            track_meta=self.track_meta[idx],
        )


@dataclass
class FinalCompactGaussians:
    """Flattened compact Gaussians ready for rendering."""

    means: torch.Tensor
    quats: torch.Tensor
    scales: torch.Tensor
    opacities: torch.Tensor
    sh_flat: torch.Tensor
    merge_weights: torch.Tensor
    source_counts: torch.Tensor
    stats: Dict[str, Any] = field(default_factory=dict)
    covered_pixels: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return int(self.means.shape[0])

    def as_render_tensors(
        self,
        sh_degree: int = 2,
        opacity_threshold: float = 0.01,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (means, quats, scales, opacities, colors) in gsplat layout."""
        sh_num_bases = (sh_degree + 1) ** 2
        colors = self.sh_flat.reshape(-1, 3, sh_num_bases).transpose(1, 2).contiguous()
        opacities = self.opacities
        if opacity_threshold > 0 and opacities.numel() > 0:
            mask = opacities > opacity_threshold
            if mask.any():
                means = self.means[mask]
                quats = self.quats[mask]
                scales = self.scales[mask]
                opacities = opacities[mask]
                colors = colors[mask]
            else:
                top_k = max(1, int(len(opacities) * 0.1))
                _, indices = torch.topk(opacities, top_k)
                mask = torch.zeros_like(opacities, dtype=torch.bool)
                mask[indices] = True
                means = self.means[mask]
                quats = self.quats[mask]
                scales = self.scales[mask]
                opacities = opacities[mask]
                colors = colors[mask]
        else:
            means = self.means
            quats = self.quats
            scales = self.scales
        return means, quats, scales, opacities, colors

    @classmethod
    def from_entries(
        cls,
        entries: Sequence[CompactGaussianEntry],
        stats: Optional[Dict[str, Any]] = None,
    ) -> "FinalCompactGaussians":
        if not entries:
            device = torch.device("cpu")
            return cls(
                means=torch.empty(0, 3, device=device),
                quats=torch.empty(0, 4, device=device),
                scales=torch.empty(0, 3, device=device),
                opacities=torch.empty(0, device=device),
                sh_flat=torch.empty(0, 0, device=device),
                merge_weights=torch.empty(0, device=device),
                source_counts=torch.empty(0, device=device),
                stats=stats or {},
            )
        return cls(
            means=torch.stack([e.mean for e in entries]),
            quats=torch.stack([e.quat for e in entries]),
            scales=torch.stack([e.scale for e in entries]),
            opacities=torch.stack([e.opacity for e in entries]),
            sh_flat=torch.stack([e.sh_flat for e in entries]),
            merge_weights=torch.stack([e.merge_weight for e in entries]),
            source_counts=torch.stack([e.source_count for e in entries]),
            stats=stats or {},
        )

    @classmethod
    def from_stacked(
        cls,
        means: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        opacities: torch.Tensor,
        sh_flat: torch.Tensor,
        merge_weights: torch.Tensor,
        source_counts: torch.Tensor,
        track_meta: Optional[List[Dict[str, Any]]] = None,
        stats: Optional[Dict[str, Any]] = None,
    ) -> "FinalCompactGaussians":
        del track_meta  # final set does not retain per-track metadata
        return cls(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            sh_flat=sh_flat,
            merge_weights=merge_weights,
            source_counts=source_counts,
            stats=stats or {},
        )
