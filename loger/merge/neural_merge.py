"""Trainable attribute compensation for fixed valid Gaussian tracks.

The first neural-merge pilot deliberately keeps the hand-filtered track
validity mask fixed.  The module learns only how to turn a valid track into a
single Gaussian; this isolates the value of post-merge attribute adaptation
before adding a learned discrete gate.
"""

from __future__ import annotations

import os
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint as _checkpoint


class NeuralTrackMerger(nn.Module):
    """Permutation-invariant track encoder and Gaussian attribute refiner."""

    def __init__(self, sh_dim: int = 27, hidden_dim: int = 128):
        super().__init__()
        self.sh_dim = int(sh_dim)
        # Populated on every forward for the hard-anchored training path.
        # These are residual-space penalties, not a second learnable branch:
        # the compact count, valid tracks and dense complement remain fixed.
        self.last_anchor_losses: dict[str, torch.Tensor] = {}
        # mean/std of position, scale, opacity and SH, plus visible fraction
        self.feature_dim = 6 + 6 + 2 + 2 * self.sh_dim + 1
        self.encoder = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mean_delta = nn.Linear(hidden_dim, 3)
        self.scale_delta = nn.Linear(hidden_dim, 3)
        self.opacity_delta = nn.Linear(hidden_dim, 1)
        self.sh_delta = nn.Linear(hidden_dim, self.sh_dim)
        # Zero initialization makes epoch zero exactly reproduce the existing
        # weighted hard merge, giving a clean identity-initialized baseline.
        for head in (self.mean_delta, self.scale_delta, self.opacity_delta, self.sh_delta):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    @staticmethod
    def _masked_stats(value: torch.Tensor, vis_f: torch.Tensor, denom: torch.Tensor):
        mean = (value * vis_f.unsqueeze(-1)).sum(dim=0) / denom.unsqueeze(-1)
        var = ((value - mean.unsqueeze(0)).square() * vis_f.unsqueeze(-1)).sum(dim=0)
        std = (var / denom.unsqueeze(-1) + 1e-8).sqrt()
        return mean, std

    def forward(
        self,
        centers: torch.Tensor,
        scales: torch.Tensor,
        opacities: torch.Tensor,
        sh: torch.Tensor,
        vis: torch.Tensor,
        base_mean: torch.Tensor,
        base_quat: torch.Tensor,
        base_scale: torch.Tensor,
        base_opacity: torch.Tensor,
        base_sh: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        vis_f = vis.to(dtype=centers.dtype)
        count = vis_f.sum(dim=0).clamp_min(1.0)
        mean_mu, std_mu = self._masked_stats(centers, vis_f, count)
        mean_scale, std_scale = self._masked_stats(scales, vis_f, count)
        mean_sh, std_sh = self._masked_stats(sh, vis_f, count)
        mean_opa = (opacities * vis_f).sum(dim=0) / count
        std_opa = (
            ((opacities - mean_opa.unsqueeze(0)).square() * vis_f).sum(dim=0)
            / count
            + 1e-8
        ).sqrt()
        visible_fraction = count / max(float(vis.shape[0]), 1.0)
        features = torch.cat(
            (mean_mu, std_mu, mean_scale, std_scale,
             mean_opa.unsqueeze(-1), std_opa.unsqueeze(-1),
             mean_sh, std_sh, visible_fraction.unsqueeze(-1)),
            dim=-1,
        )
        features = torch.nan_to_num(features, nan=0.0, posinf=1e4, neginf=-1e4)
        hidden = self.encoder(features)

        # Bounded residuals keep the first pilot stable and prevent the
        # network from escaping by producing invalid Gaussian parameters.
        delta_mean = 0.2 * torch.tanh(self.mean_delta(hidden))
        delta_scale = 0.5 * torch.tanh(self.scale_delta(hidden))
        delta_opacity = self.opacity_delta(hidden).squeeze(-1).clamp(-4.0, 4.0)
        delta_sh = 0.25 * torch.tanh(self.sh_delta(hidden))

        refined_mean = base_mean + delta_mean
        refined_scale = (base_scale.clamp_min(1e-5) * torch.exp(delta_scale)).clamp(1e-5, 10.0)
        base_logit = torch.logit(base_opacity.clamp(1e-5, 1.0 - 1e-5))
        refined_opacity = torch.sigmoid(base_logit + delta_opacity)
        refined_sh = base_sh + delta_sh
        self.last_anchor_losses = {
            "residual": (
                delta_mean.square().mean()
                + delta_scale.square().mean()
                + delta_opacity.square().mean()
                + delta_sh.square().mean()
            ),
            # Holding opacity near the hard value is a lightweight alpha/coverage
            # guard. The hard boolean ownership mask is already fixed upstream;
            # this prevents an early residual model from making compact tracks
            # transparent and relying on incidental background contribution.
            "opacity": (refined_opacity - base_opacity).square().mean(),
        }
        return refined_mean, base_quat, refined_scale, refined_opacity, refined_sh


class NeuralTrackSetMerger(nn.Module):
    """Observation-token track merger initialized to reproduce hard merge.

    Unlike :class:`NeuralTrackMerger`, this module does not reduce a track to
    first and second moments before learning.  Each visible observation remains
    an attention token, so the merger can use multimodal attribute patterns.
    """

    def __init__(self, sh_dim: int = 27, hidden_dim: int = 128, num_heads: int = 4,
                 track_chunk_size: int | None = None):
        super().__init__()
        self.sh_dim = int(sh_dim)
        self.last_anchor_losses: dict[str, torch.Tensor] = {}
        if track_chunk_size is None:
            track_chunk_size = int(os.environ.get("LOGER_NEURAL_TRACK_CHUNK_SIZE", "4096"))
        self.track_chunk_size = max(int(track_chunk_size), 0)
        token_dim = 3 + 3 + 1 + self.sh_dim + 4 + 1
        self.token_encoder = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.pool_score = nn.Linear(hidden_dim, 1)
        self.mean_delta = nn.Linear(hidden_dim, 3)
        self.quat_delta = nn.Linear(hidden_dim, 4)
        self.scale_delta = nn.Linear(hidden_dim, 3)
        self.opacity_delta = nn.Linear(hidden_dim, 1)
        self.sh_delta = nn.Linear(hidden_dim, self.sh_dim)
        for head in (
            self.mean_delta,
            self.quat_delta,
            self.scale_delta,
            self.opacity_delta,
            self.sh_delta,
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        centers: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        opacities: torch.Tensor,
        sh: torch.Tensor,
        vis: torch.Tensor,
        base_mean: torch.Tensor,
        base_quat: torch.Tensor,
        base_scale: torch.Tensor,
        base_opacity: torch.Tensor,
        base_sh: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Fuse ``(time, track, attribute)`` observations without pooling first."""
        num_tracks = int(centers.shape[1])
        chunk = self.track_chunk_size
        if chunk > 0 and num_tracks > chunk:
            outputs = []
            for start in range(0, num_tracks, chunk):
                end = min(start + chunk, num_tracks)
                inputs = (
                    centers[:, start:end], quats[:, start:end], scales[:, start:end],
                    opacities[:, start:end], sh[:, start:end], vis[:, start:end],
                    base_mean[start:end], base_quat[start:end], base_scale[start:end],
                    base_opacity[start:end], base_sh[start:end],
                )
                needs_checkpoint = torch.is_grad_enabled() and any(
                    parameter.requires_grad for parameter in self.parameters()
                )
                result = (
                    _checkpoint(self._forward_impl, *inputs, use_reentrant=False)
                    if needs_checkpoint else self._forward_impl(*inputs)
                )
                outputs.append(result)
            merged = tuple(torch.cat(values, dim=0) for values in zip(*outputs))
            self._record_anchor_from_outputs(merged, base_mean, base_quat, base_scale, base_opacity, base_sh)
            return merged
        return self._forward_impl(
            centers, quats, scales, opacities, sh, vis,
            base_mean, base_quat, base_scale, base_opacity, base_sh,
        )

    def _forward_impl(
        self,
        centers: torch.Tensor, quats: torch.Tensor, scales: torch.Tensor,
        opacities: torch.Tensor, sh: torch.Tensor, vis: torch.Tensor,
        base_mean: torch.Tensor, base_quat: torch.Tensor, base_scale: torch.Tensor,
        base_opacity: torch.Tensor, base_sh: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        eps = 1e-5
        base_quat_per_obs = base_quat.unsqueeze(0).expand_as(quats)
        quat_sign = torch.where(
            (quats * base_quat_per_obs).sum(dim=-1, keepdim=True) < 0,
            -torch.ones_like(quats[..., :1]),
            torch.ones_like(quats[..., :1]),
        )
        aligned_quats = quats * quat_sign
        tokens = torch.cat(
            (
                centers - base_mean.unsqueeze(0),
                torch.log(scales.clamp_min(eps) / base_scale.unsqueeze(0).clamp_min(eps)),
                torch.logit(opacities.clamp(eps, 1.0 - eps)).unsqueeze(-1),
                sh - base_sh.unsqueeze(0),
                aligned_quats - base_quat_per_obs,
                vis.to(dtype=centers.dtype).unsqueeze(-1),
            ),
            dim=-1,
        )
        tokens = torch.nan_to_num(tokens, nan=0.0, posinf=1e4, neginf=-1e4)
        tokens = self.token_encoder(tokens).transpose(0, 1)  # (track, time, hidden)

        visible = vis.transpose(0, 1)
        # MultiheadAttention cannot accept an all-masked sequence.  Such
        # tracks are never selected for merging, but keeping this path finite
        # makes the module safe for diagnostics and tests.
        safe_visible = visible.clone()
        no_visible = ~safe_visible.any(dim=1)
        safe_visible[no_visible, 0] = True
        key_padding_mask = ~safe_visible
        # Pooling consumes only the attended values.  Requesting weights here
        # disables PyTorch's efficient attention path and materializes a large
        # per-track matrix that is never used.
        attended, _ = self.attention(
            tokens, tokens, tokens, key_padding_mask=key_padding_mask, need_weights=False
        )
        hidden = self.attention_norm(tokens + attended)
        hidden = self.ffn_norm(hidden + self.ffn(hidden))

        scores = self.pool_score(hidden).squeeze(-1).masked_fill(~safe_visible, -1e4)
        weights = torch.softmax(scores, dim=-1)
        pooled = (hidden * weights.unsqueeze(-1)).sum(dim=1)
        pooled[no_visible] = 0.0

        delta_mean = 0.2 * torch.tanh(self.mean_delta(pooled))
        delta_quat = 0.25 * torch.tanh(self.quat_delta(pooled))
        delta_scale = 0.5 * torch.tanh(self.scale_delta(pooled))
        delta_opacity = self.opacity_delta(pooled).squeeze(-1).clamp(-4.0, 4.0)
        delta_sh = 0.25 * torch.tanh(self.sh_delta(pooled))

        refined_mean = base_mean + delta_mean
        refined_scale = (base_scale.clamp_min(eps) * torch.exp(delta_scale)).clamp(eps, 10.0)
        base_logit = torch.logit(base_opacity.clamp(eps, 1.0 - eps))
        refined_opacity = torch.sigmoid(base_logit + delta_opacity)
        refined_sh = base_sh + delta_sh
        has_quat_delta = delta_quat.square().sum(dim=-1, keepdim=True) > 0
        normalized_quat = F.normalize(base_quat + delta_quat, dim=-1)
        refined_quat = torch.where(has_quat_delta, normalized_quat, base_quat)
        self._record_anchor_losses(
            delta_mean, delta_quat, delta_scale, delta_opacity, delta_sh,
            refined_opacity, base_opacity,
        )
        return refined_mean, refined_quat, refined_scale, refined_opacity, refined_sh

    def _record_anchor_losses(
        self,
        delta_mean: torch.Tensor,
        delta_quat: torch.Tensor,
        delta_scale: torch.Tensor,
        delta_opacity: torch.Tensor,
        delta_sh: torch.Tensor,
        refined_opacity: torch.Tensor,
        base_opacity: torch.Tensor,
    ) -> None:
        """Expose hard-anchor terms for P1 K=1 residual variants.

        These terms only constrain a residual around the already fixed hard
        track result.  They cannot alter membership, ownership, or the dense
        complement, all of which have been decided before the merger call.
        """
        self.last_anchor_losses = {
            "residual": (
                delta_mean.square().mean()
                + delta_quat.square().mean()
                + delta_scale.square().mean()
                + delta_opacity.square().mean()
                + delta_sh.square().mean()
            ),
            "opacity": (refined_opacity - base_opacity).square().mean(),
        }

    def _record_anchor_from_outputs(
        self, merged: tuple[torch.Tensor, ...], base_mean: torch.Tensor,
        base_quat: torch.Tensor, base_scale: torch.Tensor, base_opacity: torch.Tensor,
        base_sh: torch.Tensor,
    ) -> None:
        mean, quat, scale, opacity, sh = merged
        self.last_anchor_losses = {
            "residual": (
                (mean - base_mean).square().mean()
                + (quat - base_quat).square().mean()
                + torch.log(scale.clamp_min(1e-5) / base_scale.clamp_min(1e-5)).square().mean()
                + (torch.logit(opacity.clamp(1e-5, 1.0 - 1e-5))
                   - torch.logit(base_opacity.clamp(1e-5, 1.0 - 1e-5))).square().mean()
                + (sh - base_sh).square().mean()
            ),
            "opacity": (opacity - base_opacity).square().mean(),
        }


class NeuralTrackFeatureMerger(NeuralTrackSetMerger):
    """Set merger conditioned on pre-head features and per-observation geometry."""

    def __init__(self, sh_dim: int = 27, hidden_dim: int = 128, num_heads: int = 4):
        super().__init__(sh_dim=sh_dim, hidden_dim=hidden_dim, num_heads=num_heads)
        # Attribute token (39) + projected head feature (16) + ray, view,
        # normalized depth and log-confidence (8).
        token_dim = 3 + 3 + 1 + self.sh_dim + 4 + 1 + 16 + 3 + 3 + 1 + 1
        self.token_encoder = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(
        self,
        centers: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        opacities: torch.Tensor,
        sh: torch.Tensor,
        merge_features: torch.Tensor,
        rays: torch.Tensor,
        views: torch.Tensor,
        depths: torch.Tensor,
        confidence: torch.Tensor,
        vis: torch.Tensor,
        base_mean: torch.Tensor,
        base_quat: torch.Tensor,
        base_scale: torch.Tensor,
        base_opacity: torch.Tensor,
        base_sh: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        num_tracks = int(centers.shape[1])
        chunk = self.track_chunk_size
        if chunk > 0 and num_tracks > chunk:
            outputs = []
            for start in range(0, num_tracks, chunk):
                end = min(start + chunk, num_tracks)
                inputs = (
                    centers[:, start:end], quats[:, start:end], scales[:, start:end],
                    opacities[:, start:end], sh[:, start:end], merge_features[:, start:end],
                    rays[:, start:end], views[:, start:end], depths[:, start:end],
                    confidence[:, start:end], vis[:, start:end], base_mean[start:end],
                    base_quat[start:end], base_scale[start:end], base_opacity[start:end], base_sh[start:end],
                )
                needs_checkpoint = torch.is_grad_enabled() and any(
                    parameter.requires_grad for parameter in self.parameters()
                )
                result = (
                    _checkpoint(self._forward_impl, *inputs, use_reentrant=False)
                    if needs_checkpoint else self._forward_impl(*inputs)
                )
                outputs.append(result)
            merged = tuple(torch.cat(values, dim=0) for values in zip(*outputs))
            self._record_anchor_from_outputs(merged, base_mean, base_quat, base_scale, base_opacity, base_sh)
            return merged
        return self._forward_impl(
            centers, quats, scales, opacities, sh, merge_features, rays, views,
            depths, confidence, vis, base_mean, base_quat, base_scale, base_opacity, base_sh,
        )

    def _forward_impl(
        self,
        centers: torch.Tensor, quats: torch.Tensor, scales: torch.Tensor,
        opacities: torch.Tensor, sh: torch.Tensor, merge_features: torch.Tensor,
        rays: torch.Tensor, views: torch.Tensor, depths: torch.Tensor,
        confidence: torch.Tensor, vis: torch.Tensor, base_mean: torch.Tensor,
        base_quat: torch.Tensor, base_scale: torch.Tensor, base_opacity: torch.Tensor,
        base_sh: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        eps = 1e-5
        base_quat_per_obs = base_quat.unsqueeze(0).expand_as(quats)
        quat_sign = torch.where(
            (quats * base_quat_per_obs).sum(dim=-1, keepdim=True) < 0,
            -torch.ones_like(quats[..., :1]),
            torch.ones_like(quats[..., :1]),
        )
        # ``torch.nanmedian`` on CUDA emits indices and has no deterministic
        # implementation. Sorting visible values followed by indexed selection
        # produces the same lower median convention without that kernel.
        visible_count = vis.sum(dim=0)
        sorted_depths = torch.sort(
            torch.where(vis, depths.clamp_min(eps), torch.full_like(depths, float("inf"))),
            dim=0,
        ).values
        median_index = ((visible_count - 1) // 2).clamp_min(0).unsqueeze(0)
        median_depth = sorted_depths.gather(0, median_index).squeeze(0)
        median_depth = torch.where(visible_count > 0, median_depth, torch.ones_like(median_depth))
        tokens = torch.cat(
            (
                centers - base_mean.unsqueeze(0),
                torch.log(scales.clamp_min(eps) / base_scale.unsqueeze(0).clamp_min(eps)),
                torch.logit(opacities.clamp(eps, 1.0 - eps)).unsqueeze(-1),
                sh - base_sh.unsqueeze(0),
                quats * quat_sign - base_quat_per_obs,
                vis.to(dtype=centers.dtype).unsqueeze(-1),
                merge_features,
                F.normalize(rays, dim=-1),
                F.normalize(views, dim=-1),
                torch.log(depths.clamp_min(eps) / median_depth.unsqueeze(0)).unsqueeze(-1),
                torch.log(confidence.clamp_min(eps)).unsqueeze(-1),
            ),
            dim=-1,
        )
        tokens = torch.nan_to_num(tokens, nan=0.0, posinf=1e4, neginf=-1e4)
        encoded = self.token_encoder(tokens).transpose(0, 1)
        visible = vis.transpose(0, 1)
        safe_visible = visible.clone()
        no_visible = ~safe_visible.any(dim=1)
        safe_visible[no_visible, 0] = True
        attended, _ = self.attention(
            encoded, encoded, encoded, key_padding_mask=~safe_visible, need_weights=False
        )
        hidden = self.attention_norm(encoded + attended)
        hidden = self.ffn_norm(hidden + self.ffn(hidden))
        scores = self.pool_score(hidden).squeeze(-1).masked_fill(~safe_visible, -1e4)
        pooled = (hidden * torch.softmax(scores, dim=-1).unsqueeze(-1)).sum(dim=1)
        pooled[no_visible] = 0.0

        delta_mean = 0.2 * torch.tanh(self.mean_delta(pooled))
        delta_quat = 0.25 * torch.tanh(self.quat_delta(pooled))
        delta_scale = 0.5 * torch.tanh(self.scale_delta(pooled))
        delta_opacity = self.opacity_delta(pooled).squeeze(-1).clamp(-4.0, 4.0)
        delta_sh = 0.25 * torch.tanh(self.sh_delta(pooled))
        refined_scale = (base_scale.clamp_min(eps) * torch.exp(delta_scale)).clamp(eps, 10.0)
        refined_opacity = torch.sigmoid(torch.logit(base_opacity.clamp(eps, 1.0 - eps)) + delta_opacity)
        has_quat_delta = delta_quat.square().sum(dim=-1, keepdim=True) > 0
        refined_quat = torch.where(has_quat_delta, F.normalize(base_quat + delta_quat, dim=-1), base_quat)
        self._record_anchor_losses(
            delta_mean, delta_quat, delta_scale, delta_opacity, delta_sh,
            refined_opacity, base_opacity,
        )
        return base_mean + delta_mean, refined_quat, refined_scale, refined_opacity, base_sh + delta_sh


class NeuralTrackDirectSlotMerger(nn.Module):
    """Feature-first track decoder that emits three direct Gaussian slots."""

    def __init__(self, sh_dim: int = 27, hidden_dim: int = 128, num_heads: int = 4, num_slots: int = 3,
                 track_chunk_size: int | None = None):
        super().__init__()
        self.sh_dim = int(sh_dim)
        self.num_slots = int(num_slots)
        # Tracks are independent.  Chunking plus non-reentrant checkpointing
        # keeps the attention graph from materializing for all tracks at once;
        # outputs and gradients are mathematically unchanged.
        if track_chunk_size is None:
            track_chunk_size = int(os.environ.get("LOGER_NEURAL_TRACK_CHUNK_SIZE", "1024"))
        self.track_chunk_size = max(int(track_chunk_size), 0)
        token_dim = 256 + 3 + 3 + 3 + 3 + 3 + 1 + 1 + 1 + 1
        self.token_encoder = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.slot_queries = nn.Parameter(torch.empty(self.num_slots, hidden_dim))
        nn.init.normal_(self.slot_queries, std=0.02)
        self.cross_attention = nn.ModuleList(
            [nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True) for _ in range(2)]
        )
        self.attention_norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(2)])
        self.ffn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, 2 * hidden_dim),
                    nn.GELU(),
                    nn.Linear(2 * hidden_dim, hidden_dim),
                )
                for _ in range(2)
            ]
        )
        self.ffn_norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(2)])
        self.mean_delta = nn.Linear(hidden_dim, 3)
        self.scale_delta = nn.Linear(hidden_dim, 3)
        self.quat_head = nn.Linear(hidden_dim, 4)
        self.opacity_head = nn.Linear(hidden_dim, 1)
        self.sh_head = nn.Linear(hidden_dim, self.sh_dim)
        for head in (self.mean_delta, self.scale_delta, self.sh_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.zeros_(self.quat_head.weight)
        nn.init.constant_(self.quat_head.bias, 0.0)
        self.quat_head.bias.data[0] = 1.0
        nn.init.zeros_(self.opacity_head.weight)
        nn.init.constant_(self.opacity_head.bias, -2.0)
        sh_mask = torch.ones(self.sh_dim)
        for index in (0, 9, 18):
            if index < self.sh_dim:
                sh_mask[index] = 0.0
        self.register_buffer("sh_residual_mask", sh_mask)

    @staticmethod
    def _masked_weighted_mean(value: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        denom = weights.sum(dim=0).clamp_min(1e-5)
        return (value * weights.unsqueeze(-1)).sum(dim=0) / denom.unsqueeze(-1)

    def forward(
        self,
        prehead_features: torch.Tensor,
        rgb: torch.Tensor,
        source_points: torch.Tensor,
        local_points: torch.Tensor,
        rays: torch.Tensor,
        views: torch.Tensor,
        depths: torch.Tensor,
        footprints: torch.Tensor,
        confidence: torch.Tensor,
        vis: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        chunk = self.track_chunk_size
        num_tracks = int(source_points.shape[1])
        if chunk <= 0 or num_tracks <= chunk:
            return self._forward_impl(
                prehead_features, rgb, source_points, local_points, rays,
                views, depths, footprints, confidence, vis,
            )
        outputs = []
        for start in range(0, num_tracks, chunk):
            end = min(start + chunk, num_tracks)
            inputs = (
                prehead_features[:, start:end], rgb[:, start:end],
                source_points[:, start:end], local_points[:, start:end],
                rays[:, start:end], views[:, start:end], depths[:, start:end],
                footprints[:, start:end], confidence[:, start:end], vis[:, start:end],
            )
            # Recompute attention activations during backward instead of
            # retaining every token/attention tensor for the full scene.
            # Checkpoint whenever either side of the merger is trainable.
            # In the merger-only schedule the dense/head tensors are detached,
            # but the merger parameters still require gradients.  Looking only
            # at input.requires_grad incorrectly retained every attention
            # activation for the full chunk and caused S2 forward OOM.
            needs_checkpoint = torch.is_grad_enabled() and (
                any(
                    isinstance(value, torch.Tensor) and value.is_floating_point()
                    and value.requires_grad for value in inputs
                )
                or any(parameter.requires_grad for parameter in self.parameters())
            )
            if needs_checkpoint:
                result = _checkpoint(self._forward_impl, *inputs, use_reentrant=False)
            else:
                result = self._forward_impl(*inputs)
            outputs.append(result)
        return tuple(torch.cat(values, dim=0) for values in zip(*outputs))

    def _forward_impl(
        self,
        prehead_features: torch.Tensor,
        rgb: torch.Tensor,
        source_points: torch.Tensor,
        local_points: torch.Tensor,
        rays: torch.Tensor,
        views: torch.Tensor,
        depths: torch.Tensor,
        footprints: torch.Tensor,
        confidence: torch.Tensor,
        vis: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Decode direct Gaussian slots from ``(time, track, feature)`` observations."""
        eps = 1e-5
        vis_f = vis.to(dtype=prehead_features.dtype)
        source_weights = vis_f * confidence.clamp_min(eps)
        origin = self._masked_weighted_mean(source_points, source_weights)
        centered_points = source_points - origin.unsqueeze(0)
        base_footprint = self._masked_weighted_mean(
            footprints.unsqueeze(-1), source_weights
        ).squeeze(-1).clamp_min(eps)
        radius = self._masked_weighted_mean(
            centered_points.norm(dim=-1, keepdim=True), source_weights
        ).squeeze(-1).maximum(base_footprint)

        visible_depths = torch.where(vis, depths.clamp_min(eps), torch.full_like(depths, float("inf")))
        sorted_depths = torch.sort(visible_depths, dim=0).values
        visible_count = vis.sum(dim=0)
        median_index = ((visible_count - 1) // 2).clamp_min(0).unsqueeze(0)
        median_depth = sorted_depths.gather(0, median_index).squeeze(0)
        median_depth = torch.where(visible_count > 0, median_depth, torch.ones_like(median_depth))
        tokens = torch.cat(
            (
                prehead_features,
                rgb,
                centered_points,
                local_points,
                F.normalize(rays, dim=-1),
                F.normalize(views, dim=-1),
                torch.log(depths.clamp_min(eps) / median_depth.unsqueeze(0)).unsqueeze(-1),
                torch.log(footprints.clamp_min(eps)).unsqueeze(-1),
                torch.log(confidence.clamp_min(eps)).unsqueeze(-1),
                vis_f.unsqueeze(-1),
            ),
            dim=-1,
        )
        encoded = self.token_encoder(torch.nan_to_num(tokens, nan=0.0, posinf=1e4, neginf=-1e4)).transpose(0, 1)
        visible = vis.transpose(0, 1)
        safe_visible = visible.clone()
        no_visible = ~safe_visible.any(dim=1)
        safe_visible[no_visible, 0] = True
        slots = self.slot_queries.unsqueeze(0).expand(encoded.shape[0], -1, -1)
        attention_weights = None
        for attention, attention_norm, ffn, ffn_norm in zip(
            self.cross_attention, self.attention_norm, self.ffn, self.ffn_norm
        ):
            attended, attention_weights = attention(
                slots, encoded, encoded, key_padding_mask=~safe_visible, need_weights=True
            )
            slots = attention_norm(slots + attended)
            slots = ffn_norm(slots + ffn(slots))

        assert attention_weights is not None
        observation_weights = attention_weights * visible.unsqueeze(1).to(attention_weights.dtype)
        observation_weights = observation_weights / observation_weights.sum(dim=-1, keepdim=True).clamp_min(eps)
        points_by_track = source_points.transpose(0, 1)
        rgb_by_track = rgb.transpose(0, 1)
        footprint_by_track = footprints.transpose(0, 1)
        barycenter = torch.einsum("mkt,mtd->mkd", observation_weights, points_by_track)
        slot_rgb = torch.einsum("mkt,mtd->mkd", observation_weights, rgb_by_track)
        slot_footprint = torch.einsum("mkt,mt->mk", observation_weights, footprint_by_track).clamp_min(eps)

        mean = barycenter + 2.0 * radius[:, None, None] * torch.tanh(self.mean_delta(slots))
        scale = slot_footprint.unsqueeze(-1) * torch.exp(1.5 * torch.tanh(self.scale_delta(slots)))
        quat = F.normalize(self.quat_head(slots), dim=-1)
        opacity = torch.sigmoid(self.opacity_head(slots).squeeze(-1))
        sh = self.sh_head(slots) * self.sh_residual_mask
        c0 = 0.28209479177387814
        for index, channel in ((0, 0), (9, 1), (18, 2)):
            if index < self.sh_dim:
                sh[..., index] = (slot_rgb[..., channel] - 0.5) / c0
        invisible_slots = no_visible[:, None]
        mean = torch.where(
            invisible_slots.unsqueeze(-1), origin[:, None], mean
        )
        scale = torch.where(
            invisible_slots.unsqueeze(-1), base_footprint[:, None, None], scale
        )
        opacity = torch.where(invisible_slots, torch.zeros_like(opacity), opacity)
        return (
            mean.reshape(-1, 3),
            quat.reshape(-1, 4),
            scale.reshape(-1, 3).clamp_min(eps),
            opacity.reshape(-1),
            sh.reshape(-1, self.sh_dim),
        )


class NeuralTrackLatentMerger(NeuralTrackDirectSlotMerger):
    """K=1 latent-fusion merger: masked attention pool over prehead tokens.

    Grilling-session design for the latent-fusion pilot (cancelled ownership,
    soft-max coverage, one output Gaussian per track). Architecturally
    identical to :class:`NeuralTrackDirectSlotMerger` restricted to a single
    slot: a learned query performs masked cross-attention pooling over the
    prehead-feature + lightweight geometric/confidence observation tokens,
    then decodes mean as a weighted barycenter plus a bounded delta, scale
    anchored to the observed footprint with an exponential delta, and
    quaternion/opacity/SH as absolute decodes. Kept as a distinct class (not
    just ``NeuralTrackDirectSlotMerger(num_slots=1)``) so checkpoints, run.py
    training wiring, and experiment logs do not conflate this K=1 pilot with
    the rejected M3 3-slot pilot.
    """

    def __init__(self, sh_dim: int = 27, hidden_dim: int = 128, num_heads: int = 4,
                 track_chunk_size: int | None = None):
        super().__init__(
            sh_dim=sh_dim, hidden_dim=hidden_dim, num_heads=num_heads,
            num_slots=1, track_chunk_size=track_chunk_size,
        )
