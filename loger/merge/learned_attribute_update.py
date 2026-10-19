"""Conservative learned attribute updater for fixed E8 strict matches.

The module deliberately cannot change Gaussian geometry.  It predicts four
per-observation blend weights (opacity plus SH-DC), which are applied only
after the causal matcher has already selected the ``update`` action.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from loger.merge.online_action_gate import FEATURE_DIM, action_features


class LearnedAttributeUpdater(nn.Module):
    """Predict bounded old/new interpolation weights for one strict match."""

    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 4),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(features))


def updater_features(*, new_means: torch.Tensor, new_scales: torch.Tensor,
                     new_opacities: torch.Tensor, new_dc: torch.Tensor,
                     old_means: torch.Tensor, old_scales: torch.Tensor,
                     old_opacities: torch.Tensor, old_dc: torch.Tensor,
                     stream: torch.Tensor, window_idx: int, num_windows: int) -> torch.Tensor:
    """Shared finite feature encoding for update teacher and online inference."""
    return torch.nan_to_num(action_features(
        new_means=new_means, new_scales=new_scales, new_opacities=new_opacities,
        new_dc=new_dc, old_means=old_means, old_scales=old_scales,
        old_opacities=old_opacities, old_dc=old_dc, stream=stream,
        window_idx=window_idx, num_windows=num_windows,
    ), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)


def predict_attributes(model: LearnedAttributeUpdater, features: torch.Tensor,
                       old_opacities: torch.Tensor, old_dc: torch.Tensor,
                       new_opacities: torch.Tensor, new_dc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return bounded attributes; means, scales and rotations are absent by design."""
    blend = model(features)
    opacity = torch.lerp(old_opacities, new_opacities, blend[:, 0])
    dc = torch.lerp(old_dc, new_dc, blend[:, 1:])
    return opacity, dc


def load_learned_attribute_updater(path: str | Path, device: torch.device) -> tuple[LearnedAttributeUpdater, torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("feature_dim", -1)) != FEATURE_DIM or payload.get("kind") != "e8_strict_attribute_updater":
        raise ValueError(f"{path}: incompatible learned attribute updater checkpoint")
    model = LearnedAttributeUpdater(hidden_dim=int(payload.get("hidden_dim", 32)))
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(), payload["feature_mean"].to(device), payload["feature_std"].to(device).clamp_min(1e-6)
