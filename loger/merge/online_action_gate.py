"""Small, bounded E8 candidate-level action gate.

The gate is intentionally restricted to attributes: its outputs select
discard/update/spawn after strict geometry/DC association has succeeded.
It never changes means, scales, or rotations.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


ACTION_NAMES = ("discard", "update", "spawn")
FEATURE_DIM = 12


def action_features(
    *, new_means: torch.Tensor, new_scales: torch.Tensor,
    new_opacities: torch.Tensor, new_dc: torch.Tensor,
    old_means: torch.Tensor, old_scales: torch.Tensor,
    old_opacities: torch.Tensor, old_dc: torch.Tensor,
    stream: torch.Tensor, window_idx: int, num_windows: int,
) -> torch.Tensor:
    """Features shared by B1 export/training/inference, shape ``(N, 12)``."""
    if new_means.numel() == 0:
        return new_means.new_empty((0, FEATURE_DIM))
    scale_ratio = torch.log(new_scales.clamp_min(1e-6) / old_scales.clamp_min(1e-6))
    window = new_means.new_full(
        (new_means.shape[0], 1), float(window_idx) / max(int(num_windows) - 1, 1)
    )
    return torch.cat((
        new_means - old_means, scale_ratio, new_dc - old_dc,
        (new_opacities - old_opacities).unsqueeze(1), stream.to(new_means.dtype).unsqueeze(1), window,
    ), dim=1)


class OnlineActionGate(nn.Module):
    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def load_action_gate(path: str | Path, device: torch.device) -> OnlineActionGate:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    hidden_dim = int(payload.get("hidden_dim", 32))
    gate = OnlineActionGate(hidden_dim=hidden_dim)
    gate.load_state_dict(payload["state_dict"])
    return gate.to(device).eval()
