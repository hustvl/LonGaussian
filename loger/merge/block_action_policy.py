"""Shared P2 block-action policy definition and inference helpers."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

ACTION_ORDER = ("discard", "update", "spawn")
FEATURE_DIM = 13


def block_action_features(*, new_means: torch.Tensor, new_scales: torch.Tensor, new_opacities: torch.Tensor,
                          new_dc: torch.Tensor, old_means: torch.Tensor, old_scales: torch.Tensor,
                          old_opacities: torch.Tensor, old_dc: torch.Tensor, strict: torch.Tensor,
                          stream: torch.Tensor, window_idx: int, num_windows: int) -> torch.Tensor:
    """Per-candidate evidence used before pooling to one block action."""
    old_means = torch.where(strict[:, None], old_means, new_means)
    old_scales = torch.where(strict[:, None], old_scales, new_scales)
    old_opacities = torch.where(strict, old_opacities, new_opacities)
    old_dc = torch.where(strict[:, None], old_dc, new_dc)
    window = new_means.new_full((new_means.shape[0], 1), float(window_idx) / max(num_windows - 1, 1))
    x = torch.cat((new_means - old_means, torch.log(new_scales.clamp_min(1e-6) / old_scales.clamp_min(1e-6)),
                   new_dc - old_dc, (new_opacities - old_opacities).unsqueeze(1), stream.float().unsqueeze(1),
                   window, strict.float().unsqueeze(1)), dim=1)
    return torch.nan_to_num(x, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)


class BlockTeacherPolicy(nn.Module):
    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.point = nn.Sequential(nn.Linear(FEATURE_DIM, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.block = nn.Sequential(nn.Linear(hidden_dim * 3 + 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3))

    def forward_block(self, x: torch.Tensor, *, is_strict: bool) -> torch.Tensor:
        h = self.point(x)
        pooled = torch.cat((h.mean(0), h.amax(0), h.std(0, unbiased=False)), dim=0)
        meta = x.new_tensor([float(x.shape[0]) / 64.0, float(is_strict)])
        return self.block(torch.cat((pooled, meta)).unsqueeze(0)).squeeze(0)


def load_block_action_policy(path: str | Path, device: torch.device) -> tuple[BlockTeacherPolicy, torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if tuple(payload.get("action_order", ())) != ACTION_ORDER or int(payload.get("feature_dim", -1)) != FEATURE_DIM:
        raise ValueError(f"{path}: incompatible block-action checkpoint")
    model = BlockTeacherPolicy(hidden_dim=int(payload.get("hidden_dim", 32)))
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(), payload["feature_mean"].to(device), payload["feature_std"].to(device).clamp_min(1e-6)
