"""Utilities for constraining predicted pointmaps to known camera rays."""

from __future__ import annotations

import torch


def expand_intrinsics(
    intrinsics: torch.Tensor,
    *,
    batch_size: int,
    num_frames: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return intrinsics as ``[B, N, 3, 3]`` without changing their values."""
    value = torch.as_tensor(intrinsics, device=device, dtype=dtype)
    if value.shape == (3, 3):
        value = value.view(1, 1, 3, 3).expand(batch_size, num_frames, -1, -1)
    elif value.ndim == 3 and value.shape[-2:] == (3, 3):
        if value.shape[0] != num_frames:
            raise ValueError(
                f"per-frame intrinsics have {value.shape[0]} frames, expected {num_frames}"
            )
        value = value.unsqueeze(0).expand(batch_size, -1, -1, -1)
    elif value.ndim == 4 and value.shape[-2:] == (3, 3):
        if value.shape[:2] != (batch_size, num_frames):
            raise ValueError(
                "batched intrinsics shape mismatch: "
                f"got {tuple(value.shape)}, expected ({batch_size}, {num_frames}, 3, 3)"
            )
    else:
        raise ValueError(
            "intrinsics must have shape [3,3], [N,3,3], or [B,N,3,3]; "
            f"got {tuple(value.shape)}"
        )
    if not torch.isfinite(value).all():
        raise ValueError("intrinsics contain non-finite values")
    if (value[..., 0, 0] <= 0).any() or (value[..., 1, 1] <= 0).any():
        raise ValueError("intrinsics focal lengths must be positive")
    return value


def calibrate_local_points_to_intrinsics(
    local_points: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    """Keep predicted z-depth while snapping x/y to the rays defined by K.

    Pixel centers follow the same ``+0.5`` convention as Pi3X's ``get_pixel``.
    ``local_points`` must be ``[B,N,H,W,3]`` and ``intrinsics`` ``[B,N,3,3]``.
    """
    if local_points.ndim != 5 or local_points.shape[-1] != 3:
        raise ValueError(
            f"local_points must have shape [B,N,H,W,3], got {tuple(local_points.shape)}"
        )
    batch_size, num_frames, height, width, _ = local_points.shape
    if intrinsics.shape != (batch_size, num_frames, 3, 3):
        raise ValueError(
            "intrinsics shape mismatch: "
            f"got {tuple(intrinsics.shape)}, expected ({batch_size}, {num_frames}, 3, 3)"
        )

    dtype = local_points.dtype
    device = local_points.device
    rows, cols = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype) + 0.5,
        torch.arange(width, device=device, dtype=dtype) + 0.5,
        indexing="ij",
    )
    cols = cols.view(1, 1, height, width)
    rows = rows.view(1, 1, height, width)
    K = intrinsics.to(device=device, dtype=dtype)
    fx = K[..., 0, 0, None, None]
    fy = K[..., 1, 1, None, None]
    cx = K[..., 0, 2, None, None]
    cy = K[..., 1, 2, None, None]
    depth = local_points[..., 2]
    x = (cols - cx) * depth / fx
    y = (rows - cy) * depth / fy
    return torch.stack((x, y, depth), dim=-1)
