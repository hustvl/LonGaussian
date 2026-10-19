"""PROTOTYPE: replace finite-depth sky geometry with a fixed world sphere.

This is deliberately post-alignment and inference-only.  It answers whether a
directional far-field Gaussian shell removes low-altitude sky artifacts while
retaining sky appearance.  It is not a production representation.
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from loger.merge.window_merge import WindowMergeResult


C0 = 0.28209479177387814


def append_fixed_sky_sphere(
    result: WindowMergeResult,
    images: torch.Tensor,
    frame_indices: Sequence[int],
    intrinsics: torch.Tensor,
    sky_masker,
    sky_sh_coeffs: Optional[torch.Tensor] = None,
    *,
    radius_factor: float = 50.0,
    sample_stride: int = 2,
    azimuth_bins: int = 512,
    elevation_bins: int = 256,
    opacity: float = 0.98,
    scale_overlap: float = 1.0,
    color_mode: Literal["rgb_dc", "predicted_sh", "rgb_dc_predicted_residual"] = "rgb_dc",
) -> WindowMergeResult:
    """Append one far-field Gaussian per occupied world-direction bin.

    ``rgb_dc`` is the stable environment-map baseline: observed RGB is stored
    only in degree-0 SH.  The other modes aggregate the head-predicted degree-2
    SH for the same world-direction bins.  ``rgb_dc_predicted_residual`` keeps
    observed RGB for DC and imports only the predicted degree-1/2 residual.
    """
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"images must be (T,3,H,W), got {tuple(images.shape)}")
    if result.camera_poses.ndim == 4:
        poses = result.camera_poses[0]
    else:
        poses = result.camera_poses
    if radius_factor <= 1 or sample_stride < 1:
        raise ValueError("radius_factor must be > 1 and sample_stride must be >= 1")
    if color_mode not in ("rgb_dc", "predicted_sh", "rgb_dc_predicted_residual"):
        raise ValueError(f"unsupported sky color mode: {color_mode}")
    if color_mode != "rgb_dc":
        if sky_sh_coeffs is None:
            raise ValueError(f"{color_mode} requires sky_sh_coeffs")
        if sky_sh_coeffs.ndim == 5 and sky_sh_coeffs.shape[0] == 1:
            sky_sh_coeffs = sky_sh_coeffs[0]
        if sky_sh_coeffs.ndim != 4 or sky_sh_coeffs.shape[0] != images.shape[0]:
            raise ValueError(
                "sky_sh_coeffs must be (T, 3*K, H, W) aligned with images, got "
                f"{tuple(sky_sh_coeffs.shape)}"
            )

    device = result.means.device
    dtype = result.means.dtype
    # Sky Gaussians must land on the same device as the scene cloud (which may
    # be CPU after streaming robust depth filtering finalize).
    frame_ids = [int(i) for i in frame_indices]
    centers = poses[frame_ids, :3, 3].to(device=device, dtype=dtype)
    sphere_center = centers.median(dim=0).values
    distance = (result.means - sphere_center).norm(dim=-1)
    finite = torch.isfinite(distance)
    if not bool(finite.any()):
        raise RuntimeError("cannot estimate sky sphere radius from non-finite scene cloud")
    scene_radius = torch.quantile(distance[finite].float(), 0.90).to(dtype).clamp_min(1e-3)
    sphere_radius = scene_radius * float(radius_factor)

    all_dirs = []
    all_rgb = []
    all_sh = []
    for frame_idx in frame_ids:
        rgb_chw = images[frame_idx].detach().clamp(0, 1)
        rgb_uint8 = (
            rgb_chw.permute(1, 2, 0).mul(255).round().to(torch.uint8).cpu().numpy()
        )
        sky = sky_masker.is_sky_rgb_uint8(rgb_uint8)
        rows, cols = np.nonzero(sky[::sample_stride, ::sample_stride])
        if rows.size == 0:
            continue
        rows = rows * sample_stride
        cols = cols * sample_stride
        uv1 = torch.from_numpy(
            np.stack((cols + 0.5, rows + 0.5, np.ones_like(cols)), axis=-1)
        ).to(device=device, dtype=dtype)
        K = intrinsics[frame_idx] if intrinsics.ndim == 3 else intrinsics
        rays_camera = F.normalize(uv1 @ torch.inverse(K.to(device=device, dtype=dtype)).T, dim=-1)
        rotation = poses[frame_idx, :3, :3].to(device=device, dtype=dtype)
        rays_world = F.normalize(rays_camera @ rotation.T, dim=-1)
        rgb = rgb_chw[:, rows, cols].T.to(device=device, dtype=dtype)
        all_dirs.append(rays_world)
        all_rgb.append(rgb)
        if sky_sh_coeffs is not None:
            all_sh.append(
                sky_sh_coeffs[frame_idx, :, rows, cols].T.to(device=device, dtype=result.sh_flat.dtype)
            )

    if not all_dirs:
        stats = dict(result.stats)
        stats["sky_sphere_prototype"] = {"applied": False, "reason": "no_sky_pixels"}
        result.stats = stats
        return result

    directions = torch.cat(all_dirs, dim=0)
    colors = torch.cat(all_rgb, dim=0)
    source_sh = torch.cat(all_sh, dim=0) if all_sh else None
    azimuth = torch.atan2(directions[:, 0], directions[:, 2])
    elevation = torch.asin(directions[:, 1].clamp(-1, 1))
    az = (((azimuth + torch.pi) / (2 * torch.pi)) * azimuth_bins).floor().long()
    el = (((elevation + torch.pi / 2) / torch.pi) * elevation_bins).floor().long()
    az = az.clamp(0, azimuth_bins - 1)
    el = el.clamp(0, elevation_bins - 1)
    keys = el * azimuth_bins + az
    unique, inverse = torch.unique(keys, sorted=True, return_inverse=True)
    count = torch.bincount(inverse, minlength=unique.numel()).to(dtype).clamp_min(1)

    fused_dirs = torch.zeros(unique.numel(), 3, device=device, dtype=dtype)
    fused_rgb = torch.zeros_like(fused_dirs)
    fused_dirs.index_add_(0, inverse, directions)
    fused_rgb.index_add_(0, inverse, colors)
    fused_dirs = F.normalize(fused_dirs / count[:, None], dim=-1)
    fused_rgb = (fused_rgb / count[:, None]).clamp(0, 1)

    sky_means = sphere_center + sphere_radius * fused_dirs
    sky_quats = torch.zeros(unique.numel(), 4, device=device, dtype=dtype)
    sky_quats[:, 0] = 1
    angular_step = max(2 * torch.pi / azimuth_bins, torch.pi / elevation_bins)
    sky_scale = sphere_radius * float(angular_step) * float(scale_overlap)
    sky_scales = torch.full(
        (unique.numel(), 3), float(sky_scale.item()), device=device, dtype=dtype
    )
    sky_opacities = torch.full(
        (unique.numel(),), float(opacity), device=device, dtype=dtype
    )
    sh_dim = int(result.sh_flat.shape[-1])
    sky_sh = torch.zeros(unique.numel(), sh_dim, device=device, dtype=result.sh_flat.dtype)
    if sh_dim < 19:
        raise ValueError("sky sphere prototype expects degree-2 RGB SH layout")
    dc = (fused_rgb.to(sky_sh.dtype) - 0.5) / C0
    if color_mode != "rgb_dc":
        if source_sh is None or source_sh.shape[-1] != sh_dim:
            raise ValueError("predicted sky SH layout does not match merged SH layout")
        sky_sh.index_add_(0, inverse, source_sh)
        sky_sh = sky_sh / count[:, None]
    if color_mode != "predicted_sh":
        sky_sh[:, 0], sky_sh[:, 9], sky_sh[:, 18] = dc[:, 0], dc[:, 1], dc[:, 2]

    stats = dict(result.stats)
    stats["sky_sphere_prototype"] = {
        "applied": True,
        "source_samples": int(directions.shape[0]),
        "num_gaussians": int(unique.numel()),
        "scene_radius": float(scene_radius.item()),
        "sphere_radius": float(sphere_radius.item()),
        "radius_factor": float(radius_factor),
        "sample_stride": int(sample_stride),
        "azimuth_bins": int(azimuth_bins),
        "elevation_bins": int(elevation_bins),
        "color_mode": color_mode,
    }
    return WindowMergeResult(
        means=torch.cat((result.means, sky_means), dim=0),
        quats=torch.cat((result.quats, sky_quats), dim=0),
        scales=torch.cat((result.scales, sky_scales), dim=0),
        opacities=torch.cat((result.opacities, sky_opacities), dim=0),
        sh_flat=torch.cat((result.sh_flat, sky_sh), dim=0),
        camera_poses=result.camera_poses,
        stats=stats,
    )
