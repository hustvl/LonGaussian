"""
DPT (Dense Prediction Transformer) Gaussian Head for per-pixel 3DGS parameter prediction.

Design references:
- splatt3r / MASt3R: PixelwiseTaskWithDPT + GaussianHead
- DPTOutputAdapter from dust3r/croco/models/dpt_block.py

Architecture:
  Takes 4 intermediate transformer features [f0, f1, f2, f3] from the gaussian decoder.
  Reassemble → align spatial resolution → Fusion → upsample → separate output heads.

Key design principles:
1. Multi-scale feature aggregation via DPT fusion (bottom-up skip connections)
2. Separate output heads per parameter group with splatt3r-aligned init
3. Per-pixel prediction at full image resolution via bilinear upsampling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Literal, Tuple


# ---------------------------------------------------------------------------
#  Gaussian parameter activations (unchanged)
# ---------------------------------------------------------------------------

def gaussian_offset_activation(raw: torch.Tensor, shift: float = 6.0) -> torch.Tensor:
    d = raw.norm(dim=1, keepdim=True).clamp(min=1e-8)
    direction = raw / d
    return direction * (torch.exp(d - shift) - torch.exp(-torch.tensor(shift, device=raw.device, dtype=raw.dtype)))


def gaussian_scale_activation(raw: torch.Tensor) -> torch.Tensor:
    return F.softplus(raw)


def gaussian_rotation_activation(raw: torch.Tensor) -> torch.Tensor:
    return F.normalize(raw, p=2, dim=1)


def gaussian_opacity_activation(raw: torch.Tensor, max_opacity: float = 1.0) -> torch.Tensor:
    return max_opacity * torch.sigmoid(raw)


def sh_coefficients_activation(raw: torch.Tensor) -> torch.Tensor:
    return raw


# ---------------------------------------------------------------------------
#  DPT reassemble + fusion modules
# ---------------------------------------------------------------------------

class ReassembleBlock(nn.Module):
    """Project a transformer token sequence to a spatial feature map,
    optionally with stride-based spatial resize."""

    def __init__(self, in_dim: int, out_dim: int, spatial_scale: float = 1.0):
        super().__init__()
        self.spatial_scale = spatial_scale
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)

        if spatial_scale > 1:
            self.resize = nn.ConvTranspose2d(out_dim, out_dim,
                                             kernel_size=int(spatial_scale * 2),
                                             stride=int(spatial_scale),
                                             padding=int(spatial_scale) // 2)
        elif spatial_scale < 1:
            self.resize = nn.Conv2d(out_dim, out_dim, kernel_size=3,
                                    stride=int(1 / spatial_scale),
                                    padding=1)
        else:
            self.resize = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.resize(x)
        return x


class ResidualConvUnit(nn.Module):
    """Two conv layers with residual connection."""

    def __init__(self, features: int, activation=nn.ReLU):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.activation = activation(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        return out + x


class FusionBlock(nn.Module):
    """Fuse a higher-level feature (path) with a skip feature, then upsample 2×."""

    def __init__(self, features: int):
        super().__init__()
        self.res_unit1 = ResidualConvUnit(features)
        self.res_unit2 = ResidualConvUnit(features)

    def forward(self, path: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        if skip is not None:
            skip = self.res_unit1(skip)
            if skip.shape != path.shape:
                skip = F.interpolate(skip, size=path.shape[2:], mode='bilinear', align_corners=False)
            path = path + skip
        path = self.res_unit2(path)
        path = F.interpolate(path, scale_factor=2, mode='bilinear', align_corners=False)
        return path


# ---------------------------------------------------------------------------
#  DPT GaussianHead
# ---------------------------------------------------------------------------

class GaussianHead(nn.Module):
    """DPT-style per-pixel gaussian parameter prediction head.

    Takes 4 intermediate transformer features from the gaussian decoder,
    fuses them bottom-up, and predicts gaussian parameters at full resolution.
    """

    def __init__(
        self,
        dim_in: int = 1024,
        feature_dim: int = 256,
        sh_degree: int = 2,
        max_opacity: float = 1.0,
        use_offsets: bool = True,
    ):
        super().__init__()

        self.sh_degree = sh_degree
        self.sh_num_coeffs = 3 * (sh_degree + 1) ** 2
        self.max_opacity = max_opacity
        self.use_offsets = use_offsets

        self.num_offset_params = 3
        self.num_scale_params = 3
        self.num_rotation_params = 4
        self.num_opacity_params = 1
        self.num_sh_params = self.sh_num_coeffs
        self.merge_feature_projection = nn.Conv2d(feature_dim, 16, kernel_size=1)

        # --- Reassemble: 4 transformer layers → 4 spatial feature maps ---
        # Hook layers 0,1,2,3 with spatial scales 4x, 2x, 1x, 0.5x
        self.reassemble = nn.ModuleList([
            ReassembleBlock(dim_in, feature_dim, spatial_scale=4.0),   # f0: 4x up
            ReassembleBlock(dim_in, feature_dim, spatial_scale=2.0),   # f1: 2x up
            ReassembleBlock(dim_in, feature_dim, spatial_scale=1.0),   # f2: 1x
            ReassembleBlock(dim_in, feature_dim, spatial_scale=0.5),   # f3: 2x down
        ])

        # --- Fusion: bottom-up refinement ---
        self.fusion_blocks = nn.ModuleList([
            FusionBlock(feature_dim),  # f3 → + f2
            FusionBlock(feature_dim),  # + f1
            FusionBlock(feature_dim),  # + f0
        ])

        # --- Final refinement ---
        self.refine = ResidualConvUnit(feature_dim)

        # --- Output heads (same as before) ---
        self._make_output_block = lambda out_ch: nn.Sequential(
            nn.Conv2d(feature_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_ch, kernel_size=1),
        )

        if self.use_offsets:
            self.offset_head = self._make_output_block(self.num_offset_params)
        else:
            self.offset_head = None

        self.scale_head = self._make_output_block(self.num_scale_params)
        self.rotation_head = self._make_output_block(self.num_rotation_params)
        self.opacity_head = self._make_output_block(self.num_opacity_params)
        self.sh_head = self._make_output_block(self.num_sh_params)

        # Initialize biases aligned with splatt3r
        self._init_output_biases()

    def _init_output_biases(self) -> None:
        """Initialize output head biases aligned with splatt3r."""
        # Offset head: shift=6.0 activation, init bias to ~0
        if self.offset_head is not None:
            last_conv = self.offset_head[-1]
            if hasattr(last_conv, 'bias') and last_conv.bias is not None:
                torch.nn.init.constant_(last_conv.bias, 0.001)
            last_conv.weight.data.mul_(0.001)

        # Scale head: softplus activation, bias=0 → softplus(0) ≈ 0.693
        last_conv = self.scale_head[-1]
        if hasattr(last_conv, 'bias') and last_conv.bias is not None:
            torch.nn.init.constant_(last_conv.bias, 0.0)

        # Opacity head: sigmoid activation, init bias to -2.0 → ~0.12
        last_conv = self.opacity_head[-1]
        if hasattr(last_conv, 'bias') and last_conv.bias is not None:
            torch.nn.init.constant_(last_conv.bias, -2.0)

    def forward(
        self,
        features: List[torch.Tensor],
        patch_h: int,
        patch_w: int,
        return_merge_features: bool = False,
        return_prehead_features: bool = False,
    ) -> dict:
        """
        Args:
            features: List of 4 tensors from transformer decoder layers,
                      each (B*N, S, D) where S = patch_h * patch_w.
            patch_h, patch_w: spatial dimensions in patches.

        Returns:
            dict with keys: offsets (or None), scales, rotations, opacities, sh_coeffs
            all at full image resolution (B*N, C, img_h, img_w).
        """
        img_h = patch_h * 14
        img_w = patch_w * 14

        # Reassemble: tokens → spatial feature maps
        spatial_feats = []
        for i, f in enumerate(features):
            BN, S, D = f.shape
            x = f.permute(0, 2, 1).reshape(BN, D, patch_h, patch_w)  # (BN, D, h, w)
            x = self.reassemble[i](x)
            spatial_feats.append(x)
        # spatial_feats now have the same spatial size (2x patch resolution)

        # Fusion: bottom-up
        path = spatial_feats[3]  # coarsest
        for i in range(2, -1, -1):
            path = self.fusion_blocks[2 - i](path, spatial_feats[i])
        # path is now at full image resolution after 3 upsample steps

        # Refine
        path = self.refine(path)

        # Bilinear upsampling to full image resolution
        path = F.interpolate(path, size=(img_h, img_w), mode='bilinear', align_corners=False)

        # Predict gaussian parameters through separate heads
        offsets = None
        if self.offset_head is not None:
            raw_offset = self.offset_head(path)
            offsets = gaussian_offset_activation(raw_offset)

        raw_scale = self.scale_head(path)
        raw_rotation = self.rotation_head(path)
        raw_opacity = self.opacity_head(path)
        raw_sh = self.sh_head(path)

        scales = gaussian_scale_activation(raw_scale)
        rotations = gaussian_rotation_activation(raw_rotation)
        opacities = gaussian_opacity_activation(raw_opacity, self.max_opacity)
        sh_coeffs = sh_coefficients_activation(raw_sh)

        outputs = {
            'offsets': offsets,
            'scales': scales,
            'rotations': rotations,
            'opacities': opacities,
            'sh_coeffs': sh_coeffs,
        }
        if return_merge_features:
            outputs['merge_features'] = self.merge_feature_projection(path)
        if return_prehead_features:
            outputs['prehead_features'] = path
        return outputs

    def pack_parameters(
        self,
        centers: torch.Tensor,
        gaussian_outputs: dict,
    ) -> dict:
        return {
            'means': centers,
            'scales': gaussian_outputs['scales'],
            'rotations': gaussian_outputs['rotations'],
            'opacities': gaussian_outputs['opacities'],
            'sh_coeffs': gaussian_outputs['sh_coeffs'],
        }
