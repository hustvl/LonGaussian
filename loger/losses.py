"""
Loss functions for training the Gaussian splatting head.

Follows the design from Splatt3r / pixelSplat:
  - MSE (L2) between rendered and GT images, with optional masking
  - LPIPS perceptual loss (VGG backbone)
  - SSIM structural loss (1 - SSIM), optional
  - Combined with configurable weights

Default weights: MSE=1.0, LPIPS=0.25, SSIM=0.15 (SSIM helps alignment/sharpness).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Dict, Tuple
import lpips


class _GaussianWindow(nn.Module):
    """Depthwise Gaussian blur for SSIM."""

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)
        self.register_buffer("kernel", kernel, persistent=False)
        self.window_size = window_size
        self._channel_kernels: Dict[int, torch.Tensor] = {}

    def _expand_kernel(self, channels: int, device: torch.device) -> torch.Tensor:
        if channels not in self._channel_kernels or self._channel_kernels[channels].device != device:
            self._channel_kernels[channels] = self.kernel.to(device).repeat(channels, 1, 1, 1)
        return self._channel_kernels[channels]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self._expand_kernel(x.shape[1], x.device)
        pad = self.window_size // 2
        return F.conv2d(x, kernel, padding=pad, groups=x.shape[1])


class SSIMLoss(nn.Module):
    """Differentiable SSIM loss: returns (1 - SSIM, SSIM)."""

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window = _GaussianWindow(window_size, sigma)
        self.c1 = 0.01 ** 2
        self.c2 = 0.03 ** 2

    def ssim_map(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mu_x = self.window(predicted)
        mu_y = self.window(target)
        mu_xx = mu_x * mu_x
        mu_yy = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_xx = self.window(predicted * predicted) - mu_xx
        sigma_yy = self.window(target * target) - mu_yy
        sigma_xy = self.window(predicted * target) - mu_xy

        num = (2 * mu_xy + self.c1) * (2 * sigma_xy + self.c2)
        den = (mu_xx + mu_yy + self.c1) * (sigma_xx + sigma_yy + self.c2)
        return num / (den + 1e-8)

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        average_over_mask: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ssim_map = self.ssim_map(predicted, target).mean(dim=1, keepdim=True)
        if mask is not None and average_over_mask:
            if mask.sum() == 0:
                ssim_val = torch.tensor(0.0, device=predicted.device)
            else:
                ssim_val = (ssim_map * mask[:, None, :, :]).sum() / mask.sum()
        else:
            ssim_val = ssim_map.mean()
        return 1.0 - ssim_val, ssim_val


class GaussianLoss(nn.Module):
    """Composite loss for training the gaussian head.

    loss = mse_weight * MSE + lpips_weight * LPIPS + ssim_weight * (1 - SSIM)

    Supports optional per-pixel masking to exclude invalid regions
    (e.g., pixels not covered by any gaussian).
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        lpips_weight: float = 0.25,
        ssim_weight: float = 0.15,
        scale_reg_weight: float = 0.01,
        scale_target_ratio: float = 0.3,
        scale_knn_samples: int = 1024,
        aniso_reg_weight: float = 0.1,
        aniso_threshold: float = 5.0,
        offset_reg_weight: float = 0.01,
        average_over_mask: bool = True,
        activation_checkpoint: bool = False,
        device: str = "cuda",
    ):
        """
        Args:
            mse_weight: Weight for the MSE (L2) loss component.
            lpips_weight: Weight for the LPIPS perceptual loss component.
            ssim_weight: Weight for the SSIM loss component (1 - SSIM).
            scale_reg_weight: Weight for neighbor-ratio scale regularization.
            aniso_reg_weight: Weight for anisotropy penalty (needle gaussians).
            aniso_threshold: Max allowed max_axis/min_axis ratio before penalty.
                              Penalizes deviation from target_ratio * nn_dist.
            scale_target_ratio: Target scale / nearest-neighbor-distance ratio.
            scale_knn_samples: Number of gaussians to sample for KNN computation.
            average_over_mask: If True, normalize loss by the sum of mask
                               values rather than total pixel count.
            activation_checkpoint: Recompute LPIPS/SSIM internals during
                backward instead of retaining their activations.  The loss and
                gradients are unchanged (up to floating-point round-off).
            device: Device for LPIPS model.
        """
        super().__init__()
        self.mse_weight = mse_weight
        self.lpips_weight = lpips_weight
        self.ssim_weight = ssim_weight
        self.scale_reg_weight = scale_reg_weight
        self.scale_target_ratio = scale_target_ratio
        self.scale_knn_samples = scale_knn_samples
        self.aniso_reg_weight = aniso_reg_weight
        self.aniso_threshold = aniso_threshold
        self.offset_reg_weight = offset_reg_weight
        self.average_over_mask = average_over_mask
        self.activation_checkpoint = activation_checkpoint

        # LPIPS with spatial=True returns per-pixel maps for masking.  Keep VGG
        # explicit because this is the canonical LoGeR rendering-loss backbone.
        if lpips_weight > 0:
            self.lpips = lpips.LPIPS("vgg", spatial=True).to(device)
            for param in self.lpips.parameters():
                param.requires_grad = False
        else:
            self.lpips = None

        if ssim_weight > 0:
            self.ssim = SSIMLoss().to(device)
        else:
            self.ssim = None

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        scales: Optional[torch.Tensor] = None,
        means: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the composite loss.

        Args:
            predicted: (V, 3, H, W) rendered RGB image, values in [0, 1].
            target:   (V, 3, H, W) ground-truth RGB image, values in [0, 1].
            mask:     (V, H, W) optional binary mask. 1 = valid, 0 = ignore.
            scales:   (M, 3) gaussian scales, for regularization.
            means:    (M, 3) gaussian centers, for KNN-based scale reg.
            offsets:  (M, 3) gaussian offsets, for offset regularization.

        Returns:
            (total_loss, dict of component losses for logging)
        """
        # --- MSE Loss ---
        squared_diff = (predicted - target) ** 2  # (V, 3, H, W)
        if mask is not None and self.average_over_mask:
            if mask.sum() == 0:
                mse = torch.tensor(0.0, device=predicted.device)
            else:
                mse = (squared_diff * mask[:, None, :, :]).sum() / mask.sum()
        else:
            mse = squared_diff.mean()

        total = self.mse_weight * mse
        losses = {"mse": mse.detach()}

        # --- LPIPS Loss ---
        if self.lpips is not None and self.lpips_weight > 0:
            # LPIPS expects inputs in [-1, 1] with normalize=True
            if self.activation_checkpoint and torch.is_grad_enabled() and predicted.requires_grad:
                lpips_out = checkpoint(
                    lambda gt, pred: self.lpips(gt, pred, normalize=True),
                    target, predicted,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                lpips_out = self.lpips(target, predicted, normalize=True)  # (V, 1, H, W)
            if mask is not None and self.average_over_mask:
                if mask.sum() == 0:
                    lpips_val = torch.tensor(0.0, device=predicted.device)
                else:
                    lpips_val = (lpips_out * mask[:, None, :, :]).sum() / mask.sum()
            else:
                lpips_val = lpips_out.mean()

            total = total + self.lpips_weight * lpips_val
            losses["lpips"] = lpips_val.detach()

        # --- SSIM Loss ---
        if self.ssim is not None and self.ssim_weight > 0:
            if self.activation_checkpoint and torch.is_grad_enabled() and predicted.requires_grad:
                def _ssim(pred, gt):
                    return self.ssim(
                        pred, gt, mask=mask, average_over_mask=self.average_over_mask,
                    )
                ssim_loss, ssim_val = checkpoint(
                    _ssim, predicted, target,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                ssim_loss, ssim_val = self.ssim(
                    predicted, target, mask=mask, average_over_mask=self.average_over_mask,
                )
            total = total + self.ssim_weight * ssim_loss
            losses["ssim_loss"] = ssim_loss.detach()
            losses["ssim"] = ssim_val.detach()

        # --- Scale-to-NN ratio regularization ---
        if self.scale_reg_weight > 0 and scales is not None and means is not None:
            # Sample subset for KNN
            M = means.shape[0]
            if M > self.scale_knn_samples:
                idx = torch.randperm(M, device=means.device)[:self.scale_knn_samples]
                means_sub = means[idx]
                scales_sub = scales[idx]
            else:
                means_sub = means
                scales_sub = scales

            # KNN distances (exclude self)
            nn_dists = torch.cdist(means_sub, means_sub).sort(dim=1).values[:, 1]  # (K,)
            scale_norms = scales_sub.norm(dim=1)  # (K,)
            ratio = scale_norms / (nn_dists + 1e-8)

            scale_reg = ((ratio - self.scale_target_ratio) ** 2).mean()
            total = total + self.scale_reg_weight * scale_reg
            losses["scale_reg"] = scale_reg.detach()
            losses["scale_ratio"] = ratio.mean().detach()

        # --- Offset regularization ---
        if self.offset_reg_weight > 0 and offsets is not None:
            offset_reg = offsets.norm(dim=1).mean()
            total = total + self.offset_reg_weight * offset_reg
            losses["offset_reg"] = offset_reg.detach()

        # --- Anisotropy regularization ---
        if self.aniso_reg_weight > 0 and scales is not None:
            # scales: (M, 3) — penalize extreme axis ratios
            max_s = scales.max(dim=1).values  # (M,)
            min_s = scales.min(dim=1).values  # (M,)
            ratio = max_s / (min_s + 1e-6)
            aniso_loss = torch.relu(ratio - self.aniso_threshold).mean()
            total = total + self.aniso_reg_weight * aniso_loss
            losses["aniso"] = aniso_loss.detach()

        losses["total"] = total.detach()
        return total, losses


class GaussianTrainer:
    """Lightweight training harness for the Gaussian head.

    Handles: freeze/ unfreeze, forward pass, rendering, loss computation.

    Usage:
        trainer = GaussianTrainer(
            model, renderer,
            mse_weight=1.0, lpips_weight=0.25, ssim_weight=0.15,
        )
        trainer.set_trainable()

        for batch in dataloader:
            loss, logs = trainer.training_step(images)
            loss.backward()
            optimizer.step()
    """

    def __init__(
        self,
        model: nn.Module,
        renderer: nn.Module,
        mse_weight: float = 1.0,
        lpips_weight: float = 0.25,
        ssim_weight: float = 0.15,
        average_over_mask: bool = True,
        device: str = "cuda",
    ):
        self.model = model
        self.renderer = renderer
        self.loss_fn = GaussianLoss(
            mse_weight=mse_weight,
            lpips_weight=lpips_weight,
            ssim_weight=ssim_weight,
            average_over_mask=average_over_mask,
            device=device,
        )
        self.device = device

    def set_trainable(self):
        """Freeze all model params except gaussian_decoder and gaussian_head."""
        for name, param in self.model.named_parameters():
            if name.startswith("gaussian_decoder") or name.startswith("gaussian_head"):
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Print summary
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(
            f"GaussianTrainer: {trainable:,} / {total:,} params trainable "
            f"({100 * trainable / total:.1f}%)"
        )

    def training_step(
        self,
        images: torch.Tensor,
        **forward_kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Single training step: predict gaussians → render → compute loss.

        Args:
            images: (B, T, 3, H, W) input image sequence.
            **forward_kwargs: Passed to model.forward() (e.g., gaussian=True).

        Returns:
            (total_loss, dict of per-component losses)
        """
        # Forward: predict gaussian parameters
        predictions = self.model(images, gaussian=True, **forward_kwargs)

        # Render all frames using their own gaussians + camera poses
        rendered = self.renderer.render_sequence(predictions)

        # Ground truth: original images from predictions dict
        target_rgb = predictions["images"]  # (1, T, H, W, 3) or (T, H, W, 3)
        if target_rgb.dim() == 5:
            target_rgb = target_rgb.squeeze(0)  # (T, H, W, 3)
        target_rgb = target_rgb.permute(0, 3, 1, 2)  # (T, 3, H, W)

        predicted_rgb = rendered["rgb"].permute(0, 3, 1, 2)  # (T, 3, H, W)

        # Simple mask: where alpha > threshold
        alpha = rendered["alpha"].squeeze(-1)  # (T, H, W)
        mask = (alpha > 0.05).float()

        loss, logs = self.loss_fn(predicted_rgb, target_rgb, mask)
        return loss, logs
