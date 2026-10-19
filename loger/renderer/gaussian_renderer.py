"""
Gaussian Splatting renderer built on gsplat 1.5.3.

Converts the per-pixel gaussian parameters predicted by Pi3 into rendered
images via differentiable rasterization. Supports single-view, multi-view,
and novel-view rendering with SH-based view-dependent color.

Key conversions handled:
  - Camera-to-world poses → world-to-camera view matrices (inverse)
  - Flat SH coefficients [3ch × K] → gsplat's [K, 3ch] per-gaussian layout
  - Spatial (B,N,H,W) → flat (M,) gaussian list with opacity filtering
  - Implicit camera model → explicit pinhole K matrix
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import warnings


class GaussianRenderer:
    """Differentiable 3DGS rasterizer wrapping gsplat's rasterization().

    Takes per-pixel gaussian parameters (as output by Pi3.forward() with
    gaussian=True) and renders them to images from arbitrary viewpoints.

    Usage:
        renderer = GaussianRenderer(sh_degree=2)
        predictions = model(imgs, gaussian=True)

        # Render from input viewpoints
        rendered = renderer.render_sequence(predictions)

        # Render from novel viewpoints
        novel_imgs = renderer.render_novel_view(
            means, quats, scales, opacities, colors,
            view_pose, intrinsics, H, W,
        )
    """

    def __init__(
        self,
        sh_degree: int = 2,
        default_focal: Optional[float] = None,
        opacity_threshold: float = 0.01,
        background_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        near_plane: float = 0.01,
        far_plane: float = 1e10,
        eps2d: float = 0.3,
        tile_size: int = 16,
        rasterize_mode: str = "classic",
    ):
        """
        Args:
            sh_degree: Spherical harmonics degree (0, 1, 2, 3).
            default_focal: Default focal length multiplier. If None, uses
                           max(H,W) * 0.7. Only used when Ks is not provided.
            opacity_threshold: Gaussians with opacity below this are filtered
                               out before rendering to save memory.
            background_color: RGB background color (range [0, 1]).
            near_plane: Near clipping plane for projection.
            far_plane: Far clipping plane for projection.
            eps2d: Epsilon for 2D covariance normalization.
            tile_size: CUDA tile size for rasterization (16 or 32).
            rasterize_mode: 'classic' or 'antialiased'.
        """
        self.sh_degree = sh_degree
        self.sh_num_bases = (sh_degree + 1) ** 2
        self.default_focal = default_focal
        self.opacity_threshold = opacity_threshold
        self.background_color = background_color
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.eps2d = eps2d
        self.tile_size = tile_size
        self.rasterize_mode = rasterize_mode

        # Lazy-import gsplat so the module is importable without CUDA
        self._gsplat = None

    @property
    def gsplat(self):
        if self._gsplat is None:
            import gsplat
            self._gsplat = gsplat
        return self._gsplat

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def prepare_gaussians(
        self,
        predictions: dict,
        frame_indices: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """Extract and flatten gaussian parameters from a model output dict.

        Args:
            predictions: Dict from Pi3.forward(), expected keys:
                points, gaussian_scales, gaussian_rotations,
                gaussian_opacities, gaussian_sh_coeffs, camera_poses.
            frame_indices: Optional subset of frames to include.

        Returns:
            Tuple of (means, quats, scales, opacities, colors, camera_poses)
            where means/quats/scales/opacities/colors are (M, ...) flat tensors
            and camera_poses is (C, 4, 4).
        """
        # Select frames
        if frame_indices is not None:
            # Use gaussian_means (points + learned offset) if available
            means_key = "gaussian_means" if "gaussian_means" in predictions else "points"
            points = predictions[means_key][:, frame_indices]
            scales_raw = predictions["gaussian_scales"][:, frame_indices]
            rots_raw = predictions["gaussian_rotations"][:, frame_indices]
            opacs_raw = predictions["gaussian_opacities"][:, frame_indices]
            sh_raw = predictions["gaussian_sh_coeffs"][:, frame_indices]
            poses = predictions["camera_poses"][:, frame_indices]
        else:
            means_key = "gaussian_means" if "gaussian_means" in predictions else "points"
            points = predictions[means_key]
            scales_raw = predictions["gaussian_scales"]
            rots_raw = predictions["gaussian_rotations"]
            opacs_raw = predictions["gaussian_opacities"]
            sh_raw = predictions["gaussian_sh_coeffs"]
            poses = predictions["camera_poses"]

        # Flatten batch and frame dims: (B, N, ...) → (B*N, ...)
        points_flat = points.reshape(-1, *points.shape[2:])        # (B*N, H, W, 3)
        scales_flat = scales_raw.reshape(-1, *scales_raw.shape[2:])  # (B*N, 3, H, W)
        rots_flat = rots_raw.reshape(-1, *rots_raw.shape[2:])        # (B*N, 4, H, W)
        opacs_flat = opacs_raw.reshape(-1, *opacs_raw.shape[2:])     # (B*N, 1, H, W)
        sh_flat = sh_raw.reshape(-1, *sh_raw.shape[2:])              # (B*N, N_sh, H, W)
        poses_flat = poses.reshape(-1, *poses.shape[2:])           # (B*N, 4, 4)

        # Spatial → flat: (B*N, C, H, W) → (B*N*H*W, C)
        # Permute each to (B*N, H, W, C) then flatten
        means = points_flat.reshape(-1, 3)                  # (M, 3)

        # scales: (B*N, 3, H, W) → (B*N, H, W, 3) → (M, 3)
        scales = scales_flat.permute(0, 2, 3, 1).reshape(-1, 3)  # (M, 3)

        # rotations: (B*N, 4, H, W) → (B*N, H, W, 4) → (M, 4)
        quats = rots_flat.permute(0, 2, 3, 1).reshape(-1, 4)    # (M, 4)

        # opacities: (B*N, 1, H, W) → (M,)
        opacities = opacs_flat.reshape(-1)                        # (M,)

        # SH: (B*N, N_sh, H, W) → (B*N, H, W, N_sh) → (M, N_sh)
        sh_2d = sh_flat.permute(0, 2, 3, 1).reshape(-1, sh_flat.shape[1])  # (M, N_sh)

        # Convert SH layout: [R0..R8, G0..G8, B0..B8] → gsplat's (M, K, 3)
        # where K = (sh_degree+1)^2. Input has 3*K channels flattened channel-first.
        colors = self._convert_sh_for_gsplat(sh_2d)              # (M, K, 3)

        # --- Opacity-based filtering ---
        if self.opacity_threshold > 0:
            mask = opacities > self.opacity_threshold
            if mask.sum() == 0:
                warnings.warn(
                    f"All gaussians filtered by opacity threshold {self.opacity_threshold}. "
                    f"Falling back to top 10%."
                )
                top_k = max(1, int(len(opacities) * 0.1))
                _, indices = torch.topk(opacities, top_k)
                mask = torch.zeros_like(opacities, dtype=torch.bool)
                mask[indices] = True
            means = means[mask]
            quats = quats[mask]
            scales = scales[mask]
            opacities = opacities[mask]
            colors = colors[mask]

        return means, quats, scales, opacities, colors, poses_flat

    def render_view(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        opacities: torch.Tensor,
        colors: torch.Tensor,
        camera_pose: torch.Tensor,
        H: int,
        W: int,
        K: Optional[torch.Tensor] = None,
        background: Optional[torch.Tensor] = None,
    ) -> dict:
        """Render gaussians from a single viewpoint.

        Args:
            means: (M, 3) gaussian centers in world coords.
            quats: (M, 4) normalized quaternions (w, x, y, z).
            scales: (M, 3) gaussian scales.
            opacities: (M,) gaussian opacities in (0, 1).
            colors: (M, K, 3) SH coefficients, K = (sh_degree+1)^2.
            camera_pose: (4, 4) camera-to-world matrix.
            H, W: image height and width.
            K: (3, 3) camera intrinsics. If None, auto-computed.
            background: (3, H, W) or (3,) background color/image.

        Returns:
            dict with keys:
                'rgb': (H, W, 3) rendered RGB image.
                'depth': (H, W, 1) expected depth map.
                'alpha': (H, W, 1) accumulated alpha.
        """
        # Ensure float32 — gsplat CUDA kernels require fp32
        means = means.float()
        quats = quats.float()
        scales = scales.float()
        opacities = opacities.float()
        colors = colors.float()
        camera_pose = camera_pose.float()

        viewmat = torch.inverse(camera_pose)  # (4, 4) world-to-camera
        viewmats = viewmat.unsqueeze(0)       # (1, 4, 4)

        if K is None:
            K = self._make_intrinsics(H, W, means.device)
        Ks = K.unsqueeze(0).float()           # (1, 3, 3)

        if background is not None:
            bg = background
        elif self.background_color == (0.0, 0.0, 0.0):
            bg = None  # default black, gsplat handles this internally
        else:
            # gsplat expects backgrounds as [..., C, D] (here C=1 camera, D=3).
            bg = torch.tensor(
                self.background_color,
                device=means.device,
                dtype=torch.float32,
            ).view(1, 3)

        rendered, alpha, info = self.gsplat.rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=W,
            height=H,
            sh_degree=self.sh_degree,
            near_plane=self.near_plane,
            far_plane=self.far_plane,
            eps2d=self.eps2d,
            tile_size=self.tile_size,
            backgrounds=bg,
            render_mode="RGB+ED",
            packed=False,
            rasterize_mode=self.rasterize_mode,
        )

        # rendered: (1, H, W, 4) channels: R, G, B, ED
        rgb = rendered[0, ..., :3]         # (H, W, 3)
        depth = rendered[0, ..., 3:4]      # (H, W, 1)
        alpha_img = alpha[0]               # (H, W, 1)

        return {
            "rgb": rgb,
            "depth": depth,
            "alpha": alpha_img,
        }

    def render_sequence(
        self,
        predictions: dict,
        target_indices: Optional[List[int]] = None,
        neighbor_frames: int = 0,
        K: Optional[torch.Tensor] = None,
    ) -> dict:
        """Render gaussians from all (or selected) input viewpoints.

        Args:
            predictions: Dict from Pi3.forward() with gaussian keys.
            target_indices: Optional subset of frame indices to render.
            neighbor_frames: Number of neighboring frames (each side) whose
                             gaussians to include when rendering each frame.
                             0 = only current frame, 1 = ±1 neighbors, etc.
            K: (3, 3) camera intrinsics. If None, auto-computed.

        Returns:
            dict with:
                'rgb': (T, H, W, 3) rendered images.
                'depth': (T, H, W, 1) expected depth maps.
                'alpha': (T, H, W, 1) alpha maps.
                'images': (T, H, W, 3) original input images (if available).
        """
        N_all = predictions["points"].shape[1]
        if target_indices is None:
            target_indices = list(range(N_all))
        H, W = predictions["gaussian_scales"].shape[-2:]

        rgbs = []
        depths = []
        alphas = []

        for frame_idx in target_indices:
            # Gather gaussians from current frame ± neighbor_frames neighbors
            start = max(0, frame_idx - neighbor_frames)
            end = min(N_all, frame_idx + neighbor_frames + 1)
            neighbor_indices = list(range(start, end))
            current_pos = frame_idx - start  # index of current frame in neighbors

            means, quats, scales, opacities, colors, poses = self.prepare_gaussians(
                predictions, frame_indices=neighbor_indices
            )
            result = self.render_view(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                camera_pose=poses[current_pos],  # current frame's camera
                H=H, W=W,
                K=K,
            )
            rgbs.append(result["rgb"])
            depths.append(result["depth"])
            alphas.append(result["alpha"])

        output = {
            "rgb": torch.stack(rgbs, dim=0),          # (T, H, W, 3)
            "depth": torch.stack(depths, dim=0),      # (T, H, W, 1)
            "alpha": torch.stack(alphas, dim=0),      # (T, H, W, 1)
        }

        # Include original images if present
        if "images" in predictions:
            imgs = predictions["images"]
            if target_indices is not None:
                imgs = imgs[:, target_indices]
            output["images"] = imgs.reshape(-1, *imgs.shape[-3:])

        return output

    def prepare_hybrid(
        self,
        merge_result,
        opacity_threshold: Optional[float] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """Convert ``WindowMergeResult`` into flat gsplat-ready tensors."""
        from loger.merge.window_merge import WindowMergeResult

        if not isinstance(merge_result, WindowMergeResult):
            raise TypeError("merge_result must be WindowMergeResult")

        means = merge_result.means.reshape(-1, 3)
        quats = merge_result.quats.reshape(-1, 4)
        scales = merge_result.scales.reshape(-1, 3)
        opacities = merge_result.opacities.reshape(-1)
        colors = self._convert_sh_for_gsplat(merge_result.sh_flat.reshape(-1, merge_result.sh_flat.shape[-1]))

        threshold = self.opacity_threshold if opacity_threshold is None else opacity_threshold
        if threshold > 0:
            mask = opacities > threshold
            if mask.sum() == 0:
                top_k = max(1, int(len(opacities) * 0.1))
                _, indices = torch.topk(opacities, top_k)
                mask = torch.zeros_like(opacities, dtype=torch.bool)
                mask[indices] = True
            means = means[mask]
            quats = quats[mask]
            scales = scales[mask]
            opacities = opacities[mask]
            colors = colors[mask]

        camera_poses = merge_result.camera_poses.reshape(-1, 4, 4)
        return means, quats, scales, opacities, colors, camera_poses

    def render_hybrid(
        self,
        merge_result,
        target_indices: Optional[List[int]] = None,
        K: Optional[torch.Tensor] = None,
        H: Optional[int] = None,
        W: Optional[int] = None,
    ) -> dict:
        """Render a global hybrid cloud from multiple camera poses."""
        means, quats, scales, opacities, colors, camera_poses = self.prepare_hybrid(
            merge_result
        )
        num_frames = camera_poses.shape[0]
        if target_indices is None:
            target_indices = list(range(num_frames))
        if H is None or W is None:
            raise ValueError("render_hybrid requires explicit H and W")

        return self.render_views_from_tensors(
            means,
            quats,
            scales,
            opacities,
            colors,
            camera_poses,
            target_indices,
            H,
            W,
            K=K,
        )

    def render_views_from_tensors(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        opacities: torch.Tensor,
        colors: torch.Tensor,
        camera_poses: torch.Tensor,
        frame_indices: List[int],
        H: int,
        W: int,
        K: Optional[torch.Tensor] = None,
        checkpoint_chunk_size: int = 0,
    ) -> dict:
        """Render a fixed gaussian set from multiple camera poses.

        ``checkpoint_chunk_size`` enables activation checkpointing over groups
        of views.  This is intentionally a *view* chunk, rather than a
        gaussian chunk: splitting gaussians would change depth ordering and
        alpha compositing.  Checkpointing retains the exact forward result and
        gradient, while rasterization intermediates are recomputed one chunk
        at a time during backward instead of being retained for every view.
        """
        if checkpoint_chunk_size > 0 and torch.is_grad_enabled() and len(frame_indices) > checkpoint_chunk_size:
            from torch.utils.checkpoint import checkpoint

            rgb_chunks = []
            depth_chunks = []
            alpha_chunks = []
            for start in range(0, len(frame_indices), checkpoint_chunk_size):
                frame_chunk = frame_indices[start:start + checkpoint_chunk_size]

                def _render_chunk(
                    chunk_means, chunk_quats, chunk_scales, chunk_opacities,
                    chunk_colors, chunk_poses, frame_chunk=tuple(frame_chunk),
                ):
                    out = self._render_views_from_tensors_impl(
                        chunk_means, chunk_quats, chunk_scales,
                        chunk_opacities, chunk_colors, chunk_poses,
                        frame_chunk, H, W, K,
                    )
                    return out["rgb"], out["depth"], out["alpha"]

                rgb, depth, alpha = checkpoint(
                    _render_chunk,
                    means, quats, scales, opacities, colors, camera_poses,
                    use_reentrant=False,
                    preserve_rng_state=False,
                    # Bind each chunk's indices: recomputation happens after
                    # the loop, when a late-bound variable names the last view.
                    determinism_check="default",
                )
                rgb_chunks.append(rgb)
                depth_chunks.append(depth)
                alpha_chunks.append(alpha)
            return {
                "rgb": torch.cat(rgb_chunks, dim=0),
                "depth": torch.cat(depth_chunks, dim=0),
                "alpha": torch.cat(alpha_chunks, dim=0),
            }

        return self._render_views_from_tensors_impl(
            means, quats, scales, opacities, colors, camera_poses,
            frame_indices, H, W, K,
        )

    def _render_views_from_tensors_impl(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        opacities: torch.Tensor,
        colors: torch.Tensor,
        camera_poses: torch.Tensor,
        frame_indices: List[int],
        H: int,
        W: int,
        K: Optional[torch.Tensor] = None,
    ) -> dict:
        """Uncheckpointed implementation of :meth:`render_views_from_tensors`."""
        if camera_poses.dim() == 4:
            camera_poses = camera_poses[0]

        rgbs = []
        depths = []
        alphas = []
        for frame_idx in frame_indices:
            result = self.render_view(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                camera_pose=camera_poses[frame_idx],
                H=H,
                W=W,
                K=K,
            )
            rgbs.append(result["rgb"])
            depths.append(result["depth"])
            alphas.append(result["alpha"])

        return {
            "rgb": torch.stack(rgbs, dim=0),
            "depth": torch.stack(depths, dim=0),
            "alpha": torch.stack(alphas, dim=0),
        }

    def render_novel_view(
        self,
        predictions: dict,
        view_pose: torch.Tensor,
        K: Optional[torch.Tensor] = None,
        H: Optional[int] = None,
        W: Optional[int] = None,
    ) -> dict:
        """Render gaussians from a novel (arbitrary) viewpoint.

        Args:
            predictions: Dict from Pi3.forward() with gaussian keys.
            view_pose: (4, 4) camera-to-world matrix for the novel view.
            K: (3, 3) camera intrinsics. Auto-computed if None.
            H, W: image dimensions. Defaults to input image size.

        Returns:
            dict with 'rgb', 'depth', 'alpha'.
        """
        means, quats, scales, opacities, colors, _ = self.prepare_gaussians(predictions)

        # Infer H, W from predictions if not provided
        if H is None or W is None:
            _, _, ph, pw = predictions["points"].shape[:4]  # approx
            if H is None:
                H = ph
            if W is None:
                W = pw

        return self.render_view(
            means=means, quats=quats, scales=scales,
            opacities=opacities, colors=colors,
            camera_pose=view_pose, H=H, W=W, K=K,
        )

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _convert_sh_for_gsplat(self, sh_flat: torch.Tensor) -> torch.Tensor:
        """Convert SH layout from channel-first to gsplat's basis-first.

        Input:  (M, 3*K) where K = (sh_degree+1)^2
                Layout: [R_0, ..., R_K-1, G_0, ..., G_K-1, B_0, ..., B_K-1]

        Output: (M, K, 3)
                Layout: [[R_0, G_0, B_0], ..., [R_K-1, G_K-1, B_K-1]]
        """
        M = sh_flat.shape[0]
        K = self.sh_num_bases
        # (M, 3*K) → (M, 3, K) → (M, K, 3)
        colors = sh_flat.reshape(M, 3, K).transpose(1, 2).contiguous()
        return colors

    def _make_intrinsics(
        self, H: int, W: int, device: torch.device
    ) -> torch.Tensor:
        """Build a pinhole camera intrinsics matrix.

        Uses the default focal length if set, otherwise computes
        fx = fy = max(H, W) * 0.7 (a reasonable default for most cameras).
        """
        if self.default_focal is not None:
            fx = fy = self.default_focal
        else:
            fx = fy = max(H, W) * 0.7

        cx = W / 2.0
        cy = H / 2.0

        K = torch.tensor(
            [[fx, 0.0, cx],
             [0.0, fy, cy],
             [0.0, 0.0, 1.0]],
            device=device,
            dtype=torch.float32,
        )
        return K
