"""In-process AllTracker dense tracking for LoGeR track merge."""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import torch

DL3DV_ALIGN_DIVISOR = 56
INFERENCE_ITERS = 4
WINDOW_LEN = 16
DEFAULT_CONF_THR = 0.1
WEIGHTS_URL = "https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth"
ALLTRACKER_SUBMODULE_RELPATH = os.path.join("submodules", "alltracker")


def alltracker_repo_root() -> str:
    """Absolute path to the vendored AllTracker inference subset."""
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ALLTRACKER_SUBMODULE_RELPATH)
    )


def ensure_alltracker_importable() -> str:
    """Put the AllTracker submodule on ``sys.path`` for ``nets.*`` imports."""
    root = alltracker_repo_root()
    nets_dir = os.path.join(root, "nets")
    if not os.path.isdir(nets_dir):
        raise FileNotFoundError(
            f"AllTracker inference files not found at {root}. "
            "Restore loger/submodules/alltracker from the source distribution."
        )
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def snap_to_divisor(value: int, divisor: int) -> int:
    """Round ``value`` up to the nearest multiple of ``divisor``."""
    return ((int(value) + divisor - 1) // divisor) * divisor


def _prepare_rgb_uint8(rgb: np.ndarray) -> np.ndarray:
    if rgb.dtype == np.uint8:
        return rgb
    out = rgb.astype(np.float32)
    if out.max() <= 1.0:
        out = out * 255.0
    return out.clip(0, 255).astype(np.uint8)


def tracks_dict_from_maps(
    traj_maps: np.ndarray,
    visconf_maps: np.ndarray,
    conf_thr: float,
    query_global_frame: int = 0,
) -> dict:
    """Build merge-compatible track dict from dense (T,H,W,*) maps."""
    t_len, h, w = traj_maps.shape[:3]
    confidence = visconf_maps[..., 1].astype(np.float32, copy=False)
    vis = confidence > conf_thr
    trajs_uv_dense = traj_maps.astype(np.float32, copy=False)
    trajs_uv = trajs_uv_dense.reshape(t_len, h * w, 2)
    vis_flat = vis.reshape(t_len, h * w)
    return {
        "trajs_uv": trajs_uv,
        "vis": vis_flat,
        "confidence": confidence.reshape(t_len, h * w),
        "trajs_uv_dense": trajs_uv_dense,
        "vis_dense": vis,
        "dense_reso": (h, w),
        "query_global_frame": int(query_global_frame),
        "orig_hw": (h, w),
    }


class AllTrackerRunner:
    """Session-scoped AllTracker wrapper (load once, track many chunks)."""

    def __init__(
        self,
        conf_thr: float = DEFAULT_CONF_THR,
        device: Optional[str] = None,
    ) -> None:
        self.conf_thr = float(conf_thr)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Optional[torch.nn.Module] = None

    def _load_model(self) -> torch.nn.Module:
        if self._model is not None:
            return self._model
        ensure_alltracker_importable()
        from nets.alltracker import Net

        model = Net(WINDOW_LEN)
        from loger.utils.checkpoint import sha256_file
        local = os.environ.get("ALLTRACKER_CHECKPOINT")
        expected_sha = os.environ.get("ALLTRACKER_SHA256")
        if local:
            digest = sha256_file(local)
            if expected_sha and digest.lower() != expected_sha.lower():
                raise ValueError("AllTracker checkpoint SHA256 mismatch")
            print(f"AllTracker SHA256={digest}", flush=True)
            state_dict = torch.load(local, map_location="cpu", weights_only=False)
        else:
            if expected_sha:
                raise ValueError("Set ALLTRACKER_CHECKPOINT when requesting hash verification")
            state_dict = torch.hub.load_state_dict_from_url(WEIGHTS_URL, map_location="cpu")
        model.load_state_dict(state_dict["model"], strict=True)
        model.to(self.device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self._model = model
        return model

    @torch.inference_mode()
    def _forward_maps(
        self,
        rgb: np.ndarray,
        query_frame: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        ensure_alltracker_importable()

        rgb_u8 = _prepare_rgb_uint8(rgb)
        t_len, h, w, _ = rgb_u8.shape
        rgbs = torch.from_numpy(rgb_u8).permute(0, 3, 1, 2).float().unsqueeze(0).to(self.device)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=self.device, dtype=torch.float32),
            torch.arange(w, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        grid_xy = torch.stack((grid_x, grid_y), dim=0).reshape(1, 1, 2, h, w)

        model = self._load_model()
        flows_e, visconf_maps_e, _, _ = model.forward_sliding(
            rgbs[:, query_frame:],
            iters=INFERENCE_ITERS,
            sw=None,
            is_training=False,
        )
        flows_e = flows_e.to(self.device)
        visconf_maps_e = visconf_maps_e.to(self.device)
        traj_maps = flows_e + grid_xy
        visconf_maps = visconf_maps_e

        if query_frame > 0:
            b_flows, b_visconf, _, _ = model.forward_sliding(
                rgbs[:, : query_frame + 1].flip([1]),
                iters=INFERENCE_ITERS,
                sw=None,
                is_training=False,
            )
            b_flows = b_flows.to(self.device)
            b_visconf = b_visconf.to(self.device)
            b_traj = b_flows + grid_xy
            b_traj = b_traj.flip([1])[:, :-1]
            b_visconf = b_visconf.flip([1])[:, :-1]
            traj_maps = torch.cat([b_traj, traj_maps], dim=1)
            visconf_maps = torch.cat([b_visconf, visconf_maps], dim=1)

        def _to_thwc(value: torch.Tensor, channels: int, name: str) -> torch.Tensor:
            """Normalize AllTracker output from BCHW/BTHWC variants to THWC."""
            if value.ndim == 5:
                value = value[0]
            if value.ndim != 4:
                raise RuntimeError(f"AllTracker {name} output has unsupported shape {tuple(value.shape)}")
            if value.shape[-1] == channels:
                return value
            if value.shape[1] == channels:
                return value.permute(0, 2, 3, 1)
            raise RuntimeError(f"AllTracker {name} output has ambiguous shape {tuple(value.shape)}")

        traj_np = _to_thwc(traj_maps, 2, "trajectory").contiguous().cpu().numpy()
        visconf_np = _to_thwc(visconf_maps, 2, "visibility").contiguous().cpu().numpy()
        if traj_np.shape[0] != t_len:
            raise RuntimeError(
                f"AllTracker returned T={traj_np.shape[0]}, expected {t_len} for query_frame={query_frame}"
            )
        return traj_np, visconf_np

    def run_chunk(self, rgb: np.ndarray, query_frame: int = 0) -> dict:
        """Track all pixels in ``rgb`` (T,H,W,3) and return merge-compatible dict."""
        traj_maps, visconf_maps = self._forward_maps(rgb, query_frame=query_frame)
        return tracks_dict_from_maps(
            traj_maps,
            visconf_maps,
            conf_thr=self.conf_thr,
            query_global_frame=query_frame,
        )

    def run_chunk_from_last(self, rgb: np.ndarray) -> dict:
        """Track all pixels anchored at the last frame via temporal flip."""
        if rgb.shape[0] <= 1:
            return self.run_chunk(rgb, query_frame=0)
        rgb_flip = np.ascontiguousarray(rgb[::-1])
        flipped = self.run_chunk(rgb_flip, query_frame=0)
        out = {
            "trajs_uv": np.ascontiguousarray(flipped["trajs_uv"][::-1]),
            "vis": np.ascontiguousarray(flipped["vis"][::-1]),
            "query_global_frame": int(rgb.shape[0] - 1),
        }
        if "confidence" in flipped:
            out["confidence"] = np.ascontiguousarray(flipped["confidence"][::-1])
        return out

    def run_chunk_dual_endpoints(self, rgb: np.ndarray) -> dict:
        """Track from first and last frames, concatenating both track banks."""
        t_len = int(rgb.shape[0])
        if t_len <= 1:
            return self.run_chunk(rgb, query_frame=0)
        tracks_first = self.run_chunk(rgb, query_frame=0)
        tracks_last = self.run_chunk_from_last(rgb)
        return merge_dual_endpoint_tracks(tracks_first, tracks_last)


def merge_dual_endpoint_tracks(tracks_first: dict, tracks_last: dict) -> dict:
    """Union forward (query=0) and backward (query=T-1) dense track banks."""
    trajs_a = tracks_first["trajs_uv"]
    trajs_b = tracks_last["trajs_uv"]
    if trajs_a.shape[0] != trajs_b.shape[0]:
        raise ValueError(
            f"dual-endpoint track length mismatch: {trajs_a.shape[0]} vs {trajs_b.shape[0]}"
        )
    vis_a = tracks_first["vis"]
    vis_b = tracks_last["vis"]
    conf_a = tracks_first.get("confidence")
    conf_b = tracks_last.get("confidence")
    merged = {
        "trajs_uv": np.concatenate([trajs_a, trajs_b], axis=1),
        "vis": np.concatenate([vis_a, vis_b], axis=1),
        "query_global_frame": tracks_first.get("query_global_frame", 0),
        "query_modes": ["first", "last"],
    }
    if conf_a is not None and conf_b is not None:
        merged["confidence"] = np.concatenate([conf_a, conf_b], axis=1)
    for key in ("trajs_uv_dense", "vis_dense", "dense_reso", "orig_hw"):
        if key in tracks_first:
            merged[f"{key}_first"] = tracks_first[key]
        if key in tracks_last:
            merged[f"{key}_last"] = tracks_last[key]
    return merged
