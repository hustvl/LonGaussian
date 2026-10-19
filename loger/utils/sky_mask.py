"""Semantic sky masks via skyseg.onnx (VGGT-compatible).

Strategy A: hard-drop sky pixels from track merge and dense complement.
The ONNX model outputs high scores for sky and low scores for non-sky.
"""

from __future__ import annotations

import os
import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Sequence, Union

import cv2
import numpy as np
import torch

# Default HF mirror used by loger.utils.visual_util
DEFAULT_SKYSEG_URL = (
    "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx"
)
DEFAULT_SCORE_THR = 160  # conservative threshold on uint8 [0,255]; high = sky

# A scene is evaluated by overlapping merge windows and then by the sky-sphere
# builder.  Both consumers see the exact same uint8 context frames, often via
# different SkyMasker instances.  Keep the masks process-wide so ONNX inference
# is performed once per distinct frame.  The bound covers the largest current
# evaluation context while preventing an unbounded multi-scene accumulation.
_MEMORY_MASK_CACHE_MAX_ITEMS = 256
_MEMORY_MASK_CACHE = OrderedDict()


def _ensure_onnx(path: Union[str, Path]) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    from loger.utils.visual_util import download_file_from_url

    print(f"[sky_mask] downloading skyseg.onnx -> {path}", flush=True)
    download_file_from_url(DEFAULT_SKYSEG_URL, str(path))
    if not path.is_file():
        raise FileNotFoundError(f"failed to download skyseg.onnx to {path}")
    return path


def _run_skyseg_bgr(
    session,
    image_bgr: np.ndarray,
    input_size: Sequence[int] = (320, 320),
) -> np.ndarray:
    """Return HxW uint8 score map (high ≈ sky), original resolution."""
    from loger.utils.visual_util import run_skyseg

    h, w = image_bgr.shape[:2]
    score = run_skyseg(session, list(input_size), image_bgr)
    if score.shape[0] != h or score.shape[1] != w:
        score = cv2.resize(score, (w, h), interpolation=cv2.INTER_LINEAR)
    return score.astype(np.uint8)


class SkyMasker:
    """Lazy ONNX sky segmenter with optional disk cache."""

    def __init__(
        self,
        onnx_path: Union[str, Path] = "skyseg.onnx",
        score_thr: float = DEFAULT_SCORE_THR,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        self.onnx_path = Path(onnx_path)
        self.score_thr = float(score_thr)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._session = None

    def _get_session(self):
        if self._session is not None:
            return self._session
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "sky_mask_drop requires onnxruntime; "
                "install with: pip install onnxruntime"
            ) from exc
        path = _ensure_onnx(self.onnx_path)
        self._session = ort.InferenceSession(str(path))
        return self._session

    def score_rgb_uint8(self, rgb: np.ndarray) -> np.ndarray:
        """(H,W,3) RGB uint8 -> (H,W) uint8 score (high=sky)."""
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return _run_skyseg_bgr(self._get_session(), bgr)

    def is_sky_rgb_uint8(self, rgb: np.ndarray) -> np.ndarray:
        """(H,W,3) RGB uint8 -> (H,W) bool, True = sky."""
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError(f"expected (H,W,3) uint8 RGB, got {rgb.shape} {rgb.dtype}")
        contiguous = np.ascontiguousarray(rgb)
        digest = hashlib.blake2b(contiguous, digest_size=16).digest()
        key = (str(self.onnx_path.resolve()), self.score_thr, contiguous.shape, digest)
        cached = _MEMORY_MASK_CACHE.get(key)
        if cached is not None:
            _MEMORY_MASK_CACHE.move_to_end(key)
            return cached

        mask = self.score_rgb_uint8(contiguous) > self.score_thr
        # Callers only read/index masks.  Marking the cached value read-only
        # prevents one consumer from silently corrupting later consumers.
        mask.setflags(write=False)
        _MEMORY_MASK_CACHE[key] = mask
        _MEMORY_MASK_CACHE.move_to_end(key)
        while len(_MEMORY_MASK_CACHE) > _MEMORY_MASK_CACHE_MAX_ITEMS:
            _MEMORY_MASK_CACHE.popitem(last=False)
        return mask

    def masks_from_rgb_batch(
        self,
        rgb_batch: np.ndarray,
        *,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """(T,H,W,3) RGB uint8 -> (T,H,W) bool torch mask, True = sky."""
        if rgb_batch.ndim != 4 or rgb_batch.shape[-1] != 3:
            raise ValueError(f"expected (T,H,W,3) rgb, got {rgb_batch.shape}")
        masks = [self.is_sky_rgb_uint8(rgb_batch[t]) for t in range(rgb_batch.shape[0])]
        out = torch.from_numpy(np.stack(masks, axis=0))
        if device is not None:
            out = out.to(device=device)
        return out


def sky_mask_from_uint8_window(
    rgb_uint8: np.ndarray,
    *,
    onnx_path: Union[str, Path] = "skyseg.onnx",
    score_thr: float = DEFAULT_SCORE_THR,
    device: Optional[torch.device] = None,
    masker: Optional[SkyMasker] = None,
) -> torch.Tensor:
    """Convenience: (Nw,H,W,3) uint8 RGB -> (Nw,H,W) bool sky mask."""
    m = masker or SkyMasker(onnx_path=onnx_path, score_thr=score_thr)
    return m.masks_from_rgb_batch(rgb_uint8, device=device)
