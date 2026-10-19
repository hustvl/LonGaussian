"""Track-RD diagnostic records and deterministic dump validation."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Sequence

import torch


TRACK_RD_DUMP_VERSION = 2
K1_REFERENCE_KEYS = ("track_ids", "means", "quats", "scales", "opacities", "sh")


def _cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().contiguous()


def build_selector_window_record(
    *,
    window_idx: int,
    global_frame_ids: Sequence[int],
    merge_track_ids: torch.Tensor,
    selected_rows: torch.Tensor,
    selected_cols: torch.Tensor,
    selected_visible: torch.Tensor,
    covered: torch.Tensor,
) -> Dict[str, Any]:
    """Record the hard-merge discrete choices for deterministic-run auditing."""
    if covered.ndim != 3:
        raise ValueError("covered must have shape (T, H, W)")
    if len(global_frame_ids) != covered.shape[0]:
        raise ValueError("global_frame_ids length must match coverage time dimension")
    expected_shape = (covered.shape[0], merge_track_ids.numel())
    if (
        selected_rows.shape != expected_shape
        or selected_cols.shape != expected_shape
        or selected_visible.shape != expected_shape
    ):
        raise ValueError("selected source indices/visibility must have shape (T, K)")
    return {
        "window_idx": int(window_idx),
        "global_frame_ids": [int(frame_id) for frame_id in global_frame_ids],
        "merge_track_ids": _cpu(merge_track_ids.to(dtype=torch.int64)),
        # Coordinates of an invisible observation do not affect the hybrid
        # cloud; use a sentinel so an irrelevant tracker coordinate cannot
        # make an otherwise identical selector trace fail R001.
        "selected_rows": _cpu(
            torch.where(
                selected_visible,
                selected_rows.to(dtype=torch.int64),
                torch.full_like(selected_rows, -1, dtype=torch.int64),
            )
        ),
        "selected_cols": _cpu(
            torch.where(
                selected_visible,
                selected_cols.to(dtype=torch.int64),
                torch.full_like(selected_cols, -1, dtype=torch.int64),
            )
        ),
        "selected_visible": _cpu(selected_visible.to(dtype=torch.bool)),
        "covered": _cpu(covered.to(dtype=torch.bool)),
    }


def selector_fingerprint(records: Sequence[Dict[str, Any]]) -> str:
    """Hash exact hard-merge IDs and dense-complement coverage in window order."""
    digest = hashlib.sha256(b"loger_track_rd_selector_v1\\0")
    for record in records:
        digest.update(int(record["window_idx"]).to_bytes(8, "little", signed=True))
        for frame_id in record["global_frame_ids"]:
            digest.update(int(frame_id).to_bytes(8, "little", signed=True))
        for key in (
            "merge_track_ids",
            "selected_rows",
            "selected_cols",
            "selected_visible",
            "covered",
        ):
            tensor = record[key].detach().cpu().contiguous()
            digest.update(key.encode("utf-8") + b"\\0")
            digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\\0")
            digest.update(str(tensor.dtype).encode("ascii") + b"\\0")
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def build_track_window_record(
    *,
    window_idx: int,
    global_frame_ids: Sequence[int],
    centers: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    sh: torch.Tensor,
    trajs_uv: torch.Tensor,
    visible: torch.Tensor,
    owned: torch.Tensor,
    k1_reference: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    """Build a CPU-only, self-contained record for one merge window."""
    if centers.ndim != 3 or centers.shape[-1] != 3:
        raise ValueError("centers must have shape (T, M, 3)")
    if visible.shape != centers.shape[:2] or owned.shape != visible.shape:
        raise ValueError("visibility/ownership shapes must match (T, M)")
    if len(global_frame_ids) != centers.shape[0]:
        raise ValueError("global_frame_ids length must match track time dimension")
    if set(K1_REFERENCE_KEYS) != set(k1_reference):
        raise ValueError("k1_reference has unexpected keys")
    return {
        "window_idx": int(window_idx),
        "global_frame_ids": [int(frame_id) for frame_id in global_frame_ids],
        "centers": _cpu(centers),
        "quats": _cpu(quats),
        "scales": _cpu(scales),
        "opacities": _cpu(opacities),
        "sh": _cpu(sh),
        "trajs_uv": _cpu(trajs_uv),
        "visible": _cpu(visible.to(torch.bool)),
        "owned": _cpu(owned.to(torch.bool)),
        "k1_reference": {key: _cpu(k1_reference[key]) for key in K1_REFERENCE_KEYS},
        "reference_complement": None,
    }


def track_window_summary(record: Dict[str, Any]) -> Dict[str, int]:
    """Return cheap integrity statistics for a serialized track window."""
    centers = record["centers"]
    visible = record["visible"]
    owned = record["owned"]
    if centers.ndim != 3 or visible.shape != centers.shape[:2]:
        raise ValueError("invalid Track-RD dump record")
    return {
        "num_frames": int(centers.shape[0]),
        "num_tracks": int(centers.shape[1]),
        "num_visible_observations": int(visible.sum().item()),
        "num_owned_observations": int(owned.sum().item()),
    }


def build_track_rd_dump(records: List[Dict[str, Any]], *, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble a versioned dump consumed by offline Track-RD oracle tools."""
    if not records:
        raise ValueError("Track-RD dump must contain at least one window")
    return {
        "format": "loger_track_rd",
        "version": TRACK_RD_DUMP_VERSION,
        "metadata": dict(metadata),
        "windows": records,
        "window_summaries": [track_window_summary(record) for record in records],
    }
