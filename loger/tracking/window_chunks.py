"""Sliding-window chunk utilities aligned with Pi3 windowing / merge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class ChunkSpec:
    """One inference window and its contribution range after Pi3-style merge."""

    chunk_idx: int
    window_start: int
    window_end: int
    global_frames: Tuple[int, ...]
    query_global_frame: int
    contribute_start: int
    contribute_end: int


def iter_sliding_windows(
    num_frames: int,
    window_size: int,
    overlap_size: int,
) -> List[Tuple[int, int]]:
    """Return (start, end) window ranges matching ``pi3.py`` forward."""
    if window_size <= 0 or window_size >= num_frames:
        return [(0, num_frames)]

    windows: List[Tuple[int, int]] = []
    step = max(window_size - overlap_size, 1)
    for start_idx in range(0, num_frames, step):
        end_idx = min(start_idx + window_size, num_frames)
        if end_idx - start_idx >= overlap_size or (end_idx == num_frames and start_idx < num_frames):
            windows.append((start_idx, end_idx))
        if end_idx == num_frames:
            break
    return windows


def _contribute_range(
    chunk_idx: int,
    num_chunks: int,
    window_start: int,
    window_end: int,
    overlap_size: int,
) -> Tuple[int, int]:
    if num_chunks == 1:
        return window_start, window_end
    if chunk_idx == 0:
        if overlap_size > 0 and (window_end - window_start) > overlap_size:
            return window_start, window_end - overlap_size
        return window_start, window_end
    if chunk_idx == num_chunks - 1:
        return window_start, window_end
    if overlap_size > 0 and (window_end - window_start) > overlap_size:
        return window_start, window_end - overlap_size
    return window_start, window_end


def chunk_specs_for_frames(
    frame_indices: Sequence[int],
    window_size: int,
    overlap_size: int,
) -> List[ChunkSpec]:
    """Build chunk specs over a subsequence of global frame indices (e.g. ctx_list)."""
    frames = tuple(int(i) for i in frame_indices)
    n = len(frames)
    if n == 0:
        return []

    windows = iter_sliding_windows(n, window_size, overlap_size)
    num_chunks = len(windows)
    specs: List[ChunkSpec] = []

    for chunk_idx, (local_start, local_end) in enumerate(windows):
        global_frames = frames[local_start:local_end]
        contrib_local_start, contrib_local_end = _contribute_range(
            chunk_idx, num_chunks, local_start, local_end, overlap_size
        )
        contribute_start = frames[contrib_local_start]
        if contrib_local_end > contrib_local_start:
            contribute_end = frames[contrib_local_end - 1] + 1
        else:
            contribute_end = contribute_start
        specs.append(
            ChunkSpec(
                chunk_idx=chunk_idx,
                window_start=local_start,
                window_end=local_end,
                global_frames=global_frames,
                query_global_frame=global_frames[0],
                contribute_start=contribute_start,
                contribute_end=contribute_end,
            )
        )
    return specs
