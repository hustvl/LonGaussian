#!/usr/bin/env python3
"""Select long RealEstate10K training sequences from the official metadata.

This script only reads the small camera metadata files.  It deliberately
rejects scene keys and source URLs present in the official test metadata, so
the resulting manifest is safe to use as a download/materialization queue.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path


def _first_line(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        return handle.readline().strip()


def _rank(seed: int, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()


def _read_candidate(path: Path, min_frames: int) -> dict | None:
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    if not lines or not lines[0].startswith(("https://", "http://")):
        return None
    camera_lines = [line for line in lines[1:] if line.strip()]
    if len(camera_lines) < min_frames:
        return None

    first = [float(value) for value in camera_lines[0].split()]
    last = [float(value) for value in camera_lines[-1].split()]
    if len(first) != 19 or len(last) != 19:
        return None
    if not all(math.isfinite(value) for value in first + last):
        return None
    fx, fy, cx, cy = first[1:5]
    if fx <= 0 or fy <= 0 or not (-1 <= cx <= 2 and -1 <= cy <= 2):
        return None
    if int(last[0]) <= int(first[0]):
        return None
    return {
        "key": path.stem,
        "url": lines[0],
        "frames": len(camera_lines),
        "timestamp_first": int(first[0]),
        "timestamp_last": int(last[0]),
        "fx_norm": fx,
        "fy_norm": fy,
        "cx_norm": cx,
        "cy_norm": cy,
        "metadata_path": str(path.resolve()),
    }


def select(
    metadata_root: Path,
    *,
    primary_count: int,
    reserve_count: int,
    min_frames: int,
    seed: int,
    workers: int = 16,
) -> dict:
    train_root = metadata_root / "train"
    test_root = metadata_root / "test"
    if not train_root.is_dir() or not test_root.is_dir():
        raise FileNotFoundError("metadata root must contain train/ and test/")

    test_files = sorted(test_root.glob("*.txt"))
    forbidden_keys = {path.stem for path in test_files}
    forbidden_urls = {_first_line(path) for path in test_files}
    forbidden_urls.discard("")

    train_files = sorted(train_root.glob("*.txt"))
    with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        parsed = pool.map(lambda path: _read_candidate(path, min_frames), train_files)
        candidates = [row for row in parsed if row is not None]

    eligible = []
    rejected_test_overlap = 0
    for candidate in candidates:
        if candidate["key"] in forbidden_keys or candidate["url"] in forbidden_urls:
            rejected_test_overlap += 1
            continue
        eligible.append(candidate)

    eligible.sort(key=lambda row: _rank(seed, row["key"]))
    requested = primary_count + reserve_count
    if len(eligible) < requested:
        raise RuntimeError(f"only {len(eligible)} eligible train scenes; need {requested}")
    chosen = eligible[:requested]
    primary = chosen[:primary_count]
    reserve = chosen[primary_count:]
    payload = {
        "format": "loger_re10k_train_long_metadata_selection_v1",
        "metadata_root": str(metadata_root.resolve()),
        "split": "official_train",
        "min_frame_count": min_frames,
        "seed": seed,
        "official_test_scene_count": len(forbidden_keys),
        "official_test_url_count": len(forbidden_urls),
        "eligible_train_scene_count": len(eligible),
        "rejected_official_test_overlap": rejected_test_overlap,
        "primary_count": len(primary),
        "reserve_count": len(reserve),
        "scene_keys_sha256": hashlib.sha256(
            json.dumps([row["key"] for row in chosen], separators=(",", ":")).encode()
        ).hexdigest(),
        "primary": primary,
        "reserve": reserve,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--primary-count", type=int, default=600)
    parser.add_argument("--reserve-count", type=int, default=600)
    parser.add_argument("--min-frame-count", type=int, default=146)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--scene-list", type=Path, help="Required fixed JSON identity list for test selection")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    payload = select(
        args.metadata_root,
        primary_count=args.primary_count,
        reserve_count=args.reserve_count,
        min_frames=args.min_frame_count,
        seed=args.seed,
        workers=args.workers,
    ) if args.split == "train" else None
    if args.split == "test":
        if not args.scene_list:
            parser.error("--scene-list is required for the fixed test split")
        keys = json.loads(args.scene_list.read_text())
        if not isinstance(keys, list) or len(keys) != len(set(keys)):
            raise ValueError("Expected unique test scene IDs")
        rows = []
        for key in keys:
            if not isinstance(key, str) or Path(key).name != key or key in {".", ".."}:
                raise ValueError("Invalid test scene identity")
            row = _read_candidate(args.metadata_root / "test" / (key + ".txt"), args.min_frame_count)
            if row is None:
                raise ValueError(f"Fixed test scene is not eligible: {key}")
            rows.append(row)
        payload = {"split": "official_test", "primary": rows, "reserve": [],
                   "primary_count": len(rows), "reserve_count": 0, "eligible_train_scene_count": 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"selected {payload['primary_count']} primary + {payload['reserve_count']} reserve "
        f"from {payload['eligible_train_scene_count']} eligible official-train scenes"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
