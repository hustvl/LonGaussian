#!/usr/bin/env python3
"""Prepare a contiguous DL3DV context-count evaluation root.

The formal project protocol counts context views, while every eighth raw frame
is held out as a target. Main-table lengths are C ∈ {64,96,128}. For a
requested context count C, this materializes the first C + (C // 7) raw frames
and holds out indices 7,15,... from the prefix; the resulting split has exactly
C context views (e.g. C=32 → 36 frames / 4 targets).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from numbers import Real
from pathlib import Path
from typing import Any

from PIL import Image


def normalize_scene_id(raw_scene_id: object) -> str:
    """Normalize an index entry without discarding an optional shard prefix.

    Official DL3DV-140 entries use ``<hash>/nerfstudio`` while local DL3DV-10K
    manifests use ``<shard>/<hash>``.  Only the storage-view suffix is removed;
    splitting on the first slash would incorrectly collapse every local entry
    in a shard to the same directory (for example, ``1K``).
    """
    value = str(raw_scene_id).strip().replace("\\", "/")
    if not value or value.startswith("/"):
        raise ValueError(f"invalid scene id: {raw_scene_id!r}")
    parts = value.split("/")
    if parts and parts[-1] == "nerfstudio":
        parts.pop()
    if (
        not parts
        or any(part in {"", ".", ".."} for part in parts)
        or any(":" in part for part in parts)
    ):
        raise ValueError(f"invalid scene id: {raw_scene_id!r}")
    return "/".join(parts)


def scene_identity(scene_id: str) -> str:
    return normalize_scene_id(scene_id).split("/")[-1]


def scene_list_sha256(scene_ids: list[str]) -> str:
    payload = json.dumps(
        scene_ids, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolved_path_sha256(path: Path) -> str:
    """Hash the canonical path string, explicitly not directory contents."""
    resolved = str(path.resolve(strict=True))
    return hashlib.sha256(resolved.encode("utf-8")).hexdigest()


def validate_frame_record(frame: object) -> str:
    """Validate the subset of a DL3DV frame record required by evaluation."""
    if not isinstance(frame, dict) or not isinstance(frame.get("file_path"), str):
        raise ValueError("invalid frame record/file_path")
    file_path = frame["file_path"]
    if not file_path.startswith("images/"):
        raise ValueError(f"non-standard frame file_path: {file_path!r}")
    relative = file_path[len("images/") :].replace("\\", "/")
    parts = relative.split("/")
    if (
        not relative
        or relative.startswith("/")
        or any(part in {"", ".", ".."} for part in parts)
        or any(":" in part for part in parts)
        or "images" in parts
    ):
        raise ValueError(f"unsafe frame file_path: {file_path!r}")

    matrix = frame.get("transform_matrix")
    if not isinstance(matrix, list) or len(matrix) != 4:
        raise ValueError("transform_matrix must be a 4x4 numeric list")
    for row in matrix:
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError("transform_matrix must be a 4x4 numeric list")
        for value in row:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError("transform_matrix contains a non-numeric value")
            if not math.isfinite(float(value)):
                raise ValueError("transform_matrix contains a non-finite value")
    return relative


def resolve_frame_image(scene_view: Path, frame: object) -> Path:
    """Resolve one frame exactly as the evaluator's images -> images_8 mapping."""
    relative = validate_frame_record(frame)
    image_root = (scene_view / "images_8").resolve(strict=True)
    image = (image_root / relative).resolve(strict=True)
    try:
        image.relative_to(image_root)
    except ValueError as exc:
        raise ValueError(f"frame image escapes images_8: {relative!r}") from exc
    if not image.is_file():
        raise FileNotFoundError(image)
    return image


def decode_image(path: Path) -> None:
    """Fully decode one image using the same Pillow backend as evaluation."""
    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            rgb.load()
    except (OSError, SyntaxError, ValueError) as exc:
        raise ValueError(f"image is not decodable: {path}") from exc


def load_scene_index(index_path: Path) -> tuple[list[str], dict[str, Any]]:
    raw_text = index_path.read_text(encoding="utf-8")
    try:
        value = json.loads(raw_text)
    except json.JSONDecodeError:
        # Fairness splits are deliberately stored as one immutable scene ID per
        # line.  Accept that minimal representation as well as the historical
        # JSON index/manifest formats without silently interpreting malformed
        # JSON as a split.
        raw_entries = [line.strip() for line in raw_text.splitlines() if line.strip()]
        if not raw_entries or any(" " in entry for entry in raw_entries):
            raise
        value = raw_entries
    metadata: dict[str, Any] = {}
    if isinstance(value, list):
        raw_entries = value
    elif isinstance(value, dict) and isinstance(value.get("scenes"), list):
        metadata = value
        raw_entries = []
        for item in value["scenes"]:
            if isinstance(item, str):
                raw_entries.append(item)
            elif isinstance(item, dict) and isinstance(item.get("entry"), str):
                raw_entries.append(item["entry"])
            else:
                raise ValueError(f"invalid scene record in {index_path}: {item!r}")
    else:
        raise ValueError(
            f"{index_path}: expected a list or a manifest with a scenes list"
        )
    scene_ids = [normalize_scene_id(entry) for entry in raw_entries]
    identities = [scene_identity(entry) for entry in scene_ids]
    if len(identities) != len(set(identities)):
        raise ValueError(f"duplicate scene identities after normalization: {index_path}")
    expected_sha = metadata.get("scene_list_sha256")
    actual_sha = scene_list_sha256(scene_ids)
    if expected_sha is not None and expected_sha != actual_sha:
        raise ValueError(
            f"{index_path}: scene_list_sha256 mismatch "
            f"(expected {expected_sha}, got {actual_sha})"
        )
    return scene_ids, metadata


def resolve_scene_view(dataset_root: Path, scene_id: str) -> Path:
    root = dataset_root.resolve(strict=True)
    scene = (root / normalize_scene_id(scene_id)).resolve(strict=True)
    try:
        scene.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{scene_id}: scene path escapes dataset root") from exc
    candidates = (scene, scene / "nerfstudio")
    for candidate in candidates:
        if (candidate / "transforms.json").is_file() and (
            candidate / "images_8"
        ).is_dir():
            return candidate
    raise RuntimeError(f"{scene_id}: missing transforms.json / images_8")


def context_split(context_count: int, target_stride: int) -> tuple[list[int], list[int]]:
    if context_count < 1 or target_stride < 2:
        raise ValueError("context-count >= 1 and target-stride >= 2 are required")
    target_count = context_count // (target_stride - 1)
    total_frames = context_count + target_count
    tgt = [target_stride - 1 + target_stride * i for i in range(target_count)]
    tgt_set = set(tgt)
    ctx = [i for i in range(total_frames) if i not in tgt_set]
    if len(ctx) != context_count:
        raise AssertionError((len(ctx), context_count, total_frames, tgt))
    return ctx, tgt


def prepare_context_eval_root(
    *,
    dataset_root: Path,
    index_path: Path,
    output_root: Path,
    context_count: int,
    target_stride: int = 8,
    sampling: str = "contiguous",
    num_scenes: int = 0,
    num_shards: int = 1,
    shard_id: int = 0,
) -> dict[str, Any]:
    if num_shards < 1 or not 0 <= shard_id < num_shards:
        raise ValueError("invalid shard specification")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    scene_ids, source_manifest = load_scene_index(index_path)
    if num_scenes > 0:
        scene_ids = scene_ids[:num_scenes]
    scene_ids = scene_ids[shard_id::num_shards]
    if sampling == "contiguous":
        ctx, tgt = context_split(context_count, target_stride)
        selected_original_indices = list(range(len(ctx) + len(tgt)))
    elif sampling == "fixed_interval":
        # Locked formal sparse protocol: C context views at I=8 and every
        # eighth raw frame is a target.  Pack only selected frames, while
        # preserving the original-index provenance for auditability.
        original_ctx = [i * target_stride for i in range(context_count)]
        window = context_count * target_stride
        original_tgt = list(range(target_stride - 1, window, target_stride))
        selected_original_indices = sorted(set(original_ctx) | set(original_tgt))
        remap = {old: new for new, old in enumerate(selected_original_indices)}
        ctx = [remap[i] for i in original_ctx]
        tgt = [remap[i] for i in original_tgt]
    else:
        raise ValueError(f"unsupported sampling mode: {sampling}")
    total_frames = max(selected_original_indices) + 1
    build_root = output_root.parent / f".{output_root.name}.building-{os.getpid()}"
    if build_root.exists():
        raise FileExistsError(f"stale build root exists: {build_root}")
    build_root.mkdir(parents=True)
    try:
        for scene_id in scene_ids:
            src_view = resolve_scene_view(dataset_root, scene_id)
            meta_path = src_view / "transforms.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            frames = meta.get("frames")
            if not isinstance(frames, list) or len(frames) < total_frames:
                count = len(frames) if isinstance(frames, list) else 0
                raise RuntimeError(f"{scene_id}: {count} < {total_frames}")
            selected = [frames[i] for i in selected_original_indices]
            for frame in selected:
                try:
                    resolve_frame_image(src_view, frame)
                except (FileNotFoundError, ValueError) as exc:
                    raise RuntimeError(f"{scene_id}: invalid referenced frame") from exc
            dst = build_root / scene_id
            dst.mkdir(parents=True)
            out_meta = dict(meta)
            out_meta["frames"] = selected
            out_meta["loger_selected_original_indices"] = selected_original_indices
            (dst / "transforms.json").write_text(
                json.dumps(out_meta, indent=2) + "\n", encoding="utf-8"
            )
            (dst / "images_8").symlink_to(
                (src_view / "images_8").resolve(strict=True),
                target_is_directory=True,
            )
            (dst / "split.json").write_text(
                json.dumps(
                    {
                        "ctx_list": ctx,
                        "tgt_list": tgt,
                        "selected_original_indices": selected_original_indices,
                        "sampling": {
                            "mode": sampling,
                            "context_count": context_count,
                            "target_stride": target_stride,
                            "source_window_frames": total_frames,
                        },
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        (build_root / "test_index.json").write_text(
            json.dumps(scene_ids, indent=2) + "\n", encoding="utf-8"
        )
        (build_root / "train_index.json").write_text("[]\n", encoding="utf-8")
        result = {
        "format": "loger_dl3dv140_context_eval_v1",
            "dataset_root": str(dataset_root.resolve(strict=True)),
            "source_index": str(index_path.resolve(strict=True)),
            "source_index_sha256": file_sha256(index_path),
            "source_manifest_role": source_manifest.get("role"),
            "scene_count": len(scene_ids),
            "scene_list_sha256": scene_list_sha256(scene_ids),
            "context_count": context_count,
            "target_stride": target_stride,
            "sampling": sampling,
            "source_window_frames": total_frames,
            "num_frames": len(selected_original_indices),
            "num_packed_frames": len(selected_original_indices),
            "num_shards": num_shards,
            "shard_id": shard_id,
            "ctx_list": ctx,
            "tgt_list": tgt,
        }
        (build_root / "manifest.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(build_root, output_root)
    except BaseException:
        shutil.rmtree(build_root, ignore_errors=True)
        raise
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--index", type=Path, default=None)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--context-count", type=int, required=True)
    p.add_argument("--target-stride", type=int, default=8)
    p.add_argument("--sampling", choices=("contiguous", "fixed_interval"), default="contiguous")
    p.add_argument("--num-scenes", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-id", type=int, default=0)
    args = p.parse_args()
    index_path = args.index or (args.dataset_root / "dl3dv140_index.json")
    result = prepare_context_eval_root(
        dataset_root=args.dataset_root,
        index_path=index_path,
        output_root=args.output_root,
        context_count=args.context_count,
        target_stride=args.target_stride,
        sampling=args.sampling,
        num_scenes=args.num_scenes,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
    )
    print(
        f"prepared {result['scene_count']} scenes: "
        f"{result['context_count']} ctx, {len(result['tgt_list'])} tgt, "
        f"{result['num_packed_frames']} packed frames ({result['sampling']})"
    )


if __name__ == "__main__":
    main()
