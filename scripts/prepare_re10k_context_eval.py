#!/usr/bin/env python3
"""Convert preprocessed RealEstate10K torch shards to LoGeR eval scenes.

The supported input is the pixelSplat-style layout ``<root>/test/*.torch``.
Each shard contains a list of scenes with JPEG bytes, normalized intrinsics,
and flattened 3x4 world-to-camera matrices.  Output scenes deliberately use
the existing LoGeR ``transforms.json + images_8 + split.json`` seam so the
same model, renderer, and metric implementation are used for DL3DV and RE10K.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image


def context_split(context_count: int, target_stride: int) -> tuple[list[int], list[int]]:
    """Return a prefix containing exactly C context views and interleaved targets."""
    if context_count < 1 or target_stride < 2:
        raise ValueError("context_count >= 1 and target_stride >= 2 are required")
    target_count = context_count // (target_stride - 1)
    total = context_count + target_count
    targets = list(range(target_stride - 1, total, target_stride))
    context = [i for i in range(total) if i not in set(targets)]
    if len(context) != context_count:
        raise AssertionError((context_count, context, targets))
    return context, targets


def _safe_key(value: Any) -> str:
    key = str(value).strip().replace("\\", "/")
    if not key or "/" in key or key in {".", ".."}:
        raise ValueError(f"unsafe RE10K scene key: {value!r}")
    return key


def _jpeg_bytes(value: Any) -> bytes:
    if isinstance(value, torch.Tensor):
        if value.ndim != 1 or value.dtype != torch.uint8:
            raise ValueError("encoded image must be a 1-D uint8 tensor")
        return value.numpy().tobytes()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    raise ValueError(f"unsupported encoded image type: {type(value).__name__}")


def _camera_record(row: torch.Tensor, width: int, height: int) -> tuple[dict, torch.Tensor]:
    row = torch.as_tensor(row, dtype=torch.float64)
    if row.shape != (18,) or not torch.isfinite(row).all():
        raise ValueError("camera must be a finite 18-vector")
    fx, fy, cx, cy = row[:4].tolist()
    w2c = torch.eye(4, dtype=torch.float64)
    w2c[:3] = row[6:].reshape(3, 4)
    c2w = torch.linalg.inv(w2c)
    if not torch.isfinite(c2w).all():
        raise ValueError("camera inversion produced non-finite values")
    intrinsics = {
        "fl_x": fx * width,
        "fl_y": fy * height,
        "cx": cx * width,
        "cy": cy * height,
    }
    return intrinsics, c2w


def _center_crop_resize(image: Image.Image, target_width: int, target_height: int) -> tuple[Image.Image, float, float, float, float]:
    """Crop to target aspect ratio, then resize; return crop/scale parameters."""
    width, height = image.size
    target_ratio = target_width / target_height
    source_ratio = width / height
    if source_ratio > target_ratio:
        crop_height, crop_width = height, int(round(height * target_ratio))
        crop_left, crop_top = (width - crop_width) // 2, 0
    else:
        crop_width, crop_height = width, int(round(width / target_ratio))
        crop_left, crop_top = 0, (height - crop_height) // 2
    image = image.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))
    scale_x, scale_y = target_width / crop_width, target_height / crop_height
    image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return image, float(crop_left), float(crop_top), scale_x, scale_y


def convert_scene(scene: dict, destination: Path, frame_count: int,
                  context: list[int], targets: list[int], target_stride: int,
                  target_width: int, target_height: int) -> dict:
    key = _safe_key(scene.get("key", ""))
    images = scene.get("images")
    cameras = scene.get("cameras")
    timestamps = torch.as_tensor(scene.get("timestamps"))
    if not isinstance(images, list) or len(images) < frame_count:
        raise ValueError(f"{key}: insufficient images")
    cameras = torch.as_tensor(cameras)
    if cameras.ndim != 2 or cameras.shape[0] < frame_count or cameras.shape[1] != 18:
        raise ValueError(f"{key}: invalid/insufficient cameras")
    if timestamps.ndim != 1 or len(timestamps) < frame_count:
        raise ValueError(f"{key}: invalid/insufficient timestamps")

    destination.mkdir(parents=True)
    image_dir = destination / "images_8"
    image_dir.mkdir()
    frames = []
    first_intrinsics = None
    for index in range(frame_count):
        encoded = _jpeg_bytes(images[index])
        with Image.open(io.BytesIO(encoded)) as image:
            image = image.convert("RGB")
            image.load()
            source_width, source_height = image.size
            image, crop_left, crop_top, scale_x, scale_y = _center_crop_resize(
                image, target_width, target_height
            )
            width, height = image.size
            name = f"{index:06d}.jpg"
            image.save(image_dir / name, format="JPEG", quality=95)
        intrinsics, c2w = _camera_record(cameras[index], source_width, source_height)
        intrinsics["cx"] = (intrinsics["cx"] - crop_left) * scale_x
        intrinsics["cy"] = (intrinsics["cy"] - crop_top) * scale_y
        intrinsics["fl_x"] *= scale_x
        intrinsics["fl_y"] *= scale_y
        if first_intrinsics is None:
            first_intrinsics = {**intrinsics, "w": width, "h": height}
        else:
            for field in ("fl_x", "fl_y", "cx", "cy"):
                if not math.isclose(intrinsics[field], first_intrinsics[field], rel_tol=1e-4, abs_tol=1e-3):
                    raise ValueError(f"{key}: varying intrinsics are unsupported ({field})")
            if (width, height) != (first_intrinsics["w"], first_intrinsics["h"]):
                raise ValueError(f"{key}: varying image dimensions are unsupported")
        frames.append({
            "file_path": f"images/{name}",
            "transform_matrix": c2w.tolist(),
            "timestamp": int(timestamps[index]),
        })
    assert first_intrinsics is not None
    meta = {
        **first_intrinsics,
        "camera_model": "PINHOLE",
        "frames": frames,
        "re10k_key": key,
        "re10k_url": str(scene.get("url", "")),
        "re10k_camera_convention": "input normalized intrinsics + w2c; output pixel intrinsics + c2w",
    }
    (destination / "transforms.json").write_text(json.dumps(meta, indent=2) + "\n")
    split = {"ctx_list": context, "tgt_list": targets,
             "sampling": {"mode": "contiguous_every_n_holdout", "target_stride": target_stride}}
    (destination / "split.json").write_text(json.dumps(split, indent=2) + "\n")
    return {"key": key, "frames": frame_count, "url": str(scene.get("url", ""))}


def iter_scenes(shards: Iterable[Path]) -> Iterable[tuple[Path, int, dict]]:
    for shard in shards:
        records = torch.load(shard, map_location="cpu", weights_only=True)
        if not isinstance(records, list):
            raise ValueError(f"{shard}: expected a list")
        for offset, scene in enumerate(records):
            if not isinstance(scene, dict):
                raise ValueError(f"{shard}:{offset}: expected a dict")
            yield shard, offset, scene


def validate_prepared_root(output_root: Path, input_root: Path, context_count: int,
                           target_stride: int, num_scenes: int) -> dict:
    """Reject stale or incomplete prepared roots before evaluation."""
    manifest_path = output_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "format": "loger_re10k_context_eval_v1",
        "source_root": str(input_root.resolve()),
        "context_count": context_count,
        "target_stride": target_stride,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(
                f"prepared-root mismatch for {field}: "
                f"expected {value!r}, got {manifest.get(field)!r}"
            )
    if num_scenes > 0 and manifest.get("scene_count") != num_scenes:
        raise ValueError(
            f"prepared-root mismatch for scene_count: expected {num_scenes}, "
            f"got {manifest.get('scene_count')!r}"
        )
    context, targets = context_split(context_count, target_stride)
    if manifest.get("ctx_list") != context or manifest.get("tgt_list") != targets:
        raise ValueError("prepared-root split does not match the requested protocol")
    keys = json.loads((output_root / "test_index.json").read_text())
    if keys != [row["key"] for row in manifest.get("scenes", [])]:
        raise ValueError("test_index.json does not match manifest scene order")
    if len(keys) != manifest.get("scene_count") or len(keys) != len(set(keys)):
        raise ValueError("prepared-root scene keys are missing or non-unique")
    for key in keys:
        scene_root = output_root / _safe_key(key)
        if not (scene_root / "transforms.json").is_file():
            raise FileNotFoundError(scene_root / "transforms.json")
        if not (scene_root / "split.json").is_file():
            raise FileNotFoundError(scene_root / "split.json")
    return manifest


def prepare(input_root: Path, output_root: Path, context_count: int,
            target_stride: int, num_scenes: int, min_frame_count: int = 0,
            scene_keys: set[str] | None = None, target_width: int = 490,
            target_height: int = 280) -> dict:
    shard_dir = input_root / "test" if (input_root / "test").is_dir() else input_root
    shards = sorted(shard_dir.glob("*.torch"))
    if not shards:
        raise FileNotFoundError(f"no .torch shards under {shard_dir}")
    context, targets = context_split(context_count, target_stride)
    frame_count = len(context) + len(targets)
    wanted = num_scenes if num_scenes > 0 else 10**18
    build = output_root.parent / f".{output_root.name}.building-{os.getpid()}"
    if output_root.exists():
        raise FileExistsError(output_root)
    if build.exists():
        raise FileExistsError(build)
    build.mkdir(parents=True)
    selected = []
    rejected = []
    seen_keys: set[str] = set()
    try:
        for shard, offset, scene in iter_scenes(shards):
            if len(selected) >= wanted:
                break
            key = str(scene.get("key", f"{shard.stem}-{offset}"))
            if scene_keys is not None and key not in scene_keys:
                continue
            if min_frame_count and len(scene.get("images", [])) < min_frame_count:
                rejected.append({"shard": shard.name, "offset": offset, "key": key,
                                 "reason": f"sequence shorter than {min_frame_count} frames"})
                continue
            safe_key = _safe_key(key)
            if safe_key in seen_keys:
                rejected.append({"shard": shard.name, "offset": offset, "key": key,
                                 "reason": "duplicate scene key"})
                continue
            destination = build / safe_key
            try:
                row = convert_scene(
                    scene, destination, frame_count, context, targets, target_stride,
                    target_width, target_height
                )
            except (ValueError, OSError, RuntimeError) as exc:
                shutil.rmtree(destination, ignore_errors=True)
                rejected.append({"shard": shard.name, "offset": offset, "key": key, "reason": str(exc)})
                continue
            row.update({"source_shard": shard.name, "source_offset": offset})
            selected.append(row)
            seen_keys.add(safe_key)
        if num_scenes > 0 and len(selected) != num_scenes:
            raise RuntimeError(f"only {len(selected)} valid scenes found; requested {num_scenes}")
        keys = [row["key"] for row in selected]
        (build / "test_index.json").write_text(json.dumps(keys, indent=2) + "\n")
        (build / "train_index.json").write_text("[]\n")
        manifest = {
            "format": "loger_re10k_context_eval_v1",
            "source_root": str(input_root.resolve()),
            "source_shard_count": len(shards),
            "scene_count": len(selected),
            "scene_list_sha256": hashlib.sha256(json.dumps(keys, separators=(",", ":")).encode()).hexdigest(),
            "context_count": context_count,
            "target_stride": target_stride,
            "packed_frame_count": frame_count,
            "target_width": target_width,
            "target_height": target_height,
            "ctx_list": context,
            "tgt_list": targets,
            "scenes": selected,
            "rejected": rejected,
        }
        (build / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(build, output_root)
        return manifest
    except BaseException:
        shutil.rmtree(build, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--context-count", type=int, default=16)
    parser.add_argument("--target-stride", type=int, default=8)
    parser.add_argument("--num-scenes", type=int, default=140)
    parser.add_argument("--min-frame-count", type=int, default=0)
    parser.add_argument("--scene-list", type=Path, default=None,
                        help="Optional JSON list of scene keys to materialize.")
    parser.add_argument("--target-width", type=int, default=490)
    parser.add_argument("--target-height", type=int, default=280)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        result = validate_prepared_root(
            args.output_root, args.input_root, args.context_count,
            args.target_stride, args.num_scenes
        )
    else:
        scene_keys = None
        if args.scene_list is not None:
            value = json.loads(args.scene_list.read_text())
            if isinstance(value, dict):
                value = value.get("scene_keys", [])
            scene_keys = {str(item) for item in value}
        result = prepare(args.input_root, args.output_root, args.context_count,
                         args.target_stride, args.num_scenes, args.min_frame_count,
                         scene_keys, args.target_width, args.target_height)
    print(f"prepared {result['scene_count']} RE10K scenes: "
          f"{result['context_count']} context + {len(result['tgt_list'])} target views")


if __name__ == "__main__":
    main()
