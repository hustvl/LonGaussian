#!/usr/bin/env python3
"""Materialize a small, resumable RE10K official-train subset.

Each source video is downloaded only for the timestamp range needed by a
centered 146-frame window.  The selected frames are center-cropped/resized to
490x280, camera intrinsics are transformed with the pixels, and the temporary
video is removed immediately after each scene.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def atomic_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_camera_window(path: Path, frame_count: int) -> tuple[str, list[list[float]]]:
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    if not lines:
        raise ValueError("empty metadata")
    rows = [[float(value) for value in line.split()] for line in lines[1:] if line.strip()]
    if len(rows) < frame_count:
        raise ValueError(f"only {len(rows)} camera rows")
    start = (len(rows) - frame_count) // 2
    rows = rows[start : start + frame_count]
    if any(len(row) != 19 or not all(math.isfinite(x) for x in row) for row in rows):
        raise ValueError("invalid camera row")
    timestamps = [int(row[0]) for row in rows]
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("camera timestamps are not strictly increasing")
    intrinsics = np.asarray([row[1:5] for row in rows], dtype=np.float64)
    if np.any(intrinsics[:, :2] <= 0):
        raise ValueError("non-positive focal length")
    if not np.allclose(intrinsics, intrinsics[:1], rtol=1e-4, atol=1e-6):
        raise ValueError("varying intrinsics are unsupported")
    return lines[0].strip(), rows


def nearest_unique_indices(frame_times_ms: list[float], target_times_ms: list[float],
                           tolerance_ms: float) -> list[int]:
    times = np.asarray(frame_times_ms, dtype=np.float64)
    targets = np.asarray(target_times_ms, dtype=np.float64)
    if times.ndim != 1 or not len(times):
        raise ValueError("no decoded frame timestamps")
    indices = np.searchsorted(times, targets)
    indices = np.clip(indices, 0, len(times) - 1)
    left = np.clip(indices - 1, 0, len(times) - 1)
    choose_left = np.abs(times[left] - targets) <= np.abs(times[indices] - targets)
    indices = np.where(choose_left, left, indices).astype(int)
    errors = np.abs(times[indices] - targets)
    if float(errors.max()) > tolerance_ms:
        raise ValueError(f"timestamp mismatch {float(errors.max()):.1f} ms")
    if any(b <= a for a, b in zip(indices.tolist(), indices.tolist()[1:])):
        raise ValueError("camera timestamps map to duplicate/non-monotonic video frames")
    return indices.tolist()


def crop_resize(image_bgr: np.ndarray, width: int, height: int) -> tuple[Image.Image, tuple]:
    source_h, source_w = image_bgr.shape[:2]
    target_ratio = width / height
    source_ratio = source_w / source_h
    if source_ratio > target_ratio:
        crop_h, crop_w = source_h, int(round(source_h * target_ratio))
        left, top = (source_w - crop_w) // 2, 0
    else:
        crop_w, crop_h = source_w, int(round(source_w / target_ratio))
        left, top = 0, (source_h - crop_h) // 2
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).crop((left, top, left + crop_w, top + crop_h))
    pil = pil.resize((width, height), Image.Resampling.LANCZOS)
    return pil, (source_w, source_h, left, top, width / crop_w, height / crop_h)


def download_segment(url: str, output_template: Path, start_sec: float, end_sec: float,
                     tool_root: Path, ffmpeg_dir: Path, deno: Path,
                     timeout_sec: float) -> Path:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tool_root) + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        sys.executable, "-m", "yt_dlp", "--no-playlist", "--no-progress",
        "--js-runtimes", f"deno:{deno}", "--remote-components", "ejs:github",
        "--socket-timeout", "20", "--retries", "2", "--fragment-retries", "2",
        "--ffmpeg-location", str(ffmpeg_dir),
        "--download-sections", f"*{start_sec:.6f}-{end_sec:.6f}",
        "--force-keyframes-at-cuts",
        "-f", "bv[height=360][vcodec^=avc1][protocol=https]/bv[height>=360][height<=480][vcodec^=avc1][protocol=https]",
        "--merge-output-format", "mp4", "-o", str(output_template), url,
    ]
    completed = subprocess.run(
        command, env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=timeout_sec,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout[-3000:])
    videos = sorted(output_template.parent.glob("source.*"))
    if len(videos) != 1 or videos[0].stat().st_size == 0:
        raise RuntimeError(f"expected one downloaded video, got {videos}")
    return videos[0]


def decode_segment(video: Path, expected_duration_sec: float) -> tuple[list[np.ndarray], list[float]]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open downloaded segment")
    frames, times = [], []
    max_frames = max(int(math.ceil(expected_duration_sec * 120)) + 120, 500)
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        times.append(float(cap.get(cv2.CAP_PROP_POS_MSEC)))
    cap.release()
    if len(frames) < 2 or any(b <= a for a, b in zip(times, times[1:])):
        raise RuntimeError(f"invalid decoded segment timestamps ({len(frames)} frames)")
    return frames, times


def materialize_scene(row: dict, output_root: Path, temp_root: Path, *, frame_count: int,
                      target_width: int, target_height: int, tool_root: Path,
                      ffmpeg_dir: Path, deno: Path, tolerance_ms: float,
                      download_timeout_sec: float) -> dict:
    key = str(row["key"])
    if Path(key).name != key or key in {".", ".."} or "/" in key or "\\" in key:
        raise ValueError("Unsafe scene identity")
    url, cameras = load_camera_window(Path(row["metadata_path"]), frame_count)
    timestamps_sec = np.asarray([camera[0] for camera in cameras], dtype=np.float64) / 1e6
    pad = 0.75
    segment_start = max(0.0, float(timestamps_sec[0]) - pad)
    segment_end = float(timestamps_sec[-1]) + pad
    scene_tmp = temp_root / key
    if scene_tmp.exists():
        shutil.rmtree(scene_tmp)
    scene_tmp.mkdir(parents=True)
    try:
        video = download_segment(
            url, scene_tmp / "source.%(ext)s", segment_start, segment_end,
            tool_root, ffmpeg_dir, deno, download_timeout_sec,
        )
        decoded, decoded_times = decode_segment(video, segment_end - segment_start)
        relative_targets_ms = ((timestamps_sec - segment_start) * 1000).tolist()
        selected = nearest_unique_indices(decoded_times, relative_targets_ms, tolerance_ms)

        image_dir = scene_tmp / "images_8"
        image_dir.mkdir()
        frames_json = []
        transformed_k = None
        for index, (camera, decoded_index) in enumerate(zip(cameras, selected)):
            image, geom = crop_resize(decoded[decoded_index], target_width, target_height)
            source_w, source_h, left, top, sx, sy = geom
            image.save(image_dir / f"{index:06d}.jpg", format="JPEG", quality=95)
            fx, fy, cx, cy = camera[1:5]
            k = {
                "fl_x": fx * source_w * sx,
                "fl_y": fy * source_h * sy,
                "cx": (cx * source_w - left) * sx,
                "cy": (cy * source_h - top) * sy,
            }
            if transformed_k is None:
                transformed_k = k
            elif any(not math.isclose(k[name], transformed_k[name], rel_tol=1e-4, abs_tol=1e-3)
                     for name in k):
                raise ValueError("transformed intrinsics vary across frames")
            w2c = np.eye(4, dtype=np.float64)
            w2c[:3] = np.asarray(camera[7:19], dtype=np.float64).reshape(3, 4)
            c2w = np.linalg.inv(w2c)
            frames_json.append({
                "file_path": f"images/{index:06d}.jpg",
                "transform_matrix": c2w.tolist(),
                "timestamp": int(camera[0]),
            })
        assert transformed_k is not None
        metadata = {
            **transformed_k, "w": target_width, "h": target_height,
            "camera_model": "PINHOLE", "frames": frames_json,
            "re10k_key": key, "re10k_url": url,
            "re10k_split": row.get("split", "official_train"),
            "re10k_camera_convention": "normalized intrinsics + w2c converted to pixel intrinsics + c2w",
            "preprocessing": "center_crop_then_resize",
            "source_segment_sec": [segment_start, segment_end],
        }
        atomic_json(scene_tmp / "transforms.json", metadata)
        targets = list(range(7, frame_count, 8))
        target_set = set(targets)
        context = [i for i in range(frame_count) if i not in target_set]
        atomic_json(scene_tmp / "split.json", {
            "ctx_list": context, "tgt_list": targets,
            "sampling": {"mode": "contiguous_every_8_holdout", "target_stride": 8},
        })
        video.unlink()
        destination = output_root / key
        if destination.exists():
            raise FileExistsError(destination)
        os.replace(scene_tmp, destination)
        return {
            "key": key, "url": url, "frames": frame_count,
            "metadata_path": row["metadata_path"],
            "source_video_size": [source_w, source_h],
            "max_timestamp_error_ms": max(
                abs(decoded_times[i] - t) for i, t in zip(selected, relative_targets_ms)
            ),
        }
    finally:
        if scene_tmp.exists():
            shutil.rmtree(scene_tmp)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-scenes", type=int, default=150)
    parser.add_argument("--frame-count", type=int, default=146)
    parser.add_argument("--target-width", type=int, default=490)
    parser.add_argument("--target-height", type=int, default=280)
    parser.add_argument("--timestamp-tolerance-ms", type=float, default=60.0)
    parser.add_argument("--download-timeout-sec", type=float, default=180.0)
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parser.add_argument("--limit-attempts", type=int, default=0)
    parser.add_argument(
        "--retry-failed", action="store_true",
        help="Retry keys recorded in a previous failed run instead of skipping them.",
    )
    parser.add_argument("--tool-root", type=Path, default=Path(".tools/yt_dlp"))
    parser.add_argument("--ffmpeg-dir", type=Path, default=Path(shutil.which("ffmpeg") or "ffmpeg").parent)
    parser.add_argument("--deno", type=Path, default=Path(shutil.which("deno") or "deno"))
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    split = selection.get("split")
    if split not in {"official_train", "official_test"}:
        raise ValueError("Selection must identify the official train or test split")
    queue = selection["primary"] + selection["reserve"]
    keys = [row["key"] for row in queue]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate selected scene IDs")
    if args.target_scenes < 1 or args.target_scenes > len(queue):
        raise ValueError("Target count exceeds the selected candidate pool")
    if not args.ffmpeg_dir.is_dir() or not args.deno.is_file():
        raise FileNotFoundError("Install ffmpeg and deno, or supply --ffmpeg-dir and --deno")
    for row in queue:
        row["split"] = split
    args.output_root.mkdir(parents=True, exist_ok=True)
    temp_root = args.output_root / ".tmp"
    temp_root.mkdir(exist_ok=True)
    state_path = args.output_root / "download_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "format": "loger_re10k_train_download_state_v1", "completed": [], "failures": []
    }
    signature = hashlib.sha256(json.dumps({"selection": selection, "frames": args.frame_count,
        "width": args.target_width, "height": args.target_height, "target": args.target_scenes,
        "tolerance_ms": args.timestamp_tolerance_ms}, sort_keys=True).encode()).hexdigest()
    if state_path.exists() and state.get("signature") != signature:
        raise ValueError("Download settings/selection changed; use a fresh output root")
    state["signature"] = signature
    completed = {row["key"]: row for row in state["completed"]}
    for key in completed:
        if key not in keys or not (args.output_root / key / "transforms.json").is_file():
            raise ValueError(f"Invalid completed download record: {key}")
    failed_keys = (
        set()
        if args.retry_failed
        else {row["key"] for row in state["failures"]} - set(completed)
    )
    attempted = 0
    for row in queue:
        if len(completed) >= args.target_scenes:
            break
        key = row["key"]
        if key in completed or key in failed_keys:
            continue
        if args.limit_attempts and attempted >= args.limit_attempts:
            break
        attempted += 1
        free_gb = shutil.disk_usage(args.output_root).free / (1024 ** 3)
        if free_gb < args.min_free_gb:
            raise RuntimeError(f"disk safety gate: only {free_gb:.1f} GiB free")
        print(f"[attempt {attempted}] {key} completed={len(completed)}/{args.target_scenes}", flush=True)
        try:
            result = materialize_scene(
                row, args.output_root, temp_root, frame_count=args.frame_count,
                target_width=args.target_width, target_height=args.target_height,
                tool_root=args.tool_root.resolve(), ffmpeg_dir=args.ffmpeg_dir.resolve(),
                deno=args.deno.resolve(),
                tolerance_ms=args.timestamp_tolerance_ms,
                download_timeout_sec=args.download_timeout_sec,
            )
        except Exception as exc:
            failure = {"key": key, "error": f"{type(exc).__name__}: {exc}"[-4000:]}
            state["failures"].append(failure)
            failed_keys.add(key)
            print(f"[failed] {key}: {failure['error'][-500:]}", flush=True)
        else:
            completed[key] = result
            state["completed"] = list(completed.values())
            atomic_json(args.output_root / "train_index.json", list(completed) if split == "official_train" else [])
            atomic_json(args.output_root / "test_index.json", list(completed) if split == "official_test" else [])
            print(f"[done] {key} completed={len(completed)}/{args.target_scenes}", flush=True)
        atomic_json(state_path, state)

    state["completed"] = list(completed.values())
    state["target_scenes"] = args.target_scenes
    state["complete"] = len(completed) >= args.target_scenes
    atomic_json(state_path, state)
    atomic_json(args.output_root / "manifest.json", {
        "format": "loger_re10k_train_long_v1", "split": split,
        "selection": str(args.selection.resolve()), "target_scene_count": args.target_scenes,
        "scene_count": len(completed), "frame_count": args.frame_count,
        "target_width": args.target_width, "target_height": args.target_height,
        "scene_keys": list(completed), "complete": state["complete"],
    })
    print(f"finished: {len(completed)}/{args.target_scenes}", flush=True)
    if not state["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
