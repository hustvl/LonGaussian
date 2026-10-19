#!/usr/bin/env python3
"""Formal reconstruction evaluation with robust foreign-only depth filtering.

Writes metrics_dl3dv_subset.json compatible with workspace/scripts/dl3dv_eval_metrics.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dl3dv_eval_metrics import (  # noqa: E402
    SceneTimer,
    get_peak_vram_gb,
    print_summary,
    summarize_scenes,
    write_metrics_json,
)


def _neighbor_window_ids(owner: int, num_windows: int, max_windows: int) -> List[int]:
    """Owner-centered contiguous window ids, clipped to ``[0, num_windows)``."""
    n = int(num_windows)
    k = int(max_windows)
    if n <= 0:
        return []
    if k <= 0 or k >= n:
        return list(range(n))
    half = k // 2
    start = max(0, int(owner) - half)
    end = min(n, start + k)
    start = max(0, end - k)
    return list(range(start, end))


def _concat_per_window_clouds(
    windows: Sequence[object],
    camera_poses: torch.Tensor,
    *,
    device: torch.device,
) -> object:
    """Concatenate per-window hybrid clouds; use shared (aligned) camera poses."""
    from loger.merge.window_merge import WindowMergeTensors, merge_result_from_tensors

    if not windows:
        raise RuntimeError("cannot concat empty per-window cloud list")
    means = torch.cat([w.means.to(device) for w in windows], dim=0)
    quats = torch.cat([w.quats.to(device) for w in windows], dim=0)
    scales = torch.cat([w.scales.to(device) for w in windows], dim=0)
    opacities = torch.cat([w.opacities.to(device) for w in windows], dim=0)
    sh_flat = torch.cat([w.sh_flat.to(device) for w in windows], dim=0)
    poses = camera_poses
    if poses.device != device:
        poses = poses.to(device)
    return merge_result_from_tensors(
        WindowMergeTensors(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            sh_flat=sh_flat,
        ),
        poses,
        stats={
            "render_source": "per_window_concat",
            "num_windows_concat": len(windows),
            "num_gaussians": int(means.shape[0]),
            "window_ids": [
                int(w.stats.get("window_idx", -1)) for w in windows if getattr(w, "stats", None)
            ],
        },
    )


def _cpu_f32(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(device="cpu", dtype=torch.float32).contiguous()


def _save_one_window_cloud(path: Path, cloud) -> Dict[str, object]:
    """Persist one per-window Gaussian cloud for later reload/render."""
    payload = {
        "means": _cpu_f32(cloud.means),
        "quats": _cpu_f32(cloud.quats),
        "scales": _cpu_f32(cloud.scales),
        "opacities": _cpu_f32(cloud.opacities),
        "sh_flat": _cpu_f32(cloud.sh_flat),
        "stats": dict(getattr(cloud, "stats", None) or {}),
        "num_gaussians": int(cloud.means.shape[0]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {
        "path": str(path),
        "num_gaussians": int(payload["num_gaussians"]),
        "stats": payload["stats"],
    }


def save_window_cloud_bundle(
    output_dir: Path,
    scene_name: str,
    merged,
    *,
    K: torch.Tensor,
    ctx_list: Sequence[int],
    tgt_list: Sequence[int],
    args,
    poses_pass1: Optional[torch.Tensor] = None,
    poses_pass2_ctx: Optional[torch.Tensor] = None,
) -> Dict[str, object]:
    """Save aligned camera trajectory + per-window compact/complement clouds.

    Layout:
      ``<output_dir>/window_clouds/<scene>/camera_trajectory.pt``
      ``<output_dir>/window_clouds/<scene>/windows/wXXX_{compact,complement}.pt``
      ``<output_dir>/window_clouds/<scene>/manifest.json``
    """
    scene_dir = Path(output_dir) / "window_clouds" / scene_name
    win_dir = scene_dir / "windows"
    win_dir.mkdir(parents=True, exist_ok=True)

    poses = merged.camera_poses.detach().cpu().float().contiguous()
    traj = {
        "schema": "loger_camera_trajectory_v1",
        "camera_poses": poses,
        "K": K.detach().cpu().float().contiguous() if torch.is_tensor(K) else K,
        "ctx_list": [int(i) for i in ctx_list],
        "tgt_list": [int(i) for i in tgt_list],
        "num_frames": int(args.num_frames),
        "window_size": int(args.window_size),
        "overlap_size": int(args.overlap_size),
        "two_pass_pose": bool(args.two_pass_pose),
        "orientation_aware_two_pass": bool(args.orientation_aware_two_pass),
        "calibrate_local_points_to_k": bool(args.calibrate_local_points_to_k),
        "merge_mode": str(args.merge_mode),
        "merge_stats": dict(getattr(merged, "stats", None) or {}),
        "poses_pass1": (
            poses_pass1.detach().cpu().float().contiguous()
            if poses_pass1 is not None
            else None
        ),
        # Exact context-only Pass-2 poses from the same forward/merge run that
        # produced the saved per-window clouds.  Keep these before
        # align_two_pass_poses() replaces merged.camera_poses with the aligned
        # full Pass-1 trajectory; owner-window reconstruction must use this
        # field rather than a separately re-run Pass-2 trajectory.
        "camera_poses_pass2_ctx": (
            poses_pass2_ctx.detach().cpu().float().contiguous()
            if poses_pass2_ctx is not None
            else None
        ),
    }
    traj_path = scene_dir / "camera_trajectory.pt"
    torch.save(traj, traj_path)

    dump_meta = list((merged.stats or {}).get("window_dump_meta") or [])
    if dump_meta:
        windows_meta = dump_meta
        total_compact = int(sum(int(w["compact"]["num_gaussians"]) for w in windows_meta))
        total_complement = int(
            sum(int(w["complement"]["num_gaussians"]) for w in windows_meta)
        )
    else:
        compact_by = {
            int(w.stats["window_idx"]): w
            for w in (merged.per_window_compact or [])
            if getattr(w, "stats", None) and "window_idx" in w.stats
        }
        complement_by = {
            int(w.stats["window_idx"]): w
            for w in (merged.per_window_complement or [])
            if getattr(w, "stats", None) and "window_idx" in w.stats
        }
        if not compact_by and not complement_by:
            raise RuntimeError(
                "save_window_clouds requires per_window_compact/complement "
                "or streamed window_dump_meta"
            )

        window_ids = sorted(set(compact_by) | set(complement_by))
        windows_meta = []
        total_compact = 0
        total_complement = 0
        for wid in window_ids:
            entry: Dict[str, object] = {"window_idx": int(wid)}
            if wid in compact_by:
                cpath = win_dir / f"w{wid:03d}_compact.pt"
                info = _save_one_window_cloud(cpath, compact_by[wid])
                entry["compact"] = info
                total_compact += int(info["num_gaussians"])
                st = info.get("stats") or {}
                for key in ("start_idx", "end_idx", "contrib_start", "contrib_end"):
                    if key in st:
                        entry[key] = int(st[key])
            if wid in complement_by:
                cpath = win_dir / f"w{wid:03d}_complement.pt"
                info = _save_one_window_cloud(cpath, complement_by[wid])
                entry["complement"] = info
                total_complement += int(info["num_gaussians"])
                st = info.get("stats") or {}
                for key in ("start_idx", "end_idx", "contrib_start", "contrib_end"):
                    if key in st and key not in entry:
                        entry[key] = int(st[key])
            windows_meta.append(entry)

    manifest = {
        "schema": "loger_window_clouds_v1",
        "scene": scene_name,
        "camera_trajectory": str(traj_path),
        "num_windows": len(windows_meta),
        "total_compact_gaussians": int(total_compact),
        "total_complement_gaussians": int(total_complement),
        "windows": windows_meta,
    }
    man_path = scene_dir / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest

# OOM localization: last successful stage / window written to jsonl + stderr.
_OOM_DIAG_PATH: Optional[Path] = None
_OOM_DIAG_LAST: Dict[str, object] = {}


def _vram_gb() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    free_b, total_b = torch.cuda.mem_get_info()
    alloc = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    peak = torch.cuda.max_memory_allocated()
    return {
        "alloc_gb": alloc / (1024**3),
        "reserved_gb": reserved / (1024**3),
        "peak_alloc_gb": peak / (1024**3),
        "free_gb": free_b / (1024**3),
        "total_gb": total_b / (1024**3),
    }


def oom_diag(stage: str, **extra) -> None:
    """Record a diagnostic checkpoint (enabled when --oom_diag / LOGER_OOM_DIAG)."""
    if _OOM_DIAG_PATH is None and not os.environ.get("LOGER_OOM_DIAG"):
        return
    rec = {
        "ts": time.time(),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stage": stage,
        "vram": _vram_gb(),
        **extra,
    }
    _OOM_DIAG_LAST.clear()
    _OOM_DIAG_LAST.update(rec)
    line = json.dumps(rec, ensure_ascii=False)
    print(f"[oom_diag] {line}", flush=True)
    path = _OOM_DIAG_PATH
    if path is None and os.environ.get("LOGER_OOM_DIAG"):
        path = Path(os.environ["LOGER_OOM_DIAG"])
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())


from scripts.evaluation_utils import (  # noqa: E402
    align_two_pass_poses,
    compute_view_metrics,
    configure_reproducibility,
    ctx_merge_owner_groups,
    load_scene_list,
    packed_merge_owner_groups,
    prepare_merge_inputs,
)


def parse_filter_variant(name: str) -> Dict[str, object]:
    """Parse ``m01_v1_r30`` into the released depth-filter thresholds."""
    parts = name.split("_")
    if len(parts) != 3 or not all(parts[i].startswith(p) for i, p in enumerate(("m", "v", "r"))):
        raise ValueError(f"malformed depth-filter variant: {name}")
    return {
        "name": name,
        "rel_margin": float(parts[0][1:]) / 100.0,
        "min_votes": int(parts[1][1:]),
        "ratio_threshold": float(parts[2][1:]) / 100.0,
    }
from scripts.reconstruction_pipeline import (  # noqa: E402
    build_merge_config,
    forward_track_merge,
    load_gaussian_checkpoint,
    load_split,
)


def eval_scene(
    model,
    renderer,
    scene_path: Path,
    args,
    merge_config,
    spec: Dict[str, object],
    lpips_fn,
) -> dict:
    from loger.merge.robust_foreign_depth_filter import robust_foreign_depth_filter
    from run import load_dl3dv_scene, model_sliding_window_kwargs

    if getattr(args, "inference_scene", None):
        ctx_list, tgt_list = list(range(args.num_frames)), []
    else:
        ctx_list, tgt_list = load_split(scene_path)
    oom_diag(
        "load_split",
        ctx=len(ctx_list),
        tgt=len(tgt_list),
        num_frames=args.num_frames,
        window_size=args.window_size,
        overlap_size=args.overlap_size,
    )
    loaded = load_dl3dv_scene(str(scene_path), args.num_frames, image_scale=args.image_scale)
    if loaded is None:
        raise RuntimeError(f"failed to load {scene_path}")
    images, _, K = loaded
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # Long sequences: keep the full RGB tensor on CPU and let Pi3 move each
    # sliding-window slice to GPU (see imgs_w.to(self.image_mean.device)).
    images_on_cpu = bool(getattr(args, "images_on_cpu", False)) or (
        int(args.num_frames) >= 1000
    )
    if images_on_cpu:
        images = images.cpu().contiguous()
        # Keep K on CPU too so windowed intrinsics slices match RGB until Pi3
        # moves both onto the model device.
        K = K.cpu().contiguous()
        oom_diag(
            "images_on_cpu",
            n=int(images.shape[0]),
            H=int(images.shape[-2]),
            W=int(images.shape[-1]),
            shape=list(images.shape),
            bytes_gb=float(images.numel() * images.element_size()) / (1024**3),
        )
    else:
        images = images.to(device)
        K = K.to(device)
        oom_diag(
            "images_on_device",
            n=int(images.shape[0]),
            H=int(images.shape[-2]),
            W=int(images.shape[-1]),
            shape=list(images.shape),
        )
    H, W = images.shape[-2:]
    win = min(11, H, W)
    if win % 2 == 0:
        win -= 1
    win = max(win, 3)

    sw_kwargs = model_sliding_window_kwargs(
        args.num_frames,
        window_size=args.window_size,
        overlap_size=args.overlap_size,
        sim3=args.sim3,
        se3=not args.sim3,
        turn_off_ttt=args.no_ttt,
        turn_off_swa=args.no_swa,
    )
    if args.gaussian_head_batch_size > 0:
        sw_kwargs["gaussian_head_batch_size"] = args.gaussian_head_batch_size

    merge_images, merge_ctx, ctx_isolated = prepare_merge_inputs(
        images, ctx_list, args.two_pass_pose
    )
    oom_diag(
        "prepare_merge_inputs",
        merge_n=int(merge_images.shape[0]),
        merge_ctx=len(merge_ctx),
        ctx_isolated=bool(ctx_isolated),
    )

    with torch.no_grad():
        poses_pass1 = None
        pred_all = None
        if args.two_pass_pose and not args.save_pass2_trajectory_only:
            oom_diag("pass1_start", track_merge=False, frames=int(images.shape[0]))
            # Long-seq export: skip Gaussian head on pass-1 (poses + points only).
            poses_only = bool(
                getattr(args, "save_window_clouds_only", False)
                or int(args.num_frames) >= 1000
            )
            pred_all = model(
                images[None],
                gaussian=True,
                track_merge=False,
                poses_only=poses_only,
                intrinsics=K,
                calibrate_local_points_to_k=args.calibrate_local_points_to_k,
                **sw_kwargs,
            )
            # Clone the small pose tensor so the much larger pass-1 prediction
            # dictionary can be released by the export-only path.
            poses_pass1 = pred_all["camera_poses"][0].to(device).clone()
            oom_diag(
                "pass1_done",
                poses=list(poses_pass1.shape),
                pred_keys=sorted(list(pred_all.keys()))[:20],
            )
            # Drop bulky pass-1 predictions unless sky needs predicted SH.
            need_pred_all = bool(
                args.sky_sphere and args.sky_sphere_color_mode != "rgb_dc"
            )
            if args.save_gaussian_sample_only or not need_pred_all:
                del pred_all
                pred_all = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                oom_diag("pass1_pred_all_released", need_pred_all=need_pred_all)

        oom_diag("track_merge_start", merge_n=int(merge_images.shape[0]))
        if args.save_pass2_trajectory_only:
            # Run the context-only Pass-2 merger in streaming/export mode, but
            # discard each window cloud instead of writing another ~30 GB dump.
            # The assembled camera poses returned by finalize() are in exactly
            # the same world frame as the already-exported per-window clouds.
            merge_config.window_cloud_dump_dir = None
            merge_config.window_cloud_export_only = True
        if args.save_window_clouds or args.save_window_clouds_only:
            dump_root = (
                Path(args.output_dir) / "window_clouds" / scene_path.name / "windows"
            )
            dump_root.mkdir(parents=True, exist_ok=True)
            merge_config.window_cloud_dump_dir = str(dump_root)
            merge_config.window_cloud_export_only = bool(args.save_window_clouds_only)
            oom_diag(
                "window_cloud_stream_dump",
                dump_dir=str(dump_root),
                export_only=bool(merge_config.window_cloud_export_only),
            )
        merged = forward_track_merge(
            model, merge_images, merge_ctx, merge_config, args, intrinsics=K
        )
        oom_diag(
            "track_merge_done",
            num_gaussians=int(getattr(merged, "num_gaussians", -1)),
            num_windows=int((merged.stats or {}).get("num_windows", -1))
            if getattr(merged, "stats", None)
            else -1,
            per_window=len(list(merged.per_window_results or [])),
        )
        poses_pass2_ctx_same_run = (
            merged.camera_poses.detach().cpu().float().contiguous().clone()
            if args.two_pass_pose
            else None
        )
        if args.save_pass2_trajectory_only:
            trajectory_dir = Path(args.output_dir) / "pass2_trajectory"
            trajectory_dir.mkdir(parents=True, exist_ok=True)
            trajectory_path = trajectory_dir / f"{scene_path.name}.pt"
            trajectory_payload = {
                "schema": "loger_pass2_context_trajectory_v1",
                "scene": scene_path.name,
                "camera_poses": merged.camera_poses.detach().cpu().float().contiguous(),
                "K": K.detach().cpu().float().contiguous(),
                "ctx_list": [int(i) for i in ctx_list],
                "tgt_list": [int(i) for i in tgt_list],
                "num_frames": int(args.num_frames),
                "window_size": int(args.window_size),
                "overlap_size": int(args.overlap_size),
                "context_input_isolation": bool(ctx_isolated),
                "calibrate_local_points_to_k": bool(args.calibrate_local_points_to_k),
                "merge_mode": str(args.merge_mode),
                "merge_stats": dict(getattr(merged, "stats", None) or {}),
            }
            torch.save(trajectory_payload, trajectory_path)
            oom_diag("save_pass2_trajectory_done", path=str(trajectory_path))
            return {
                "scene": scene_path.name,
                "scene_path": str(scene_path),
                "ctx_list": ctx_list,
                "tgt_list": tgt_list,
                "views": [],
                "psnr_mean": float("nan"),
                "ssim_mean": float("nan"),
                "lpips_mean": float("nan"),
                "num_gaussians": 0,
                "pass2_trajectory": str(trajectory_path),
                "merge_stats": dict(getattr(merged, "stats", None) or {}),
            }
        if args.two_pass_pose:
            if poses_pass1 is None:
                raise RuntimeError("two_pass_pose enabled but pass-1 poses missing")
            oom_diag("align_two_pass_start")
            align_two_pass_poses(
                merged,
                poses_pass1,
                ctx_list,
                pass2_ctx_isolated=ctx_isolated,
                orientation_aware=args.orientation_aware_two_pass,
            )
            oom_diag("align_two_pass_done")

        window_cloud_manifest = None
        if args.save_window_clouds or args.save_window_clouds_only:
            oom_diag(
                "save_window_clouds_start",
                compact=len(list(merged.per_window_compact or [])),
                complement=len(list(merged.per_window_complement or [])),
            )
            window_cloud_manifest = save_window_cloud_bundle(
                Path(args.output_dir),
                scene_path.name,
                merged,
                K=K,
                ctx_list=ctx_list,
                tgt_list=tgt_list,
                args=args,
                poses_pass1=poses_pass1,
                poses_pass2_ctx=poses_pass2_ctx_same_run,
            )
            oom_diag(
                "save_window_clouds_done",
                num_windows=int(window_cloud_manifest.get("num_windows", -1)),
                total_compact=int(
                    window_cloud_manifest.get("total_compact_gaussians", -1)
                ),
                total_complement=int(
                    window_cloud_manifest.get("total_complement_gaussians", -1)
                ),
            )
            if args.save_window_clouds_only:
                return {
                    "scene": scene_path.name,
                    "scene_path": str(scene_path),
                    "ctx_list": ctx_list,
                    "tgt_list": tgt_list,
                    "views": [],
                    "psnr_mean": float("nan"),
                    "ssim_mean": float("nan"),
                    "lpips_mean": float("nan"),
                    "num_gaussians": int(getattr(merged, "num_gaussians", 0)),
                    "window_cloud_manifest": window_cloud_manifest,
                    "merge_stats": dict(getattr(merged, "stats", None) or {}),
                }

        if args.save_gaussian_sample_only:
            # The export-only path returns before sky construction and target
            # rendering, so these image/pose tensors are no longer needed.
            del merge_images
            if poses_pass1 is not None:
                del poses_pass1
            del images
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        filter_stats = None
        filtered = merged
        per_window = list(merged.per_window_results or [])
        render_max_windows = int(getattr(args, "render_max_windows", 0) or 0)
        if render_max_windows > 0 and not per_window:
            raise RuntimeError("render_max_windows requires per_window_results")
        if not args.disable_depth_filter:
            if not per_window:
                raise RuntimeError("missing per_window_results")
            if ctx_isolated:
                owner_groups = ctx_merge_owner_groups(
                    ctx_list, args.window_size, args.overlap_size
                )
                render_map = {int(g): i for i, g in enumerate(ctx_list)}
            else:
                owner_groups = packed_merge_owner_groups(
                    ctx_list, args.num_frames, args.window_size, args.overlap_size
                )
                render_map = None

            oom_diag(
                "depth_filter_start",
                per_window=len(per_window),
                owner_groups=len(owner_groups),
                num_gaussians=int(getattr(merged, "num_gaussians", -1)),
                stream_hybrid=bool(getattr(args, "max_gpu_merge_windows", 0) or 0),
            )
            depth_filter_stream = int(getattr(args, "max_gpu_merge_windows", 0) or 0) > 0
            filtered, filter_stats = robust_foreign_depth_filter(
                merged, per_window, renderer, owner_groups, K, H, W,
                rel_margin=float(spec["rel_margin"]),
                min_votes=int(spec["min_votes"]),
                visible_ratio_threshold=float(args.visible_ratio_threshold),
                decisive_ratio_threshold=float(spec["ratio_threshold"]),
                min_rel_gap=float(args.min_rel_gap),
                alpha_threshold=float(args.alpha_threshold),
                neighborhood_radius=int(args.neighborhood_radius),
                min_valid_neighbors=int(args.min_valid_neighbors),
                max_rel_depth_spread=float(args.max_rel_depth_spread),
                render_chunk=int(args.render_chunk), allow_empty=True,
                collect_diag=False, render_index_map=render_map,
                stream_hybrid=depth_filter_stream,
                gaussian_chunk_size=int(getattr(args, "depth_filter_gaussian_chunk_size", 500_000)),
                max_gpu_windows=int(
                    getattr(args, "depth_filter_max_gpu_windows", 0)
                    or getattr(args, "max_gpu_merge_windows", 0)
                    or 1
                ),
            )
            oom_diag(
                "depth_filter_done",
                num_gaussians=int(getattr(filtered, "num_gaussians", -1)),
                filter_keys=sorted(list((filter_stats or {}).keys()))[:15],
                stream_hybrid=depth_filter_stream,
            )
            # Filtered cloud may be on CPU after streaming robust depth filtering; move for rasterization.
            if (
                torch.cuda.is_available()
                and getattr(filtered, "means", None) is not None
                and filtered.means.device.type == "cpu"
            ):
                from loger.merge.window_merge import (
                    WindowMergeTensors,
                    merge_result_from_tensors,
                    window_merge_tensors_to_device,
                )

                oom_diag(
                    "depth_filter_to_cuda_start",
                    num_gaussians=int(filtered.num_gaussians),
                )
                cuda = torch.device(args.device if torch.cuda.is_available() else "cpu")
                moved = window_merge_tensors_to_device(
                    WindowMergeTensors(
                        means=filtered.means,
                        quats=filtered.quats,
                        scales=filtered.scales,
                        opacities=filtered.opacities,
                        sh_flat=filtered.sh_flat,
                    ),
                    cuda,
                )
                poses = filtered.camera_poses
                if poses.device != cuda:
                    poses = poses.to(cuda)
                filtered = merge_result_from_tensors(
                    moved, poses, stats=dict(filtered.stats or {})
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                oom_diag(
                    "depth_filter_to_cuda_done",
                    num_gaussians=int(filtered.num_gaussians),
                )

        # Keep the finite scene before the optional far-field sky shell so a
        # point-cloud visualization is not dominated by the shell radius.
        finite_scene = filtered

        if args.save_gaussian_sample:
            oom_diag("save_gaussian_sample_start", num_gaussians=int(finite_scene.num_gaussians))
            sample_dir = Path(args.output_dir) / "gaussian_samples"
            sample_dir.mkdir(parents=True, exist_ok=True)
            try:
                count = int(finite_scene.num_gaussians)
                sample_count = min(count, int(args.gaussian_sample_max_points))
                # Avoid float32 linspace index overflow for >2^24 Gaussians.
                if sample_count <= 1 or count <= 1:
                    indices = torch.zeros(sample_count, dtype=torch.long)
                else:
                    indices = (
                        torch.arange(sample_count, dtype=torch.int64)
                        * int(count - 1)
                        // int(sample_count - 1)
                    ).clamp(0, count - 1)
                means_src = finite_scene.means.detach()
                sh_src = finite_scene.sh_flat.detach()
                opac_src = finite_scene.opacities.detach()
                # Index on CPU to dodge CUDA assert after large kernels.
                idx_cpu = indices.cpu()
                means = means_src.cpu()[idx_cpu].float().numpy()
                # The flattened SH layout is channel-major:
                # [R0..R(K-1), G0..G(K-1), B0..B(K-1)].  Select the DC
                # coefficient of each colour channel rather than the first
                # three entries (which are R0, R1, and R2).
                sh_sample = sh_src.cpu()[idx_cpu].float()
                if sh_sample.shape[-1] % 3 != 0:
                    raise ValueError(
                        f"Expected 3*K flattened SH channels, got {sh_sample.shape[-1]}"
                    )
                sh_bases = sh_sample.shape[-1] // 3
                sh_dc = sh_sample[:, [0, sh_bases, 2 * sh_bases]].numpy()
                colors = np.clip(0.5 + 0.28209479177387814 * sh_dc, 0.0, 1.0)
                opacities = opac_src.cpu()[idx_cpu].float().numpy()
                camera_centers = (
                    finite_scene.camera_poses.reshape(-1, 4, 4)[:, :3, 3]
                    .detach().cpu().float().numpy()
                )
                np.savez_compressed(
                    sample_dir / f"{scene_path.name}.npz",
                    means=means,
                    colors=colors,
                    opacities=opacities,
                    camera_centers=camera_centers,
                    total_finite_gaussians=np.asarray(count, dtype=np.int64),
                )
                oom_diag("save_gaussian_sample_done", sample_count=sample_count)
            except Exception as exc:
                oom_diag("save_gaussian_sample_failed", error=str(exc))
                print(f"[warn] save_gaussian_sample failed: {exc}", flush=True)
            if args.save_gaussian_sample_only:
                return {
                    "scene": scene_path.name,
                    "scene_path": str(scene_path),
                    "views": [],
                    "psnr_mean": float("nan"),
                    "ssim_mean": float("nan"),
                    "lpips_mean": float("nan"),
                    "num_gaussians": int(getattr(finite_scene, "num_gaussians", 0)),
                }

        if args.sky_sphere:
            from loger.merge.sky_sphere_prototype import append_fixed_sky_sphere
            from loger.utils.sky_mask import SkyMasker

            filtered = append_fixed_sky_sphere(
                filtered, images, ctx_list, K,
                SkyMasker(onnx_path=args.sky_mask_onnx, score_thr=args.sky_mask_score_thr),
                sky_sh_coeffs=(
                    pred_all["gaussian_sh_coeffs"][0]
                    if pred_all is not None and args.sky_sphere_color_mode != "rgb_dc"
                    else None
                ),
                radius_factor=args.sky_sphere_radius_factor,
                sample_stride=args.sky_sphere_sample_stride,
                azimuth_bins=args.sky_sphere_azimuth_bins,
                elevation_bins=args.sky_sphere_elevation_bins,
                opacity=args.sky_sphere_opacity,
                scale_overlap=args.sky_sphere_scale_overlap,
                color_mode=args.sky_sphere_color_mode,
            )

        if getattr(args, "inference_scene", None):
            from loger.utils.ply_export import export_scene_ply
            destination = Path(args.output_dir) / "scene.ply"
            export_scene_ply(filtered, destination)
            return {"scene": scene_path.name, "num_gaussians": int(filtered.num_gaussians),
                    "merge_stats": dict(filtered.stats), "ply": destination.name,
                    "context_count": len(ctx_list), "depth_filter": filter_stats,
                    "sky_sphere": bool(args.sky_sphere)}

        views = []
        render_gs_counts: List[int] = []
        oom_diag(
            "render_start",
            tgt=len(tgt_list),
            render_chunk=int(args.render_chunk),
            render_max_windows=int(render_max_windows),
        )
        if render_max_windows > 0:
            from loger.merge.window_merge import (
                owner_window_for_frame,
                sliding_window_ranges,
            )

            win_ranges = sliding_window_ranges(
                int(args.num_frames), int(args.window_size), int(args.overlap_size)
            )
            by_idx = {
                int(w.stats["window_idx"]): w
                for w in per_window
                if getattr(w, "stats", None) and "window_idx" in w.stats
            }
            num_pw = max(by_idx.keys()) + 1 if by_idx else 0
            # Use two-pass-aligned poses from the global merge result.
            aligned_poses = filtered.camera_poses
            groups: Dict[int, List[int]] = defaultdict(list)
            for frame_idx in tgt_list:
                owner = owner_window_for_frame(
                    int(frame_idx), win_ranges, int(args.overlap_size)
                )
                groups[int(owner)].append(int(frame_idx))

            for owner in sorted(groups):
                frames = groups[owner]
                sel = _neighbor_window_ids(owner, num_pw, render_max_windows)
                local_windows = [by_idx[i] for i in sel if i in by_idx]
                if not local_windows:
                    raise RuntimeError(
                        f"no per-window clouds for owner={owner} sel={sel}"
                    )
                local = _concat_per_window_clouds(
                    local_windows,
                    aligned_poses,
                    device=torch.device(
                        args.device if torch.cuda.is_available() else "cpu"
                    ),
                )
                oom_diag(
                    "render_local_windows",
                    owner=int(owner),
                    windows=sel,
                    frames=list(frames),
                    num_gaussians=int(local.num_gaussians),
                )
                for start in range(0, len(frames), args.render_chunk):
                    chunk = frames[start : start + args.render_chunk]
                    out = renderer.render_hybrid(
                        local, target_indices=list(chunk), K=K, H=H, W=W
                    )
                    pred = out["rgb"].clamp(0, 1).permute(0, 3, 1, 2)
                    gt = images[chunk]
                    for offset, frame_idx in enumerate(chunk):
                        metrics = compute_view_metrics(
                            pred[offset], gt[offset], lpips_fn, win
                        )
                        views.append(
                            {
                                "target_index": int(frame_idx),
                                "owner_window": int(owner),
                                "render_windows": list(sel),
                                "num_gaussians_render": int(local.num_gaussians),
                                **metrics,
                            }
                        )
                        render_gs_counts.append(int(local.num_gaussians))
                        if args.save_images:
                            render_dir = (
                                Path(args.output_dir) / "renders" / scene_path.name
                            )
                            render_dir.mkdir(parents=True, exist_ok=True)
                            for image, suffix in (
                                (pred[offset], "pred"),
                                (gt[offset], "gt"),
                            ):
                                array = (
                                    image.detach().cpu().permute(1, 2, 0).numpy()
                                    * 255.0
                                ).round().clip(0, 255).astype(np.uint8)
                                Image.fromarray(array).save(
                                    render_dir
                                    / f"frame_{int(frame_idx):02d}_{suffix}.png"
                                )
                del local
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            for start in range(0, len(tgt_list), args.render_chunk):
                chunk = tgt_list[start : start + args.render_chunk]
                oom_diag("render_chunk", start=start, chunk=list(chunk))
                out = renderer.render_hybrid(
                    filtered, target_indices=list(chunk), K=K, H=H, W=W
                )
                pred = out["rgb"].clamp(0, 1).permute(0, 3, 1, 2)
                gt = images[chunk]
                for offset, frame_idx in enumerate(chunk):
                    metrics = compute_view_metrics(
                        pred[offset], gt[offset], lpips_fn, win
                    )
                    views.append({"target_index": int(frame_idx), **metrics})
                    render_gs_counts.append(int(filtered.num_gaussians))
                    if args.save_images:
                        render_dir = Path(args.output_dir) / "renders" / scene_path.name
                        render_dir.mkdir(parents=True, exist_ok=True)
                        for image, suffix in ((pred[offset], "pred"), (gt[offset], "gt")):
                            array = (
                                image.detach().cpu().permute(1, 2, 0).numpy() * 255.0
                            ).round().clip(0, 255).astype(np.uint8)
                            Image.fromarray(array).save(
                                render_dir / f"frame_{int(frame_idx):02d}_{suffix}.png"
                            )

    psnr_mean = float(np.mean([v["psnr"] for v in views]))
    ssim_mean = float(np.mean([v["ssim"] for v in views]))
    lpips_mean = float(np.mean([v["lpips"] for v in views]))
    report_gs = (
        int(round(float(np.mean(render_gs_counts))))
        if render_gs_counts
        else int(filtered.num_gaussians)
    )
    return {
        "scene": scene_path.name,
        "scene_path": str(scene_path),
        "ctx_list": ctx_list,
        "tgt_list": tgt_list,
        "num_gaussians": report_gs,
        "num_gaussians_global": int(filtered.num_gaussians),
        "render_max_windows": int(render_max_windows),
        "merge_stats": dict(filtered.stats),
        "depth_filter_stats": filter_stats,
        "two_pass": {
            "enabled": bool(args.two_pass_pose),
            "context_count": len(ctx_list),
            "target_count": len(tgt_list),
            "pose_alignment": (
                "SE3-SO3-context-orientations-plus-center-translation"
                if args.orientation_aware_two_pass
                else "SE3-Kabsch-on-context-camera-centers"
            ),
            "target_feature_leakage": False,
            "context_input_isolation": ctx_isolated,
        },
        "views": views,
        "psnr_mean": psnr_mean,
        "ssim_mean": ssim_mean,
        "lpips_mean": lpips_mean,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference_scene", help="Prepared scene with images_8 and transforms.json; use every input frame as context and export final PLY")
    parser.add_argument("--gaussian_head", required=True)
    parser.add_argument("--model_name", default="ckpts/LoGeR_star/latest.pt")
    parser.add_argument("--config", default="ckpts/LoGeR_star/original_config.yaml")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_scenes", type=int, default=0)
    parser.add_argument("--scene_offset", type=int, default=0)
    parser.add_argument(
        "--scene_ids",
        default="",
        help="Comma-separated scene directory names; overrides --num_scenes/--scene_offset.",
    )
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--save_gaussian_sample", action="store_true")
    parser.add_argument("--save_gaussian_sample_only", action="store_true")
    parser.add_argument("--gaussian_sample_max_points", type=int, default=250000)
    parser.add_argument(
        "--save_window_clouds",
        action="store_true",
        help="After merge(+two-pass), dump camera trajectory and per-window "
        "compact/complement Gaussian .pt files under output_dir/window_clouds/.",
    )
    parser.add_argument(
        "--save_window_clouds_only",
        action="store_true",
        help="Like --save_window_clouds, then exit before robust depth filtering/sky/render.",
    )
    parser.add_argument(
        "--save_pass2_trajectory_only",
        action="store_true",
        help=(
            "Run context-only Pass-2 streaming merge, discard window clouds, "
            "and save the assembled context trajectory in the Gaussian frame."
        ),
    )
    parser.add_argument(
        "--render_max_windows",
        type=int,
        default=0,
        help="If >0, render each target from owner-centered concat of at most "
        "N per-window hybrid clouds instead of the global merged cloud.",
    )
    parser.add_argument(
        "--images_on_cpu",
        action="store_true",
        help="Keep the full RGB tensor on CPU (Pi3 streams window slices to GPU). "
        "Auto-enabled when --num_frames >= 1000.",
    )
    parser.add_argument("--num_frames", type=int, default=146)
    parser.add_argument("--image_scale", type=float, default=1.0)
    parser.add_argument("--window_size", type=int, default=12)
    parser.add_argument("--overlap_size", type=int, default=3)
    parser.add_argument("--gaussian_head_batch_size", type=int, default=0)
    parser.add_argument("--two_pass_pose", action="store_true")
    parser.add_argument(
        "--orientation_aware_two_pass",
        action="store_true",
        help="Align two-pass rotation from context camera orientations instead of centers.",
    )
    parser.add_argument(
        "--calibrate_local_points_to_k",
        action="store_true",
        help="Keep predicted z while snapping local x/y to the rays defined by dataset K.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--variant", default="m10_v2_r65")
    parser.add_argument("--disable_depth_filter", action="store_true")
    parser.add_argument("--min_rel_gap", type=float, default=0.10)
    parser.add_argument("--visible_ratio_threshold", type=float, default=0.50)
    parser.add_argument("--neighborhood_radius", type=int, default=1)
    parser.add_argument("--min_valid_neighbors", type=int, default=5)
    parser.add_argument("--max_rel_depth_spread", type=float, default=0.15)
    parser.add_argument("--alpha_threshold", type=float, default=0.5)
    parser.add_argument("--render_chunk", type=int, default=8)
    parser.add_argument("--merge_mode", default="neural")
    parser.add_argument("--rotation_mode", default="svd")
    parser.add_argument("--track_anchor_mode", default="median_nn")
    parser.add_argument("--track_anchor_radius", type=float, default=0.15)
    parser.add_argument("--track_anchor_beta", type=float, default=6.0)
    parser.add_argument("--merge_hybrid_budget_ratio", type=float, default=0.0)
    parser.add_argument("--alltracker_conf_thr", type=float, default=0.1)
    parser.add_argument("--track_query_mode", default="first")
    parser.add_argument("--no_track_dense_complement", action="store_true")
    parser.add_argument(
        "--dense_accumulation",
        action="store_true",
        help=(
            "Ablation: skip track-guided compact merge; spawn dense complement for "
            "uncovered non-sky pixels. Forces global_complement_voxel_factor=0."
        ),
    )
    parser.add_argument("--sim3", action="store_true")
    parser.add_argument("--no_ttt", action="store_true")
    parser.add_argument("--no_swa", action="store_true")
    parser.add_argument("--global_complement_voxel_factor", type=float, default=4.0)
    parser.add_argument("--global_complement_voxel_schedule", default="fixed")
    parser.add_argument("--global_complement_voxel_policy", default="first")
    parser.add_argument("--global_complement_voxel_min_windows", type=int, default=3)
    parser.add_argument("--global_complement_voxel_min_complement", type=int, default=0)
    parser.add_argument(
        "--max_gpu_merge_windows",
        type=int,
        default=0,
        help="Keep at most N newest merge-window clouds on GPU; spill older chunks to CPU. 0=disabled.",
    )
    parser.add_argument(
        "--depth_filter_max_gpu_windows",
        type=int,
        default=0,
        help="robust depth filtering streaming: max owner-window clouds kept on GPU (0=use max_gpu_merge_windows or 1).",
    )
    parser.add_argument(
        "--depth_filter_gaussian_chunk_size",
        type=int,
        default=500000,
        help="robust depth filtering streaming: hybrid means voted in chunks of this size.",
    )
    parser.add_argument("--sky_mask_drop", action="store_true")
    parser.add_argument("--sky_mask_onnx", default="skyseg.onnx")
    parser.add_argument("--sky_mask_score_thr", type=float, default=160.0)
    parser.add_argument("--sky_sphere", action="store_true")
    parser.add_argument("--sky_sphere_radius_factor", type=float, default=50.0)
    parser.add_argument("--sky_sphere_sample_stride", type=int, default=2)
    parser.add_argument("--sky_sphere_azimuth_bins", type=int, default=512)
    parser.add_argument("--sky_sphere_elevation_bins", type=int, default=256)
    parser.add_argument("--sky_sphere_opacity", type=float, default=0.98)
    parser.add_argument("--sky_sphere_scale_overlap", type=float, default=1.0)
    parser.add_argument(
        "--sky_sphere_color_mode",
        choices=("rgb_dc", "predicted_sh", "rgb_dc_predicted_residual"),
        default="rgb_dc",
    )
    parser.add_argument("--global_online_voxel_factor", type=float, default=0.0)
    parser.add_argument("--global_online_geometry_factor", type=float, default=0.0)
    parser.add_argument("--global_online_dc_l2_threshold", type=float, default=-1.0)
    parser.add_argument("--global_online_update_alpha", type=float, default=0.0)
    parser.add_argument(
        "--global_online_matched_action",
        choices=("auto", "discard", "spawn", "update"),
        default="auto",
    )
    parser.add_argument("--global_online_teacher_max_records", type=int, default=0)
    parser.add_argument("--global_online_teacher_dump_path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--oom_diag",
        action="store_true",
        help="Log stage/window VRAM checkpoints to output_dir/oom_diag.jsonl for OOM localization.",
    )
    args = parser.parse_args()

    configure_reproducibility(args.seed, args.deterministic)
    spec = parse_filter_variant(args.variant)

    from loger.renderer import GaussianRenderer
    from run import load_pi3_model

    import lpips

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global _OOM_DIAG_PATH
    if args.oom_diag:
        _OOM_DIAG_PATH = output_dir / "oom_diag.jsonl"
        if _OOM_DIAG_PATH.exists():
            _OOM_DIAG_PATH.unlink()
        os.environ["LOGER_OOM_DIAG"] = str(_OOM_DIAG_PATH)
        oom_diag("eval_start", output_dir=str(output_dir))

    model = load_pi3_model(args.model_name, args.config, gaussian=True)
    model = model.to(device).eval()
    load_gaussian_checkpoint(model, args.gaussian_head, str(device))
    renderer = GaussianRenderer(sh_degree=2)
    lpips_fn = None if args.inference_scene else lpips.LPIPS(net="vgg").cpu().eval()
    merge_config = build_merge_config(args)

    if args.inference_scene:
        row = eval_scene(model, renderer, Path(args.inference_scene), args, merge_config, spec, lpips_fn)
        (output_dir / "inference.json").write_text(json.dumps(row, indent=2) + "\n")
        return

    scenes = load_scene_list(
        dataset_root,
        0 if args.scene_ids else args.num_scenes,
        0 if args.scene_ids else args.scene_offset,
    )
    if args.scene_ids:
        requested = [item.strip() for item in args.scene_ids.split(",") if item.strip()]
        available = {path.name: path for path in scenes}
        # ``num_scenes=0`` returns every scene, so explicit selection is
        # independent of the filesystem ordering used by smoke evaluations.
        missing = [scene_id for scene_id in requested if scene_id not in available]
        if missing:
            raise ValueError(f"requested scene_ids not found under {dataset_root}: {missing}")
        scenes = [available[scene_id] for scene_id in requested]
    scene_results = []

    print(
        f"robust depth filtering formal eval variant={args.variant} scenes={len(scenes)} "
        f"frames={args.num_frames} two_pass={args.two_pass_pose} "
        f"gap={args.min_rel_gap} vis_ratio={args.visible_ratio_threshold}",
        flush=True,
    )

    for scene_path in tqdm(scenes, desc="Evaluating"):
        with SceneTimer() as timer:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            try:
                row = eval_scene(
                    model, renderer, scene_path, args, merge_config, spec, lpips_fn
                )
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                last = dict(_OOM_DIAG_LAST) if _OOM_DIAG_LAST else {}
                # Prefer last jsonl line (includes window_* from pi3).
                if _OOM_DIAG_PATH is not None and _OOM_DIAG_PATH.exists():
                    try:
                        lines = [
                            ln for ln in _OOM_DIAG_PATH.read_text(encoding="utf-8").splitlines()
                            if ln.strip()
                        ]
                        if lines:
                            last = json.loads(lines[-1])
                    except Exception:
                        pass
                oom_diag(
                    "cuda_oom",
                    error=str(exc),
                    last_stage=last.get("stage"),
                    last_record=last,
                )
                row = {
                    "scene": scene_path.name,
                    "scene_path": str(scene_path),
                    "error": f"cuda_oom: {exc}",
                    "oom_last_stage": last.get("stage"),
                    "oom_last_record": last,
                    "views": [],
                    "psnr_mean": float("nan"),
                    "ssim_mean": float("nan"),
                    "lpips_mean": float("nan"),
                }
            except Exception as exc:
                oom_diag("eval_scene_failed", error=f"{type(exc).__name__}: {exc}")
                row = {
                    "scene": scene_path.name,
                    "scene_path": str(scene_path),
                    "error": f"{type(exc).__name__}: {exc}",
                    "views": [],
                    "psnr_mean": float("nan"),
                    "ssim_mean": float("nan"),
                    "lpips_mean": float("nan"),
                }
            row["time_sec"] = float(timer.elapsed_sec)
            row["peak_vram_gb"] = get_peak_vram_gb()
        scene_results.append(row)
        print(
            f"[scene] {scene_path.parent.name[:16]} psnr={row.get('psnr_mean', float('nan')):.3f}",
            flush=True,
        )

    summary = summarize_scenes(scene_results)
    payload = {
        "method": "LoGeR+robust depth filtering",
        "protocol": "DL3DV140-longaussian-v1",
        "gaussian_head": args.gaussian_head,
        "model_name": args.model_name,
        "dataset_root": str(dataset_root),
        "depth_filter_variant": args.variant,
        "depth_filter_enabled": not args.disable_depth_filter,
        "sky_mask_drop": args.sky_mask_drop,
        "sky_sphere": args.sky_sphere,
        "sky_sphere_color_mode": args.sky_sphere_color_mode,
        "sky_mask_score_thr": args.sky_mask_score_thr,
        "min_rel_gap": args.min_rel_gap,
        "visible_ratio_threshold": args.visible_ratio_threshold,
        "max_rel_depth_spread": args.max_rel_depth_spread,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "num_frames": args.num_frames,
        "scene_ids": args.scene_ids,
        "image_scale": args.image_scale,
        "window_size": args.window_size,
        "overlap_size": args.overlap_size,
        "gaussian_head_batch_size": args.gaussian_head_batch_size,
        "two_pass_pose": args.two_pass_pose,
        "orientation_aware_two_pass": args.orientation_aware_two_pass,
        "calibrate_local_points_to_k": args.calibrate_local_points_to_k,
        "track_merge": True,
        "merge_mode": args.merge_mode,
        "dense_accumulation": bool(
            getattr(args, "dense_accumulation", False)
            or float(getattr(args, "alltracker_conf_thr", 0.0) or 0.0) > 1.0
        ),
        "track_dense_complement": not args.no_track_dense_complement,
        "global_complement_voxel_factor": (
            0.0
            if bool(getattr(args, "dense_accumulation", False))
            or float(getattr(args, "alltracker_conf_thr", 0.0) or 0.0) > 1.0
            else args.global_complement_voxel_factor
        ),
        "save_window_clouds": bool(args.save_window_clouds or args.save_window_clouds_only),
        "save_window_clouds_only": bool(args.save_window_clouds_only),
        "save_pass2_trajectory_only": bool(args.save_pass2_trajectory_only),
        "render_max_windows": int(getattr(args, "render_max_windows", 0) or 0),
        "max_gpu_merge_windows": int(getattr(args, "max_gpu_merge_windows", 0) or 0),
        "summary": summary,
        "scenes": scene_results,
    }
    out_json = output_dir / "metrics_dl3dv_subset.json"
    write_metrics_json(out_json, payload)
    print_summary("LoGeR+robust depth filtering", summary)
    print(f"wrote {out_json}", flush=True)
    if not scene_results or any("error" in row for row in scene_results):
        raise RuntimeError("Evaluation incomplete: see per-scene errors in metrics JSON")


if __name__ == "__main__":
    main()
