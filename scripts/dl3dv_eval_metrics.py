"""Unified DL3DV-TEST evaluation metrics schema and helpers."""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import torch


def reset_peak_vram() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def get_peak_vram_gb() -> float | None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024**3)
    return None


class SceneTimer:
    def __init__(self) -> None:
        self._start: float | None = None

    def __enter__(self) -> "SceneTimer":
        self._start = time.perf_counter()
        reset_peak_vram()
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    @property
    def elapsed_sec(self) -> float:
        if self._start is None:
            raise RuntimeError("SceneTimer has not been started")
        return time.perf_counter() - self._start


def summarize_scenes(scenes: list[dict[str, Any]]) -> dict[str, Any]:
    all_psnr: list[float] = []
    all_ssim: list[float] = []
    all_lpips: list[float] = []
    gaussians: list[float] = []
    times: list[float] = []
    vram_peaks: list[float] = []

    for scene in scenes:
        for view in scene.get("views", []):
            all_psnr.append(float(view["psnr"]))
            all_ssim.append(float(view["ssim"]))
            all_lpips.append(float(view["lpips"]))
        if "num_gaussians" in scene and scene["num_gaussians"] is not None:
            gaussians.append(float(scene["num_gaussians"]))
        if "time_sec" in scene and scene["time_sec"] is not None:
            times.append(float(scene["time_sec"]))
        if scene.get("peak_vram_gb") is not None:
            vram_peaks.append(float(scene["peak_vram_gb"]))

    summary: dict[str, Any] = {
        "num_scenes": len(scenes),
        "num_test_views": len(all_psnr),
    }
    if all_psnr:
        summary.update(
            {
                "psnr_mean": float(np.mean(all_psnr)),
                "psnr_std": float(np.std(all_psnr)),
                "ssim_mean": float(np.mean(all_ssim)),
                "ssim_std": float(np.std(all_ssim)),
                "lpips_mean": float(np.mean(all_lpips)),
                "lpips_std": float(np.std(all_lpips)),
            }
        )
    if gaussians:
        summary["gaussians_per_scene_mean"] = float(np.mean(gaussians))
    if times:
        summary["time_per_scene_mean_sec"] = float(np.mean(times))
    if vram_peaks:
        summary["peak_vram_gb"] = float(max(vram_peaks))
    return summary


def print_summary(method: str, summary: dict[str, Any]) -> None:
    print(f"\n{'=' * 60}")
    print(f"{method} DL3DV-TEST subset results")
    print(f"{'=' * 60}")
    print(f"  Scenes:      {summary.get('num_scenes', 0)}")
    print(f"  Test views:  {summary.get('num_test_views', 0)}")
    if "gaussians_per_scene_mean" in summary:
        gaussians = summary["gaussians_per_scene_mean"]
        print(f"  GS/scene:    {gaussians / 1e6:.2f}M ({gaussians:,.0f})")
    if "time_per_scene_mean_sec" in summary:
        print(f"  Time/scene:  {summary['time_per_scene_mean_sec']:.2f}s")
    if "peak_vram_gb" in summary:
        print(f"  Peak VRAM:   {summary['peak_vram_gb']:.2f} GB")
    if "psnr_mean" in summary:
        print(f"  PSNR:  {summary['psnr_mean']:.2f} ± {summary['psnr_std']:.2f}")
        print(f"  SSIM:  {summary['ssim_mean']:.4f} ± {summary['ssim_std']:.4f}")
        print(f"  LPIPS: {summary['lpips_mean']:.4f} ± {summary['lpips_std']:.4f}")
    print(f"{'=' * 60}")


def write_metrics_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
