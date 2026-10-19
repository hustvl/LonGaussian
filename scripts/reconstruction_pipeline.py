#!/usr/bin/env python3
"""Shared checkpoint loading, merge configuration and reconstruction helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple


def load_split(scene_path: Path) -> Tuple[List[int], List[int]]:
    split_path = scene_path / "split.json"
    if not split_path.is_file():
        split_path = scene_path.parent / "split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    return [int(i) for i in split["ctx_list"]], [int(i) for i in split["tgt_list"]]


def load_gaussian_checkpoint(model, gaussian_head: str, device: str) -> None:
    from loger.utils.checkpoint import load_gaussian_weights
    load_gaussian_weights(model, gaussian_head, require_merger=True)
    model.to(device)
    if getattr(model, "neural_track_merger", None) is not None:
        model.neural_track_merger.to(device)


def build_merge_config(args) -> "MergeConfig":
    from loger.merge.merge_config import MergeConfig

    raw_window_actions = getattr(args, "global_online_window_action", None) or []
    window_actions = {}
    for spec in raw_window_actions:
        try:
            raw_idx, action = str(spec).split(":", 1)
            window_actions[int(raw_idx)] = action
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "--global_online_window_action must have form WINDOW:ACTION"
            ) from exc
    raw_unmatched_actions = getattr(args, "global_online_window_unmatched_action", None) or []
    unmatched_actions = {}
    for spec in raw_unmatched_actions:
        try:
            raw_idx, action = str(spec).split(":", 1)
            unmatched_actions[int(raw_idx)] = action
        except (TypeError, ValueError) as exc:
            raise ValueError("--global_online_window_unmatched_action must have form WINDOW:ACTION") from exc
    def parse_block_actions(raw_specs, *, name):
        parsed = {}
        for spec in raw_specs or []:
            try:
                raw_window, raw_block, action = str(spec).split(":", 2)
                parsed.setdefault(int(raw_window), {})[int(raw_block)] = action
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must have form WINDOW:BLOCK:ACTION") from exc
        return parsed
    strict_block_actions = parse_block_actions(
        getattr(args, "global_online_strict_block_action", None), name="--global_online_strict_block_action"
    )
    unmatched_block_actions = parse_block_actions(
        getattr(args, "global_online_unmatched_block_action", None), name="--global_online_unmatched_block_action"
    )
    return MergeConfig(
        merge_mode=args.merge_mode,
        rotation_mode=args.rotation_mode,
        track_anchor_mode=args.track_anchor_mode,
        track_anchor_radius=args.track_anchor_radius,
        track_anchor_beta=args.track_anchor_beta,
        hybrid_budget_ratio=args.merge_hybrid_budget_ratio,
        track_query_mode=args.track_query_mode,
        dense_accumulation=bool(getattr(args, "dense_accumulation", False)),
        global_complement_voxel_factor=(
            0.0
            if bool(getattr(args, "dense_accumulation", False))
            else args.global_complement_voxel_factor
        ),
        global_complement_voxel_schedule=args.global_complement_voxel_schedule,
        global_complement_voxel_policy=args.global_complement_voxel_policy,
        global_complement_voxel_min_windows=args.global_complement_voxel_min_windows,
        global_complement_voxel_min_complement=args.global_complement_voxel_min_complement,
        max_gpu_merge_windows=int(getattr(args, "max_gpu_merge_windows", 0) or 0),
        window_cloud_dump_dir=getattr(args, "window_cloud_dump_dir", None),
        window_cloud_export_only=bool(getattr(args, "window_cloud_export_only", False)),
        complement_conf_gate=bool(getattr(args, "complement_conf_gate", False)),
        complement_conf_thr=float(getattr(args, "complement_conf_thr", 0.0) or 0.0),
        complement_conf_gate_mode=str(getattr(args, "complement_conf_gate_mode", "hard")),
        premerge_conf_gate=bool(getattr(args, "premerge_conf_gate", False)),
        premerge_conf_thr=float(getattr(args, "premerge_conf_thr", 0.0) or 0.0),
        sky_mask_drop=bool(getattr(args, "sky_mask_drop", False)),
        sky_mask_onnx=str(getattr(args, "sky_mask_onnx", "skyseg.onnx")),
        sky_mask_score_thr=float(getattr(args, "sky_mask_score_thr", 160.0) or 160.0),
        global_online_voxel_factor=getattr(args, "global_online_voxel_factor", 0.0),
        global_online_geometry_factor=getattr(args, "global_online_geometry_factor", 0.0),
        global_online_dc_l2_threshold=getattr(args, "global_online_dc_l2_threshold", -1.0),
        global_online_update_alpha=getattr(args, "global_online_update_alpha", 0.0),
        global_online_attribute_updater_checkpoint=getattr(args, "global_online_attribute_updater_checkpoint", None),
        global_online_matched_action=getattr(args, "global_online_matched_action", "auto"),
        global_online_window_actions=window_actions or None,
        global_online_action_gate_checkpoint=getattr(args, "global_online_action_gate_checkpoint", None),
        global_online_unmatched_action=getattr(args, "global_online_unmatched_action", "spawn"),
        global_online_window_unmatched_actions=unmatched_actions or None,
        global_online_teacher_all_dump_path=getattr(args, "global_online_teacher_all_dump_path", None),
        global_online_teacher_block_voxel_factor=getattr(args, "global_online_teacher_block_voxel_factor", 4.0),
        global_online_strict_block_actions=strict_block_actions or None,
        global_online_unmatched_block_actions=unmatched_block_actions or None,
        global_online_block_action_policy_checkpoint=getattr(args, "global_online_block_action_policy_checkpoint", None),
        global_online_teacher_max_records=getattr(args, "global_online_teacher_max_records", 0),
        global_online_teacher_strict_records_per_window=getattr(args, "global_online_teacher_strict_records_per_window", 0),
        global_online_teacher_dump_path=getattr(args, "global_online_teacher_dump_path", None),
    )


def forward_track_merge(model, images, ctx_list, merge_config, args, intrinsics=None):
    return model(
        images[None],
        gaussian=True,
        track_merge=True,
        merge_config=merge_config,
        merge_ctx_indices=ctx_list,
        neural_merger=model.neural_track_merger,
        track_dense_complement=not args.no_track_dense_complement,
        alltracker_conf_thr=args.alltracker_conf_thr,
        record_per_window_clouds=True,
        window_size=args.window_size,
        overlap_size=args.overlap_size,
        se3=not args.sim3,
        sim3=args.sim3,
        turn_off_ttt=args.no_ttt,
        turn_off_swa=args.no_swa,
        no_detach=True,
        intrinsics=intrinsics,
        calibrate_local_points_to_k=bool(
            getattr(args, "calibrate_local_points_to_k", False)
        ),
    )
