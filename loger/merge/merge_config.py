"""Configuration for track-guided Gaussian merge."""

from __future__ import annotations

from typing import Any, Literal


MergeMode = Literal[
    "hard",
    "hard_medoid",
    "hard_tracklet",
    "hard_appearance",
    "hard_flatk",
    "hard_nested",
    "hard_raw_keep",
    "neural",
    "neural_set",
    "neural_feature",
    "neural_direct_slots",
    "neural_latent",
    "dsga_v0",
]
RotationMode = Literal["svd", "argmax", "chordal"]


class MergeConfig:
    """Merge knobs; accepts extra kwargs from newer run.py builds."""

    _DEFAULTS: dict[str, Any] = {
        "min_track_len": 2,
        "min_visible_frames": 2,
        "min_mean_opacity": 0.05,
        "max_center_variance": 0.3,
        "uv_thresh": 3.0,
        "merge_3d_thresh": 0.15,
        "scale_variance_boost": 1.0,
        "merge_mode": "hard",
        "rotation_mode": "svd",
        "soft_coverage": False,
        "soft_validity_temp": 0.1,
        "hybrid_budget_ratio": 0.0,
        "hybrid_budget_selection": "topk",
        "hybrid_budget_explore_fraction": 0.0,
        "hybrid_budget_sampling_alpha": 1.0,
        # Ablation: skip track-guided compact merge; spawn dense complement only.
        # Forces global_complement_voxel_factor=0 in __post_init__.
        "dense_accumulation": False,
        "tracklet_mad_multiplier": 3.0,
        "tracklet_min_jump": 0.15,
        "flatk_fraction_k4": 0.15,
        "flatk_fraction_k2": 0.35,
        "nested_fraction_k4": 0.15,
        "nested_fraction_k2": 0.35,
        "raw_keep_k": 1,
        "track_anchor_mode": "median_nn",
        "track_anchor_radius": 0.15,
        "track_anchor_beta": 6.0,
        "track_anchor_r_min": 0.02,
        "track_anchor_r_max": 0.5,
        "track_anchor_nn_samples": 5000,
        "track_source": "alltracker",
        "track_query_mode": "first",
        "geom_oracle_tau_3d": 0.15,
        "geom_oracle_conf_thr": 0.0,
        "gate_opacity_margin": 0.0,
        "gate_margin_loss_weight": 0.0,
        "complement_mode": "dense",
        # E12-A1: source-side conf gate for dense complement spawn only.
        # thr is applied to sigmoid(pred['conf']); disabled by default.
        "complement_conf_gate": False,
        "complement_conf_thr": 0.0,
        "complement_conf_gate_mode": "hard",
        # E12-A1c: pre-merge conf mask — low-conf pixels are suppressed from
        # track merge AND marked covered so they never spawn complement.
        "premerge_conf_gate": False,
        "premerge_conf_thr": 0.0,
        # Strategy A: semantic sky hard-drop (skyseg.onnx). Sky pixels are
        # suppressed from track merge and marked covered (no complement spawn).
        "sky_mask_drop": False,
        "sky_mask_onnx": "skyseg.onnx",
        "sky_mask_score_thr": 160.0,
        "octree_max_depth": 5,
        "octree_max_points": 10,
        "octree_root_edge": None,
        "octree_max_voxels": 8192,
        "octree_merger_mode": "shared",
        "global_complement_voxel_factor": 0.0,
        "global_complement_voxel_schedule": "fixed",
        "global_complement_voxel_policy": "max_opacity",
        "global_complement_voxel_min_windows": 0,
        "global_complement_voxel_min_complement": 0,
        # Bound GPU-resident merge clouds: older window chunks are spilled to CPU.
        # 0 disables spilling (legacy: keep all windows on GPU).
        "max_gpu_merge_windows": 0,
        # Stream per-window compact/complement to disk during merge (avoids
        # host-RAM blowup when exporting thousands of frames).
        "window_cloud_dump_dir": None,
        # When True with dump_dir: do not accumulate a global hybrid / in-memory
        # per-window records; only poses + on-disk window clouds are retained.
        "window_cloud_export_only": False,
        # E8 premise ablation: reject a completed window's points when its
        # world voxel was already occupied by an earlier window.
        "global_online_voxel_factor": 0.0,
        "global_online_geometry_factor": 0.0,
        "global_online_dc_l2_threshold": -1.0,
        # Conservative E8 attribute assimilation for strict online matches.
        # Geometry remains immutable; only opacity and SH-DC use a fixed EMA.
        "global_online_update_alpha": 0.0,
        # Optional learned replacement for the fixed strict-match EMA.  It is
        # consulted only when the hard matched action resolves to ``update``.
        # The updater is attribute-only (opacity + SH-DC), never geometry.
        "global_online_attribute_updater_checkpoint": None,
        # B0 tri-action control. ``auto`` preserves prior behavior: strict
        # discard without EMA, update when a positive EMA weight is requested.
        "global_online_matched_action": "auto",
        # B1 group counterfactual: retain a strict-discard default while
        # intervening on one completed window's matched-candidate group.
        # Keys are window indices and values use the same action vocabulary.
        "global_online_window_actions": None,
        # M2 learned candidate action gate.  Opt-in; without a checkpoint the
        # exact B0/B1 hand-coded behavior is preserved.
        "global_online_action_gate_checkpoint": None,
        # B1 data collection is opt-in and bounded to avoid evaluator bloat.
        "global_online_teacher_max_records": 0,
        "global_online_teacher_dump_path": None,
        # E8-P2: unmatched candidates are novel by default, but a B1/P2
        # counterfactual may reject one completed window's unmatched group.
        "global_online_unmatched_action": "spawn",
        "global_online_window_unmatched_actions": None,
        "global_online_teacher_all_dump_path": None,
        # P2 block-local counterfactual intervention.  Block ids are stable
        # within a completed window (stream x quantized 3D voxel) and are
        # exported with the all-candidate teacher dump.
        "global_online_teacher_block_voxel_factor": 4.0,
        # Optional per-window strict-match reservation for P2 block labels.
        # Zero preserves the legacy uniform all-candidate sample.
        "global_online_teacher_strict_records_per_window": 0,
        "global_online_strict_block_actions": None,
        "global_online_unmatched_block_actions": None,
        # Opt-in learned P2 policy.  It emits one action per local block and
        # association class; absent a checkpoint, all legacy behavior is exact.
        "global_online_block_action_policy_checkpoint": None,
    }

    def __init__(self, **kwargs: Any) -> None:
        for key, value in self._DEFAULTS.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.__post_init__()

    def __post_init__(self) -> None:
        if bool(getattr(self, "dense_accumulation", False)):
            # Dense ablation must not voxel-dedup the complement stream.
            self.global_complement_voxel_factor = 0.0
        if self.hybrid_budget_ratio < 0:
            raise ValueError("hybrid_budget_ratio must be non-negative")
        if self.tracklet_mad_multiplier < 0 or self.tracklet_min_jump < 0:
            raise ValueError("tracklet split thresholds must be non-negative")
        if self.flatk_fraction_k4 < 0 or self.flatk_fraction_k2 < 0:
            raise ValueError("flat-K fractions must be non-negative")
        if self.nested_fraction_k4 < 0 or self.nested_fraction_k2 < 0:
            raise ValueError("nested fractions must be non-negative")
        if getattr(self, "raw_keep_k", 1) < 1:
            raise ValueError("raw_keep_k must be >= 1")
        if getattr(self, "global_complement_voxel_factor", 0.0) < 0:
            raise ValueError("global_complement_voxel_factor must be non-negative")
        policy = getattr(self, "global_complement_voxel_policy", "max_opacity")
        if policy not in ("max_opacity", "first"):
            raise ValueError("unsupported global_complement_voxel_policy")
        if getattr(self, "global_complement_voxel_min_windows", 0) < 0:
            raise ValueError("global_complement_voxel_min_windows must be >= 0")
        if getattr(self, "global_complement_voxel_min_complement", 0) < 0:
            raise ValueError("global_complement_voxel_min_complement must be >= 0")
        conf_mode = str(getattr(self, "complement_conf_gate_mode", "hard"))
        if conf_mode not in ("hard", "soft_opacity"):
            raise ValueError("complement_conf_gate_mode must be hard or soft_opacity")
        if float(getattr(self, "complement_conf_thr", 0.0) or 0.0) < 0:
            raise ValueError("complement_conf_thr must be non-negative")
        if float(getattr(self, "premerge_conf_thr", 0.0) or 0.0) < 0:
            raise ValueError("premerge_conf_thr must be non-negative")
        sky_thr = float(getattr(self, "sky_mask_score_thr", 160.0) or 0.0)
        if sky_thr < 0 or sky_thr > 255:
            raise ValueError("sky_mask_score_thr must be in [0, 255]")
        if getattr(self, "global_online_voxel_factor", 0.0) < 0:
            raise ValueError("global_online_voxel_factor must be non-negative")
        if getattr(self, "global_online_geometry_factor", 0.0) < 0:
            raise ValueError("global_online_geometry_factor must be non-negative")
        if getattr(self, "global_online_dc_l2_threshold", -1.0) < -1.0:
            raise ValueError("global_online_dc_l2_threshold must be >= -1")
        update_alpha = float(getattr(self, "global_online_update_alpha", 0.0) or 0.0)
        if not 0.0 <= update_alpha <= 1.0:
            raise ValueError("global_online_update_alpha must be in [0, 1]")
        if update_alpha > 0 and (
            getattr(self, "global_online_geometry_factor", 0.0) <= 0
            or getattr(self, "global_online_dc_l2_threshold", -1.0) < 0
        ):
            raise ValueError("global_online_update_alpha requires geometry and DC gates")
        updater = getattr(self, "global_online_attribute_updater_checkpoint", None)
        if updater and (
            getattr(self, "global_online_geometry_factor", 0.0) <= 0
            or getattr(self, "global_online_dc_l2_threshold", -1.0) < 0
        ):
            raise ValueError("learned attribute updater requires geometry and DC gates")
        matched_action = str(getattr(self, "global_online_matched_action", "auto"))
        if matched_action not in ("auto", "discard", "spawn", "update"):
            raise ValueError("global_online_matched_action must be auto, discard, spawn, or update")
        if matched_action == "update" and (
            getattr(self, "global_online_geometry_factor", 0.0) <= 0
            or getattr(self, "global_online_dc_l2_threshold", -1.0) < 0
        ):
            raise ValueError("global_online_matched_action=update requires geometry and DC gates")
        window_actions = getattr(self, "global_online_window_actions", None)
        if window_actions is not None:
            if not isinstance(window_actions, dict):
                raise ValueError("global_online_window_actions must be a dict or None")
            for raw_window, action in window_actions.items():
                if int(raw_window) < 0:
                    raise ValueError("global_online_window_actions keys must be non-negative")
                if str(action) not in ("discard", "spawn", "update"):
                    raise ValueError("global_online_window_actions values must be discard, spawn, or update")
                if str(action) == "update" and (
                    getattr(self, "global_online_geometry_factor", 0.0) <= 0
                    or getattr(self, "global_online_dc_l2_threshold", -1.0) < 0
                ):
                    raise ValueError("global_online_window_actions=update requires geometry and DC gates")
        if int(getattr(self, "global_online_teacher_max_records", 0) or 0) < 0:
            raise ValueError("global_online_teacher_max_records must be non-negative")
        if int(getattr(self, "global_online_teacher_strict_records_per_window", 0) or 0) < 0:
            raise ValueError("global_online_teacher_strict_records_per_window must be non-negative")
        unmatched_action = str(getattr(self, "global_online_unmatched_action", "spawn"))
        if unmatched_action not in ("discard", "spawn"):
            raise ValueError("global_online_unmatched_action must be discard or spawn")
        unmatched_overrides = getattr(self, "global_online_window_unmatched_actions", None)
        if unmatched_overrides is not None:
            if not isinstance(unmatched_overrides, dict):
                raise ValueError("global_online_window_unmatched_actions must be a dict or None")
            for raw_window, action in unmatched_overrides.items():
                if int(raw_window) < 0 or str(action) not in ("discard", "spawn"):
                    raise ValueError("unmatched window actions require non-negative keys and discard/spawn values")
        block_factor = float(getattr(self, "global_online_teacher_block_voxel_factor", 4.0) or 0.0)
        if block_factor <= 0:
            raise ValueError("global_online_teacher_block_voxel_factor must be positive")
        for attr, actions, allowed in (
            ("global_online_strict_block_actions", getattr(self, "global_online_strict_block_actions", None), ("discard", "spawn", "update")),
            ("global_online_unmatched_block_actions", getattr(self, "global_online_unmatched_block_actions", None), ("discard", "spawn")),
        ):
            if actions is None:
                continue
            if not isinstance(actions, dict):
                raise ValueError(f"{attr} must be a dict or None")
            for raw_window, blocks in actions.items():
                if int(raw_window) < 0 or not isinstance(blocks, dict):
                    raise ValueError(f"{attr} requires non-negative window keys mapped to block-action dicts")
                for raw_block, action in blocks.items():
                    if int(raw_block) < 0 or str(action) not in allowed:
                        raise ValueError(f"{attr} has an invalid block action")
                    if str(action) == "update" and (
                        getattr(self, "global_online_geometry_factor", 0.0) <= 0
                        or getattr(self, "global_online_dc_l2_threshold", -1.0) < 0
                    ):
                        raise ValueError("strict block update requires geometry and DC gates")
        if self.merge_mode == "dsga_v0":
            self.soft_coverage = True
            if self.rotation_mode == "svd":
                self.rotation_mode = "chordal"
