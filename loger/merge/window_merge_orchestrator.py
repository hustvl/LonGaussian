"""Orchestrate per-window AllTracker merge inside Pi3 forward."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

import os
import hashlib
from dataclasses import asdict, replace

import numpy as np
import torch

from loger.merge.merge_config import MergeConfig
from loger.merge.window_merge import (
    GlobalCloudAccumulator,
    PerWindowCloudRecord,
    WindowMergeResult,
    WindowMergeTensors,
    assemble_camera_poses,
    extract_complement_tensors,
    global_contribute_range,
    merge_within_window,
    merge_within_window_dsga,
    pred_low_conf_mask,
    slice_predictions,
    stack_window_hybrid_tensors,
    merge_result_from_tensors,
    apply_global_complement_voxel_dedup,
    window_merge_tensors_to_device,
)
from loger.merge.streaming_refine import (
    GlobalVoxelMap,
    StreamingRefineModule,
    apply_chunk_delta,
    dense_pixel_attributes,
)
from loger.merge.global_voxel_dedup import OnlineFirstWinsVoxelDeduper
from loger.merge.online_action_gate import action_features, load_action_gate
from loger.merge.block_action_policy import block_action_features, load_block_action_policy
from loger.merge.learned_attribute_update import (
    load_learned_attribute_updater, predict_attributes, updater_features,
)
from loger.merge.window_se3 import align_window_to_prev_se3
from loger.tracking.alltracker_runner import AllTrackerRunner, DEFAULT_CONF_THR


def _rgb_window_to_uint8(imgs_w: torch.Tensor) -> np.ndarray:
    """(Nw, 3, H, W) float tensor -> (Nw, H, W, 3) uint8."""
    rgb = imgs_w.detach().float().permute(0, 2, 3, 1).cpu().numpy()
    if rgb.max() <= 1.0:
        rgb = (rgb * 255.0).clip(0, 255)
    return rgb.astype(np.uint8)


def resolve_online_matched_action(action: str, update_alpha: float) -> str:
    """Resolve the backward-compatible action alias without touching tensors."""
    if action == "auto":
        return "update" if float(update_alpha) > 0 else "discard"
    if action not in ("discard", "spawn", "update"):
        raise ValueError(f"unsupported online matched action: {action}")
    return action


def candidate_spatial_block_ids(means: torch.Tensor, stream: torch.Tensor, edge: float) -> torch.Tensor:
    """Return deterministic local ids for ``stream x 3D voxel`` candidate blocks.

    IDs are intentionally local to a completed window.  They are exported in
    the teacher dump and used only with that dump's window/action run, avoiding
    brittle global hashes of floating-point world coordinates.
    """
    if means.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=means.device)
    if edge <= 0:
        raise ValueError("candidate block edge must be positive")
    voxel = torch.floor(means / float(edge)).to(torch.int64)
    key = torch.cat((stream.to(torch.int64).unsqueeze(1), voxel), dim=1)
    _, inverse = torch.unique(key, dim=0, sorted=True, return_inverse=True)
    return inverse.to(torch.long)


class WindowMergeOrchestrator:
    """Accumulate per-window hybrid pieces into one global cloud."""

    def __init__(
        self,
        *,
        bundle_num_frames: int,
        ctx_global_indices: Sequence[int],
        overlap_size: int,
        num_windows: int,
        merge_config: Optional[MergeConfig] = None,
        alltracker_conf_thr: float = DEFAULT_CONF_THR,
        track_dense_complement: bool = True,
        track_rd_dump_path: Optional[str] = None,
        track_rd_selection_fingerprint: bool = False,
        neural_merger: Optional[torch.nn.Module] = None,
        record_per_window_clouds: bool = False,
        streaming_refine: Optional[StreamingRefineModule] = None,
        **_: object,
    ) -> None:
        self.bundle_num_frames = int(bundle_num_frames)
        self.ctx_set: Set[int] = {int(i) for i in ctx_global_indices}
        self.overlap_size = int(overlap_size)
        self.num_windows = int(num_windows)
        self.config = merge_config or MergeConfig()
        # conf_thr above AllTracker confidence domain [0,1] zeros all visibility.
        # Promote to the explicit dense path so thr=1.1 queue jobs skip tracking
        # and still satisfy compact==0 / voxel-off smoke checks.
        if float(alltracker_conf_thr) > 1.0 and not bool(
            getattr(self.config, "dense_accumulation", False)
        ):
            self.config.dense_accumulation = True
            self.config.global_complement_voxel_factor = 0.0
        self.runner = AllTrackerRunner(conf_thr=alltracker_conf_thr)
        self.track_dense_complement = bool(track_dense_complement)
        self.track_rd_dump_path = track_rd_dump_path
        self.track_rd_selection_fingerprint = bool(track_rd_selection_fingerprint)
        self.neural_merger = neural_merger
        self.streaming_refine = streaming_refine
        self.refine_map: Optional[GlobalVoxelMap] = None
        self._refine_diagnostics: List[dict] = []
        if self.streaming_refine is not None:
            self.streaming_refine.reset_episode()
        self.record_per_window_clouds = bool(record_per_window_clouds)
        self._per_window_records: List[PerWindowCloudRecord] = []
        self._window_dump_meta: List[Dict[str, object]] = []
        self._track_rd_records: List[dict] = []
        self._selector_trace_records: List[dict] = []
        self.cloud = GlobalCloudAccumulator()
        self.online_voxel_deduper: Optional[OnlineFirstWinsVoxelDeduper] = None
        self.online_action_gate: Optional[torch.nn.Module] = None
        self.block_action_policy: Optional[torch.nn.Module] = None
        self._block_action_feature_mean: Optional[torch.Tensor] = None
        self._block_action_feature_std: Optional[torch.Tensor] = None
        self.attribute_updater: Optional[torch.nn.Module] = None
        self._attribute_updater_feature_mean: Optional[torch.Tensor] = None
        self._attribute_updater_feature_std: Optional[torch.Tensor] = None
        self._online_voxel_windows: List[dict] = []
        self._teacher_records: List[dict[str, torch.Tensor]] = []
        self._teacher_all_records: List[dict[str, torch.Tensor]] = []
        self.pose_by_frame: Dict[int, torch.Tensor] = {}
        self._prev_overlap_pred: Optional[dict] = None
        self.num_dense_before = len(ctx_global_indices)  # updated on first window
        # Dense bypass for geometric regs (contrib ∩ ctx only; keep grad).
        self._dense_scales: List[torch.Tensor] = []
        self._dense_means: List[torch.Tensor] = []
        self._dense_offsets: List[torch.Tensor] = []
        self._sky_masker = None
        self._gpu_spill_events: int = 0
        self._gpu_spill_last: Dict[str, object] = {}
        self._budget_topk_kept = 0
        self._budget_explore_kept = 0
        self._budget_selection_windows = 0
        self._budget_selection_digests: List[str] = []
        self._sky_stats = {
            "sky_mask_drop": bool(getattr(self.config, "sky_mask_drop", False)),
            "sky_pixels": 0,
            "sky_frames": 0,
            "sky_pixel_total": 0,
        }

    def _gate_actions(
        self, deduper: OnlineFirstWinsVoxelDeduper, *, means: torch.Tensor,
        scales: torch.Tensor, opacities: torch.Tensor, dc: torch.Tensor,
        slots: torch.Tensor, stream: torch.Tensor, window_idx: int,
    ) -> torch.Tensor:
        """Return discard/update/spawn indices for strict matched candidates."""
        if slots.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=means.device)
        checkpoint = getattr(self.config, "global_online_action_gate_checkpoint", None)
        if not checkpoint:
            return torch.empty(0, dtype=torch.long, device=means.device)
        if self.online_action_gate is None:
            self.online_action_gate = load_action_gate(str(checkpoint), means.device)
        old = deduper.representative_features(slots)
        features = action_features(
            new_means=means, new_scales=scales, new_opacities=opacities, new_dc=dc,
            old_means=old["old_means"], old_scales=old["old_scales"],
            old_opacities=old["old_opacities"], old_dc=old["old_dc"], stream=stream,
            window_idx=window_idx, num_windows=self.num_windows,
        )
        with torch.no_grad():
            return self.online_action_gate(features).argmax(dim=1)

    def _learned_block_actions(
        self, deduper: OnlineFirstWinsVoxelDeduper, *, means: torch.Tensor, scales: torch.Tensor,
        opacities: torch.Tensor, dc: torch.Tensor, slots: torch.Tensor, strict: torch.Tensor,
        stream: torch.Tensor, block_ids: torch.Tensor, window_idx: int,
    ) -> tuple[dict[int, str], dict[int, str]]:
        """Infer one legal action per ``block x association class``."""
        checkpoint = getattr(self.config, "global_online_block_action_policy_checkpoint", None)
        if not checkpoint or means.numel() == 0:
            return {}, {}
        if self.block_action_policy is None:
            policy, mean, std = load_block_action_policy(str(checkpoint), means.device)
            self.block_action_policy, self._block_action_feature_mean, self._block_action_feature_std = policy, mean, std
        old_means = torch.zeros_like(means); old_scales = torch.zeros_like(scales)
        old_opacities = torch.zeros_like(opacities); old_dc = torch.zeros_like(dc)
        if bool(strict.any()):
            old = deduper.representative_features(slots[strict])
            old_means[strict], old_scales[strict] = old["old_means"], old["old_scales"]
            old_opacities[strict], old_dc[strict] = old["old_opacities"], old["old_dc"]
        x = block_action_features(new_means=means, new_scales=scales, new_opacities=opacities, new_dc=dc,
                                  old_means=old_means, old_scales=old_scales, old_opacities=old_opacities,
                                  old_dc=old_dc, strict=strict, stream=stream, window_idx=window_idx,
                                  num_windows=self.num_windows)
        x = (x - self._block_action_feature_mean) / self._block_action_feature_std
        strict_out, unmatched_out = {}, {}
        for is_strict, target in ((True, strict_out), (False, unmatched_out)):
            group = strict == is_strict
            for block in torch.unique(block_ids[group]).tolist():
                rows = group & (block_ids == int(block))
                with torch.no_grad():
                    logits = self.block_action_policy.forward_block(x[rows], is_strict=is_strict)
                    if not is_strict:
                        logits[1] = float("-inf")
                    action = int(logits.argmax().item())
                target[int(block)] = ("discard", "update", "spawn")[action]
        return strict_out, unmatched_out

    def _updated_attributes(
        self, deduper: OnlineFirstWinsVoxelDeduper, *, means: torch.Tensor,
        scales: torch.Tensor, opacities: torch.Tensor, dc: torch.Tensor,
        slots: torch.Tensor, stream: torch.Tensor, window_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict strict-match attributes, or exactly retain incoming EMA data.

        This function has no geometry output.  It is called only from the
        fixed ``update`` path after association, so it cannot alter action
        selection or create a new match.
        """
        checkpoint = getattr(self.config, "global_online_attribute_updater_checkpoint", None)
        if not checkpoint or slots.numel() == 0:
            return opacities, dc
        if self.attribute_updater is None:
            model, mean, std = load_learned_attribute_updater(str(checkpoint), means.device)
            self.attribute_updater = model
            self._attribute_updater_feature_mean, self._attribute_updater_feature_std = mean, std
        old = deduper.representative_features(slots)
        x = updater_features(
            new_means=means, new_scales=scales, new_opacities=opacities, new_dc=dc,
            old_means=old["old_means"], old_scales=old["old_scales"],
            old_opacities=old["old_opacities"], old_dc=old["old_dc"], stream=stream,
            window_idx=window_idx, num_windows=self.num_windows,
        )
        with torch.no_grad():
            return predict_attributes(self.attribute_updater, (x - self._attribute_updater_feature_mean) / self._attribute_updater_feature_std,
                                      old["old_opacities"], old["old_dc"], opacities, dc)

    def _run_tracks_for_window(self, rgb: np.ndarray) -> dict:
        mode = getattr(self.config, "track_query_mode", "first")
        if mode == "first":
            return self.runner.run_chunk(rgb, query_frame=0)
        if mode == "dual_endpoints":
            return self.runner.run_chunk_dual_endpoints(rgb)
        raise ValueError(f"unsupported track_query_mode: {mode}")

    def _get_sky_masker(self):
        if self._sky_masker is not None:
            return self._sky_masker
        from loger.utils.sky_mask import SkyMasker

        self._sky_masker = SkyMasker(
            onnx_path=str(getattr(self.config, "sky_mask_onnx", "skyseg.onnx")),
            score_thr=float(getattr(self.config, "sky_mask_score_thr", 160.0) or 160.0),
        )
        return self._sky_masker

    def _sky_mask_for_ctx(
        self,
        rgb: np.ndarray,
        ctx_local: Sequence[int],
        *,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Return (len(ctx_local), H, W) bool sky mask (True=sky), or None."""
        if not bool(getattr(self.config, "sky_mask_drop", False)):
            return None
        if not ctx_local:
            return None
        sky = self._get_sky_masker().masks_from_rgb_batch(
            rgb[list(ctx_local)], device=device
        )
        self._sky_stats["sky_pixels"] += int(sky.sum().item())
        self._sky_stats["sky_frames"] += int(sky.shape[0])
        self._sky_stats["sky_pixel_total"] += int(sky.numel())
        return sky

    def sky_mask_stats(self) -> dict:
        total = int(self._sky_stats.get("sky_pixel_total") or 0)
        sky_px = int(self._sky_stats.get("sky_pixels") or 0)
        return {
            **self._sky_stats,
            "sky_pixel_frac": float(sky_px) / float(max(total, 1)),
        }

    def _append_dense_reg(self, pred_contrib: dict) -> None:
        """Flatten contribution-ctx dense attrs for aniso / offset regularization."""
        if bool(getattr(self.config, "window_cloud_export_only", False)):
            return
        # points / scales: (1, T, H, W, 3) or (1, T, 3, H, W)
        means = pred_contrib["points"][0].reshape(-1, 3)
        scales = pred_contrib["gaussian_scales"][0].permute(0, 2, 3, 1).reshape(-1, 3)
        self._dense_means.append(means)
        self._dense_scales.append(scales)
        offsets = pred_contrib.get("gaussian_offsets")
        if offsets is not None:
            offs = offsets[0].permute(0, 2, 3, 1).reshape(-1, 3)
            self._dense_offsets.append(offs)

    def _refine_against_map(
        self,
        pred_ctx: dict,
        aligned: dict,
        ctx_local: Sequence[int],
        window_idx: int,
        height: int,
        width: int,
    ) -> Optional[torch.Tensor]:
        """Decide update-or-spawn for every ctx pixel of this window.

        Returns a ``(T, H, W)`` mask of pixels absorbed by the streaming map, or
        ``None`` before the map exists (first window spawns everything).
        """
        if (
            self.streaming_refine is None
            or not self.streaming_refine.config.enabled
            or self.refine_map is None
        ):
            return None
        device = pred_ctx["points"].device
        refine = torch.zeros(len(ctx_local), height, width, dtype=torch.bool, device=device)
        for local_j, window_t in enumerate(ctx_local):
            means, opacities, sh_dc = dense_pixel_attributes(pred_ctx, local_j)
            mask, _, diagnostics = self.streaming_refine.decide(
                self.refine_map,
                means,
                opacities,
                sh_dc,
                aligned["camera_poses"][0, window_t],
            )
            refine[local_j] = mask.reshape(height, width)
            self._refine_diagnostics.append(
                {"window_idx": window_idx, "frame": int(window_t), **diagnostics}
            )
        return refine

    def _init_refine_map(self, compact: WindowMergeTensors) -> None:
        """Size the map's voxel edge the same way complement dedup does."""
        if (
            self.streaming_refine is None
            or not self.streaming_refine.config.enabled
            or self.refine_map is not None
        ):
            return
        if compact.num_gaussians == 0:
            return
        from loger.merge.vectorized_merge import estimate_median_1nn

        median_nn = float(estimate_median_1nn(compact.means))
        if not median_nn > 0:
            return
        self.refine_map = GlobalVoxelMap(
            edge=self.streaming_refine.config.voxel_factor * median_nn,
            slots_per_voxel=self.streaming_refine.config.slots_per_voxel,
            device=compact.means.device,
        )

    def _ingest_into_map(self, tensors: WindowMergeTensors, stream: str) -> None:
        if self.refine_map is not None:
            self.refine_map.ingest(tensors, stream=stream)

    @staticmethod
    def _masked_tensors(tensors: WindowMergeTensors, keep: torch.Tensor) -> WindowMergeTensors:
        conf = tensors.conf_scores
        return replace(
            tensors,
            means=tensors.means[keep],
            quats=tensors.quats[keep],
            scales=tensors.scales[keep],
            opacities=tensors.opacities[keep],
            sh_flat=tensors.sh_flat[keep],
            conf_scores=(conf[keep] if conf is not None and conf.numel() == tensors.num_gaussians else conf),
        )

    def _deduplicate_completed_window(
        self, compact: WindowMergeTensors, complement: WindowMergeTensors, window_idx: int
    ) -> Tuple[WindowMergeTensors, WindowMergeTensors]:
        """Reject only points matching the pre-window global occupancy map."""
        factor = float(getattr(self.config, "global_online_voxel_factor", 0.0) or 0.0)
        if factor <= 0:
            return compact, complement
        if self.online_voxel_deduper is None:
            from loger.merge.vectorized_merge import estimate_median_1nn

            reference = compact if compact.num_gaussians else complement
            if reference.num_gaussians < 2:
                return compact, complement
            median_nn = float(estimate_median_1nn(reference.means))
            if not median_nn > 0:
                return compact, complement
            self.online_voxel_deduper = OnlineFirstWinsVoxelDeduper(
                edge=factor * median_nn, device=reference.means.device
            )

        deduper = self.online_voxel_deduper
        geometry_factor = float(getattr(self.config, "global_online_geometry_factor", 0.0) or 0.0)
        dc_l2_threshold = float(getattr(self.config, "global_online_dc_l2_threshold", -1.0))
        compact_keep, compact_slots = deduper.associate(compact.means, scales=compact.scales, sh_flat=compact.sh_flat, geometry_factor=geometry_factor, dc_l2_threshold=dc_l2_threshold)
        complement_keep, complement_slots = deduper.associate(complement.means, scales=complement.scales, sh_flat=complement.sh_flat, geometry_factor=geometry_factor, dc_l2_threshold=dc_l2_threshold)
        matched_action = str(getattr(self.config, "global_online_matched_action", "auto"))
        window_actions = getattr(self.config, "global_online_window_actions", None) or {}
        # A B1 counterfactual intervenes on exactly one completed window while
        # leaving the policy for every other window unchanged.
        matched_action = str(window_actions.get(int(window_idx), matched_action))
        compact_matched = ~compact_keep
        complement_matched = ~complement_keep
        candidate_stream = torch.cat((
            torch.zeros(compact.num_gaussians, dtype=torch.int8, device=compact.means.device),
            torch.ones(complement.num_gaussians, dtype=torch.int8, device=compact.means.device),
        ))
        candidate_means = torch.cat((compact.means, complement.means))
        block_factor = float(getattr(self.config, "global_online_teacher_block_voxel_factor", 4.0) or 0.0)
        candidate_block_ids = candidate_spatial_block_ids(candidate_means, candidate_stream, deduper.edge * block_factor)
        compact_block_ids = candidate_block_ids[:compact.num_gaussians]
        complement_block_ids = candidate_block_ids[compact.num_gaussians:]
        update_alpha = float(getattr(self.config, "global_online_update_alpha", 0.0) or 0.0)
        matched_action = resolve_online_matched_action(matched_action, update_alpha)
        self._record_teacher_candidates(
            deduper, compact, complement, compact_slots, complement_slots,
            compact_matched, complement_matched, window_idx,
        )
        self._record_teacher_all_candidates(
            deduper, compact, complement, compact_slots, complement_slots,
            compact_matched, complement_matched, window_idx, candidate_block_ids,
        )
        learned_strict_blocks, learned_unmatched_blocks = self._learned_block_actions(
            deduper, means=candidate_means, scales=torch.cat((compact.scales, complement.scales)),
            opacities=torch.cat((compact.opacities, complement.opacities)),
            dc=torch.cat((compact.sh_flat[:, :3], complement.sh_flat[:, :3])),
            slots=torch.cat((compact_slots, complement_slots)),
            strict=torch.cat((compact_matched, complement_matched)), stream=candidate_stream,
            block_ids=candidate_block_ids, window_idx=window_idx,
        )
        updated_targets = 0
        # Learned gate is active only when no B1 window intervention overrides
        # the hand-coded policy. It operates after strict association, so it
        # cannot create a geometry match or move any geometry.
        gate_actions = self._gate_actions(
            deduper,
            means=torch.cat((compact.means[compact_matched], complement.means[complement_matched])),
            scales=torch.cat((compact.scales[compact_matched], complement.scales[complement_matched])),
            opacities=torch.cat((compact.opacities[compact_matched], complement.opacities[complement_matched])),
            dc=torch.cat((compact.sh_flat[compact_matched, :3], complement.sh_flat[complement_matched, :3])),
            slots=torch.cat((compact_slots[compact_matched], complement_slots[complement_matched])),
            stream=torch.cat((
                torch.zeros(int(compact_matched.sum().item()), dtype=torch.int8, device=compact.means.device),
                torch.ones(int(complement_matched.sum().item()), dtype=torch.int8, device=compact.means.device),
            )), window_idx=window_idx,
        ) if (getattr(self.config, "global_online_action_gate_checkpoint", None) and int(window_idx) not in window_actions) else torch.empty(0, dtype=torch.long, device=compact.means.device)
        if gate_actions.numel() > 0:
            n_compact = int(compact_matched.sum().item())
            compact_match_idx = compact_matched.nonzero(as_tuple=True)[0]
            complement_match_idx = complement_matched.nonzero(as_tuple=True)[0]
            compact_keep[compact_match_idx] = gate_actions[:n_compact] == 2
            complement_keep[complement_match_idx] = gate_actions[n_compact:] == 2
            update_slots = torch.cat((
                compact_slots[compact_match_idx[gate_actions[:n_compact] == 1]],
                complement_slots[complement_match_idx[gate_actions[n_compact:] == 1]],
            ))
            update_incoming_opacity = torch.cat((
                compact.opacities[compact_match_idx[gate_actions[:n_compact] == 1]],
                complement.opacities[complement_match_idx[gate_actions[n_compact:] == 1]],
            ))
            update_incoming_sh = torch.cat((
                compact.sh_flat[compact_match_idx[gate_actions[:n_compact] == 1]],
                complement.sh_flat[complement_match_idx[gate_actions[n_compact:] == 1]],
            ))
            if update_slots.numel() > 0 and update_alpha > 0:
                streams, chunks, locals_ = deduper.representative_locations(update_slots)
                updated_targets = self.cloud.ema_update_attributes(
                    streams=streams, chunks=chunks, local_indices=locals_,
                    incoming_opacities=update_incoming_opacity, incoming_sh_flat=update_incoming_sh,
                    alpha=update_alpha,
                )
            matched_action = "learned_gate"
        elif matched_action == "spawn":
            # B0 control: even high-confidence associations become new points.
            # Their association is still recorded, so the decision surface is
            # identical to the discard/update controls.
            compact_keep = torch.ones_like(compact_keep)
            complement_keep = torch.ones_like(complement_keep)
        elif matched_action == "update" and update_alpha > 0:
            matched_slots = torch.cat((compact_slots[compact_matched], complement_slots[complement_matched]))
            if int(matched_slots.numel()) > 0:
                streams, chunks, locals_ = deduper.representative_locations(matched_slots)
                incoming_opacity = torch.cat((compact.opacities[compact_matched], complement.opacities[complement_matched]))
                incoming_sh = torch.cat((compact.sh_flat[compact_matched], complement.sh_flat[complement_matched]))
                matched_means = torch.cat((compact.means[compact_matched], complement.means[complement_matched]))
                matched_scales = torch.cat((compact.scales[compact_matched], complement.scales[complement_matched]))
                matched_stream = torch.cat((
                    torch.zeros(int(compact_matched.sum().item()), dtype=torch.int8, device=compact.means.device),
                    torch.ones(int(complement_matched.sum().item()), dtype=torch.int8, device=compact.means.device),
                ))
                predicted_opacity, predicted_dc = self._updated_attributes(
                    deduper, means=matched_means, scales=matched_scales, opacities=incoming_opacity,
                    dc=incoming_sh[:, :3], slots=matched_slots, stream=matched_stream, window_idx=window_idx,
                )
                incoming_sh = incoming_sh.clone(); incoming_sh[:, :3] = predicted_dc
                updated_targets = self.cloud.ema_update_attributes(
                    streams=streams, chunks=chunks, local_indices=locals_,
                    incoming_opacities=predicted_opacity, incoming_sh_flat=incoming_sh,
                    # Learned outputs are final target attributes.  The
                    # accumulator still averages duplicate observations once.
                    alpha=1.0 if getattr(self.config, "global_online_attribute_updater_checkpoint", None) else update_alpha,
                )
        unmatched_action = str(getattr(self.config, "global_online_unmatched_action", "spawn"))
        unmatched_windows = getattr(self.config, "global_online_window_unmatched_actions", None) or {}
        unmatched_action = str(unmatched_windows.get(int(window_idx), unmatched_action))
        if unmatched_action == "discard":
            compact_keep[~compact_matched] = False
            complement_keep[~complement_matched] = False
        strict_block_actions = dict(learned_strict_blocks)
        strict_block_actions.update((getattr(self.config, "global_online_strict_block_actions", None) or {}).get(int(window_idx), {}))
        unmatched_block_actions = dict(learned_unmatched_blocks)
        unmatched_block_actions.update((getattr(self.config, "global_online_unmatched_block_actions", None) or {}).get(int(window_idx), {}))
        block_updates = []
        for raw_block, action in strict_block_actions.items():
            block = int(raw_block)
            compact_mask = compact_matched & (compact_block_ids == block)
            complement_mask = complement_matched & (complement_block_ids == block)
            if action == "spawn":
                compact_keep[compact_mask] = True; complement_keep[complement_mask] = True
            elif action == "discard":
                compact_keep[compact_mask] = False; complement_keep[complement_mask] = False
            elif action == "update":
                compact_keep[compact_mask] = False; complement_keep[complement_mask] = False
                block_updates.append((compact_mask, complement_mask))
        for raw_block, action in unmatched_block_actions.items():
            block = int(raw_block)
            compact_mask = (~compact_matched) & (compact_block_ids == block)
            complement_mask = (~complement_matched) & (complement_block_ids == block)
            if action == "spawn":
                compact_keep[compact_mask] = True; complement_keep[complement_mask] = True
            else:
                compact_keep[compact_mask] = False; complement_keep[complement_mask] = False
        if block_updates and update_alpha > 0:
            cmask = torch.stack([x[0] for x in block_updates]).any(0)
            xmask = torch.stack([x[1] for x in block_updates]).any(0)
            slots = torch.cat((compact_slots[cmask], complement_slots[xmask]))
            streams, chunks, locals_ = deduper.representative_locations(slots)
            updated_targets += self.cloud.ema_update_attributes(
                streams=streams, chunks=chunks, local_indices=locals_,
                incoming_opacities=torch.cat((compact.opacities[cmask], complement.opacities[xmask])),
                incoming_sh_flat=torch.cat((compact.sh_flat[cmask], complement.sh_flat[xmask])), alpha=update_alpha,
            )
        compact_out = self._masked_tensors(compact, compact_keep)
        complement_out = self._masked_tensors(complement, complement_keep)
        # Commit once after both streams have been checked: points created by
        # one completed window never suppress each other.
        compact_chunk = len(self.cloud.compact_means)
        complement_chunk = len(self.cloud.complement_means)
        committed_means = torch.cat((compact_out.means, complement_out.means), dim=0)
        committed_scales = torch.cat((compact_out.scales, complement_out.scales), dim=0)
        committed_sh = torch.cat((compact_out.sh_flat, complement_out.sh_flat), dim=0)
        compact_n = compact_out.num_gaussians
        deduper.commit(
            committed_means, scales=committed_scales, sh_flat=committed_sh,
            opacities=torch.cat((compact_out.opacities, complement_out.opacities), dim=0),
            source_stream=torch.cat((
                torch.zeros(compact_n, dtype=torch.int8, device=committed_means.device),
                torch.ones(complement_out.num_gaussians, dtype=torch.int8, device=committed_means.device),
            )),
            source_chunk=torch.cat((
                torch.full((compact_n,), compact_chunk, dtype=torch.long, device=committed_means.device),
                torch.full((complement_out.num_gaussians,), complement_chunk, dtype=torch.long, device=committed_means.device),
            )),
            source_local=torch.cat((
                torch.arange(compact_n, device=committed_means.device),
                torch.arange(complement_out.num_gaussians, device=committed_means.device),
            )),
        )
        self._online_voxel_windows.append({
            "window_idx": int(window_idx),
            "compact_input": compact.num_gaussians,
            "compact_rejected": int((~compact_keep).sum().item()),
            "compact_matched": int(compact_matched.sum().item()),
            "complement_input": complement.num_gaussians,
            "complement_rejected": int((~complement_keep).sum().item()),
            "complement_matched": int(complement_matched.sum().item()),
            "matched_action": matched_action,
            "attribute_updater_enabled": bool(getattr(self.config, "global_online_attribute_updater_checkpoint", None)),
            "strict_block_actions": {int(k): str(v) for k, v in strict_block_actions.items()},
            "unmatched_block_actions": {int(k): str(v) for k, v in unmatched_block_actions.items()},
            "updated_targets": updated_targets,
        })
        return compact_out, complement_out

    def _record_teacher_candidates(
        self,
        deduper: OnlineFirstWinsVoxelDeduper,
        compact: WindowMergeTensors,
        complement: WindowMergeTensors,
        compact_slots: torch.Tensor,
        complement_slots: torch.Tensor,
        compact_matched: torch.Tensor,
        complement_matched: torch.Tensor,
        window_idx: int,
    ) -> None:
        """Persist a bounded, action-agnostic B1 candidate table at finalize."""
        limit = int(getattr(self.config, "global_online_teacher_max_records", 0) or 0)
        if limit <= 0:
            return
        recorded = sum(int(x["slot"].numel()) for x in self._teacher_records)
        remaining = limit - recorded
        if remaining <= 0:
            return
        slots = torch.cat((compact_slots[compact_matched], complement_slots[complement_matched]))
        if int(slots.numel()) == 0:
            return
        means = torch.cat((compact.means[compact_matched], complement.means[complement_matched]))
        scales = torch.cat((compact.scales[compact_matched], complement.scales[complement_matched]))
        opacities = torch.cat((compact.opacities[compact_matched], complement.opacities[complement_matched]))
        dc = torch.cat((compact.sh_flat[compact_matched, :3], complement.sh_flat[complement_matched, :3]))
        stream = torch.cat((
            torch.zeros(int(compact_matched.sum().item()), dtype=torch.int8, device=slots.device),
            torch.ones(int(complement_matched.sum().item()), dtype=torch.int8, device=slots.device),
        ))
        # Reserve capacity for later windows: the teacher set must expose
        # temporal variation rather than being filled entirely by window 1.
        per_window = max(1, (limit + self.num_windows - 1) // self.num_windows)
        take = min(remaining, per_window, int(slots.numel()))
        if take < int(slots.numel()):
            index = torch.linspace(0, int(slots.numel()) - 1, steps=take, device=slots.device).round().long()
            slots, means, scales, opacities, dc, stream = (
                slots[index], means[index], scales[index], opacities[index], dc[index], stream[index]
            )
        old = deduper.representative_features(slots)
        self._teacher_records.append({
            "slot": slots.detach().cpu(), "new_means": means.detach().cpu(),
            "new_scales": scales.detach().cpu(), "new_opacities": opacities.detach().cpu(),
            "new_dc": dc.detach().cpu(), "stream": stream.detach().cpu(),
            "window_idx": torch.full((slots.numel(),), int(window_idx), dtype=torch.int16),
            **{key: value.cpu() for key, value in old.items()},
        })

    def _record_teacher_all_candidates(
        self, deduper: OnlineFirstWinsVoxelDeduper, compact: WindowMergeTensors,
        complement: WindowMergeTensors, compact_slots: torch.Tensor,
        complement_slots: torch.Tensor, compact_matched: torch.Tensor,
        complement_matched: torch.Tensor, window_idx: int, block_ids: torch.Tensor,
    ) -> None:
        """Export bounded all-candidate P2 rows; unmatched rows have no old point."""
        path = getattr(self.config, "global_online_teacher_all_dump_path", None)
        limit = int(getattr(self.config, "global_online_teacher_max_records", 0) or 0)
        if not path or limit <= 0:
            return
        recorded = sum(int(x["is_strict_match"].numel()) for x in self._teacher_all_records)
        remaining = limit - recorded
        if remaining <= 0:
            return
        means = torch.cat((compact.means, complement.means)); scales = torch.cat((compact.scales, complement.scales))
        opacities = torch.cat((compact.opacities, complement.opacities)); dc = torch.cat((compact.sh_flat[:, :3], complement.sh_flat[:, :3]))
        slots = torch.cat((compact_slots, complement_slots)); matched = torch.cat((compact_matched, complement_matched))
        stream = torch.cat((torch.zeros(compact.num_gaussians, dtype=torch.int8, device=means.device), torch.ones(complement.num_gaussians, dtype=torch.int8, device=means.device)))
        per_window = max(1, (limit + self.num_windows - 1) // self.num_windows)
        take = min(remaining, per_window, int(means.shape[0]))
        if take < int(means.shape[0]):
            strict_quota = int(getattr(self.config, "global_online_teacher_strict_records_per_window", 0) or 0)
            if strict_quota > 0:
                strict_idx = matched.nonzero(as_tuple=True)[0]
                novel_idx = (~matched).nonzero(as_tuple=True)[0]
                strict_take = min(int(strict_idx.numel()), strict_quota, take)
                novel_take = min(int(novel_idx.numel()), take - strict_take)
                def evenly(indices: torch.Tensor, count: int) -> torch.Tensor:
                    if count <= 0:
                        return indices[:0]
                    if count >= int(indices.numel()):
                        return indices
                    return indices[torch.linspace(0, int(indices.numel()) - 1, steps=count, device=means.device).round().long()]
                index = torch.cat((evenly(strict_idx, strict_take), evenly(novel_idx, novel_take))).sort().values
            else:
                index = torch.linspace(0, int(means.shape[0]) - 1, steps=take, device=means.device).round().long()
            means, scales, opacities, dc, slots, matched, stream, block_ids = (means[index], scales[index], opacities[index], dc[index], slots[index], matched[index], stream[index], block_ids[index])
        old_means = torch.zeros_like(means); old_scales = torch.zeros_like(scales); old_dc = torch.zeros_like(dc); old_opacities = torch.zeros_like(opacities)
        if bool(matched.any()):
            old = deduper.representative_features(slots[matched])
            old_means[matched], old_scales[matched], old_dc[matched], old_opacities[matched] = old["old_means"], old["old_scales"], old["old_dc"], old["old_opacities"]
        self._teacher_all_records.append({
            "new_means": means.detach().cpu(), "new_scales": scales.detach().cpu(), "new_opacities": opacities.detach().cpu(), "new_dc": dc.detach().cpu(),
            "old_means": old_means.detach().cpu(), "old_scales": old_scales.detach().cpu(), "old_opacities": old_opacities.detach().cpu(), "old_dc": old_dc.detach().cpu(),
            "is_strict_match": matched.detach().cpu(), "stream": stream.detach().cpu(),
            "block_id": block_ids.detach().cpu(),
            "window_idx": torch.full((means.shape[0],), int(window_idx), dtype=torch.int16),
        })

    def _splice_refine_deltas(self) -> Dict[str, object]:
        """Fold accumulated deltas back into the accumulator chunks."""
        if self.refine_map is None:
            return {"applied": False, "reason": "no_map"}
        for stream in ("compact", "complement"):
            chunks = self.refine_map.chunk_deltas(stream)
            means = getattr(self.cloud, f"{stream}_means")
            quats = getattr(self.cloud, f"{stream}_quats")
            scales = getattr(self.cloud, f"{stream}_scales")
            opacities = getattr(self.cloud, f"{stream}_opacities")
            sh_flat = getattr(self.cloud, f"{stream}_sh_flat")
            if len(chunks) != len(means):
                raise RuntimeError(
                    f"streaming refine tracked {len(chunks)} {stream} chunks but the "
                    f"accumulator holds {len(means)}"
                )
            for i, delta in enumerate(chunks):
                updated = apply_chunk_delta(
                    WindowMergeTensors(
                        means[i], quats[i], scales[i], opacities[i], sh_flat[i]
                    ),
                    delta["means"],
                    delta["opacities"],
                    delta["sh_dc"],
                )
                means[i] = updated.means
                opacities[i] = updated.opacities
                sh_flat[i] = updated.sh_flat
        stats: Dict[str, object] = {"applied": True, **self.refine_map.stats()}
        if self._refine_diagnostics:
            stats["mean_refine_fraction"] = float(
                np.mean([d["refine_fraction"] for d in self._refine_diagnostics])
            )
            stats["mean_candidate_fraction"] = float(
                np.mean([d["candidate_fraction"] for d in self._refine_diagnostics])
            )
            stats["num_refine_frames"] = len(self._refine_diagnostics)
        return stats

    def _commit_window_clouds(
        self,
        *,
        window_idx: int,
        start_idx: int,
        end_idx: int,
        contrib_start: int,
        contrib_end: int,
        compact: WindowMergeTensors,
        complement: WindowMergeTensors,
    ) -> None:
        """Persist and/or accumulate one window's compact/complement clouds."""
        export_only = bool(getattr(self.config, "window_cloud_export_only", False))
        self._record_per_window(
            window_idx=window_idx,
            start_idx=start_idx,
            end_idx=end_idx,
            contrib_start=contrib_start,
            contrib_end=contrib_end,
            compact=compact,
            complement=complement,
        )
        if complement.budget_selection_stats is not None:
            budget_stats = complement.budget_selection_stats
            self._budget_topk_kept += int(budget_stats["topk_kept"])
            self._budget_explore_kept += int(budget_stats["explore_kept"])
            self._budget_selection_windows += 1
            payload = torch.cat([
                complement.means.detach().float().cpu().reshape(-1),
                complement.opacities.detach().float().cpu().reshape(-1),
            ]).numpy().tobytes()
            self._budget_selection_digests.append(
                hashlib.sha256(payload).hexdigest()
            )
        if export_only:
            return
        self.cloud.append(compact, is_complement=False, window_idx=window_idx)
        self.cloud.append(complement, is_complement=True, window_idx=window_idx)
        self._init_refine_map(compact)
        self._ingest_into_map(compact, "compact")
        self._ingest_into_map(complement, "complement")

    def _record_per_window(
        self,
        *,
        window_idx: int,
        start_idx: int,
        end_idx: int,
        contrib_start: int,
        contrib_end: int,
        compact: WindowMergeTensors,
        complement: WindowMergeTensors,
    ) -> None:
        dump_dir = getattr(self.config, "window_cloud_dump_dir", None)
        export_only = bool(getattr(self.config, "window_cloud_export_only", False))
        if dump_dir:
            from pathlib import Path

            win_dir = Path(str(dump_dir))
            win_dir.mkdir(parents=True, exist_ok=True)

            def _save(variant: str, tensors: WindowMergeTensors) -> Dict[str, object]:
                path = win_dir / f"w{int(window_idx):03d}_{variant}.pt"
                payload = {
                    "means": tensors.means.detach().to("cpu", dtype=torch.float32).contiguous(),
                    "quats": tensors.quats.detach().to("cpu", dtype=torch.float32).contiguous(),
                    "scales": tensors.scales.detach().to("cpu", dtype=torch.float32).contiguous(),
                    "opacities": tensors.opacities.detach().to("cpu", dtype=torch.float32).contiguous(),
                    "sh_flat": tensors.sh_flat.detach().to("cpu", dtype=torch.float32).contiguous(),
                    "stats": {
                        "window_idx": int(window_idx),
                        "start_idx": int(start_idx),
                        "end_idx": int(end_idx),
                        "contrib_start": int(contrib_start),
                        "contrib_end": int(contrib_end),
                        "variant": variant,
                        "num_gaussians_compact": int(compact.num_gaussians),
                        "num_gaussians_complement": int(complement.num_gaussians),
                        "num_gaussians": int(tensors.num_gaussians),
                    },
                    "num_gaussians": int(tensors.num_gaussians),
                }
                torch.save(payload, path)
                return {
                    "path": str(path),
                    "num_gaussians": int(tensors.num_gaussians),
                    "stats": payload["stats"],
                }

            entry: Dict[str, object] = {
                "window_idx": int(window_idx),
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "contrib_start": int(contrib_start),
                "contrib_end": int(contrib_end),
                "compact": _save("compact", compact),
                "complement": _save("complement", complement),
            }
            self._window_dump_meta.append(entry)
            if export_only:
                return
        if not self.record_per_window_clouds:
            return
        self._per_window_records.append(
            PerWindowCloudRecord(
                window_idx=window_idx,
                start_idx=start_idx,
                end_idx=end_idx,
                contrib_start=contrib_start,
                contrib_end=contrib_end,
                compact=compact,
                complement=complement,
            )
        )

    def build_per_window_results(self) -> List[WindowMergeResult]:
        """Materialize per-window hybrid clouds for diagnostic rendering."""
        return self._build_per_window_variant_results("hybrid")

    def build_per_window_compact_results(self) -> List[WindowMergeResult]:
        return self._build_per_window_variant_results("compact")

    def build_per_window_complement_results(self) -> List[WindowMergeResult]:
        return self._build_per_window_variant_results("complement")

    def _build_per_window_variant_results(self, variant: str) -> List[WindowMergeResult]:
        if not self._per_window_records:
            return []
        camera_poses = assemble_camera_poses(self.pose_by_frame, self.bundle_num_frames)
        results: List[WindowMergeResult] = []
        for rec in self._per_window_records:
            if variant == "compact":
                tensors = rec.compact
            elif variant == "complement":
                tensors = rec.complement
            else:
                tensors = stack_window_hybrid_tensors(rec.compact, rec.complement)
            results.append(
                merge_result_from_tensors(
                    tensors,
                    camera_poses,
                    stats={
                        "window_idx": rec.window_idx,
                        "start_idx": rec.start_idx,
                        "end_idx": rec.end_idx,
                        "contrib_start": rec.contrib_start,
                        "contrib_end": rec.contrib_end,
                        "variant": variant,
                        "num_gaussians_compact": rec.compact.num_gaussians,
                        "num_gaussians_complement": rec.complement.num_gaussians,
                        "num_gaussians": tensors.num_gaussians,
                    },
                )
            )
        return results

    def process_window(
        self,
        window_idx: int,
        start_idx: int,
        end_idx: int,
        pred_raw: dict,
        imgs_w_raw: torch.Tensor,
    ) -> None:
        if window_idx == 0 or self._prev_overlap_pred is None:
            aligned = pred_raw
        else:
            aligned = align_window_to_prev_se3(
                pred_raw, self._prev_overlap_pred, self.overlap_size
            )

        contrib_start, contrib_end = global_contribute_range(
            window_idx, self.num_windows, start_idx, end_idx, self.overlap_size
        )

        for global_f in range(contrib_start, contrib_end):
            local_t = global_f - start_idx
            self.pose_by_frame[global_f] = aligned["camera_poses"][0, local_t].detach()

        ctx_local = [i for i in range(end_idx - start_idx) if (start_idx + i) in self.ctx_set]
        if not ctx_local:
            self._store_overlap_tail(aligned, end_idx - start_idx)
            self._maybe_spill_old_windows_to_cpu(window_idx)
            return

        if window_idx == 0:
            height, width = aligned["points"].shape[2:4]
            self.num_dense_before = len(self.ctx_set) * height * width

        rgb = _rgb_window_to_uint8(imgs_w_raw)
        pred_ctx = slice_predictions(aligned, ctx_local)
        global_ids_ctx = [start_idx + i for i in ctx_local]
        height, width = pred_ctx["points"].shape[2:4]
        device = pred_ctx["points"].device
        sh_dim = int(pred_ctx["gaussian_sh_coeffs"].shape[2])
        complement = WindowMergeTensors.empty(device, sh_dim=sh_dim)
        dense_accumulation = bool(getattr(self.config, "dense_accumulation", False))

        if dense_accumulation:
            # Explicit dense ablation: no track-guided compact, no track coverage.
            # Sky pixels (if sky_mask_drop) stay covered so they never spawn
            # complement; depth filter / sky sphere stay outside this path.
            covered = torch.zeros(
                len(ctx_local), height, width, dtype=torch.bool, device=device
            )
            sky_mask = self._sky_mask_for_ctx(rgb, ctx_local, device=device)
            if sky_mask is not None:
                covered |= sky_mask
            compact = WindowMergeTensors.empty(device, sh_dim=sh_dim)
            contrib_ctx_local = [
                j for j, g in enumerate(global_ids_ctx) if contrib_start <= g < contrib_end
            ]
            if contrib_ctx_local:
                pred_contrib = slice_predictions(pred_ctx, contrib_ctx_local)
                self._append_dense_reg(pred_contrib)
                if self.track_dense_complement:
                    complement = extract_complement_tensors(
                        pred_contrib,
                        covered[contrib_ctx_local],
                        conf_gate=bool(getattr(self.config, "complement_conf_gate", False)),
                        conf_thr=float(getattr(self.config, "complement_conf_thr", 0.0) or 0.0),
                        conf_gate_mode=str(
                            getattr(self.config, "complement_conf_gate_mode", "hard")
                        ),
                    )
            compact, complement = self._deduplicate_completed_window(
                compact, complement, window_idx
            )
            self._commit_window_clouds(
                window_idx=window_idx,
                start_idx=start_idx,
                end_idx=end_idx,
                contrib_start=contrib_start,
                contrib_end=contrib_end,
                compact=compact,
                complement=complement,
            )
            self._store_overlap_tail(aligned, end_idx - start_idx)
            self._maybe_spill_old_windows_to_cpu(window_idx)
            return

        tracks = self._run_tracks_for_window(rgb)
        tracks_ctx = {
            "trajs_uv": tracks["trajs_uv"][ctx_local],
            "vis": tracks["vis"][ctx_local],
        }

        if self.config.merge_mode == "dsga_v0":
            soft_coverage = torch.zeros(
                len(ctx_local), height, width, dtype=pred_ctx["points"].dtype, device=device
            )
            compact = merge_within_window_dsga(
                pred_ctx, tracks_ctx, self.config, soft_coverage, global_ids_ctx
            )

            contrib_ctx_local = [
                j for j, g in enumerate(global_ids_ctx) if contrib_start <= g < contrib_end
            ]
            if contrib_ctx_local:
                pred_contrib = slice_predictions(pred_ctx, contrib_ctx_local)
                self._append_dense_reg(pred_contrib)
                if self.track_dense_complement:
                    from loger.merge.soft_coverage import extract_soft_complement_tensors

                    soft_contrib = soft_coverage[contrib_ctx_local]
                    max_complement = None
                    if self.config.hybrid_budget_ratio > 0:
                        dense_count = len(contrib_ctx_local) * height * width
                        hybrid_budget = round(
                            self.config.hybrid_budget_ratio * dense_count
                        )
                        max_complement = max(
                            0, hybrid_budget - compact.num_gaussians
                        )
                    complement = extract_soft_complement_tensors(
                        pred_contrib,
                        soft_contrib,
                        max_gaussians=max_complement,
                    )
            compact, complement = self._deduplicate_completed_window(
                compact, complement, window_idx
            )
            self._commit_window_clouds(
                window_idx=window_idx,
                start_idx=start_idx,
                end_idx=end_idx,
                contrib_start=contrib_start,
                contrib_end=contrib_end,
                compact=compact,
                complement=complement,
            )
        else:
            refined = self._refine_against_map(
                pred_ctx, aligned, ctx_local, window_idx, height, width
            )
            covered = torch.zeros(len(ctx_local), height, width, dtype=torch.bool, device=device)
            if refined is not None:
                covered |= refined
            low_conf = None
            if bool(getattr(self.config, "premerge_conf_gate", False)):
                low_conf = pred_low_conf_mask(
                    pred_ctx,
                    float(getattr(self.config, "premerge_conf_thr", 0.0) or 0.0),
                )
                if low_conf is not None:
                    covered |= low_conf
            sky_mask = self._sky_mask_for_ctx(rgb, ctx_local, device=device)
            if sky_mask is not None:
                covered |= sky_mask
            suppressed = refined
            if low_conf is not None:
                suppressed = low_conf if suppressed is None else (suppressed | low_conf)
            if sky_mask is not None:
                suppressed = sky_mask if suppressed is None else (suppressed | sky_mask)
            compact = merge_within_window(
                pred_ctx,
                tracks_ctx,
                self.config,
                covered,
                global_ids_ctx,
                track_rd_records=self._track_rd_records if self.track_rd_dump_path else None,
                selector_trace_records=(
                    self._selector_trace_records if self.track_rd_selection_fingerprint else None
                ),
                window_idx=window_idx,
                neural_merger=self.neural_merger,
                suppressed=suppressed,
            )

            contrib_ctx_local = [
                j for j, g in enumerate(global_ids_ctx) if contrib_start <= g < contrib_end
            ]
            if contrib_ctx_local:
                pred_contrib = slice_predictions(pred_ctx, contrib_ctx_local)
                self._append_dense_reg(pred_contrib)
                if self.track_dense_complement:
                    covered_contrib = covered[contrib_ctx_local]
                    drop_lowest_renderable = None
                    if self.config.merge_mode in ("hard_tracklet", "hard_appearance", "hard_flatk", "hard_nested"):
                        # Each split replaces one compact K=1 by two. Remove
                        # exactly that many renderable lowest-opacity residuals.
                        # Failure is explicit if the window cannot pay its rate.
                        extra = compact.num_gaussians - int(compact.base_compact_count or 0)
                        drop_lowest_renderable = extra
                    max_complement = None
                    if self.config.hybrid_budget_ratio > 0:
                        dense_count = len(contrib_ctx_local) * height * width
                        hybrid_budget = round(
                            self.config.hybrid_budget_ratio * dense_count
                        )
                        max_complement = max(
                            0, hybrid_budget - compact.num_gaussians
                        )
                    complement = extract_complement_tensors(
                        pred_contrib, covered_contrib,
                        max_gaussians=max_complement,
                        budget_selection=str(self.config.hybrid_budget_selection),
                        budget_explore_fraction=float(
                            self.config.hybrid_budget_explore_fraction
                        ),
                        budget_sampling_alpha=float(
                            self.config.hybrid_budget_sampling_alpha
                        ),
                        drop_lowest_renderable=drop_lowest_renderable,
                        conf_gate=bool(getattr(self.config, "complement_conf_gate", False)),
                        conf_thr=float(getattr(self.config, "complement_conf_thr", 0.0) or 0.0),
                        conf_gate_mode=str(
                            getattr(self.config, "complement_conf_gate_mode", "hard")
                        ),
                    )
                    if self.track_rd_dump_path:
                        self._track_rd_records[-1]["reference_complement"] = {
                            "means": complement.means.detach().cpu().contiguous(),
                            "quats": complement.quats.detach().cpu().contiguous(),
                            "scales": complement.scales.detach().cpu().contiguous(),
                            "opacities": complement.opacities.detach().cpu().contiguous(),
                            "sh": complement.sh_flat.detach().cpu().contiguous(),
                        }

            compact, complement = self._deduplicate_completed_window(
                compact, complement, window_idx
            )
            self._commit_window_clouds(
                window_idx=window_idx,
                start_idx=start_idx,
                end_idx=end_idx,
                contrib_start=contrib_start,
                contrib_end=contrib_end,
                compact=compact,
                complement=complement,
            )

        self._store_overlap_tail(aligned, end_idx - start_idx)
        self._maybe_spill_old_windows_to_cpu(window_idx)

    def _maybe_spill_old_windows_to_cpu(self, window_idx: int) -> None:
        """Bound GPU-resident merge state to ``max_gpu_merge_windows``."""
        keep = int(getattr(self.config, "max_gpu_merge_windows", 0) or 0)
        if keep <= 0:
            return
        spill_stats = self.cloud.spill_old_chunks_to_cpu(keep)
        # Dense-reg bypass lists are one chunk per processed window with ctx.
        for lst in (self._dense_means, self._dense_scales, self._dense_offsets):
            spill_stats[f"dense_{id(lst)}"] = GlobalCloudAccumulator._spill_list_to_cpu(
                lst, keep
            )
        # Per-window robust depth filtering backups: move older records' tensors to CPU (same storage
        # as cloud chunks when shared; still needed when lists diverge).
        spilled_records = 0
        for rec in self._per_window_records:
            if int(rec.window_idx) <= int(window_idx) - keep:
                if rec.compact.means.device.type != "cpu":
                    rec.compact = window_merge_tensors_to_device(rec.compact, torch.device("cpu"))
                    spilled_records += 1
                if rec.complement.means.device.type != "cpu":
                    rec.complement = window_merge_tensors_to_device(
                        rec.complement, torch.device("cpu")
                    )
        spill_stats["per_window_records_spilled"] = spilled_records
        spill_stats["window_idx"] = int(window_idx)
        self._gpu_spill_events += 1
        self._gpu_spill_last = spill_stats
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _store_overlap_tail(self, aligned: dict, window_len: int) -> None:
        if self.overlap_size <= 0 or window_len <= 0:
            self._prev_overlap_pred = None
            return
        tail = max(window_len - self.overlap_size, 0)
        local_indices = list(range(tail, window_len))
        if not local_indices:
            self._prev_overlap_pred = None
            return
        self._prev_overlap_pred = slice_predictions(aligned, local_indices)

    def finalize(self) -> WindowMergeResult:
        export_only = bool(getattr(self.config, "window_cloud_export_only", False))
        if export_only:
            camera_poses = assemble_camera_poses(self.pose_by_frame, self.bundle_num_frames)
            empty = WindowMergeTensors.empty(torch.device("cpu"))
            total_compact = int(
                sum(int(e["compact"]["num_gaussians"]) for e in self._window_dump_meta)
            )
            total_complement = int(
                sum(int(e["complement"]["num_gaussians"]) for e in self._window_dump_meta)
            )
            stats = {
                "num_gaussians_before": self.num_dense_before,
                "num_gaussians_compact": total_compact,
                "num_gaussians_complement": total_complement,
                "num_gaussians_hybrid": total_compact + total_complement,
                "compression_ratio": (total_compact + total_complement)
                / max(self.num_dense_before, 1),
                "num_windows": self.num_windows,
                "merge_mode": self.config.merge_mode,
                "window_cloud_export_only": True,
                "window_cloud_dump_dir": getattr(self.config, "window_cloud_dump_dir", None),
                "window_dump_meta": list(self._window_dump_meta),
                "max_gpu_merge_windows": int(
                    getattr(self.config, "max_gpu_merge_windows", 0) or 0
                ),
            }
            return WindowMergeResult(
                means=empty.means,
                quats=empty.quats,
                scales=empty.scales,
                opacities=empty.opacities,
                sh_flat=empty.sh_flat,
                camera_poses=camera_poses,
                stats=stats,
            )

        refine_stats: Dict[str, object] = {"applied": False}
        if self.streaming_refine is not None and self.streaming_refine.config.enabled:
            refine_stats = self._splice_refine_deltas()
        voxel_stats: Dict[str, object] = {"applied": False}
        factor = float(getattr(self.config, "global_complement_voxel_factor", 0.0) or 0.0)
        if factor > 0:
            voxel_stats = apply_global_complement_voxel_dedup(
                self.cloud,
                factor=factor,
                policy=str(getattr(self.config, "global_complement_voxel_policy", "first")),
                min_windows=int(getattr(self.config, "global_complement_voxel_min_windows", 0)),
                num_windows=self.num_windows,
                min_complement=int(getattr(self.config, "global_complement_voxel_min_complement", 0)),
            )
        online_voxel_stats: Dict[str, object] = {"applied": False}
        if self.online_voxel_deduper is not None:
            online_voxel_stats = self.online_voxel_deduper.stats()
            online_voxel_stats["windows"] = self._online_voxel_windows
        teacher_dump_path = getattr(self.config, "global_online_teacher_dump_path", None)
        if teacher_dump_path:
            if not self._teacher_records:
                raise RuntimeError("teacher dump requested but no matched candidates were recorded")
            parent = os.path.dirname(os.path.abspath(str(teacher_dump_path)))
            if parent:
                os.makedirs(parent, exist_ok=True)
            keys = tuple(self._teacher_records[0])
            payload = {key: torch.cat([record[key] for record in self._teacher_records], dim=0) for key in keys}
            payload["schema_version"] = 1
            payload["num_records"] = int(payload["slot"].numel())
            torch.save(payload, str(teacher_dump_path))
            online_voxel_stats["teacher_candidate_dump_path"] = str(teacher_dump_path)
            online_voxel_stats["teacher_candidate_records"] = int(payload["num_records"])
        teacher_all_path = getattr(self.config, "global_online_teacher_all_dump_path", None)
        if teacher_all_path:
            if not self._teacher_all_records:
                raise RuntimeError("all-candidate teacher dump requested but no candidates were recorded")
            parent = os.path.dirname(os.path.abspath(str(teacher_all_path)))
            if parent:
                os.makedirs(parent, exist_ok=True)
            keys = tuple(self._teacher_all_records[0])
            payload = {key: torch.cat([record[key] for record in self._teacher_all_records], dim=0) for key in keys}
            payload["schema_version"] = 3
            payload["num_records"] = int(payload["is_strict_match"].numel())
            torch.save(payload, str(teacher_all_path))
            online_voxel_stats["teacher_all_candidate_dump_path"] = str(teacher_all_path)
            online_voxel_stats["teacher_all_candidate_records"] = int(payload["num_records"])

        max_gpu = int(getattr(self.config, "max_gpu_merge_windows", 0) or 0)
        if max_gpu > 0:
            # Spill every chunk to CPU before concatenate so finalize never
            # materializes the full hybrid on CUDA.
            self.cloud.spill_old_chunks_to_cpu(0)
            for lst in (self._dense_means, self._dense_scales, self._dense_offsets):
                GlobalCloudAccumulator._spill_list_to_cpu(lst, 0)
            for rec in self._per_window_records:
                if rec.compact.means.device.type != "cpu":
                    rec.compact = window_merge_tensors_to_device(
                        rec.compact, torch.device("cpu")
                    )
                if rec.complement.means.device.type != "cpu":
                    rec.complement = window_merge_tensors_to_device(
                        rec.complement, torch.device("cpu")
                    )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        global_compact = self.cloud.stacked_compact()
        global_complement = self.cloud.stacked_complement()
        global_compact_window_ids = self.cloud.stacked_compact_window_ids()
        global_complement_window_ids = self.cloud.stacked_complement_window_ids()
        stacked = self.cloud.stacked()
        # When GPU window residency is bounded, keep the finalized global cloud on
        # CPU so robust depth filtering / render can stream slices instead of pinning ~all Gaussians.
        if max_gpu > 0:
            cpu = torch.device("cpu")
            if global_compact.means.device.type != "cpu":
                global_compact = window_merge_tensors_to_device(global_compact, cpu)
            if global_complement.means.device.type != "cpu":
                global_complement = window_merge_tensors_to_device(global_complement, cpu)
            if stacked.means.device.type != "cpu":
                stacked = window_merge_tensors_to_device(stacked, cpu)
            if global_compact_window_ids.numel() and global_compact_window_ids.device.type != "cpu":
                global_compact_window_ids = global_compact_window_ids.to(cpu)
            if (
                global_complement_window_ids.numel()
                and global_complement_window_ids.device.type != "cpu"
            ):
                global_complement_window_ids = global_complement_window_ids.to(cpu)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        camera_poses = assemble_camera_poses(self.pose_by_frame, self.bundle_num_frames)
        total = stacked.num_gaussians

        def _cat_maybe(chunks: List[torch.Tensor]) -> Optional[torch.Tensor]:
            if not chunks:
                return None
            device = next((t.device for t in reversed(chunks) if t.is_cuda), chunks[0].device)
            return torch.cat([t.to(device, non_blocking=False) for t in chunks], dim=0)

        dense_scales = _cat_maybe(self._dense_scales)
        dense_means = _cat_maybe(self._dense_means)
        dense_offsets = _cat_maybe(self._dense_offsets)
        stats = {
            "num_gaussians_before": self.num_dense_before,
            "num_gaussians_compact": self.cloud.num_compact,
            "num_gaussians_complement": self.cloud.num_complement,
            "num_gaussians_hybrid": total,
            "compression_ratio": total / max(self.num_dense_before, 1),
            "num_windows": self.num_windows,
            "num_dense_reg": int(dense_scales.shape[0]) if dense_scales is not None else 0,
            "merge_mode": self.config.merge_mode,
            "dense_accumulation": bool(
                getattr(self.config, "dense_accumulation", False)
            ),
            "max_gpu_merge_windows": int(
                getattr(self.config, "max_gpu_merge_windows", 0) or 0
            ),
            "gpu_spill_events": int(self._gpu_spill_events),
            "gpu_spill_last": dict(self._gpu_spill_last),
            "rotation_mode": self.config.rotation_mode,
            "hybrid_budget_ratio": self.config.hybrid_budget_ratio,
            "hybrid_budget_selection": self.config.hybrid_budget_selection,
            "hybrid_budget_explore_fraction": self.config.hybrid_budget_explore_fraction,
            "hybrid_budget_sampling_alpha": self.config.hybrid_budget_sampling_alpha,
            "budget_topk_kept": self._budget_topk_kept,
            "budget_explore_kept": self._budget_explore_kept,
            "budget_selection_windows": self._budget_selection_windows,
            "budget_selection_fingerprint": (
                hashlib.sha256(
                    "".join(self._budget_selection_digests).encode()
                ).hexdigest()
                if self._budget_selection_digests else None
            ),
            "num_tracklet_splits": self.cloud.num_tracklet_splits,
            "num_tracklet_base_compact": self.cloud.num_tracklet_base_compact,
            "num_flatk_k2": self.cloud.num_flatk_k2,
            "num_flatk_k4": self.cloud.num_flatk_k4,
            "flatk_route_fingerprint": (
                hashlib.sha256("".join(self.cloud.flatk_trace_digests).encode()).hexdigest()
                if self.cloud.flatk_trace_digests else None
            ),
            "num_nested_k2": self.cloud.num_nested_k2,
            "num_nested_k4": self.cloud.num_nested_k4,
            "nested_route_fingerprint": (
                hashlib.sha256("".join(self.cloud.nested_trace_digests).encode()).hexdigest()
                if self.cloud.nested_trace_digests else None
            ),
            "global_complement_voxel": voxel_stats,
            "global_online_voxel": online_voxel_stats,
            "streaming_refine": refine_stats,
            "sky_mask": self.sky_mask_stats(),
        }
        if self.track_rd_dump_path:
            from loger.merge.track_rd import build_track_rd_dump

            parent = os.path.dirname(os.path.abspath(self.track_rd_dump_path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            dump = build_track_rd_dump(
                self._track_rd_records,
                metadata={
                    "bundle_num_frames": self.bundle_num_frames,
                    "ctx_global_indices": sorted(self.ctx_set),
                    "merge_mode": self.config.merge_mode,
                    "rotation_mode": self.config.rotation_mode,
                    "merge_config": asdict(self.config),
                    "reference_counts": {
                        "compact": self.cloud.num_compact,
                        "complement": self.cloud.num_complement,
                        "hybrid": total,
                    },
                },
            )
            torch.save(dump, self.track_rd_dump_path)
            stats["track_rd_dump_path"] = self.track_rd_dump_path
            stats["track_rd_num_windows"] = len(self._track_rd_records)
        if self.track_rd_selection_fingerprint:
            from loger.merge.track_rd import selector_fingerprint

            stats["selector_fingerprint"] = selector_fingerprint(self._selector_trace_records)
            stats["selector_trace_windows"] = len(self._selector_trace_records)
        per_window_results = (
            self.build_per_window_results() if self.record_per_window_clouds else None
        )
        per_window_compact = (
            self.build_per_window_compact_results() if self.record_per_window_clouds else None
        )
        per_window_complement = (
            self.build_per_window_complement_results() if self.record_per_window_clouds else None
        )
        return WindowMergeResult(
            means=stacked.means,
            quats=stacked.quats,
            scales=stacked.scales,
            opacities=stacked.opacities,
            sh_flat=stacked.sh_flat,
            camera_poses=camera_poses,
            stats=stats,
            dense_scales=dense_scales,
            dense_means=dense_means,
            dense_offsets=dense_offsets,
            per_window_results=per_window_results,
            per_window_compact=per_window_compact,
            per_window_complement=per_window_complement,
            global_compact=global_compact,
            global_complement=global_complement,
            global_compact_window_ids=global_compact_window_ids,
            global_complement_window_ids=global_complement_window_ids,
            global_complement_conf=self.cloud.stacked_complement_conf_scores(),
        )
