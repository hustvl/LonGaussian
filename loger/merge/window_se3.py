"""Per-window SE3 alignment for in-forward track merge."""

from __future__ import annotations

import torch

from loger.utils.gaussian_transform import apply_sim3_to_gaussian_predictions
from loger.utils.geometry import homogenize_points
from loger.utils.overlap_se3 import estimate_overlap_se3


def align_window_to_prev_se3(
    pred: dict,
    prev_aligned: dict,
    overlap_size: int,
) -> dict:
    """SE3-align ``pred`` to the world frame established by ``prev_aligned``."""
    sample = pred.get("points")
    if sample is None:
        sample = pred.get("camera_poses")
    if sample is None:
        return pred

    device = sample.device
    dtype = sample.dtype
    batch_size = sample.shape[0]
    identity_rot = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
    zero_trans = torch.zeros(batch_size, 3, device=device, dtype=dtype)
    one_scale = torch.ones(batch_size, device=device, dtype=dtype)

    if overlap_size <= 0:
        current_rot = identity_rot
        current_trans = zero_trans
        current_scale = one_scale
    else:
        prev_cam = prev_aligned.get("camera_poses")
        curr_cam = pred.get("camera_poses")
        if prev_cam is None or curr_cam is None or prev_cam.shape[1] == 0 or curr_cam.shape[1] == 0:
            current_rot = identity_rot
            current_trans = zero_trans
            current_scale = one_scale
        else:
            actual_overlap = min(overlap_size, prev_cam.shape[1], curr_cam.shape[1])
            prev_pose = prev_cam[:, prev_cam.shape[1] - actual_overlap:]
            curr_pose = curr_cam[:, :actual_overlap]
            current_rot, current_trans = estimate_overlap_se3(prev_pose, curr_pose)
            current_scale = one_scale

    aligned_pred: dict = {}
    original_local_points = pred.get("local_points")
    aligned_local_points = original_local_points
    aligned_pred["local_points"] = aligned_local_points

    def _transform_camera(cam_tensor):
        if cam_tensor is None:
            return None
        frames = cam_tensor.shape[1]
        rot_local = cam_tensor[..., :3, :3]
        trans_local = cam_tensor[..., :3, 3]
        rot_global = torch.matmul(
            current_rot.unsqueeze(1).expand(-1, frames, -1, -1),
            rot_local,
        )
        rotated_trans = torch.matmul(
            current_rot.unsqueeze(1).expand(-1, frames, -1, -1),
            trans_local.unsqueeze(-1),
        ).squeeze(-1)
        trans_global = rotated_trans + current_trans.unsqueeze(1)
        cam_out = cam_tensor.clone()
        cam_out[..., :3, :3] = rot_global
        cam_out[..., :3, 3] = trans_global
        return cam_out

    camera_global = _transform_camera(pred.get("camera_poses", None))
    aligned_pred["camera_poses"] = camera_global
    aligned_pred["local_camera_poses"] = _transform_camera(pred.get("local_camera_poses", None))

    if camera_global is not None and aligned_local_points is not None:
        aligned_points = torch.einsum(
            "bnij, bnhwj -> bnhwi",
            camera_global,
            homogenize_points(aligned_local_points),
        )[..., :3]
    else:
        points = pred.get("points", None)
        if points is not None:
            rotated_points = torch.einsum("bij, bnhwj -> bnhwi", current_rot, points)
            aligned_points = rotated_points + current_trans.view(batch_size, 1, 1, 1, 3)
        else:
            aligned_points = None
    aligned_pred["points"] = aligned_points

    aligned_pred["conf"] = pred.get("conf", None)
    gaussian_keys = (
        "gaussian_means",
        "gaussian_offsets",
        "gaussian_rotations",
        "gaussian_scales",
        "gaussian_opacities",
        "gaussian_sh_coeffs",
    )
    if any(pred.get(key) is not None for key in gaussian_keys):
        aligned_pred.update(
            apply_sim3_to_gaussian_predictions(
                pred,
                current_rot,
                current_trans,
                current_scale,
                allow_scale=False,
                aligned_points=aligned_points,
            )
        )

    for key, value in pred.items():
        if key in aligned_pred:
            continue
        aligned_pred[key] = value

    return aligned_pred
