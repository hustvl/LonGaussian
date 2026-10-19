#!/usr/bin/env python
"""Inference script: predict gaussians and save as .ply point cloud.

Usage:
    # Single image folder
    python infer.py \
        --input data/examples/office \
        --output results/scene.ply

    # With trained gaussian head
    python infer.py \
        --input data/examples/office \
        --output results/scene.ply \
        --gaussian_head training_output/gaussian_head.pt

    # Dataset scene
    python infer.py \
        --input /path/to/dataset/scene/images \
        --output results/scene.ply \
        --num_frames 30 \
        --neighbor_frames 2
"""

import argparse
import os
import sys
import glob
import numpy as np
import torch
from natsort import natsorted
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(__file__))


def extract_model_state(checkpoint):
    """Return a tensor state dict from current or legacy training checkpoints."""
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint does not contain a model state dictionary")
    state = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    if not isinstance(state, dict) or not state or not any(
        isinstance(value, torch.Tensor) for value in state.values()
    ):
        raise ValueError("checkpoint does not contain a usable model state dictionary")
    if all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


def load_trained_state(model, checkpoint_path, device):
    """Require complete Gaussian decoder/head parameters."""
    from loger.utils.checkpoint import load_gaussian_weights
    return load_gaussian_weights(model, checkpoint_path, require_merger=False)


def unpack_dl3dv_scene(result):
    """Validate the loader contract instead of silently unpacking the wrong shape."""
    if not isinstance(result, (tuple, list)) or len(result) != 3:
        raise ValueError("load_dl3dv_scene must return three values: images, poses, intrinsics")
    return result[0], result[1], result[2]


def load_images_from_paths(image_paths, target_w=None, target_h=None):
    """Load and resize images to a tensor."""
    if not image_paths:
        raise ValueError("Input directory contains no PNG/JPEG images")
    if target_w is None or target_h is None:
        with Image.open(image_paths[0]) as img:
            W, H = img.size
        import math
        pixel_limit = 255000
        scale = math.sqrt(pixel_limit / (W * H))
        target_w = int(max(1, round(W * scale / 14)) * 14)
        target_h = int(max(1, round(H * scale / 14)) * 14)

    out = torch.empty((len(image_paths), 3, target_h, target_w), dtype=torch.float32)
    to_tensor = transforms.ToTensor()
    for i, p in enumerate(image_paths):
        img = Image.open(p).convert("RGB").resize((target_w, target_h), Image.Resampling.LANCZOS)
        out[i] = to_tensor(img)
    return out


def export_ply(means, scales, quats, opacities, sh_coeffs, sh_degree, path):
    """Write gaussians to standard 3DGS .ply format."""
    N = means.shape[0]
    K = (sh_degree + 1) ** 2
    rest_count = max(0, K - 1)

    sh = sh_coeffs.reshape(N, 3, K).transpose(0, 2, 1)
    dc = sh[:, 0]          # (N, 3)
    rest = sh[:, 1:]       # (N, K-1, 3)

    opa_inv = np.log(np.clip(opacities, 1e-7, 1 - 1e-7) / (1 - np.clip(opacities, 1e-7, 1 - 1e-7)))

    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property float nx\nproperty float ny\nproperty float nz\n")
        for i in range(3):
            f.write(f"property float f_dc_{i}\n")
        for i in range(rest_count * 3):
            f.write(f"property float f_rest_{i}\n")
        f.write("property float opacity\n")
        f.write("property float scale_0\nproperty float scale_1\nproperty float scale_2\n")
        f.write("property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n")
        f.write("end_header\n")
        for i in range(N):
            f.write(f"{means[i,0]:.6f} {means[i,1]:.6f} {means[i,2]:.6f} 0 0 0 ")
            for c in range(3):
                f.write(f"{dc[i,c]:.6f} ")
            for j in range(rest_count):
                for c in range(3):
                    f.write(f"{rest[i,j,c]:.6f} ")
            f.write(f"{opa_inv[i]:.6f} ")
            f.write(f"{scales[i,0]:.6f} {scales[i,1]:.6f} {scales[i,2]:.6f} ")
            f.write(f"{quats[i,0]:.6f} {quats[i,1]:.6f} {quats[i,2]:.6f} {quats[i,3]:.6f}\n")
    print(f"Saved {N} gaussians to {path}")


def main():
    parser = argparse.ArgumentParser(description="Pi3 gaussian inference")
    parser.add_argument("--input", required=True, help="Image folder or DL3DV scene path")
    parser.add_argument("--output", default="results/scene.ply", help="Output .ply path")
    parser.add_argument("--model_name", default="ckpts/LoGeR/latest.pt")
    parser.add_argument("--config", default="ckpts/LoGeR/original_config.yaml")
    parser.add_argument("--gaussian_head", default=None, help="Trained gaussian_head.pt")
    parser.add_argument("--num_frames", type=int, default=-1, help="Frames to process (-1 = all)")
    parser.add_argument("--window_size", type=int, default=12, help="Window size (-1 = all frames)")
    parser.add_argument("--overlap_size", type=int, default=3, help="Window overlap")
    parser.add_argument("--opacity_filter", type=float, default=0.0, help="Min opacity to include")
    parser.add_argument("--max_gaussians", type=int, default=5_000_000, help="Max gaussians in output")
    parser.add_argument("--sh_degree", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from run import load_pi3_model, model_sliding_window_kwargs

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- Load model ----
    model = load_pi3_model(args.model_name, args.config, gaussian=True)
    if model is None:
        print("Failed to load model."); return
    model = model.to(device).eval()

    if args.gaussian_head:
        print(f"Loading gaussian head: {args.gaussian_head}")
        load_trained_state(model, args.gaussian_head, device)

    # ---- Load images ----
    is_dl3dv = os.path.isdir(os.path.join(args.input, "images_8"))
    if is_dl3dv:
        from run import load_dl3dv_scene
        import json
        with open(os.path.join(args.input, "transforms.json")) as f:
            meta = json.load(f)
        total_frames = len(meta["frames"])
        nf = total_frames if args.num_frames <= 0 else min(args.num_frames, total_frames)
        result = load_dl3dv_scene(args.input, nf, image_scale=1.0)
        if result is None:
            print("Failed to load scene."); return
        images, _, _ = unpack_dl3dv_scene(result)
    else:
        paths = natsorted(glob.glob(os.path.join(args.input, "*.png")) +
                          glob.glob(os.path.join(args.input, "*.jpg")) +
                          glob.glob(os.path.join(args.input, "*.jpeg")))
        if args.num_frames > 0:
            paths = paths[:args.num_frames]
        images = load_images_from_paths(paths)

    images = images.to(device)
    N = images.shape[0]
    ws = N if args.window_size <= 0 else min(args.window_size, N)
    ov = min(args.overlap_size, max(N // 10, 1))
    print(f"Images: {images.shape}, window_size={ws}, overlap={ov}")

    # ---- Forward ----
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    with torch.no_grad(), torch.amp.autocast(
        device_type="cuda", enabled=(device == "cuda"), dtype=dtype
    ):
        pred = model(
            images[None], gaussian=True,
            **model_sliding_window_kwargs(N, ws, ov),
        )

    # ---- Collect gaussians from all frames ----
    all_means, all_scales, all_quats, all_opacs, all_sh = [], [], [], [], []
    mean_key = "gaussian_means" if "gaussian_means" in pred else "points"

    for i in range(pred["points"].shape[1]):
        p = pred[mean_key][0, i]
        s = pred["gaussian_scales"][0, i]
        r = pred["gaussian_rotations"][0, i]
        o = pred["gaussian_opacities"][0, i]
        sh = pred["gaussian_sh_coeffs"][0, i]

        mask = o.reshape(-1) > args.opacity_filter
        H, W = p.shape[:2]

        all_means.append(p.reshape(-1, 3)[mask].cpu().numpy())
        all_scales.append(s.permute(1, 2, 0).reshape(-1, 3)[mask].cpu().numpy())
        all_quats.append(r.permute(1, 2, 0).reshape(-1, 4)[mask].cpu().numpy())
        all_opacs.append(o.reshape(-1)[mask].cpu().numpy())
        all_sh.append(sh.permute(1, 2, 0).reshape(-1, 27)[mask].cpu().numpy())

    means_np = np.concatenate(all_means, axis=0)
    scales_np = np.concatenate(all_scales, axis=0)
    quats_np = np.concatenate(all_quats, axis=0)
    opacs_np = np.concatenate(all_opacs, axis=0)
    sh_np = np.concatenate(all_sh, axis=0)
    print(f"Total gaussians: {means_np.shape[0]}")

    # --- Print gaussian parameter stats ---
    s_flat = scales_np.reshape(-1)
    o_flat = opacs_np.reshape(-1)
    k = min(1000, means_np.shape[0])
    idx = np.random.choice(means_np.shape[0], k, replace=False)
    pts_sample = means_np[idx]
    s_sample = np.linalg.norm(scales_np[idx], axis=1)
    nn_dists = np.sort(np.linalg.norm(pts_sample[:, None] - pts_sample[None, :], axis=-1), axis=1)[:, 1]
    ratios = s_sample / (nn_dists + 1e-8)
    print(f"  scales:   p50={np.percentile(s_flat, 50):.4f}  "
          f"p05={np.percentile(s_flat, 5):.4f}  p95={np.percentile(s_flat, 95):.4f}")
    print(f"  opacity:  p50={np.percentile(o_flat, 50):.4f}  "
          f"p95={np.percentile(o_flat, 95):.4f}")
    print(f"  pts nn:   p50={np.median(nn_dists):.4f}  "
          f"p95={np.percentile(nn_dists, 95):.4f}")
    print(f"  scale/nn ratio: p50={np.median(ratios):.4f}  p95={np.percentile(ratios, 95):.4f}")

    # Subsample: weighted by opacity so visible gaussians are prioritized
    if means_np.shape[0] > args.max_gaussians:
        weights = opacs_np.astype(np.float64)
        weights /= weights.sum()
        idx = np.random.choice(means_np.shape[0], args.max_gaussians,
                               replace=False, p=weights)
        means_np, scales_np, quats_np, opacs_np, sh_np = (
            means_np[idx], scales_np[idx], quats_np[idx], opacs_np[idx], sh_np[idx])
        print(f"Subsampled {means_np.shape[0]} → {args.max_gaussians} (weighted by opacity)")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    export_ply(means_np, scales_np, quats_np, opacs_np, sh_np, args.sh_degree, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
