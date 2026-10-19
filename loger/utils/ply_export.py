"""Standard 3DGS binary PLY export of the final merged scene."""
import numpy as np
from plyfile import PlyData, PlyElement


def export_scene_ply(scene, path):
    arrays = [x.detach().float().cpu().numpy() for x in
              (scene.means, scene.scales, scene.quats, scene.opacities, scene.sh_flat)]
    means, scales, quats, opacity, sh = arrays
    n = len(means)
    if not n or not all(np.isfinite(x).all() for x in arrays):
        raise ValueError("Cannot export empty or non-finite Gaussian scene")
    sh = sh.reshape(n, 3, -1)
    # Standard 3DGS stores the residual SH coefficients channel-major.
    residual = sh[:, :, 1:].reshape(n, -1)
    opacity = np.clip(opacity.reshape(n, 1), 1e-7, 1 - 1e-7)
    values = np.concatenate((means, np.zeros_like(means), sh[:, :, 0], residual,
                             np.log(opacity / (1 - opacity)),
                             np.log(np.maximum(scales, 1e-12)), quats), axis=1)
    names = ["x", "y", "z", "nx", "ny", "nz"]
    names += [f"f_dc_{i}" for i in range(3)]
    names += [f"f_rest_{i}" for i in range(residual.shape[1])]
    names += ["opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    vertices = np.empty(n, dtype=[(name, "<f4") for name in names])
    for i, name in enumerate(names):
        vertices[name] = values[:, i]
    with open(path, "xb") as stream:
        PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(stream)
