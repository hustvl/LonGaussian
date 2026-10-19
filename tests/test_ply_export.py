import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from plyfile import PlyData
from loger.utils.ply_export import export_scene_ply


class PlyExportTests(unittest.TestCase):
    def test_standard_scale_opacity_and_channel_major_sh(self):
        scene = SimpleNamespace(means=torch.zeros(1, 3), scales=torch.full((1, 3), 2.),
            quats=torch.tensor([[1., 0., 0., 0.]]), opacities=torch.tensor([0.5]),
            sh_flat=torch.arange(27.).reshape(1, 27))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.ply"
            export_scene_ply(scene, path)
            row = PlyData.read(path)["vertex"][0]
            self.assertAlmostEqual(float(row["scale_0"]), float(np.log(2)), places=6)
            self.assertEqual(float(row["opacity"]), 0.)
            self.assertEqual(float(row["f_dc_1"]), 9.)
            self.assertEqual(float(row["f_rest_8"]), 10.)
            with self.assertRaises(FileExistsError):
                export_scene_ply(scene, path)
