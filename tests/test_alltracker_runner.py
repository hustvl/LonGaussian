"""Keep the published AllTracker inference subset self-contained."""

import unittest

import numpy as np
import torch

from loger.tracking.alltracker_runner import AllTrackerRunner


class _ZeroFlowTracker:
    def forward_sliding(self, rgb, **_kwargs):
        batch, frames, _channels, height, width = rgb.shape
        zeros = torch.zeros((batch, frames, 2, height, width), device=rgb.device)
        return zeros, zeros, None, None


class AllTrackerRunnerTests(unittest.TestCase):
    def test_forward_maps_uses_pixel_xy_grid_without_unbundled_modules(self):
        rgb = np.zeros((2, 2, 3, 3), dtype=np.uint8)
        runner = AllTrackerRunner(device="cpu")
        runner._load_model = lambda: _ZeroFlowTracker()

        tracks, confidence = runner._forward_maps(rgb)

        self.assertEqual(tracks.shape, (2, 2, 3, 2))
        self.assertEqual(confidence.shape, (2, 2, 3, 2))
        np.testing.assert_array_equal(tracks[0, 0, :, 0], [0, 1, 2])
        np.testing.assert_array_equal(tracks[0, :, 0, 1], [0, 1])
        np.testing.assert_array_equal(tracks[1], tracks[0])


if __name__ == "__main__":
    unittest.main()
