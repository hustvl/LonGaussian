"""Checkpoint recomputation must preserve the view and its derivatives."""
import unittest
import torch
from loger.renderer.gaussian_renderer import GaussianRenderer


class RendererCheckpointTests(unittest.TestCase):
    def test_view_chunks_preserve_outputs_and_gradients(self):
        class AnalyticRenderer(GaussianRenderer):
            def render_view(self, means, quats, scales, opacities, colors,
                            camera_pose, H, W, K=None, background=None):
                value = torch.sin(means * camera_pose[0, 0]).sum()
                return {key: value.expand(H, W, n)
                        for key, n in [('rgb', 3), ('depth', 1), ('alpha', 1)]}

        renderer = AnalyticRenderer()
        reference = None
        for chunk in (0, 1, 2):
            with self.subTest(chunk=chunk):
                means = torch.tensor([[0.2, 0.5, 0.7]], requires_grad=True)
                poses = torch.eye(4).repeat(3, 1, 1)
                poses[:, 0, 0] = torch.tensor([1., 2., 3.])
                poses.requires_grad_()
                result = renderer.render_views_from_tensors(
                    means, means, means, means, means, poses, [0, 1, 2],
                    2, 2, checkpoint_chunk_size=chunk)
                sum(value.sum() for value in result.values()).backward()
                observed = [result[k].detach() for k in ('rgb', 'depth', 'alpha')]
                observed += [means.grad, poses.grad]
                if reference is None:
                    reference = observed
                else:
                    for actual, expected in zip(observed, reference):
                        torch.testing.assert_close(actual, expected)
