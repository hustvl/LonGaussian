import unittest
import pathlib

import torch

from infer import extract_model_state, unpack_dl3dv_scene


class InferLoadingTests(unittest.TestCase):
    def test_inference_uses_current_autocast_api(self):
        source = (pathlib.Path(__file__).resolve().parents[1] / "infer.py").read_text()
        self.assertIn("torch.amp.autocast", source)
        self.assertNotIn("torch.cuda.amp.autocast", source)

    def test_extracts_model_state_from_training_checkpoint(self):
        weight = torch.tensor([1.0])
        state = extract_model_state({"model": {"gaussian_head.weight": weight}, "optimizer": {}})
        self.assertEqual(list(state), ["gaussian_head.weight"])
        self.assertIs(state["gaussian_head.weight"], weight)

    def test_accepts_legacy_flat_state_dict(self):
        weight = torch.tensor([2.0])
        state = extract_model_state({"gaussian_head.weight": weight})
        self.assertIs(state["gaussian_head.weight"], weight)

    def test_rejects_checkpoint_without_tensor_state(self):
        with self.assertRaisesRegex(ValueError, "model state"):
            extract_model_state({"optimizer": {"state": {}}})

    def test_unpacks_current_dl3dv_loader_result(self):
        images, poses, intrinsics = object(), object(), object()
        self.assertEqual(
            unpack_dl3dv_scene((images, poses, intrinsics)),
            (images, poses, intrinsics),
        )

    def test_rejects_changed_dl3dv_loader_contract(self):
        with self.assertRaisesRegex(ValueError, "three values"):
            unpack_dl3dv_scene((object(), object()))


if __name__ == "__main__":
    unittest.main()
