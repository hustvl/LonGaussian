import tempfile
import unittest
from pathlib import Path
import torch
from loger.utils.checkpoint import load_gaussian_weights


class CheckpointContract(unittest.TestCase):
    def test_complete_partial_and_shape_mismatch(self):
        model = torch.nn.Module()
        for name in ("gaussian_decoder", "gaussian_head", "neural_track_merger"):
            setattr(model, name, torch.nn.Linear(2, 2))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            state = {key: torch.ones_like(value) for key, value in model.state_dict().items()}
            torch.save({"model": state}, path)
            load_gaussian_weights(model, path)
            self.assertTrue(torch.equal(model.gaussian_head.weight, torch.ones(2, 2)))
            del state["neural_track_merger.weight"]
            torch.save({"model": state}, path)
            with self.assertRaisesRegex(ValueError, "Incomplete neural_track_merger"):
                load_gaussian_weights(model, path)
            state["neural_track_merger.weight"] = torch.ones(3, 2)
            torch.save({"model": state}, path)
            with self.assertRaisesRegex(ValueError, "shape/type"):
                load_gaussian_weights(model, path)
