import os
import pathlib
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class ScriptEnvironmentTests(unittest.TestCase):
    def test_evaluation_enables_cublas_reproducibility(self):
        script = (ROOT / "scripts" / "evaluate_dl3dv140.sh").read_text()
        self.assertIn("CUBLAS_WORKSPACE_CONFIG", script)
        self.assertIn("LOGER_DISABLE_TORCH_COMPILE", script)

    def test_evaluation_sets_memory_allocator_before_python(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            for name in ("base.pt", "stage2.pt", "sky.onnx"):
                (root / name).touch()
            (root / "dataset").mkdir()
            fake_python = root / "fake_python.sh"
            fake_python.write_text(
                '#!/usr/bin/env bash\nprintf "%s" "${PYTORCH_CUDA_ALLOC_CONF:-}" > "$CAPTURE"\n'
            )
            fake_python.chmod(0o755)
            env = dict(os.environ)
            env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
            env.update({
                "PYTHON": str(fake_python),
                "MODEL_NAME": str(root / "base.pt"),
                "GAUSSIAN_HEAD": str(root / "stage2.pt"),
                "SKY_MASK_ONNX": str(root / "sky.onnx"),
                "DATASET_ROOT": str(root / "dataset"),
                "OUTPUT_DIR": str(root / "output"),
                "CAPTURE": str(root / "capture.txt"),
            })
            subprocess.run(
                ["bash", str(ROOT / "scripts" / "evaluate_dl3dv140.sh")],
                env=env, check=True, capture_output=True, text=True,
            )
            self.assertEqual((root / "capture.txt").read_text(), "expandable_segments:True")


if __name__ == "__main__":
    unittest.main()
