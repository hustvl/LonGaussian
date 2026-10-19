import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from scripts import download_re10k_train_long as re10k
from scripts.evaluate_main_table import packed_split, validate_metrics, validate_root


class MainTableTests(unittest.TestCase):
    def test_fixed_splits(self):
        for count, frames, targets in ((64, 73, 9), (96, 109, 13), (128, 146, 18)):
            ctx, tgt = packed_split(count)
            self.assertEqual((len(ctx), len(tgt)), (count, targets))
            self.assertEqual(len(ctx) + len(tgt), frames)
            self.assertEqual(tgt, list(range(7, frames, 8)))

    def test_root_and_metrics_reject_wrong_split_or_nonfinite_view(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scene = root / "scene-a"
            scene.mkdir()
            (scene / "images_8").mkdir()
            (root / "test_index.json").write_text('["scene-a"]')
            ctx, tgt = packed_split(64)
            (scene / "split.json").write_text(json.dumps({"ctx_list": ctx, "tgt_list": tgt}))
            (scene / "transforms.json").write_text(json.dumps({"frames": [{}] * 73}))
            validate_root(root, "dl3dv", 64, ["scene-a"])
            views = [{"target_index": i, "psnr": 20.0, "ssim": 0.6, "lpips": 0.3} for i in tgt]
            metrics = {"summary": {"num_scenes": 1, "num_test_views": 9,
                       "psnr_mean": 20.0, "ssim_mean": 0.6, "lpips_mean": 0.3},
                       "scenes": [{"scene": "scene-a", "tgt_list": tgt, "views": views,
                                   "psnr_mean": 20.0, "ssim_mean": 0.6, "lpips_mean": 0.3}]}
            result = root / "metrics.json"
            result.write_text(json.dumps(metrics))
            self.assertEqual(validate_metrics(result, "dl3dv", 64, ["scene-a"])["num_test_views"], 9)
            with self.assertRaisesRegex(ValueError, "gaussian_head differs"):
                validate_metrics(result, "dl3dv", 64, ["scene-a"],
                                 expected_sources={"gaussian_head": root / "stage2b.pt"})
            metrics["scenes"][0]["views"][0]["psnr"] = float("nan")
            result.write_text(json.dumps(metrics))
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                validate_metrics(result, "dl3dv", 64, ["scene-a"])
            (scene / "split.json").write_text(json.dumps({"ctx_list": ctx, "tgt_list": tgt[:-1]}))
            with self.assertRaisesRegex(ValueError, "invalid split"):
                validate_root(root, "dl3dv", 64, ["scene-a"])


class Re10kDownloadTests(unittest.TestCase):
    def test_timestamp_crop_and_camera_record(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "output"
            output.mkdir()
            source = root / "camera.txt"
            pose = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
            rows = [[time, 0.8, 0.8, 0.5, 0.5, 0, 0, *pose] for time in (0, 40000, 80000)]
            source.write_text("https://example.invalid/video\n" + "\n".join(
                " ".join(str(value) for value in row) for row in rows
            ) + "\n")
            frames = [np.full((360, 640, 3), index * 30, np.uint8) for index in range(3)]

            def fake_download(_url, output_template, *_args):
                path = output_template.parent / "source.mp4"
                path.write_bytes(b"video")
                return path

            with mock.patch.object(re10k, "download_segment", side_effect=fake_download), mock.patch.object(
                re10k, "decode_segment", return_value=(frames, [0.0, 40.0, 80.0])
            ):
                result = re10k.materialize_scene(
                    {"key": "scene-a", "metadata_path": str(source), "split": "official_test"},
                    output, root / "tmp", frame_count=3, target_width=490, target_height=280,
                    tool_root=root, ffmpeg_dir=root, deno=root / "deno", tolerance_ms=1.0,
                    download_timeout_sec=1.0,
                )
            self.assertEqual(result["max_timestamp_error_ms"], 0.0)
            metadata = json.loads((output / "scene-a/transforms.json").read_text())
            self.assertEqual((metadata["w"], metadata["h"]), (490, 280))
            self.assertEqual([frame["timestamp"] for frame in metadata["frames"]], [0, 40000, 80000])
            self.assertEqual(metadata["re10k_split"], "official_test")
            with Image.open(output / "scene-a/images_8/000000.jpg") as image:
                self.assertEqual(image.size, (490, 280))

    def test_fixed_test_video_failure_is_recorded_not_replaced(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            selection = root / "selection.json"
            selection.write_text(json.dumps({"split": "official_test", "primary": [
                {"key": "fixed-scene", "metadata_path": "unused.txt"}], "reserve": []}))
            ffmpeg = root / "ffmpeg"
            ffmpeg.mkdir()
            deno = root / "deno"
            deno.write_text("")
            args = ["download_re10k_train_long.py", "--selection", str(selection),
                    "--output-root", str(root / "output"), "--target-scenes", "1",
                    "--min-free-gb", "0", "--ffmpeg-dir", str(ffmpeg), "--deno", str(deno)]
            with mock.patch.object(sys, "argv", args), mock.patch.object(
                re10k, "materialize_scene", side_effect=RuntimeError("unavailable video")
            ):
                with self.assertRaises(SystemExit) as caught:
                    re10k.main()
            self.assertEqual(caught.exception.code, 2)
            state = json.loads((root / "output/download_state.json").read_text())
            self.assertEqual(state["completed"], [])
            self.assertFalse(state["complete"])
            self.assertEqual(state["failures"][0]["key"], "fixed-scene")
            self.assertFalse((root / "output/test_index.json").exists())


if __name__ == "__main__":
    unittest.main()
