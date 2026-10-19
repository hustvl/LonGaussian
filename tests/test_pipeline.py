import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.prepare_training_data import main, read_ids
from scripts.train_pipeline import build_command, validate_roots, verify_checkpoint
from scripts.download_assets import safe_member


def scene(root, key, url="", split=""):
    folder = root / key
    (folder / "images_8").mkdir(parents=True)
    frames = []
    for i in range(146):
        name = f"{i:06d}.jpg"
        (folder / "images_8" / name).write_bytes(b"fixture")
        frames.append({"file_path": "images/" + name,
                       "transform_matrix": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]})
    (folder / "transforms.json").write_text(json.dumps({"w":490,"h":280,"fl_x":300,"fl_y":300,
        "cx":245,"cy":140,"frames":frames,"re10k_url":url,"re10k_split":split}))


class PipelineTests(unittest.TestCase):
    def test_splits_and_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            for name in ("metadata/train", "metadata/test", "dl", "re"):
                (p/name).mkdir(parents=True)
            (p/"metadata/test/test.txt").write_text("https://example.test/test\n")
            for key in ("a", "b", "heldout"):
                scene(p/"dl", key)
            for key in ("c", "d"):
                url = "https://example.test/" + key
                scene(p/"re", key, url, "official_train")
                (p/"metadata/train"/(key+".txt")).write_text(url+"\n")
            (p/"test.json").write_text('["heldout"]')
            argv = ["prepare", "--dl3dv-root",str(p/"dl"),"--re10k-root",str(p/"re"),
                    "--re10k-metadata",str(p/"metadata"),"--dl3dv-test-index",str(p/"test.json"),
                    "--output",str(p/"out"),"--smoke"]
            with patch("sys.argv", argv):
                main()
            validate_roots(p/"out", True)
            self.assertEqual(set(read_ids(p/"out/stage1/train_index.json")), {"a","b"})
            (p/"out/stage1/train_index.json").write_text('["heldout","a"]')
            with self.assertRaises(ValueError):
                validate_roots(p/"out", True)

    def test_paper_recipe_and_smoke_cannot_be_confused(self):
        for stage, steps in (("stage1",2148),("stage2a",900),("stage2b",300)):
            cmd, actual = build_command(stage, Path("data"),Path("weights"),Path("out"),"0,1,2,3")
            self.assertEqual(actual, steps)
            self.assertEqual(cmd[cmd.index("--scenes_per_epoch")+1],str(steps*4))
            if stage == "stage2b":
                self.assertIn("--sky_mask_onnx",cmd)
                self.assertIn("--balanced_train_root_b",cmd)
        with self.assertRaises(ValueError):
            build_command("stage1",Path("d"),Path("w"),Path("o"),"0,1",False)

    def test_incomplete_stage_is_rejected(self):
        import torch
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"stage.pt"
            torch.save({"global_step":0,"model":{}},path)
            with self.assertRaisesRegex(RuntimeError,"Incomplete"):
                verify_checkpoint(path,1)

    def test_activation_offload_only_changes_memory_policy(self):
        for stage in ("stage1", "stage2a", "stage2b"):
            args = (stage, Path("d"), Path("w"), Path("o"), "0,1")
            enabled, steps = build_command(*args, accumulation=2)
            disabled, other_steps = build_command(*args, accumulation=2,
                                                   activation_offload=False)
            self.assertEqual(steps, other_steps)
            self.assertEqual("--train_save_activations_on_cpu" in enabled,
                             stage != "stage1")
            self.assertEqual([x for x in enabled if x != "--train_save_activations_on_cpu"],
                             disabled)

    def test_two_gpu_accumulated_recipe(self):
        for stage, steps in (("stage1", 2148), ("stage2a", 900), ("stage2b", 300)):
            cmd, actual = build_command(stage, Path("d"), Path("w"), Path("o"), "0,1", accumulation=2)
            self.assertEqual(actual, steps)
            self.assertEqual(cmd[cmd.index("--scenes_per_epoch") + 1], str(steps * 4))
            self.assertEqual(cmd[cmd.index("--gradient_accumulation_steps") + 1], "2")
        with self.assertRaises(ValueError):
            build_command("stage1", Path("d"), Path("w"), Path("o"), "0,1", accumulation=1)

    def test_accumulated_gradient_matches_global_batch(self):
        import copy
        import torch
        torch.manual_seed(7)
        full = torch.nn.Linear(3, 2).double()
        accumulated = copy.deepcopy(full)
        opt_full = torch.optim.AdamW(full.parameters(), lr=1e-5)
        opt_acc = torch.optim.AdamW(accumulated.parameters(), lr=1e-5)
        x, y = torch.randn(4, 3).double(), torch.randn(4, 2).double()
        for _ in range(3):
            opt_full.zero_grad(set_to_none=True)
            opt_acc.zero_grad(set_to_none=True)
            (full(x) - y).square().mean().backward()
            # Two ranks' average on each of two microsteps.
            for micro in range(2):
                for rank in range(2):
                    i = 2 * micro + rank
                    ((accumulated(x[i:i+1]) - y[i:i+1]).square().mean() / 4).backward()
            for a, b in zip(full.parameters(), accumulated.parameters()):
                torch.testing.assert_close(a.grad, b.grad, rtol=1e-12, atol=1e-12)
            torch.nn.utils.clip_grad_norm_(full.parameters(), 1)
            torch.nn.utils.clip_grad_norm_(accumulated.parameters(), 1)
            opt_full.step()
            opt_acc.step()
            for a, b in zip(full.parameters(), accumulated.parameters()):
                torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)

    def test_archive_traversal_is_rejected(self):
        for name in ("../escape","/absolute","a/../../x","a\\b"):
            with self.assertRaises(ValueError):
                safe_member(name)
