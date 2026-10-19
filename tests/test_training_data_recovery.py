import ast
import contextlib
import io
import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from typing import List, Tuple, Optional, Sequence
from unittest.mock import patch

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def loader_scope():
    names = {'_load_dl3dv_frames_meta', '_dl3dv_target_size', '_build_dl3dv_K',
             '_load_dl3dv_frame_tensors', 'load_dl3dv_scene_indices', 'load_dl3dv_scene_split',
             '_even_subset', 'dense_every8_holdout_min_span', 'try_load_dl3dv_bucket_split'}
    tree = ast.parse((ROOT / 'run.py').read_text())
    scope = dict(globals())
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                                 and n.name in names], type_ignores=[]), str(ROOT / 'run.py'), 'exec'), scope)
    return scope


class TrainingDataRecoveryTests(unittest.TestCase):
    def make_scene(self, root, corrupt):
        (root / 'images_8').mkdir()
        frames = []
        for i in range(180):
            name = f'{i:04d}.png'
            path = root / 'images_8' / name
            if i in corrupt:
                path.write_bytes(b'nonempty but broken PNG')
            else:
                Image.new('RGB', (14, 14), (i % 255, 30, 40)).save(path)
            frames.append({'file_path': 'images/' + name, 'transform_matrix': torch.eye(4).tolist()})
        (root / 'transforms.json').write_text(json.dumps({'w':14,'h':14,'fl_x':10,'fl_y':10,
                                                       'cx':7,'cy':7,'frames':frames}))

    def test_same_scene_recovery_preserves_dense_and_sparse_topology(self):
        scope = loader_scope()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_scene(root, {0})
            for context, interval in ((16,1), (16,8), (64,1), (96,1), (128,1)):
                with patch.object(random, 'randint', return_value=0), contextlib.redirect_stdout(io.StringIO()):
                    result = scope['try_load_dl3dv_bucket_split'](str(root), context, interval, 3, [14,14], 1)
                self.assertIsNotNone(result, 'A bad sampled frame must not discard an otherwise usable scene')
                images, _, _, _, _, ctx, holdout = result
                self.assertEqual(len(ctx), context)
                self.assertTrue(set(ctx).isdisjoint(holdout))
                self.assertNotIn(0, ctx + holdout)
                self.assertEqual(images.shape[0], len(ctx) + len(holdout))
                if interval > 1:
                    self.assertTrue(all(b-a == interval for a,b in zip(ctx,ctx[1:])))
                else:
                    self.assertTrue(all((x-ctx[0]) % 8 != 7 for x in ctx))

    def test_fully_broken_scene_still_fails(self):
        scope = loader_scope()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_scene(root, set(range(180)))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = scope['try_load_dl3dv_bucket_split'](str(root),16,1,3,[14,14],1)
            self.assertIsNone(result)
            self.assertIn('0 valid starts', output.getvalue())
            self.assertIn('refusing partial update', output.getvalue())

    def test_clean_scene_keeps_original_sampling(self):
        scope = loader_scope()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_scene(root, set())
            with patch.object(random, 'randint', return_value=0), patch(
                'loger.utils.training_frames.choose_decodable_start',
                side_effect=AssertionError('Clean scene must not invoke recovery'),
            ):
                result = scope['try_load_dl3dv_bucket_split'](str(root),16,1,3,[14,14],1)
            self.assertEqual(result[5], [i for i in range(18) if i not in (7,15)])
            self.assertEqual(result[6], [7,15])

    def test_recovery_is_seeded_and_does_not_modify_data(self):
        scope = loader_scope()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_scene(root, {0})
            chosen = []
            before = (root/'images_8/0000.png').read_bytes()
            for _ in range(2):
                random.seed(12)
                with patch.object(random, 'randint', return_value=0), contextlib.redirect_stdout(io.StringIO()):
                    result = scope['try_load_dl3dv_bucket_split'](str(root),16,1,3,[14,14],1)
                chosen.append(result[5:])
            self.assertEqual(chosen[0], chosen[1])
            self.assertEqual(before, (root/'images_8/0000.png').read_bytes())
