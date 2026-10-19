"""CPU-only recovery of valid temporal windows in partially corrupt scenes."""
from pathlib import Path
from PIL import Image


def choose_decodable_start(scene_path, frames, offsets, span, rng):
    """Uniformly choose a start whose requested frames all decode strictly.

    Called only after the normal loader fails. Original frame indices and
    temporal offsets are retained; no image is padded, repaired or dropped.
    Deliberately uncached so that changes to files cannot leave stale validity.
    """
    valid = []
    for frame in frames:
        path = Path(scene_path) / frame['file_path'].replace('images/', 'images_8/', 1)
        try:
            with Image.open(path) as image:
                image.convert('RGB').load()
            valid.append(True)
        except (OSError, ValueError, SyntaxError):
            valid.append(False)
    starts = [start for start in range(len(frames) - span + 1)
              if all(valid[start + offset] for offset in offsets)]
    print(f"[training-data] scene={scene_path}: {valid.count(False)} undecodable frames; "
          f"{len(starts)} valid starts for span={span}", flush=True)
    return rng.choice(starts) if starts else None
