#!/usr/bin/env python3
"""Deterministic paper-configuration splits, independent of historical checkpoints."""
import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from PIL import Image


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def identity(entry):
    parts = Path(entry).parts
    return parts[-2] if parts[-1] == "nerfstudio" else parts[-1]


def read_ids(path):
    entries = json.loads(Path(path).read_text())
    if not isinstance(entries, list) or not all(isinstance(x, str) for x in entries):
        raise ValueError(f"Expected JSON list of scene paths: {path}")
    if any(Path(x).is_absolute() or ".." in Path(x).parts for x in entries):
        raise ValueError(f"Unsafe scene paths in {path}")
    if len({identity(x) for x in entries}) != len(entries):
        raise ValueError(f"Duplicate scene identities: {path}")
    return entries


def inspect_scene(path, min_frames, decode=False):
    meta = json.loads((path / "transforms.json").read_text())
    frames = meta.get("frames", [])
    if len(frames) < min_frames:
        raise ValueError(f"only {len(frames)} frames, need {min_frames}")
    for key in ("w", "h", "fl_x", "fl_y", "cx", "cy"):
        if key not in meta or not math.isfinite(float(meta[key])):
            raise ValueError(f"missing/nonfinite intrinsics: {key}")
    if any(meta[key] <= 0 for key in ("w", "h", "fl_x", "fl_y")):
        raise ValueError("nonpositive dimensions or focal lengths")
    seen = set()
    for frame in frames:
        rel = frame["file_path"]
        if not rel.startswith("images/") or ".." in Path(rel).parts or rel in seen:
            raise ValueError("unsafe, unsupported or duplicated frame path")
        seen.add(rel)
        image = path / rel.replace("images/", "images_8/", 1)
        if not image.is_file() or image.stat().st_size == 0:
            raise ValueError(f"missing image: {rel}")
        matrix = frame["transform_matrix"]
        if len(matrix) != 4 or any(len(row) != 4 or not all(math.isfinite(float(v)) for v in row) for row in matrix):
            raise ValueError("invalid camera-to-world matrix")
        if decode:
            with Image.open(image) as handle:
                handle.load()
    return {"frames": len(frames), "metadata_sha256": hashlib.sha256((path / "transforms.json").read_bytes()).hexdigest(),
            "url": meta.get("re10k_url", ""), "split": meta.get("re10k_split", "")}


def scan(root, excluded, min_frames, decode, limit=0):
    root = root.resolve()
    index = root / "train_index.json"
    entries = read_ids(index) if index.exists() else sorted(str(p.parent.relative_to(root)) for p in root.rglob("transforms.json"))
    rows, rejected, seen = [], [], set()
    for index, entry in enumerate(entries, 1):
        if index % 500 == 0:
            print(f"Scanned {index - 1}/{len(entries)} scenes: {len(rows)} eligible, "
                  f"{len(rejected)} rejected", file=sys.stderr, flush=True)
        key = identity(entry)
        if key in seen:
            raise ValueError(f"Repeated scene identity: {key}")
        seen.add(key)
        if key in excluded:
            rejected.append({"id": key, "reason": "evaluation identity"})
            continue
        scene = root / entry
        if not (scene / "transforms.json").exists():
            scene = scene / "nerfstudio"
        try:
            info = inspect_scene(scene, min_frames, decode)
        except (ValueError, KeyError, OSError, TypeError) as exc:
            rejected.append({"id": key, "reason": str(exc)})
            continue
        rows.append({"id": key, "entry": str(scene.relative_to(root)), **info})
        if limit and len(rows) >= limit:
            break
    return rows, rejected


def select(rows, count, seed, label):
    ordered = sorted(rows, key=lambda row: digest([seed, label, row["id"]]))
    if len(ordered) < count:
        raise ValueError(f"{label}: only {len(ordered)} eligible scenes, require {count}")
    return ordered[:count]


def materialize(root, source, rows, role, seed):
    root.mkdir(parents=True)
    ids = []
    for row in rows:
        key = row["id"]
        (root / key).symlink_to(os.path.relpath(source / row["entry"], root), target_is_directory=True)
        ids.append(key)
    (root / "train_index.json").write_text(json.dumps(ids, indent=2) + "\n")
    (root / "test_index.json").write_text("[]\n")
    manifest = {"format": "longaussian_paper_split_v1", "role": role, "seed": seed,
                "selection_rule": "sha256(seed,label,identity)", "count": len(ids),
                "scene_list_sha256": digest(ids), "scenes": rows}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dl3dv-root", type=Path, required=True)
    parser.add_argument("--re10k-root", type=Path, required=True)
    parser.add_argument("--re10k-metadata", type=Path, required=True)
    parser.add_argument("--dl3dv-test-index", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--decode-images", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="2/2-scene test split; never a paper result")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing existing output: {args.output}")
    dl_test = {identity(x) for p in args.dl3dv_test_index for x in read_ids(p)}
    test_files = sorted((args.re10k_metadata / "test").glob("*.txt"))
    if not test_files:
        raise ValueError("Official RE10K test metadata is required to audit train/test URLs")
    re_test = {p.stem for p in test_files}
    test_urls = {p.read_text().splitlines()[0].strip() for p in test_files}
    dl, dl_rejected = scan(args.dl3dv_root, dl_test, 146 if args.smoke else 19, args.decode_images, 8 if args.smoke else 0)
    re, re_rejected = scan(args.re10k_root, re_test, 146, args.decode_images)
    eligible_re = []
    for row in re:
        original = args.re10k_metadata / "train" / (row["id"] + ".txt")
        if not original.is_file():
            re_rejected.append({"id": row["id"], "reason": "not in official train metadata"})
            continue
        url = original.read_text().splitlines()[0].strip()
        if url in test_urls or row["url"] != url or row["split"] != "official_train":
            re_rejected.append({"id": row["id"], "reason": "URL/split mismatch or test overlap"})
            continue
        eligible_re.append(row)
    stage1 = select(dl, 2 if args.smoke else 8590, args.seed, "stage1")
    stage2a = [row for row in stage1 if row["frames"] >= 146]
    if not stage2a:
        raise ValueError("No long DL3DV scenes for Stage IIa")
    adapt = select(stage2a, 2 if args.smoke else 118, args.seed, "stage2b_dl3dv")
    re_adapt = select(eligible_re, 2 if args.smoke else 150, args.seed, "stage2b_re10k")
    args.output.mkdir(parents=True)
    for name, source, rows in [("stage1", args.dl3dv_root, stage1), ("stage2a", args.dl3dv_root, stage2a),
                               ("stage2b_dl3dv", args.dl3dv_root, adapt), ("stage2b_re10k", args.re10k_root, re_adapt)]:
        materialize(args.output / name, source.resolve(), rows, name, args.seed)
    audit = {"format": "longaussian_split_audit_v1", "smoke": args.smoke, "seed": args.seed,
             "counts": {"stage1": len(stage1), "stage2a": len(stage2a), "stage2b_dl3dv": len(adapt), "stage2b_re10k": len(re_adapt)},
             "excluded_dl3dv_ids": sorted(dl_test), "excluded_re10k_ids": sorted(re_test),
             "excluded_re10k_url_sha256": sorted(digest(x) for x in test_urls),
             "dl3dv_rejected": dl_rejected, "re10k_rejected": re_rejected,
             "note": "New deterministic paper-configuration split, not a claim about historical checkpoints"}
    (args.output / "split_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit["counts"]))


if __name__ == "__main__":
    main()
