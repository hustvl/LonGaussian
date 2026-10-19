#!/usr/bin/env python3
"""Download official inputs with pinned revisions and safe archive extraction."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tarfile
import urllib.request
import zipfile


def checksum(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_member(name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe archive member: {name}")
    return path


def extract_zip(source, destination):
    # DL3DV training downloads may be per-scene zip archives.
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        for item in archive.infolist():
            relative = safe_member(item.filename)
            if ((item.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError("Archive symlinks are unsupported")
            target = destination / relative
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(target)
            with archive.open(item) as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["weights", "dl3dv", "benchmark", "re10k-metadata", "unpack-dl3dv"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default="main", help="Resolved to a fixed commit before download")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--benchmark-scene", help="Download one benchmark scene for a transfer smoke test")
    args = parser.parse_args()
    if args.benchmark_scene and args.kind != "benchmark":
        parser.error("--benchmark-scene is only valid for benchmark downloads")
    if args.benchmark_scene and not re.fullmatch(r"[0-9a-f]{64}", args.benchmark_scene):
        parser.error("--benchmark-scene must be a 64-character lowercase scene ID")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.kind == "unpack-dl3dv":
        if not args.source:
            parser.error("--source is required for unpack-dl3dv")
        archives = sorted(args.source.rglob("*.zip"))
        if not archives:
            raise ValueError("No zip archives found; if already extracted pass the raw root directly to preparation")
        for path in archives:
            target = args.output / path.relative_to(args.source).with_suffix("")
            marker = target / ".archive.sha256"
            sha = checksum(path)
            if marker.exists() and marker.read_text().strip() == sha:
                continue
            if target.exists():
                raise FileExistsError(f"Incomplete/different extraction at {target}; use a fresh destination")
            extract_zip(path, target)
            marker.write_text(sha + "\n")
        return
    if args.kind == "re10k-metadata":
        url = "https://storage.googleapis.com/realestate10k-public/RealEstate10K.tgz"
        archive = args.output / "RealEstate10K.tgz"
        if not archive.exists():
            partial = archive.with_suffix(".partial")
            urllib.request.urlretrieve(url, partial)
            os.replace(partial, archive)
        destination = args.output / "metadata"
        if destination.exists():
            raise FileExistsError(f"Metadata already exists: {destination}")
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                safe_member(member.name)
                if not member.isdir() and not member.isfile():
                    raise ValueError("Only regular files/directories are accepted")
            tar.extractall(destination, filter="data")
        (args.output / "download.json").write_text(json.dumps({"url": url, "sha256": checksum(archive)}, indent=2))
        return
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    api = HfApi()
    records = []
    if args.kind == "weights":
        specs = [("Junyi42/LoGeR", "LoGeR_star/latest.pt", "68a0ae71408eb5605b8409296e7d5bf43c87853595766195257af97c5e60373c"),
                 ("aharley/alltracker", "alltracker.pth", "ffd9ebcfb6d206d594b646999a150540f92c049cf9b2bf940facf7123f62aa1d"),
                 ("JianyuanWang/skyseg", "skyseg.onnx", "ab9c34c64c3d821220a2886a4a06da4642ffa14d5b30e8d5339056a089aa1d39")]
        for repo, filename, expected in specs:
            revision = api.model_info(repo, revision=args.revision).sha
            path = hf_hub_download(repo, filename, revision=revision, local_dir=args.output)
            sha = checksum(path)
            if sha != expected:
                raise ValueError(f"Upstream weight differs from verified version: {repo}/{filename}")
            records.append({"repo": repo, "revision": revision, "file": filename, "sha256": sha})
    else:
        repo = "DL3DV/DL3DV-ALL-480P" if args.kind == "dl3dv" else "DL3DV/DL3DV-Benchmark"
        revision = api.dataset_info(repo, revision=args.revision).sha
        patterns = None if args.kind == "dl3dv" else ["*/nerfstudio/images_8/*", "*/nerfstudio/transforms.json", "benchmark-meta.csv"]
        if args.benchmark_scene:
            scene = args.benchmark_scene
            image_prefix = f"{scene}/nerfstudio/images_8/"
            image_files = sorted(item.path for item in api.list_repo_tree(
                repo, repo_type="dataset", revision=revision,
                path_in_repo=image_prefix.rstrip("/"), recursive=True,
            ) if item.path.startswith(image_prefix))
            if len(image_files) < 146:
                raise ValueError(f"Benchmark scene has only {len(image_files)} images_8 files")
            files = [f"{scene}/nerfstudio/transforms.json", *image_files]
            with ThreadPoolExecutor(max_workers=8) as pool:
                downloaded = list(pool.map(lambda filename: hf_hub_download(
                    repo, filename, repo_type="dataset", revision=revision, local_dir=args.output,
                ), files))
            records.append({"repo": repo, "revision": revision, "complete": False,
                            "smoke_scene": scene, "files": len(downloaded),
                            "downloaded_bytes": sum(Path(path).stat().st_size for path in downloaded)})
        else:
            snapshot_download(repo, repo_type="dataset", revision=revision, local_dir=args.output, allow_patterns=patterns)
            records.append({"repo": repo, "revision": revision, "allow_patterns": patterns, "complete": True})
    (args.output / "download.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
