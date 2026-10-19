"""Minimal training provenance + RNG helpers for two-stage reproduction.

Does not change the model forward or loss. Intended for forensic Stage I/II
runs that do not use the full M0-CLEAN train-scene-manifest contract.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from loger.util.clean_training_state import (
    RNG_STATE_FORMAT,
    capture_rng_state,
    file_sha256,
    restore_rng_state,
    seed_all_rng,
)

MINIMAL_PROVENANCE_FORMAT = "loger_minimal_train_provenance_v1"
PROVENANCE_FILENAME = "training_provenance.json"


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return _bytes_sha256(payload.encode("utf-8"))


def git_head_and_diff_hash(repo_root: str | Path) -> dict[str, str | None]:
    root = Path(repo_root)
    head = None
    diff_hash = None
    try:
        head = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        head = None
    try:
        diff = subprocess.check_output(
            ["git", "diff", "HEAD"], cwd=root, stderr=subprocess.DEVNULL
        )
        diff_hash = _bytes_sha256(diff) if diff else _bytes_sha256(b"")
    except (subprocess.CalledProcessError, FileNotFoundError):
        diff_hash = None
    return {"git_head": head, "git_diff_sha256": diff_hash}


def source_file_sha256(path: str | Path) -> dict[str, str]:
    p = Path(path)
    return {"path": str(p.resolve()), "sha256": file_sha256(p)}


def require_empty_or_missing_dir(path: str | Path, *, label: str = "output_folder") -> Path:
    """Fail closed: refuse to train into a non-empty directory (blocks silent resume)."""
    out = Path(path)
    if out.exists():
        if not out.is_dir():
            raise FileExistsError(f"{label} exists and is not a directory: {out}")
        contents = list(out.iterdir())
        if contents:
            raise FileExistsError(
                f"{label} is not empty ({len(contents)} entries): {out}. "
                "Refuse to resume into a dirty directory; pick a new empty path."
            )
    return out


def build_minimal_provenance(
    *,
    resolved_args: Mapping[str, Any],
    repo_root: str | Path,
    key_source_files: list[str | Path],
    dependency_checkpoints: Mapping[str, str | Path | None],
    dataset_manifest: str | Path | None = None,
) -> dict[str, Any]:
    args_dict = {k: resolved_args[k] for k in sorted(resolved_args)}
    # Make JSON-safe
    safe_args: dict[str, Any] = {}
    for k, v in args_dict.items():
        if isinstance(v, Path):
            safe_args[k] = str(v)
        elif isinstance(v, (str, int, float, bool)) or v is None:
            safe_args[k] = v
        else:
            safe_args[k] = str(v)

    deps: dict[str, Any] = {}
    for name, path in dependency_checkpoints.items():
        if path is None or path == "" or path == "None":
            deps[name] = None
            continue
        p = Path(path)
        deps[name] = source_file_sha256(p) if p.is_file() else {"path": str(p), "sha256": None}

    manifest_rec = None
    if dataset_manifest is not None:
        mp = Path(dataset_manifest)
        manifest_rec = source_file_sha256(mp) if mp.is_file() else {"path": str(mp), "sha256": None}

    record = {
        "format": MINIMAL_PROVENANCE_FORMAT,
        "resolved_args": safe_args,
        "resolved_args_sha256": canonical_json_sha256(safe_args),
        "code": {
            **git_head_and_diff_hash(repo_root),
            "source_files": [source_file_sha256(p) for p in key_source_files],
        },
        "dependency_checkpoints": deps,
        "dataset_manifest": manifest_rec,
    }
    return record


def write_minimal_provenance(output_folder: str | Path, record: Mapping[str, Any]) -> Path:
    if record.get("format") != MINIMAL_PROVENANCE_FORMAT:
        raise ValueError("invalid minimal provenance format")
    out = Path(output_folder)
    out.mkdir(parents=True, exist_ok=True)
    path = out / PROVENANCE_FILENAME
    if path.exists():
        raise FileExistsError(f"provenance already exists: {path}")
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return path


def attach_rng_to_checkpoint(checkpoint: dict[str, Any], *, include_cuda: bool = True) -> dict[str, Any]:
    checkpoint = dict(checkpoint)
    checkpoint["rng_state"] = capture_rng_state(include_cuda=include_cuda)
    return checkpoint


def restore_rng_from_checkpoint(checkpoint: Mapping[str, Any], *, include_cuda: bool = True) -> bool:
    state = checkpoint.get("rng_state")
    if state is None:
        return False
    if not isinstance(state, Mapping) or state.get("format") != RNG_STATE_FORMAT:
        raise ValueError("checkpoint rng_state has invalid format")
    restore_rng_state(state, require_cuda=include_cuda and ("torch_cuda" in state and state["torch_cuda"] is not None))
    return True


__all__ = [
    "MINIMAL_PROVENANCE_FORMAT",
    "PROVENANCE_FILENAME",
    "attach_rng_to_checkpoint",
    "build_minimal_provenance",
    "canonical_json_sha256",
    "git_head_and_diff_hash",
    "require_empty_or_missing_dir",
    "restore_rng_from_checkpoint",
    "seed_all_rng",
    "source_file_sha256",
    "write_minimal_provenance",
]
