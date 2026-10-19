"""Fail-closed data and resume contracts for clean multi-scene training.

This module intentionally has no dependency on :mod:`run`.  It contains the
CPU-testable pieces needed by the M0-CLEAN gate: loading a frozen training
scene manifest, accounting for DDP sampler padding / optimizer updates, and
capturing the per-rank random-number-generator state required for an exact
resume.

The training entry point wires these helpers into argparse, the dataset loop,
and checkpoint I/O.  Keeping the contract here makes those later call-site
changes small and, more importantly, independently testable.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image


TRAIN_SCENE_MANIFEST_FORMAT = "loger_m0_scene_manifest_v1"
TRAIN_MANIFEST_CHECKPOINT_FORMAT = "loger_train_manifest_checkpoint_v1"
EXPOSURE_PLAN_FORMAT = "loger_ddp_exposure_plan_v1"
EPOCH_SAMPLING_STATE_FORMAT = "loger_epoch_sampling_state_v1"
REALIZED_ACCOUNTING_FORMAT = "loger_realized_training_accounting_v1"
TRAINING_PROVENANCE_FORMAT = "loger_training_provenance_v1"
CLEAN_CHECKPOINT_FORMAT = "loger_clean_training_checkpoint_v1"
RNG_STATE_FORMAT = "loger_rng_state_v1"
RANK_RNG_BUNDLE_FORMAT = "loger_rank_rng_bundle_v1"

_SCENE_ID_PATTERN = re.compile(r"[0-9a-f]{64}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

# These are the only two data tiers authorized by the M0-CLEAN plan.  Tests
# exercise small synthetic tiers by monkeypatching this mapping; production
# callers cannot silently relabel an arbitrary-sized manifest as clean.
TRAIN_ROLE_COUNTS: Mapping[str, int] = MappingProxyType(
    {
        "train_clean_500": 500,
        "train_clean_2000": 2000,
    }
)
REQUIRED_EXCLUSION_SOURCE_ROLES = frozenset(
    {
        "canonical_140",
        "historical_exposure",
        "adapt_heldout20",
        "smoke5",
        "final_confirm20",
    }
)
EXCLUSION_SOURCE_CONTRACT: Mapping[str, Mapping[str, object]] = MappingProxyType(
    {
        "canonical_140": MappingProxyType(
            {"identity_field": "index_entries", "exact_count": 140}
        ),
        "historical_exposure": MappingProxyType(
            {"identity_field": "excluded_identities", "min_count": 1}
        ),
        "adapt_heldout20": MappingProxyType(
            {"identity_field": "scenes", "exact_count": 20}
        ),
        "smoke5": MappingProxyType(
            {"identity_field": "scenes", "exact_count": 5}
        ),
        "final_confirm20": MappingProxyType(
            {"identity_field": "scenes", "exact_count": 20}
        ),
    }
)
TRAIN_SELECTION_MODE = "source_order_after_exclusions_and_eligibility"
TRAIN_REQUIRED_FRAMES = 146
TRAIN_ELIGIBILITY_CONTRACT: Mapping[str, object] = MappingProxyType(
    {
        "min_frames": TRAIN_REQUIRED_FRAMES,
        "transforms_json": "readable",
        "frame_records": "all frames have finite 4x4 poses",
        "images_8": "all referenced files Pillow-decodable",
    }
)
EPOCH_SAMPLER_VERSION = "loger_epoch_sampler_v1"
TRAIN_UPDATE_PHASES = frozenset(
    {"joint", "head", "merger", "alternating", "dense"}
)
CLEAN_OPTIMIZER_CLASS = "torch.optim.AdamW"
CLEAN_SCHEDULER_CLASS = "torch.optim.lr_scheduler.LambdaLR"
_ADAMW_GROUP_FIELDS = frozenset(
    {
        "amsgrad",
        "betas",
        "capturable",
        "differentiable",
        "eps",
        "foreach",
        "fused",
        "lr",
        "maximize",
        "params",
        "weight_decay",
    }
)
_LAMBDA_LR_STATE_FIELDS = frozenset(
    {
        "_get_lr_called_within_step",
        "_last_lr",
        "_step_count",
        "base_lrs",
        "last_epoch",
        "lr_lambdas",
        "verbose",
    }
)
REQUIRED_PROVENANCE_TRAINING_ARGS = frozenset(
    {
        "dataset_root",
        "deterministic",
        "epochs",
        "sampler_version",
        "scenes_per_epoch",
        "train_scene_manifest",
        "train_seed",
        "train_update_phase",
        "world_size",
    }
)


class SceneEligibilityError(ValueError):
    """A stable reason-coded failure used to audit skipped source entries."""

    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


def file_sha256(path: Path) -> str:
    """Return the SHA256 of file bytes without loading the whole file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_sha256(value: object) -> str:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not canonically JSON-serializable") from exc
    return _bytes_sha256(payload)


def _read_json_bytes(path: Path, *, label: str) -> tuple[object, str]:
    """Parse and hash the same bytes, avoiding a parse/hash TOCTOU split."""
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read {label}: {path}") from exc
    digest = _bytes_sha256(payload)
    try:
        return json.loads(payload), digest
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label} JSON: {path}") from exc


def normalize_scene_entry(raw_entry: object) -> str:
    """Normalize a DL3DV entry while rejecting path traversal/ambiguity."""
    if not isinstance(raw_entry, str):
        raise ValueError(f"scene entry must be a string: {raw_entry!r}")
    value = raw_entry.strip().replace("\\", "/")
    if not value or value.startswith("/"):
        raise ValueError(f"invalid scene entry: {raw_entry!r}")
    parts = value.split("/")
    if parts and parts[-1] == "nerfstudio":
        parts.pop()
    if (
        not parts
        or any(part in {"", ".", ".."} for part in parts)
        or any(":" in part for part in parts)
    ):
        raise ValueError(f"invalid scene entry: {raw_entry!r}")
    return "/".join(parts)


def scene_identity(entry: str) -> str:
    """Return the final, globally compared DL3DV scene identity."""
    identity = normalize_scene_entry(entry).split("/")[-1]
    if _SCENE_ID_PATTERN.fullmatch(identity) is None:
        raise ValueError(f"invalid DL3DV scene identity: {identity!r}")
    return identity


def scene_list_sha256(entries: Sequence[str]) -> str:
    """Hash a normalized ordered scene list using the frozen M0 encoding."""
    normalized = [normalize_scene_entry(entry) for entry in entries]
    payload = json.dumps(
        normalized, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def identity_list_sha256(identities: Sequence[str], *, sort_values: bool = False) -> str:
    """Hash a validated identity sequence using the frozen compact encoding."""
    checked = []
    for identity in identities:
        if not isinstance(identity, str) or _SCENE_ID_PATTERN.fullmatch(identity) is None:
            raise ValueError(f"invalid DL3DV scene identity: {identity!r}")
        checked.append(identity)
    if sort_values:
        checked.sort()
    return _canonical_json_sha256(checked)


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA256 hex digest")
    return value


def _require_nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _require_positive_int(value: object, field: str) -> int:
    result = _require_nonnegative_int(value, field)
    if result == 0:
        raise ValueError(f"{field} must be positive")
    return result


def _require_finite_real(value: object, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field} must be a finite real number")
    return float(value)


def _resolve_recorded_file(record: Mapping[str, Any], *, label: str) -> tuple[Path, object]:
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label}.path must be a non-empty absolute path")
    unexpanded = Path(raw_path).expanduser()
    if not unexpanded.is_absolute():
        raise ValueError(f"{label}.path must be absolute")
    path = unexpanded.resolve(strict=True)
    if raw_path != str(path):
        raise ValueError(f"{label}.path must be the resolved absolute path")
    if not path.is_file():
        raise ValueError(f"{label}.path is not a file: {path}")
    expected_sha = _require_sha256(record.get("sha256"), f"{label}.sha256")
    value, actual_sha = _read_json_bytes(path, label=label)
    if expected_sha != actual_sha:
        raise ValueError(
            f"{label} SHA256 mismatch ({expected_sha} != {actual_sha})"
        )
    return path, value


def _entries_from_index(value: object, *, label: str) -> tuple[str, ...]:
    if isinstance(value, list):
        records = value
    elif isinstance(value, dict) and isinstance(value.get("scenes"), list):
        records = value["scenes"]
    else:
        raise ValueError(f"{label} must be a list or an object with a scenes list")

    entries: list[str] = []
    identities: list[str] = []
    for index, record in enumerate(records):
        if isinstance(record, str):
            raw_entry = record
            recorded_identity = None
        elif isinstance(record, dict) and isinstance(record.get("entry"), str):
            raw_entry = record["entry"]
            recorded_identity = record.get("identity")
        else:
            raise ValueError(f"invalid {label} record at index {index}")
        entry = normalize_scene_entry(raw_entry)
        identity = scene_identity(entry)
        if recorded_identity is not None and recorded_identity != identity:
            raise ValueError(f"{label} identity mismatch for {entry}")
        entries.append(entry)
        identities.append(identity)
    if len(entries) != len(set(entries)):
        raise ValueError(f"{label} contains duplicate normalized entries")
    if len(identities) != len(set(identities)):
        raise ValueError(f"{label} contains duplicate scene identities")
    if isinstance(value, dict) and value.get("scene_list_sha256") is not None:
        recorded_sha = _require_sha256(
            value.get("scene_list_sha256"), f"{label}.scene_list_sha256"
        )
        actual_sha = scene_list_sha256(entries)
        if recorded_sha != actual_sha:
            raise ValueError(f"{label} scene-list SHA256 mismatch")
    return tuple(entries)


def _identities_from_exclusion_source(
    value: object,
    *,
    identity_field: str,
    label: str,
) -> tuple[str, ...]:
    if identity_field == "index_entries":
        return tuple(scene_identity(entry) for entry in _entries_from_index(value, label=label))
    if identity_field == "scenes":
        if not isinstance(value, dict) or not isinstance(value.get("scenes"), list):
            raise ValueError(f"{label} must contain a scenes list")
        return tuple(
            scene_identity(entry)
            for entry in _entries_from_index(
                {"scenes": value["scenes"], "scene_list_sha256": value.get("scene_list_sha256")},
                label=label,
            )
        )
    if identity_field == "excluded_identities":
        if not isinstance(value, dict) or not isinstance(
            value.get("excluded_identities"), list
        ):
            raise ValueError(f"{label} must contain excluded_identities")
        identities = []
        for identity in value["excluded_identities"]:
            if not isinstance(identity, str) or _SCENE_ID_PATTERN.fullmatch(identity) is None:
                raise ValueError(f"invalid identity in {label}: {identity!r}")
            identities.append(identity)
        if len(identities) != len(set(identities)):
            raise ValueError(f"{label} contains duplicate scene identities")
        return tuple(identities)
    raise ValueError(
        f"{label}.identity_field must be index_entries, scenes, or excluded_identities"
    )


def _resolve_scene_view(dataset_root: Path, entry: str) -> Path:
    root = dataset_root.resolve(strict=True)
    expected_identity = scene_identity(entry)
    try:
        scene = (root / normalize_scene_entry(entry)).resolve(strict=True)
    except FileNotFoundError as exc:
        raise SceneEligibilityError("missing_scene_view", repr(entry)) from exc
    try:
        scene.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"scene path escapes dataset root: {entry!r}") from exc
    if scene.name != expected_identity:
        raise ValueError(
            f"resolved scene identity mismatch: {scene.name!r} != "
            f"{expected_identity!r} for {entry!r}"
        )
    for candidate in (scene, scene / "nerfstudio"):
        try:
            resolved_candidate = candidate.resolve(strict=True)
        except FileNotFoundError:
            continue
        try:
            resolved_candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"resolved scene view escapes dataset root: {entry!r}"
            ) from exc
        resolved_identity = (
            resolved_candidate.parent.name
            if resolved_candidate.name == "nerfstudio"
            else resolved_candidate.name
        )
        if resolved_identity != expected_identity:
            raise ValueError(
                f"resolved scene-view identity mismatch: {resolved_identity!r} != "
                f"{expected_identity!r} for {entry!r}"
            )
        if (resolved_candidate / "transforms.json").is_file() and (
            resolved_candidate / "images_8"
        ).is_dir():
            return resolved_candidate
    raise SceneEligibilityError(
        "missing_scene_view", f"{entry!r} has no transforms.json/images_8 view"
    )


def _validate_pose(value: object, *, entry: str, frame_index: int) -> None:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(not isinstance(row, list) or len(row) != 4 for row in value)
    ):
        raise SceneEligibilityError(
            "invalid_pose", f"{entry} frame {frame_index} is not 4x4"
        )
    for row in value:
        for item in row:
            if (
                isinstance(item, bool)
                or not isinstance(item, Real)
                or not math.isfinite(float(item))
            ):
                raise SceneEligibilityError(
                    "invalid_pose", f"{entry} frame {frame_index} is non-finite"
                )


def _frame_image_path(images_root: Path, raw_file_path: object, *, entry: str) -> Path:
    if not isinstance(raw_file_path, str) or not raw_file_path.startswith("images/"):
        raise SceneEligibilityError(
            "invalid_frame_record", f"{entry} has a non-standard file_path"
        )
    relative = raw_file_path[len("images/") :].replace("\\", "/")
    parts = relative.split("/")
    if (
        not relative
        or relative.startswith("/")
        or any(part in {"", ".", ".."} for part in parts)
        or any(":" in part for part in parts)
    ):
        raise SceneEligibilityError(
            "invalid_frame_record", f"{entry} has an unsafe file_path"
        )
    try:
        image_path = (images_root / relative).resolve(strict=True)
    except FileNotFoundError as exc:
        raise SceneEligibilityError(
            "missing_referenced_image", f"{entry}: {relative}"
        ) from exc
    try:
        image_path.relative_to(images_root)
    except ValueError as exc:
        raise SceneEligibilityError(
            "image_path_escape", f"{entry}: {relative}"
        ) from exc
    if not image_path.is_file():
        raise SceneEligibilityError(
            "missing_referenced_image", f"{entry}: {relative}"
        )
    return image_path


def _decode_image(path: Path, *, entry: str) -> None:
    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            rgb.load()
    except (OSError, SyntaxError, ValueError) as exc:
        raise SceneEligibilityError(
            "undecodable_image", f"{entry}: {path}"
        ) from exc


def _validate_scene_eligibility(
    dataset_root: Path,
    entry: str,
    *,
    required_frames: int,
    manifest_record: Mapping[str, Any] | None = None,
) -> tuple[Path, str, int]:
    """Fully validate one selected scene against the frozen clean contract."""
    view = _resolve_scene_view(dataset_root, entry)
    try:
        transforms_path = (view / "transforms.json").resolve(strict=True)
    except FileNotFoundError as exc:
        raise SceneEligibilityError(
            "invalid_transforms_json", f"{entry}: missing transforms.json"
        ) from exc
    try:
        transforms_path.relative_to(view)
    except ValueError as exc:
        raise SceneEligibilityError(
            "transforms_path_escape", f"{entry}: transforms.json escapes the scene view"
        ) from exc
    if not transforms_path.is_file():
        raise SceneEligibilityError(
            "invalid_transforms_json", f"{entry}: transforms.json is not a file"
        )
    try:
        payload = transforms_path.read_bytes()
        transforms = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SceneEligibilityError(
            "invalid_transforms_json", f"{entry}: {transforms_path}"
        ) from exc
    transforms_sha = _bytes_sha256(payload)
    if not isinstance(transforms, dict) or not isinstance(transforms.get("frames"), list):
        raise SceneEligibilityError(
            "invalid_transforms_json", f"{entry}: frames must be a list"
        )
    frames = transforms["frames"]
    if len(frames) < required_frames:
        raise SceneEligibilityError(
            "insufficient_frames", f"{entry}: {len(frames)} < {required_frames}"
        )

    images_root = (view / "images_8").resolve(strict=True)
    try:
        images_root.relative_to(view)
    except ValueError as exc:
        raise SceneEligibilityError(
            "image_path_escape", f"{entry}: images_8 escapes the scene view"
        ) from exc
    decoded: set[Path] = set()
    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise SceneEligibilityError(
                "invalid_frame_record", f"{entry} frame {frame_index} is not an object"
            )
        _validate_pose(frame.get("transform_matrix"), entry=entry, frame_index=frame_index)
        image_path = _frame_image_path(
            images_root, frame.get("file_path"), entry=entry
        )
        if image_path not in decoded:
            _decode_image(image_path, entry=entry)
            decoded.add(image_path)

    if manifest_record is not None:
        recorded_sha = _require_sha256(
            manifest_record.get("transforms_sha256"),
            f"scene {entry} transforms_sha256",
        )
        if recorded_sha != transforms_sha:
            raise SceneEligibilityError(
                "transforms_sha256_mismatch", entry
            )
        if manifest_record.get("num_frames") != len(frames):
            raise SceneEligibilityError(
                "num_frames_mismatch",
                f"{entry}: {manifest_record.get('num_frames')!r} != {len(frames)}",
            )
        if manifest_record.get("required_frames_checked") != len(frames):
            raise SceneEligibilityError(
                "eligibility_check_count_mismatch",
                f"{entry}: every one of {len(frames)} frames must be checked",
            )
    return view, transforms_sha, len(frames)


@dataclass(frozen=True)
class TrainSceneManifest:
    """Validated frozen training-scene manifest and resolved scene paths."""

    path: str
    sha256: str
    role: str
    dataset_root: str
    source_index_path: str
    source_index_sha256: str
    scene_list_sha256: str
    excluded_identities_sha256: str
    exclusion_sources_sha256: str
    eligibility_contract_sha256: str
    selection_mode: str
    required_frames: int
    entries: tuple[str, ...]
    identities: tuple[str, ...]
    scene_paths: tuple[str, ...]
    source_positions: tuple[int, ...]
    excluded_identities: tuple[str, ...]

    @property
    def scene_count(self) -> int:
        return len(self.entries)

    def checkpoint_record(self) -> dict[str, Any]:
        """Return the immutable data binding persisted in each checkpoint."""
        return {
            "format": TRAIN_MANIFEST_CHECKPOINT_FORMAT,
            "path": self.path,
            "sha256": self.sha256,
            "role": self.role,
            "dataset_root": self.dataset_root,
            "source_index": {
                "path": self.source_index_path,
                "sha256": self.source_index_sha256,
            },
            "scene_count": self.scene_count,
            "scene_list_sha256": self.scene_list_sha256,
            "excluded_identities_sha256": self.excluded_identities_sha256,
            "exclusion_sources_sha256": self.exclusion_sources_sha256,
            "eligibility_contract_sha256": self.eligibility_contract_sha256,
            "selection_mode": self.selection_mode,
            "required_frames": self.required_frames,
        }


def validate_train_scene_selection_args(
    train_scene_manifest: str | Path | None,
    train_scene_limit: int,
) -> None:
    """Enforce the CLI mutual exclusion required by the clean-data gate."""
    if train_scene_manifest is not None and not str(train_scene_manifest).strip():
        raise ValueError("--train_scene_manifest cannot be empty")
    if (
        isinstance(train_scene_limit, bool)
        or not isinstance(train_scene_limit, int)
        or train_scene_limit < 0
    ):
        raise ValueError("--train_scene_limit must be a non-negative integer")
    has_manifest = train_scene_manifest is not None
    if has_manifest and train_scene_limit > 0:
        raise ValueError(
            "--train_scene_manifest and --train_scene_limit are mutually exclusive"
        )


def _validate_exclusion_contract(
    value: Mapping[str, Any],
    *,
    trusted_sources: Mapping[str, Mapping[str, str]] | None,
) -> tuple[tuple[str, ...], str, str]:
    if not isinstance(trusted_sources, Mapping):
        raise ValueError("trusted exclusion-source bindings are required")
    trusted_roles = frozenset(trusted_sources)
    if trusted_roles != REQUIRED_EXCLUSION_SOURCE_ROLES:
        raise ValueError("trusted exclusion-source roles are incomplete")
    trusted_bindings: dict[str, dict[str, str]] = {}
    for role in sorted(REQUIRED_EXCLUSION_SOURCE_ROLES):
        binding = trusted_sources[role]
        if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
            raise ValueError(f"trusted exclusion-source binding is invalid: {role}")
        trusted_path = _validate_file_binding(
            binding, label=f"trusted exclusion source {role}"
        )
        trusted_bindings[role] = {
            "path": str(trusted_path),
            "sha256": str(binding["sha256"]),
        }
    records = value.get("exclusion_sources")
    if not isinstance(records, list):
        raise ValueError("manifest exclusion_sources must be a list")
    role_to_identities: dict[str, tuple[str, ...]] = {}
    normalized_bindings: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for index, record in enumerate(records):
        label = f"exclusion_sources[{index}]"
        if not isinstance(record, Mapping):
            raise ValueError(f"{label} must be an object")
        role = record.get("role")
        if not isinstance(role, str) or role not in REQUIRED_EXCLUSION_SOURCE_ROLES:
            raise ValueError(f"unexpected exclusion source role: {role!r}")
        if role in role_to_identities:
            raise ValueError(f"duplicate exclusion source role: {role}")
        path, source_value = _resolve_recorded_file(record, label=label)
        if {
            "path": str(path),
            "sha256": str(record["sha256"]),
        } != trusted_bindings[role]:
            raise ValueError(
                f"{label} does not match the trusted binding for role {role}"
            )
        if path in seen_paths:
            raise ValueError(f"duplicate exclusion source path: {path}")
        seen_paths.add(path)
        identity_field = record.get("identity_field")
        if not isinstance(identity_field, str):
            raise ValueError(f"{label}.identity_field must be a string")
        source_contract = EXCLUSION_SOURCE_CONTRACT[role]
        if identity_field != source_contract["identity_field"]:
            raise ValueError(
                f"{label}.identity_field mismatch for role {role}: "
                f"{identity_field!r} != {source_contract['identity_field']!r}"
            )
        identities = _identities_from_exclusion_source(
            source_value,
            identity_field=identity_field,
            label=label,
        )
        exact_count = source_contract.get("exact_count")
        min_count = source_contract.get("min_count")
        if exact_count is not None and len(identities) != exact_count:
            raise ValueError(
                f"{label} must contain exactly {exact_count} identities for {role}"
            )
        if min_count is not None and len(identities) < min_count:
            raise ValueError(
                f"{label} must contain at least {min_count} identities for {role}"
            )
        expected_identity_sha = _require_sha256(
            record.get("identity_list_sha256"), f"{label}.identity_list_sha256"
        )
        actual_identity_sha = identity_list_sha256(identities)
        if expected_identity_sha != actual_identity_sha:
            raise ValueError(f"{label} identity-list SHA256 mismatch")
        role_to_identities[role] = identities
        normalized_bindings.append(
            {
                "role": role,
                "path": str(path),
                "sha256": record["sha256"],
                "identity_field": identity_field,
                "identity_list_sha256": actual_identity_sha,
            }
        )

    roles = frozenset(role_to_identities)
    if roles != REQUIRED_EXCLUSION_SOURCE_ROLES:
        missing = sorted(REQUIRED_EXCLUSION_SOURCE_ROLES - roles)
        extra = sorted(roles - REQUIRED_EXCLUSION_SOURCE_ROLES)
        raise ValueError(
            f"exclusion source roles mismatch: missing={missing}, extra={extra}"
        )

    identity_roles: dict[str, set[str]] = {}
    for role, identities in role_to_identities.items():
        for identity in identities:
            identity_roles.setdefault(identity, set()).add(role)
    union = tuple(sorted(identity_roles))

    excluded = value.get("excluded_identities")
    if not isinstance(excluded, list):
        raise ValueError("manifest excluded_identities must be a list")
    if excluded != list(union):
        raise ValueError(
            "manifest excluded_identities must equal the sorted union of exclusion_sources"
        )

    evidence = value.get("exclusion_evidence")
    if not isinstance(evidence, list):
        raise ValueError("manifest exclusion_evidence must be a list")
    evidence_by_identity: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(evidence):
        if not isinstance(record, Mapping):
            raise ValueError(f"exclusion_evidence[{index}] must be an object")
        identity = record.get("identity")
        if not isinstance(identity, str) or _SCENE_ID_PATTERN.fullmatch(identity) is None:
            raise ValueError(f"invalid exclusion evidence identity: {identity!r}")
        if identity in evidence_by_identity:
            raise ValueError(f"duplicate exclusion evidence identity: {identity}")
        source_roles = record.get("source_roles")
        if (
            not isinstance(source_roles, list)
            or any(not isinstance(item, str) for item in source_roles)
            or source_roles != sorted(identity_roles.get(identity, set()))
        ):
            raise ValueError(f"exclusion evidence source_roles mismatch for {identity}")
        reason = record.get("reason")
        references = record.get("evidence")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"exclusion evidence reason is empty for {identity}")
        if (
            not isinstance(references, list)
            or not references
            or any(not isinstance(item, str) or not item.strip() for item in references)
        ):
            raise ValueError(f"exclusion evidence references are empty for {identity}")
        evidence_by_identity[identity] = record
    if set(evidence_by_identity) != set(union):
        raise ValueError("exclusion_evidence identities do not match exclusion union")

    return (
        union,
        identity_list_sha256(union),
        _canonical_json_sha256(normalized_bindings),
    )


def _validate_eligibility_contract(value: Mapping[str, Any]) -> tuple[str, int, str]:
    selection = value.get("selection")
    eligibility = value.get("eligibility")
    if not isinstance(selection, Mapping):
        raise ValueError("manifest selection must be an object")
    if not isinstance(eligibility, Mapping):
        raise ValueError("manifest eligibility must be an object")
    if set(selection) != {"mode", "required_frames"}:
        raise ValueError("manifest selection has an unexpected schema")
    mode = selection.get("mode")
    required_frames = selection.get("required_frames")
    if mode != TRAIN_SELECTION_MODE:
        raise ValueError(f"unexpected training selection mode: {mode!r}")
    if required_frames != TRAIN_REQUIRED_FRAMES:
        raise ValueError(
            f"selection.required_frames must equal {TRAIN_REQUIRED_FRAMES}"
        )
    if dict(eligibility) != dict(TRAIN_ELIGIBILITY_CONTRACT):
        raise ValueError("manifest eligibility contract mismatch")
    return mode, required_frames, _canonical_json_sha256(dict(eligibility))


def _parse_eligibility_rejections(
    value: Mapping[str, Any],
    *,
    source_entries: Sequence[str],
    excluded_identities: set[str],
) -> dict[str, str]:
    records = value.get("eligibility_rejections")
    if not isinstance(records, list):
        raise ValueError("manifest eligibility_rejections must be a list")
    source_set = set(source_entries)
    result: dict[str, str] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"eligibility_rejections[{index}] must be an object")
        entry = normalize_scene_entry(record.get("entry"))
        identity = scene_identity(entry)
        if record.get("identity") != identity:
            raise ValueError(f"eligibility rejection identity mismatch for {entry}")
        if entry not in source_set:
            raise ValueError(f"eligibility rejection is absent from source index: {entry}")
        if identity in excluded_identities:
            raise ValueError(f"eligibility rejection duplicates an exclusion: {entry}")
        reason = record.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"eligibility rejection reason is empty for {entry}")
        if entry in result:
            raise ValueError(f"duplicate eligibility rejection: {entry}")
        result[entry] = reason
    return result


def load_train_scene_manifest(
    manifest_path: str | Path,
    dataset_root: str | Path,
    *,
    trusted_exclusion_sources: Mapping[str, Mapping[str, str]] | None = None,
) -> TrainSceneManifest:
    """Load and fully bind a non-empty clean-training manifest.

    The loader deliberately does not accept a bare list and never falls back
    to a dataset index.  If the caller supplied ``--train_scene_manifest``, any
    malformed, empty, moved, or drifted source is a fatal error.
    """
    path = Path(manifest_path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"training scene manifest is not a file: {path}")
    value_raw, manifest_sha = _read_json_bytes(path, label="training scene manifest")
    if not isinstance(value_raw, dict):
        raise ValueError("training scene manifest must be a JSON object, not a list")
    value: Mapping[str, Any] = value_raw
    if value.get("format") != TRAIN_SCENE_MANIFEST_FORMAT:
        raise ValueError(f"unexpected training manifest format: {value.get('format')!r}")

    role = value.get("role")
    if not isinstance(role, str) or role not in TRAIN_ROLE_COUNTS:
        raise ValueError(f"manifest role is not clean training: {role!r}")
    expected_scene_count = TRAIN_ROLE_COUNTS[role]

    requested_root = Path(dataset_root).expanduser().resolve(strict=True)
    recorded_root_raw = value.get("dataset_root")
    if not isinstance(recorded_root_raw, str) or not recorded_root_raw:
        raise ValueError("manifest dataset_root must be a non-empty path string")
    recorded_root = Path(recorded_root_raw).expanduser().resolve(strict=True)
    if recorded_root != requested_root:
        raise ValueError(
            f"manifest dataset_root mismatch: {recorded_root} != {requested_root}"
        )

    source = value.get("source_index")
    if not isinstance(source, Mapping):
        raise ValueError("manifest source_index must be an object")
    source_path, source_value = _resolve_recorded_file(source, label="source_index")
    expected_source_path = requested_root / "train_index.json"
    if source_path != expected_source_path:
        raise ValueError(
            f"manifest source_index.path mismatch: {source_path} != {expected_source_path}"
        )
    actual_source_sha = str(source["sha256"])
    source_entries = _entries_from_index(source_value, label="source_index")

    selection_mode, required_frames, eligibility_sha = _validate_eligibility_contract(
        value
    )
    (
        excluded_identities,
        excluded_identities_sha,
        exclusion_sources_sha,
    ) = _validate_exclusion_contract(
        value,
        trusted_sources=trusted_exclusion_sources,
    )
    excluded_set = set(excluded_identities)
    rejection_reasons = _parse_eligibility_rejections(
        value,
        source_entries=source_entries,
        excluded_identities=excluded_set,
    )

    records = value.get("scenes")
    if not isinstance(records, list):
        raise ValueError("manifest scenes must be a list")
    if not records:
        raise ValueError("explicit training scene manifest is empty; refusing fallback")

    entries: list[str] = []
    identities: list[str] = []
    records_by_entry: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"manifest scene {index} must be an object")
        entry = normalize_scene_entry(record.get("entry"))
        identity = scene_identity(entry)
        if record.get("identity") != identity:
            raise ValueError(f"manifest identity mismatch for {entry}")
        _require_sha256(
            record.get("transforms_sha256"),
            f"scenes[{index}].transforms_sha256",
        )
        entries.append(entry)
        identities.append(identity)
        records_by_entry[entry] = record

    if len(entries) != len(set(entries)):
        raise ValueError("manifest contains duplicate normalized scene entries")
    if len(identities) != len(set(identities)):
        raise ValueError("manifest contains duplicate scene identities")
    if set(identities) & excluded_set:
        raise ValueError("manifest scenes overlap excluded_identities")

    recorded_count = value.get("scene_count")
    if recorded_count != len(entries):
        raise ValueError(
            f"manifest scene_count mismatch: {recorded_count!r} != {len(entries)}"
        )
    if len(entries) != expected_scene_count:
        raise ValueError(
            f"manifest role/count mismatch: {role!r} has {len(entries)} scenes"
        )
    recorded_scene_sha = _require_sha256(
        value.get("scene_list_sha256"), "scene_list_sha256"
    )
    actual_scene_sha = scene_list_sha256(entries)
    if recorded_scene_sha != actual_scene_sha:
        raise ValueError(
            "manifest scene-list SHA256 mismatch "
            f"({recorded_scene_sha} != {actual_scene_sha})"
        )

    # Prove that the manifest is exactly the first requested number of
    # source-order candidates after authoritative exclusions and verified
    # eligibility rejections.  A source entry cannot be silently skipped.
    expected_entries: list[str] = []
    source_positions: list[int] = []
    used_rejections: set[str] = set()
    for source_position, entry in enumerate(source_entries):
        identity = scene_identity(entry)
        if identity in excluded_set:
            continue
        if entry in rejection_reasons:
            try:
                _validate_scene_eligibility(
                    requested_root,
                    entry,
                    required_frames=required_frames,
                )
            except SceneEligibilityError as exc:
                if exc.reason != rejection_reasons[entry]:
                    raise ValueError(
                        f"eligibility rejection reason drift for {entry}: "
                        f"{rejection_reasons[entry]!r} != {exc.reason!r}"
                    ) from exc
            else:
                raise ValueError(
                    f"manifest rejects an eligible source scene: {entry}"
                )
            used_rejections.add(entry)
            continue
        expected_entries.append(entry)
        source_positions.append(source_position)
        if len(expected_entries) == expected_scene_count:
            break
    if len(expected_entries) != expected_scene_count:
        raise ValueError(
            f"source index has only {len(expected_entries)} eligible/unexcluded scenes; "
            f"{expected_scene_count} required"
        )
    if tuple(entries) != tuple(expected_entries):
        raise ValueError(
            "manifest scenes are not the exact source-order prefix after exclusions/eligibility"
        )
    if used_rejections != set(rejection_reasons):
        unused = sorted(set(rejection_reasons) - used_rejections)
        raise ValueError(f"unused eligibility_rejections outside selected prefix: {unused}")

    scene_paths: list[str] = []
    seen_views: set[Path] = set()
    for entry in entries:
        try:
            view, _, _ = _validate_scene_eligibility(
                requested_root,
                entry,
                required_frames=required_frames,
                manifest_record=records_by_entry[entry],
            )
        except SceneEligibilityError as exc:
            raise ValueError(f"selected scene is not eligible: {entry}: {exc}") from exc
        if view in seen_views:
            raise ValueError(f"duplicate resolved scene path: {view}")
        seen_views.add(view)
        scene_paths.append(str(view))

    return TrainSceneManifest(
        path=str(path),
        sha256=manifest_sha,
        role=role,
        dataset_root=str(requested_root),
        source_index_path=str(source_path),
        source_index_sha256=actual_source_sha,
        scene_list_sha256=actual_scene_sha,
        excluded_identities_sha256=excluded_identities_sha,
        exclusion_sources_sha256=exclusion_sources_sha,
        eligibility_contract_sha256=eligibility_sha,
        selection_mode=selection_mode,
        required_frames=required_frames,
        entries=tuple(entries),
        identities=tuple(identities),
        scene_paths=tuple(scene_paths),
        source_positions=tuple(source_positions),
        excluded_identities=excluded_identities,
    )


def validate_train_clean_prefix(
    clean500: TrainSceneManifest,
    clean2000: TrainSceneManifest,
) -> None:
    """Prove the frozen 500-scene tier is the exact prefix of the 2000 tier."""
    if clean500.role != "train_clean_500" or clean500.scene_count != 500:
        raise ValueError("first manifest must be train_clean_500 with 500 scenes")
    if clean2000.role != "train_clean_2000" or clean2000.scene_count != 2000:
        raise ValueError("second manifest must be train_clean_2000 with 2000 scenes")
    for field in (
        "dataset_root",
        "source_index_path",
        "source_index_sha256",
        "excluded_identities_sha256",
        "exclusion_sources_sha256",
        "eligibility_contract_sha256",
        "selection_mode",
        "required_frames",
    ):
        if getattr(clean500, field) != getattr(clean2000, field):
            raise ValueError(f"clean500/2000 contract mismatch: {field}")
    if clean500.entries != clean2000.entries[:500]:
        raise ValueError("TRAIN-CLEAN-500 is not the exact TRAIN-CLEAN-2000 prefix")
    if clean500.identities != clean2000.identities[:500]:
        raise ValueError("TRAIN-CLEAN-500 identity prefix mismatch")
    if clean500.scene_paths != clean2000.scene_paths[:500]:
        raise ValueError("TRAIN-CLEAN-500 resolved-path prefix mismatch")
    if clean500.source_positions != clean2000.source_positions[:500]:
        raise ValueError("TRAIN-CLEAN-500 source-position prefix mismatch")
    if clean500.excluded_identities != clean2000.excluded_identities:
        raise ValueError("TRAIN-CLEAN-500/2000 excluded identity lists differ")
    if clean500.scene_list_sha256 != scene_list_sha256(clean500.entries):
        raise ValueError("TRAIN-CLEAN-500 scene-list binding is inconsistent")
    if clean2000.scene_list_sha256 != scene_list_sha256(clean2000.entries):
        raise ValueError("TRAIN-CLEAN-2000 scene-list binding is inconsistent")


@dataclass(frozen=True)
class ExposurePlan:
    """Static DDP exposure and optimizer-step accounting."""

    format: str
    manifest_unique_scenes: int
    selected_unique_scenes_per_epoch: int
    world_size: int
    epochs: int
    update_multiplier: int
    per_rank_exposures_per_epoch: int
    sampler_padding_exposures_per_epoch: int
    effective_exposures_per_epoch: int
    planned_optimizer_updates: int
    effective_exposures_total: int
    per_rank_exposures_total: tuple[int, ...]

    def checkpoint_record(self) -> dict[str, Any]:
        result = asdict(self)
        result["per_rank_exposures_total"] = list(self.per_rank_exposures_total)
        return result


def plan_ddp_exposures(
    *,
    train_scene_count: int,
    scenes_per_epoch: int,
    world_size: int,
    epochs: int,
    update_multiplier: int = 1,
) -> ExposurePlan:
    """Compute scheduler steps and effective exposures before training.

    ``planned_optimizer_updates`` is the number of synchronized DDP updates,
    i.e. ``epochs * ceil(selected_scenes/world_size) * update_multiplier``.
    The effective exposure count includes deterministic sampler padding.
    """
    train_scene_count = _require_positive_int(
        train_scene_count, "train_scene_count"
    )
    if isinstance(scenes_per_epoch, bool) or not isinstance(scenes_per_epoch, int):
        raise ValueError("scenes_per_epoch must be an integer")
    world_size = _require_positive_int(world_size, "world_size")
    epochs = _require_positive_int(epochs, "epochs")
    update_multiplier = _require_positive_int(
        update_multiplier, "update_multiplier"
    )
    selected = (
        min(scenes_per_epoch, train_scene_count)
        if scenes_per_epoch > 0
        else train_scene_count
    )
    local_base = math.ceil(selected / world_size)
    padding_base = local_base * world_size - selected
    local_effective = local_base * update_multiplier
    effective_epoch = local_effective * world_size
    padding_effective = padding_base * update_multiplier
    planned_updates = epochs * local_effective
    per_rank_total = tuple(planned_updates for _ in range(world_size))
    return ExposurePlan(
        format=EXPOSURE_PLAN_FORMAT,
        manifest_unique_scenes=train_scene_count,
        selected_unique_scenes_per_epoch=selected,
        world_size=world_size,
        epochs=epochs,
        update_multiplier=update_multiplier,
        per_rank_exposures_per_epoch=local_effective,
        sampler_padding_exposures_per_epoch=padding_effective,
        effective_exposures_per_epoch=effective_epoch,
        planned_optimizer_updates=planned_updates,
        effective_exposures_total=effective_epoch * epochs,
        per_rank_exposures_total=per_rank_total,
    )


def epoch_ddp_shards(
    entries: Sequence[str],
    *,
    world_size: int,
    update_multiplier: int = 1,
) -> tuple[list[list[str]], dict[str, Any]]:
    """Return exact equal-length rank shards plus padding identities.

    ``entries`` is the already sampled/shuffled global epoch order.  Padding
    repeats entries from its prefix, matching the existing training sampler.
    """
    if not entries:
        raise ValueError("epoch entries must be non-empty")
    world_size = _require_positive_int(world_size, "world_size")
    update_multiplier = _require_positive_int(
        update_multiplier, "update_multiplier"
    )
    normalized = [normalize_scene_entry(entry) for entry in entries]
    identities = [scene_identity(entry) for entry in normalized]
    if len(identities) != len(set(identities)):
        raise ValueError("epoch entries contain duplicate identities before padding")
    local_base = math.ceil(len(normalized) / world_size)
    padding_count = local_base * world_size - len(normalized)
    padding_entries = [
        normalized[index % len(normalized)] for index in range(padding_count)
    ]
    padded = normalized + padding_entries
    shards: list[list[str]] = []
    for rank in range(world_size):
        shard = padded[rank * local_base : (rank + 1) * local_base]
        if update_multiplier > 1:
            shard = [entry for entry in shard for _ in range(update_multiplier)]
        shards.append(shard)
    accounting = {
        "selected_unique_scenes": len(normalized),
        "world_size": world_size,
        "update_multiplier": update_multiplier,
        "sampler_padding_entries": padding_entries,
        "sampler_padding_identities": [scene_identity(entry) for entry in padding_entries],
        "sampler_padding_exposures": padding_count * update_multiplier,
        "effective_exposures": len(padded) * update_multiplier,
        "per_rank_exposures": [len(shard) for shard in shards],
    }
    return shards, accounting


def _epoch_sampler_seed(seed: int, epoch: int) -> int:
    """Derive the versioned epoch seed used by the clean-training sampler."""
    # ``seed=0`` intentionally preserves the existing ``Random(epoch)`` order.
    # The sampler version stored beside it makes this arithmetic part of the
    # checkpoint contract rather than an undocumented implementation detail.
    return seed + epoch


def _sample_epoch_order(
    entries: Sequence[str],
    *,
    sampling_mode: str,
    requested_scenes_per_epoch: int,
    seed: int,
    epoch: int,
) -> list[str]:
    """Select and shuffle the global epoch order from frozen sampler inputs."""
    normalized = [normalize_scene_entry(entry) for entry in entries]
    identities = [scene_identity(entry) for entry in normalized]
    if not normalized:
        raise ValueError("sampler entries must be non-empty")
    if len(identities) != len(set(identities)):
        raise ValueError("sampler entries contain duplicate scene identities")
    rng = random.Random(_epoch_sampler_seed(seed, epoch))
    if sampling_mode == "sample_cap":
        return rng.sample(
            normalized,
            min(requested_scenes_per_epoch, len(normalized)),
        )
    result = list(normalized)
    rng.shuffle(result)
    return result


@dataclass(frozen=True)
class EpochSamplingState:
    """Exact epoch-order/shard binding required for a mid-epoch resume."""

    format: str
    sampler_version: str
    sampling_mode: str
    requested_scenes_per_epoch: int
    seed: int
    epoch: int
    sampler_input_scene_list_sha256: str
    global_order_sha256: str
    selected_unique_scenes: int
    world_size: int
    update_multiplier: int
    per_rank_shard_sha256: tuple[str, ...]
    sampler_padding_identities: tuple[str, ...]
    per_rank_exposures: tuple[int, ...]

    def checkpoint_record(self) -> dict[str, Any]:
        result = asdict(self)
        result["per_rank_shard_sha256"] = list(self.per_rank_shard_sha256)
        result["sampler_padding_identities"] = list(
            self.sampler_padding_identities
        )
        result["per_rank_exposures"] = list(self.per_rank_exposures)
        return result


def build_epoch_sampling_state(
    entries: Sequence[str],
    *,
    world_size: int,
    update_multiplier: int,
    sampling_mode: str,
    requested_scenes_per_epoch: int,
    seed: int,
    epoch: int,
) -> tuple[list[list[str]], EpochSamplingState]:
    """Build shards and bind the exact sampler invocation that produced them."""
    if sampling_mode not in {"shuffle_all", "sample_cap"}:
        raise ValueError("sampling_mode must be shuffle_all or sample_cap")
    if isinstance(requested_scenes_per_epoch, bool) or not isinstance(
        requested_scenes_per_epoch, int
    ):
        raise ValueError("requested_scenes_per_epoch must be an integer")
    if sampling_mode == "shuffle_all" and requested_scenes_per_epoch > 0:
        raise ValueError("shuffle_all requires a non-positive scenes_per_epoch request")
    if sampling_mode == "sample_cap" and requested_scenes_per_epoch <= 0:
        raise ValueError("sample_cap requires a positive scenes_per_epoch request")
    seed = _require_nonnegative_int(seed, "seed")
    epoch = _require_nonnegative_int(epoch, "epoch")
    global_order = _sample_epoch_order(
        entries,
        sampling_mode=sampling_mode,
        requested_scenes_per_epoch=requested_scenes_per_epoch,
        seed=seed,
        epoch=epoch,
    )
    shards, accounting = epoch_ddp_shards(
        global_order,
        world_size=world_size,
        update_multiplier=update_multiplier,
    )
    state = EpochSamplingState(
        format=EPOCH_SAMPLING_STATE_FORMAT,
        sampler_version=EPOCH_SAMPLER_VERSION,
        sampling_mode=sampling_mode,
        requested_scenes_per_epoch=requested_scenes_per_epoch,
        seed=seed,
        epoch=epoch,
        sampler_input_scene_list_sha256=scene_list_sha256(entries),
        global_order_sha256=scene_list_sha256(global_order),
        selected_unique_scenes=len(global_order),
        world_size=world_size,
        update_multiplier=update_multiplier,
        per_rank_shard_sha256=tuple(scene_list_sha256(shard) for shard in shards),
        sampler_padding_identities=tuple(
            accounting["sampler_padding_identities"]
        ),
        per_rank_exposures=tuple(accounting["per_rank_exposures"]),
    )
    return shards, state


@dataclass(frozen=True)
class RealizedTrainingAccounting:
    """Cumulative processed/successful/skipped exposure accounting."""

    format: str
    completed_epochs: int
    next_scene_index: int
    processed_optimizer_opportunities: int
    processed_exposures: int
    successful_optimizer_updates: int
    successful_exposures: int
    skipped_optimizer_updates: int
    skipped_exposures: int
    skip_reason_counts: tuple[tuple[str, int], ...]
    per_rank_processed_exposures: tuple[int, ...]
    per_rank_successful_optimizer_updates: tuple[int, ...]
    per_rank_skipped_exposures: tuple[int, ...]

    def checkpoint_record(self) -> dict[str, Any]:
        result = asdict(self)
        result["skip_reason_counts"] = {
            reason: count for reason, count in self.skip_reason_counts
        }
        result["per_rank_processed_exposures"] = list(
            self.per_rank_processed_exposures
        )
        result["per_rank_successful_optimizer_updates"] = list(
            self.per_rank_successful_optimizer_updates
        )
        result["per_rank_skipped_exposures"] = list(
            self.per_rank_skipped_exposures
        )
        return result


def build_realized_training_accounting(
    *,
    exposure_plan: ExposurePlan,
    epoch_sampling_state: EpochSamplingState,
    completed_epochs: int,
    next_scene_index: int,
    successful_optimizer_updates: int,
    skip_reason_counts: Mapping[str, int],
) -> RealizedTrainingAccounting:
    """Validate and freeze exact progress at one scene-boundary checkpoint."""
    _validate_exposure_plan(exposure_plan)
    _validate_epoch_sampling_state(epoch_sampling_state, exposure_plan)
    completed_epochs = _require_nonnegative_int(completed_epochs, "completed_epochs")
    next_scene_index = _require_nonnegative_int(next_scene_index, "next_scene_index")
    successful_optimizer_updates = _require_nonnegative_int(
        successful_optimizer_updates, "successful_optimizer_updates"
    )
    if completed_epochs > exposure_plan.epochs:
        raise ValueError("completed_epochs exceeds the exposure plan")
    if next_scene_index > exposure_plan.per_rank_exposures_per_epoch:
        raise ValueError("next_scene_index exceeds the per-rank epoch shard")
    if completed_epochs == exposure_plan.epochs and next_scene_index != 0:
        raise ValueError("a completed plan cannot have an active scene index")
    expected_sampling_epoch = (
        completed_epochs if next_scene_index > 0 else max(completed_epochs - 1, 0)
    )
    if epoch_sampling_state.epoch != expected_sampling_epoch:
        raise ValueError(
            "epoch sampling state does not match completed_epochs/next_scene_index"
        )
    if epoch_sampling_state.world_size != exposure_plan.world_size:
        raise ValueError("sampling/exposure world_size mismatch")
    if epoch_sampling_state.update_multiplier != exposure_plan.update_multiplier:
        raise ValueError("sampling/exposure update_multiplier mismatch")
    if tuple(epoch_sampling_state.per_rank_exposures) != tuple(
        exposure_plan.per_rank_exposures_per_epoch
        for _ in range(exposure_plan.world_size)
    ):
        raise ValueError("sampling/exposure per-rank exposure mismatch")

    processed_opportunities = (
        completed_epochs * exposure_plan.per_rank_exposures_per_epoch
        + next_scene_index
    )
    if successful_optimizer_updates > processed_opportunities:
        raise ValueError("successful updates exceed processed exposures")
    if successful_optimizer_updates > exposure_plan.planned_optimizer_updates:
        raise ValueError("successful updates exceed planned optimizer updates")
    skipped_updates = processed_opportunities - successful_optimizer_updates
    normalized_reasons: list[tuple[str, int]] = []
    for reason, count in sorted(skip_reason_counts.items()):
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("skip reason names must be non-empty strings")
        normalized_reasons.append(
            (reason, _require_nonnegative_int(count, f"skip reason {reason}"))
        )
    if sum(count for _, count in normalized_reasons) != skipped_updates:
        raise ValueError("skip reason counts do not equal skipped optimizer updates")
    world_size = exposure_plan.world_size
    return RealizedTrainingAccounting(
        format=REALIZED_ACCOUNTING_FORMAT,
        completed_epochs=completed_epochs,
        next_scene_index=next_scene_index,
        processed_optimizer_opportunities=processed_opportunities,
        processed_exposures=processed_opportunities * world_size,
        successful_optimizer_updates=successful_optimizer_updates,
        successful_exposures=successful_optimizer_updates * world_size,
        skipped_optimizer_updates=skipped_updates,
        skipped_exposures=skipped_updates * world_size,
        skip_reason_counts=tuple(normalized_reasons),
        per_rank_processed_exposures=tuple(
            processed_opportunities for _ in range(world_size)
        ),
        per_rank_successful_optimizer_updates=tuple(
            successful_optimizer_updates for _ in range(world_size)
        ),
        per_rank_skipped_exposures=tuple(
            skipped_updates for _ in range(world_size)
        ),
    )


def validate_resume_manifest(
    checkpoint: Mapping[str, Any],
    manifest: TrainSceneManifest,
) -> None:
    """Fail closed if a checkpoint is not bound to the exact manifest bytes/list."""
    recorded = checkpoint.get("train_scene_manifest")
    if not isinstance(recorded, Mapping):
        raise ValueError("resume checkpoint has no train_scene_manifest provenance")
    if recorded.get("format") != TRAIN_MANIFEST_CHECKPOINT_FORMAT:
        raise ValueError("resume checkpoint has an invalid manifest provenance format")
    expected = manifest.checkpoint_record()
    for field in (
        "path",
        "sha256",
        "scene_list_sha256",
        "excluded_identities_sha256",
        "exclusion_sources_sha256",
        "eligibility_contract_sha256",
        "selection_mode",
        "required_frames",
        "role",
        "dataset_root",
        "scene_count",
    ):
        if recorded.get(field) != expected[field]:
            raise ValueError(
                f"resume train manifest {field} mismatch: "
                f"{recorded.get(field)!r} != {expected[field]!r}"
            )
    if recorded.get("source_index") != expected["source_index"]:
        raise ValueError("resume train manifest source_index binding mismatch")


def validate_resume_exposure_plan(
    checkpoint: Mapping[str, Any],
    exposure_plan: ExposurePlan,
) -> None:
    """Reject world-size/epoch/sampling changes that alter the LR schedule."""
    _validate_exposure_plan(exposure_plan)
    recorded = checkpoint.get("training_exposure_plan")
    expected = exposure_plan.checkpoint_record()
    if recorded != expected:
        raise ValueError(
            "resume training exposure plan mismatch; refusing scheduler/data drift"
        )


def _validate_exposure_plan(exposure_plan: ExposurePlan) -> None:
    """Recompute every derived exposure field rather than trusting a dataclass."""
    if exposure_plan.format != EXPOSURE_PLAN_FORMAT:
        raise ValueError("invalid training exposure plan format")
    expected = plan_ddp_exposures(
        train_scene_count=exposure_plan.manifest_unique_scenes,
        scenes_per_epoch=exposure_plan.selected_unique_scenes_per_epoch,
        world_size=exposure_plan.world_size,
        epochs=exposure_plan.epochs,
        update_multiplier=exposure_plan.update_multiplier,
    )
    if exposure_plan != expected:
        raise ValueError("training exposure plan is internally inconsistent")


def _file_binding(path: str | Path, *, label: str) -> dict[str, str]:
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} is not a file: {resolved}")
    return {"path": str(resolved), "sha256": file_sha256(resolved)}


def _validate_file_binding(record: object, *, label: str) -> Path:
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} binding is missing")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label}.path must be a non-empty absolute path")
    unresolved = Path(raw_path).expanduser()
    if not unresolved.is_absolute():
        raise ValueError(f"{label}.path must be absolute")
    path = unresolved.resolve(strict=True)
    if raw_path != str(path) or not path.is_file():
        raise ValueError(f"{label}.path must resolve to a file")
    expected = _require_sha256(record.get("sha256"), f"{label}.sha256")
    actual = file_sha256(path)
    if expected != actual:
        raise ValueError(f"{label} SHA256 mismatch ({expected} != {actual})")
    return path


def _validate_provenance_training_args(args: Mapping[str, Any]) -> None:
    missing = sorted(REQUIRED_PROVENANCE_TRAINING_ARGS - set(args))
    if missing:
        raise ValueError(f"training provenance is missing resume-critical args: {missing}")
    if args.get("deterministic") is not True:
        raise ValueError("clean training provenance requires deterministic=true")
    _require_nonnegative_int(
        args.get("train_seed"), "training provenance train_seed"
    )
    _require_positive_int(args.get("epochs"), "training provenance epochs")
    _require_positive_int(args.get("world_size"), "training provenance world_size")
    if isinstance(args.get("scenes_per_epoch"), bool) or not isinstance(
        args.get("scenes_per_epoch"), int
    ):
        raise ValueError("training provenance scenes_per_epoch must be an integer")
    if args.get("train_update_phase") not in TRAIN_UPDATE_PHASES:
        raise ValueError(
            "training provenance train_update_phase must be one of "
            f"{sorted(TRAIN_UPDATE_PHASES)}"
        )
    for field in ("dataset_root", "train_scene_manifest"):
        if not isinstance(args.get(field), str) or not args.get(field):
            raise ValueError(f"training provenance {field} must be a path string")
    if args.get("sampler_version") != EPOCH_SAMPLER_VERSION:
        raise ValueError("training provenance sampler_version mismatch")


def build_training_provenance(
    *,
    base_checkpoint: str | Path,
    config: str | Path,
    initial_gaussian_checkpoint: str | Path | None,
    code_fingerprint: str,
    training_args: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind every external starting artifact and the resume-relevant CLI."""
    _require_sha256(code_fingerprint, "code_fingerprint")
    args_copy = dict(training_args)
    _validate_provenance_training_args(args_copy)
    args_sha = _canonical_json_sha256(args_copy)
    return {
        "format": TRAINING_PROVENANCE_FORMAT,
        "base_checkpoint": _file_binding(base_checkpoint, label="base checkpoint"),
        "config": _file_binding(config, label="config"),
        "initial_gaussian_checkpoint": (
            None
            if initial_gaussian_checkpoint is None
            else _file_binding(
                initial_gaussian_checkpoint,
                label="initial Gaussian checkpoint",
            )
        ),
        "code_fingerprint": code_fingerprint,
        "training_args": args_copy,
        "training_args_sha256": args_sha,
    }


def _validate_training_provenance_record(record: object) -> Mapping[str, Any]:
    if not isinstance(record, Mapping) or record.get("format") != TRAINING_PROVENANCE_FORMAT:
        raise ValueError("invalid training provenance record")
    for field in ("base_checkpoint", "config"):
        _validate_file_binding(
            record.get(field), label=f"training provenance {field}"
        )
    initial = record.get("initial_gaussian_checkpoint")
    if initial is not None:
        _validate_file_binding(
            initial, label="training provenance initial_gaussian_checkpoint"
        )
    _require_sha256(record.get("code_fingerprint"), "training provenance code_fingerprint")
    args = record.get("training_args")
    if not isinstance(args, Mapping):
        raise ValueError("training provenance training_args must be an object")
    _validate_provenance_training_args(args)
    expected_args_sha = _require_sha256(
        record.get("training_args_sha256"),
        "training provenance training_args_sha256",
    )
    actual_args_sha = _canonical_json_sha256(dict(args))
    if expected_args_sha != actual_args_sha:
        raise ValueError("training provenance training_args SHA256 mismatch")
    return record


def seed_all_rng(seed: int, *, include_cuda: bool = True) -> None:
    """Seed Python, NumPy, Torch CPU and (when available) CUDA RNGs."""
    seed = _require_nonnegative_int(seed, "seed")
    if seed > 2**32 - 1:
        raise ValueError("seed must fit NumPy's uint32 seed range")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if include_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state(
    *,
    include_cuda: bool = True,
    cuda_device: torch.device | str | int | None = None,
) -> dict[str, Any]:
    """Capture one rank's complete RNG state without touching other GPUs."""
    cuda_state = None
    if include_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA RNG state requested but CUDA is unavailable")
        device = torch.cuda.current_device() if cuda_device is None else cuda_device
        cuda_state = torch.cuda.get_rng_state(device=device).cpu().clone()
    numpy_state = np.random.get_state()
    numpy_safe = {
        "bit_generator": str(numpy_state[0]),
        "keys": torch.from_numpy(numpy_state[1].copy()),
        "position": int(numpy_state[2]),
        "has_gauss": int(numpy_state[3]),
        "cached_gaussian": float(numpy_state[4]),
    }
    return {
        "format": RNG_STATE_FORMAT,
        "python": random.getstate(),
        "numpy": numpy_safe,
        "torch_cpu": torch.get_rng_state().cpu().clone(),
        "torch_cuda": cuda_state,
    }


def _validate_rng_state(state: object, *, require_cuda: bool) -> Mapping[str, Any]:
    if not isinstance(state, Mapping) or state.get("format") != RNG_STATE_FORMAT:
        raise ValueError("invalid or missing RNG state")
    python_state = state.get("python")
    try:
        random.Random().setstate(python_state)
    except (TypeError, ValueError) as exc:
        raise ValueError("RNG state has an invalid Python state") from exc
    numpy_state = state.get("numpy")
    if not isinstance(numpy_state, Mapping):
        raise ValueError("RNG state is missing NumPy state")
    if numpy_state.get("bit_generator") != "MT19937":
        raise ValueError("NumPy RNG bit generator must be MT19937")
    numpy_keys = numpy_state.get("keys")
    if (
        not isinstance(numpy_keys, torch.Tensor)
        or numpy_keys.dtype != torch.uint32
        or numpy_keys.ndim != 1
        or numpy_keys.numel() != 624
    ):
        raise ValueError("NumPy RNG keys must be a 624-element uint32 tensor")
    position = _require_nonnegative_int(numpy_state.get("position"), "NumPy RNG position")
    if position > 624:
        raise ValueError("NumPy RNG position is out of range")
    has_gauss = _require_nonnegative_int(
        numpy_state.get("has_gauss"), "NumPy RNG has_gauss"
    )
    if has_gauss not in {0, 1}:
        raise ValueError("NumPy RNG has_gauss must be zero or one")
    cached_gaussian = numpy_state.get("cached_gaussian")
    if (
        isinstance(cached_gaussian, bool)
        or not isinstance(cached_gaussian, Real)
        or not math.isfinite(float(cached_gaussian))
    ):
        raise ValueError("NumPy RNG cached_gaussian must be finite")
    try:
        probe = np.random.RandomState()
        probe.set_state(
            (
                "MT19937",
                numpy_keys.detach().cpu().numpy().astype(np.uint32, copy=True),
                position,
                has_gauss,
                float(cached_gaussian),
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("RNG state has an invalid NumPy state") from exc

    cpu_state = state.get("torch_cpu")
    if (
        not isinstance(cpu_state, torch.Tensor)
        or cpu_state.dtype != torch.uint8
        or cpu_state.ndim != 1
        or cpu_state.numel() == 0
    ):
        raise ValueError("RNG state is missing Torch CPU state")
    try:
        torch.Generator(device="cpu").set_state(cpu_state.detach().cpu())
    except RuntimeError as exc:
        raise ValueError("RNG state has an invalid Torch CPU state") from exc
    cuda_state = state.get("torch_cuda")
    if require_cuda and not isinstance(cuda_state, torch.Tensor):
        raise ValueError("RNG state is missing Torch CUDA state")
    if cuda_state is not None and (
        not isinstance(cuda_state, torch.Tensor)
        or cuda_state.dtype != torch.uint8
        or cuda_state.ndim != 1
        or cuda_state.numel() == 0
    ):
        raise ValueError("Torch CUDA RNG state must be a non-empty uint8 tensor or null")
    return state


def restore_rng_state(
    state: object,
    *,
    require_cuda: bool = True,
    cuda_device: torch.device | str | int | None = None,
) -> None:
    """Restore one rank's RNG state, remapping CUDA state to its current device."""
    checked = _validate_rng_state(state, require_cuda=require_cuda)
    cuda_state = checked["torch_cuda"]
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint has CUDA RNG state but CUDA is unavailable")
        device = torch.cuda.current_device() if cuda_device is None else cuda_device
        try:
            torch.Generator(device=device).set_state(cuda_state.detach().cpu())
        except RuntimeError as exc:
            raise ValueError("RNG state has an invalid Torch CUDA state") from exc
    numpy_state = checked["numpy"]
    decoded_numpy_state = (
        "MT19937",
        numpy_state["keys"].detach().cpu().numpy().astype(np.uint32, copy=True),
        int(numpy_state["position"]),
        int(numpy_state["has_gauss"]),
        float(numpy_state["cached_gaussian"]),
    )
    # All validation, including CUDA availability, has completed before any
    # global generator is mutated.  Restore is therefore fail-atomic.
    random.setstate(checked["python"])
    np.random.set_state(decoded_numpy_state)
    torch.set_rng_state(checked["torch_cpu"].detach().cpu())
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state.cpu(), device=device)


def _clone_rng_state(state: Mapping[str, Any]) -> dict[str, Any]:
    numpy_state = state["numpy"]
    return {
        "format": RNG_STATE_FORMAT,
        "python": state["python"],
        "numpy": {
            "bit_generator": numpy_state["bit_generator"],
            "keys": numpy_state["keys"].detach().cpu().clone(),
            "position": int(numpy_state["position"]),
            "has_gauss": int(numpy_state["has_gauss"]),
            "cached_gaussian": float(numpy_state["cached_gaussian"]),
        },
        "torch_cpu": state["torch_cpu"].detach().cpu().clone(),
        "torch_cuda": (
            None
            if state["torch_cuda"] is None
            else state["torch_cuda"].detach().cpu().clone()
        ),
    }


def pack_rank_rng_states(
    states: Sequence[object],
    *,
    world_size: int,
    require_cuda: bool = True,
) -> dict[str, Any]:
    """Validate and pack the state gathered from every DDP rank."""
    world_size = _require_positive_int(world_size, "world_size")
    if len(states) != world_size:
        raise ValueError(
            f"expected RNG state for {world_size} ranks, received {len(states)}"
        )
    checked = [
        _clone_rng_state(_validate_rng_state(state, require_cuda=require_cuda))
        for state in states
    ]
    return {
        "format": RANK_RNG_BUNDLE_FORMAT,
        "world_size": world_size,
        "states": checked,
    }


def rank_rng_state_for_resume(
    checkpoint: Mapping[str, Any],
    *,
    rank: int,
    world_size: int,
    require_cuda: bool = True,
) -> Mapping[str, Any]:
    """Select one rank's RNG state and reject incomplete/world-size drift."""
    world_size = _require_positive_int(world_size, "world_size")
    rank = _require_nonnegative_int(rank, "rank")
    bundle = checkpoint.get("rng_state_by_rank")
    if not isinstance(bundle, Mapping) or bundle.get("format") != RANK_RNG_BUNDLE_FORMAT:
        raise ValueError("resume checkpoint has no complete per-rank RNG bundle")
    if bundle.get("world_size") != world_size:
        raise ValueError(
            "resume RNG world_size mismatch: "
            f"{bundle.get('world_size')!r} != {world_size}"
        )
    states = bundle.get("states")
    if not isinstance(states, Sequence) or isinstance(states, (str, bytes)):
        raise ValueError("resume RNG bundle states must be a sequence")
    if len(states) != world_size:
        raise ValueError("resume RNG bundle is incomplete")
    if not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size)")
    checked = [
        _validate_rng_state(state, require_cuda=require_cuda) for state in states
    ]
    return checked[rank]


def _validate_epoch_sampling_state(
    state: EpochSamplingState,
    exposure_plan: ExposurePlan,
) -> None:
    if state.format != EPOCH_SAMPLING_STATE_FORMAT:
        raise ValueError("invalid epoch sampling state format")
    if state.sampler_version != EPOCH_SAMPLER_VERSION:
        raise ValueError("epoch sampler version mismatch")
    if state.sampling_mode not in {"shuffle_all", "sample_cap"}:
        raise ValueError("invalid epoch sampling mode")
    if isinstance(state.requested_scenes_per_epoch, bool) or not isinstance(
        state.requested_scenes_per_epoch, int
    ):
        raise ValueError("epoch sampling request must be an integer")
    if state.sampling_mode == "shuffle_all" and state.requested_scenes_per_epoch > 0:
        raise ValueError("shuffle_all sampling request drift")
    if state.sampling_mode == "sample_cap" and state.requested_scenes_per_epoch <= 0:
        raise ValueError("sample_cap sampling request drift")
    _require_nonnegative_int(state.seed, "epoch sampling seed")
    _require_nonnegative_int(state.epoch, "epoch sampling epoch")
    _require_sha256(
        state.sampler_input_scene_list_sha256,
        "sampler input scene-list SHA256",
    )
    _require_sha256(state.global_order_sha256, "epoch global_order_sha256")
    if state.selected_unique_scenes != exposure_plan.selected_unique_scenes_per_epoch:
        raise ValueError("epoch sampling selected-scene count mismatch")
    expected_selected = (
        exposure_plan.manifest_unique_scenes
        if state.sampling_mode == "shuffle_all"
        else min(
            state.requested_scenes_per_epoch,
            exposure_plan.manifest_unique_scenes,
        )
    )
    if state.selected_unique_scenes != expected_selected:
        raise ValueError("epoch sampling mode/request selected-scene mismatch")
    if state.world_size != exposure_plan.world_size:
        raise ValueError("epoch sampling world_size mismatch")
    if state.update_multiplier != exposure_plan.update_multiplier:
        raise ValueError("epoch sampling update_multiplier mismatch")
    if len(state.per_rank_shard_sha256) != state.world_size:
        raise ValueError("epoch sampling shard hashes are incomplete")
    for digest in state.per_rank_shard_sha256:
        _require_sha256(digest, "per-rank shard SHA256")
    expected_padding = (
        math.ceil(state.selected_unique_scenes / state.world_size) * state.world_size
        - state.selected_unique_scenes
    )
    if len(state.sampler_padding_identities) != expected_padding:
        raise ValueError("epoch sampling padding identities are incomplete")
    for identity in state.sampler_padding_identities:
        if _SCENE_ID_PATTERN.fullmatch(identity) is None:
            raise ValueError("epoch sampling has an invalid padding identity")
    expected_exposures = tuple(
        exposure_plan.per_rank_exposures_per_epoch
        for _ in range(exposure_plan.world_size)
    )
    if state.per_rank_exposures != expected_exposures:
        raise ValueError("epoch sampling per-rank exposures mismatch")


def _validate_realized_accounting(
    accounting: RealizedTrainingAccounting,
    *,
    exposure_plan: ExposurePlan,
    epoch_sampling_state: EpochSamplingState,
) -> None:
    recomputed = build_realized_training_accounting(
        exposure_plan=exposure_plan,
        epoch_sampling_state=epoch_sampling_state,
        completed_epochs=accounting.completed_epochs,
        next_scene_index=accounting.next_scene_index,
        successful_optimizer_updates=accounting.successful_optimizer_updates,
        skip_reason_counts=dict(accounting.skip_reason_counts),
    )
    if accounting != recomputed:
        raise ValueError("realized training accounting is internally inconsistent")


def _validate_model_state_schema(model_state: object) -> None:
    if not isinstance(model_state, Mapping) or not model_state:
        raise ValueError("resume checkpoint is missing model state")
    for name, tensor in model_state.items():
        if not isinstance(name, str) or not name:
            raise ValueError("resume checkpoint model state has an invalid key")
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("resume checkpoint model state has a non-tensor value")


def _validate_optimizer_state_schema(optimizer: object) -> None:
    if not isinstance(optimizer, Mapping) or not optimizer:
        raise ValueError("resume checkpoint is missing optimizer state")
    state = optimizer.get("state")
    groups = optimizer.get("param_groups")
    if not isinstance(state, Mapping):
        raise ValueError("resume checkpoint optimizer state must contain state")
    if (
        not isinstance(groups, Sequence)
        or isinstance(groups, (str, bytes))
        or not groups
    ):
        raise ValueError("resume checkpoint optimizer param_groups are missing")
    parameter_ids: list[int] = []
    parameter_amsgrad: dict[int, bool] = {}
    for group_index, group in enumerate(groups):
        if not isinstance(group, Mapping):
            raise ValueError(f"optimizer param_groups[{group_index}] is not an object")
        missing_fields = sorted(_ADAMW_GROUP_FIELDS - set(group))
        if missing_fields:
            raise ValueError(
                f"optimizer param_groups[{group_index}] is missing AdamW fields: "
                f"{missing_fields}"
            )
        learning_rate = _require_finite_real(
            group["lr"], f"optimizer param_groups[{group_index}].lr"
        )
        if learning_rate < 0:
            raise ValueError("optimizer learning rates must be non-negative")
        betas = group["betas"]
        if (
            not isinstance(betas, Sequence)
            or isinstance(betas, (str, bytes))
            or len(betas) != 2
        ):
            raise ValueError("optimizer AdamW betas must contain two values")
        beta_values = tuple(
            _require_finite_real(beta, "optimizer AdamW beta") for beta in betas
        )
        if any(beta < 0 or beta >= 1 for beta in beta_values):
            raise ValueError("optimizer AdamW betas must be in [0, 1)")
        if _require_finite_real(group["eps"], "optimizer AdamW eps") <= 0:
            raise ValueError("optimizer AdamW eps must be positive")
        if (
            _require_finite_real(
                group["weight_decay"], "optimizer AdamW weight_decay"
            )
            < 0
        ):
            raise ValueError("optimizer AdamW weight_decay must be non-negative")
        for flag in ("amsgrad", "maximize", "capturable", "differentiable"):
            if not isinstance(group[flag], bool):
                raise ValueError(f"optimizer AdamW {flag} must be boolean")
        for optional_flag in ("foreach", "fused"):
            if group[optional_flag] is not None and not isinstance(
                group[optional_flag], bool
            ):
                raise ValueError(
                    f"optimizer AdamW {optional_flag} must be boolean or null"
                )
        params = group.get("params")
        if (
            not isinstance(params, Sequence)
            or isinstance(params, (str, bytes))
            or not params
        ):
            raise ValueError(
                f"optimizer param_groups[{group_index}].params are missing"
            )
        for parameter_id in params:
            checked_id = _require_nonnegative_int(
                parameter_id,
                f"optimizer param_groups[{group_index}] parameter id",
            )
            parameter_ids.append(checked_id)
            parameter_amsgrad[checked_id] = bool(group["amsgrad"])
    if len(parameter_ids) != len(set(parameter_ids)):
        raise ValueError("optimizer parameter ids are duplicated across param_groups")
    known_parameters = set(parameter_ids)
    for parameter_id, parameter_state in state.items():
        checked_id = _require_nonnegative_int(
            parameter_id, "optimizer state parameter id"
        )
        if checked_id not in known_parameters:
            raise ValueError("optimizer state refers to an unknown parameter id")
        if not isinstance(parameter_state, Mapping):
            raise ValueError("optimizer per-parameter state must be an object")
        required_state = {"step", "exp_avg", "exp_avg_sq"}
        if parameter_amsgrad[checked_id]:
            required_state.add("max_exp_avg_sq")
        missing_state = sorted(required_state - set(parameter_state))
        if missing_state:
            raise ValueError(
                f"optimizer AdamW state for parameter {checked_id} is missing: "
                f"{missing_state}"
            )
        step = parameter_state["step"]
        if (
            not isinstance(step, torch.Tensor)
            or step.numel() != 1
            or not torch.isfinite(step.detach().cpu()).item()
            or float(step.detach().cpu().item()) < 0
        ):
            raise ValueError("optimizer AdamW step must be a finite scalar tensor")
        first_moment = parameter_state["exp_avg"]
        second_moment = parameter_state["exp_avg_sq"]
        if (
            not isinstance(first_moment, torch.Tensor)
            or not isinstance(second_moment, torch.Tensor)
            or first_moment.shape != second_moment.shape
        ):
            raise ValueError("optimizer AdamW moment tensors have an invalid schema")
        if parameter_amsgrad[checked_id]:
            maximum_second_moment = parameter_state["max_exp_avg_sq"]
            if (
                not isinstance(maximum_second_moment, torch.Tensor)
                or maximum_second_moment.shape != second_moment.shape
            ):
                raise ValueError("optimizer AdamW AMSGrad tensor has an invalid schema")


def _validate_scheduler_state_schema(scheduler: object) -> Mapping[str, Any]:
    if not isinstance(scheduler, Mapping) or not scheduler:
        raise ValueError("resume checkpoint is missing scheduler state")
    missing = sorted(_LAMBDA_LR_STATE_FIELDS - set(scheduler))
    if missing:
        raise ValueError(f"resume LambdaLR scheduler state is missing: {missing}")
    base_lrs = scheduler["base_lrs"]
    last_lrs = scheduler["_last_lr"]
    lambdas = scheduler["lr_lambdas"]
    if any(
        not isinstance(values, Sequence) or isinstance(values, (str, bytes))
        for values in (base_lrs, last_lrs, lambdas)
    ):
        raise ValueError("resume LambdaLR scheduler lists have an invalid schema")
    if not base_lrs or len(base_lrs) != len(last_lrs) or len(base_lrs) != len(lambdas):
        raise ValueError("resume LambdaLR scheduler group counts mismatch")
    for index, value in enumerate(base_lrs):
        if _require_finite_real(value, f"scheduler base_lrs[{index}]") < 0:
            raise ValueError("scheduler base learning rates must be non-negative")
    for index, value in enumerate(last_lrs):
        if _require_finite_real(value, f"scheduler _last_lr[{index}]") < 0:
            raise ValueError("scheduler last learning rates must be non-negative")
    if any(value is not None for value in lambdas):
        raise ValueError("weights-only-safe LambdaLR state requires null lr_lambdas")
    last_epoch = _require_nonnegative_int(scheduler["last_epoch"], "scheduler last_epoch")
    step_count = _require_positive_int(scheduler["_step_count"], "scheduler step count")
    if step_count != last_epoch + 1:
        raise ValueError("resume LambdaLR scheduler step count is inconsistent")
    for flag in ("verbose", "_get_lr_called_within_step"):
        if not isinstance(scheduler[flag], bool):
            raise ValueError(f"resume LambdaLR scheduler {flag} must be boolean")
    return scheduler


def validate_resume_contract(
    checkpoint: Mapping[str, Any],
    *,
    manifest: TrainSceneManifest,
    exposure_plan: ExposurePlan,
    epoch_sampling_state: EpochSamplingState,
    realized_accounting: RealizedTrainingAccounting,
    training_provenance: Mapping[str, Any],
    rank: int,
    world_size: int,
    require_cuda_rng: bool = True,
) -> Mapping[str, Any]:
    """Validate all clean-training resume invariants before loading state."""
    if checkpoint.get("format") != CLEAN_CHECKPOINT_FORMAT:
        raise ValueError("resume checkpoint is not a clean-training checkpoint")
    _validate_exposure_plan(exposure_plan)
    if exposure_plan.manifest_unique_scenes != manifest.scene_count:
        raise ValueError("resume exposure plan/manifest scene-count mismatch")
    if epoch_sampling_state.sampler_input_scene_list_sha256 != (
        manifest.scene_list_sha256
    ):
        raise ValueError("resume sampler input/manifest scene-list mismatch")
    _, recomputed_sampling_state = build_epoch_sampling_state(
        manifest.entries,
        world_size=epoch_sampling_state.world_size,
        update_multiplier=epoch_sampling_state.update_multiplier,
        sampling_mode=epoch_sampling_state.sampling_mode,
        requested_scenes_per_epoch=(
            epoch_sampling_state.requested_scenes_per_epoch
        ),
        seed=epoch_sampling_state.seed,
        epoch=epoch_sampling_state.epoch,
    )
    if epoch_sampling_state != recomputed_sampling_state:
        raise ValueError("resume epoch sampling state is not sampler-derived")
    if world_size != exposure_plan.world_size:
        raise ValueError("resume world_size does not match the exposure plan")
    _validate_model_state_schema(checkpoint.get("model"))
    if checkpoint.get("optimizer_class") != CLEAN_OPTIMIZER_CLASS:
        raise ValueError("resume checkpoint optimizer class mismatch")
    if checkpoint.get("scheduler_class") != CLEAN_SCHEDULER_CLASS:
        raise ValueError("resume checkpoint scheduler class mismatch")
    optimizer = checkpoint.get("optimizer")
    _validate_optimizer_state_schema(optimizer)
    scheduler = _validate_scheduler_state_schema(checkpoint.get("scheduler"))
    if len(optimizer["param_groups"]) != len(scheduler["base_lrs"]):
        raise ValueError("resume optimizer/scheduler parameter-group count mismatch")

    validate_resume_manifest(checkpoint, manifest)
    validate_resume_exposure_plan(checkpoint, exposure_plan)
    _validate_epoch_sampling_state(epoch_sampling_state, exposure_plan)
    if checkpoint.get("epoch_sampling_state") != epoch_sampling_state.checkpoint_record():
        raise ValueError("resume epoch sampling/shard state mismatch")
    _validate_realized_accounting(
        realized_accounting,
        exposure_plan=exposure_plan,
        epoch_sampling_state=epoch_sampling_state,
    )
    realized_record = realized_accounting.checkpoint_record()
    if checkpoint.get("realized_training_accounting") != realized_record:
        raise ValueError("resume realized-training accounting mismatch")
    if checkpoint.get("completed_epochs") != realized_accounting.completed_epochs:
        raise ValueError("resume completed_epochs mismatch")
    if checkpoint.get("next_scene_index") != realized_accounting.next_scene_index:
        raise ValueError("resume next_scene_index mismatch")
    if checkpoint.get("global_step") != realized_accounting.successful_optimizer_updates:
        raise ValueError("resume global_step mismatch")
    if realized_accounting.successful_optimizer_updates > 0 and not optimizer["state"]:
        raise ValueError("resume AdamW state is empty after successful updates")
    scheduler_last_epoch = scheduler.get("last_epoch")
    if (
        isinstance(scheduler_last_epoch, bool)
        or not isinstance(scheduler_last_epoch, int)
        or scheduler_last_epoch
        != realized_accounting.successful_optimizer_updates
    ):
        raise ValueError("resume scheduler/global_step mismatch")

    expected_provenance = dict(_validate_training_provenance_record(training_provenance))
    recorded_provenance = checkpoint.get("training_provenance")
    _validate_training_provenance_record(recorded_provenance)
    if recorded_provenance != expected_provenance:
        raise ValueError("resume training provenance mismatch")
    provenance_args = expected_provenance["training_args"]
    if provenance_args["dataset_root"] != manifest.dataset_root:
        raise ValueError("training provenance dataset_root mismatch")
    if provenance_args["train_scene_manifest"] != manifest.path:
        raise ValueError("training provenance train_scene_manifest mismatch")
    if provenance_args["epochs"] != exposure_plan.epochs:
        raise ValueError("training provenance epochs mismatch")
    if provenance_args["world_size"] != exposure_plan.world_size:
        raise ValueError("training provenance world_size mismatch")
    if provenance_args["scenes_per_epoch"] != (
        epoch_sampling_state.requested_scenes_per_epoch
    ):
        raise ValueError("training provenance scenes_per_epoch mismatch")
    if provenance_args["train_seed"] != epoch_sampling_state.seed:
        raise ValueError("training provenance train_seed mismatch")
    expected_multiplier = (
        2 if provenance_args["train_update_phase"] == "alternating" else 1
    )
    if expected_multiplier != exposure_plan.update_multiplier:
        raise ValueError("training provenance update phase/multiplier mismatch")

    return rank_rng_state_for_resume(
        checkpoint,
        rank=rank,
        world_size=world_size,
        require_cuda=require_cuda_rng,
    )


def identity_counts(entries: Iterable[str]) -> dict[str, int]:
    """Small logging helper for manifest/exposure diagnostics."""
    normalized = [normalize_scene_entry(entry) for entry in entries]
    return {
        "entries": len(normalized),
        "unique_entries": len(set(normalized)),
        "unique_identities": len({scene_identity(entry) for entry in normalized}),
    }
