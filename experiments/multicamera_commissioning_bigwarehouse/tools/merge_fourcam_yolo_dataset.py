#!/usr/bin/env python3
"""Fail-closed merger and dataset-card generator for four-camera YOLO data.

The four source captures remain untouched.  A successful invocation creates a
new, camera-prefixed dataset with an auditable source-to-output inventory.  An
existing output is never resumed or replaced: retries must use a new path.

This utility is intentionally study-local.  Its camera/model/topic contract is
the frozen ``warehouse_full_4cam`` commissioning setup, not a generic dataset
concatenator. Training-grade output additionally requires capture-code and
simulation-asset inventories plus success-only completion markers. The one
legacy override is diagnostic-only and is branded non-training-grade in every
merged metadata surface.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import cv2
import yaml


SCHEMA_VERSION = "fourcam_yolo_merge.v1"
CARD_SCHEMA_VERSION = "fourcam_yolo_dataset_card.v1"
CAPTURE_INVOCATION_SCHEMA_VERSION = "yolo_capture_invocation.v1"
SIMULATION_ASSET_INVENTORY_SCHEMA_VERSION = "yolo_capture_simulation_assets.v1"
CAPTURE_COMPLETION_SCHEMA_VERSION = "yolo_capture_completion.v1"
CAPTURE_TRANSPORT_SCHEMA_VERSION = "yolo_capture_transport.v1"
TRAINING_CAPTURE_TRANSPORT_VALUES = {
    "ROS_LOCALHOST_ONLY": "1",
    "IGN_IP": "127.0.0.1",
    "GZ_IP": "127.0.0.1",
}
CAPTURE_TRANSPORT_VARIABLES = (
    *TRAINING_CAPTURE_TRANSPORT_VALUES,
    "ROS_DOMAIN_ID",
    "IGN_PARTITION",
)
VERIFIED_CAPTURE_PROVENANCE = "verified_training_grade"
LEGACY_CAPTURE_PROVENANCE = "legacy_missing_diagnostic_only"
CAMERA_CONTRACTS: dict[str, dict[str, str]] = {
    "camera_A": {
        "camera_model": "external_camera",
        "image_topic": "/external_camera/image_raw",
        "labels_topic": "/external_camera/segmentation/labels_map",
    },
    "camera_B": {
        "camera_model": "external_camera_b",
        "image_topic": "/external_camera_b/image_raw",
        "labels_topic": "/external_camera_b/segmentation/labels_map",
    },
    "camera_C": {
        "camera_model": "external_camera_c",
        "image_topic": "/external_camera_c/image_raw",
        "labels_topic": "/external_camera_c/segmentation/labels_map",
    },
    "camera_D": {
        "camera_model": "external_camera_d",
        "image_topic": "/external_camera_d/image_raw",
        "labels_topic": "/external_camera_d/segmentation/labels_map",
    },
}
CAMERAS = tuple(CAMERA_CONTRACTS)
SPLITS = ("train", "val")
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp"})
SAFE_STEM = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
QUALIFICATION_FIELDS = (
    "sample_id",
    "camera_id",
    "sample_kind",
    "split",
    "image",
    "label",
    "mask",
    "robot_x",
    "robot_y",
    "robot_yaw",
    "camera_range_m",
    "range_bin",
    "mask_area_policy",
    "min_mask_area_applied_px",
    "mask_area_px",
    "localization_qualified",
    "occlusion_state",
    "occlusion_policy",
    "visible_height_fraction",
    "bottom_occlusion_px",
    "geometry_certified_no_opportunity",
    "no_opportunity_reason",
    "semantic_robot_pixels",
)


class DatasetMergeError(RuntimeError):
    """A source dataset or requested merge is unsafe or inconsistent."""


@dataclass(frozen=True)
class MergeConfig:
    expected_world: str = "warehouse_full_4cam.world.sdf"
    range_edges_m: tuple[float, ...] = (5.0, 8.0, 12.0, 16.0)
    min_train_per_camera: int = 1
    min_val_per_camera: int = 1
    min_train_per_core_range_bin: int = 1
    min_val_per_core_range_bin: int = 1
    max_cross_input_duplicate_hashes: int = 0
    min_negative_train_per_camera: int = 1
    allow_legacy_diagnostic_provenance: bool = False

    def __post_init__(self) -> None:
        if not str(self.expected_world).strip():
            raise DatasetMergeError("expected_world must not be empty")
        edges = tuple(float(value) for value in self.range_edges_m)
        if len(edges) < 2:
            raise DatasetMergeError("At least two range edges are required")
        if any(not math.isfinite(value) or value < 0.0 for value in edges):
            raise DatasetMergeError(f"Range edges must be finite and non-negative: {edges}")
        if any(right <= left for left, right in zip(edges, edges[1:])):
            raise DatasetMergeError(f"Range edges must be strictly increasing: {edges}")
        object.__setattr__(self, "range_edges_m", edges)
        for field_name in (
            "min_train_per_camera",
            "min_val_per_camera",
            "min_train_per_core_range_bin",
            "min_val_per_core_range_bin",
            "max_cross_input_duplicate_hashes",
            "min_negative_train_per_camera",
        ):
            value = int(getattr(self, field_name))
            if value < 0:
                raise DatasetMergeError(f"{field_name} must be non-negative, got {value}")
            object.__setattr__(self, field_name, value)
        if not isinstance(self.allow_legacy_diagnostic_provenance, bool):
            raise DatasetMergeError(
                "allow_legacy_diagnostic_provenance must be a Boolean"
            )


@dataclass(frozen=True)
class SourceFile:
    path: Path
    relative_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class CapturePolicy:
    min_mask_area_px: float
    far_range_start_m: float
    far_min_mask_area_px: float
    min_visible_height_fraction: float
    max_bottom_occlusion_px: float
    occlusion_policy: str

    def min_mask_area_for_range(self, range_m: float) -> float:
        if range_m >= self.far_range_start_m:
            return self.far_min_mask_area_px
        return self.min_mask_area_px

    def mask_area_policy_for_range(self, range_m: float) -> str:
        return "far" if range_m >= self.far_range_start_m else "near"

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_mask_area_px": self.min_mask_area_px,
            "far_range_start_m": self.far_range_start_m,
            "far_min_mask_area_px": self.far_min_mask_area_px,
            "min_visible_height_fraction": self.min_visible_height_fraction,
            "max_bottom_occlusion_px": self.max_bottom_occlusion_px,
            "occlusion_policy": self.occlusion_policy,
        }


@dataclass(frozen=True)
class AcceptedSample:
    camera_id: str
    sample_kind: str
    split: str
    image: SourceFile
    label: SourceFile
    mask: SourceFile | None
    robot_x: float
    robot_y: float
    robot_yaw: float
    pose_group: str
    camera_range_m: float
    range_bin: str
    min_mask_area_applied_px: float
    mask_area_px: float
    mask_area_policy: str
    localization_qualified: bool
    occlusion_state: str
    visible_height_fraction: float | None
    bottom_occlusion_px: float | None
    geometry_certified_no_opportunity: bool
    no_opportunity_reason: str
    semantic_robot_pixels: int
    image_width: int
    image_height: int


@dataclass(frozen=True)
class CameraAudit:
    camera_id: str
    source_dir: Path
    manifest: dict[str, Any]
    manifest_file: SourceFile
    diagnostics_file: SourceFile
    data_yaml_file: SourceFile
    completion_file: SourceFile | None
    capture_policy: CapturePolicy
    provenance_status: str
    samples: tuple[AcceptedSample, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise DatasetMergeError(f"Cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _source_file(path: Path, root: Path) -> SourceFile:
    if path.is_symlink() or not path.is_file():
        raise DatasetMergeError(f"Expected a regular, non-symlink file: {path}")
    try:
        relative = path.relative_to(root).as_posix()
        size = int(path.stat().st_size)
    except (OSError, ValueError) as exc:
        raise DatasetMergeError(f"Cannot inventory source file {path}: {exc}") from exc
    return SourceFile(path=path, relative_path=relative, sha256=_sha256(path), size_bytes=size)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetMergeError(f"Cannot read JSON object {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DatasetMergeError(f"Expected a JSON mapping in {path}")
    return payload


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise DatasetMergeError(f"Cannot read YAML mapping {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DatasetMergeError(f"Expected a YAML mapping in {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        encoded = json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        ) + "\n"
        path.write_text(encoded, encoding="utf-8")
    except (OSError, TypeError, ValueError) as exc:
        raise DatasetMergeError(f"Cannot write JSON file {path}: {exc}") from exc


def _finite_float(value: Any, context: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DatasetMergeError(f"{context} must be numeric, got {value!r}") from exc
    if not math.isfinite(number):
        raise DatasetMergeError(f"{context} must be finite, got {value!r}")
    return number


def _nonnegative_int(value: Any, context: str) -> int:
    if isinstance(value, bool):
        raise DatasetMergeError(f"{context} must be an integer, got {value!r}")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise DatasetMergeError(f"{context} must be an integer, got {value!r}") from exc
    if result < 0 or str(value).strip() not in {str(result), f"{result}.0"}:
        raise DatasetMergeError(f"{context} must be a non-negative integer, got {value!r}")
    return result


def _require_text(mapping: Mapping[str, Any], key: str, context: str) -> str:
    value = str(mapping.get(key, "")).strip()
    if not value:
        raise DatasetMergeError(f"{context} is missing non-empty {key!r}")
    return value


def _require_sha256(mapping: Mapping[str, Any], key: str, context: str) -> str:
    value = _require_text(mapping, key, context).lower()
    if not SHA256_RE.fullmatch(value):
        raise DatasetMergeError(f"{context}.{key} is not a lowercase SHA-256 digest: {value!r}")
    return value


def _canonical_inventory_sha256(entries: Iterable[Mapping[str, Any]]) -> str:
    canonical = [
        {
            "logical_path": str(entry["logical_path"]),
            "roles": sorted(str(role) for role in entry["roles"]),
            "sha256": str(entry["sha256"]),
            "size_bytes": int(entry["size_bytes"]),
        }
        for entry in entries
    ]
    canonical.sort(key=lambda entry: (entry["logical_path"], entry["roles"]))
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_capture_provenance(
    manifest: Mapping[str, Any],
    camera_id: str,
    config: MergeConfig,
) -> str:
    """Validate capture code/assets or admit an explicit diagnostic legacy input."""

    context = f"{camera_id} dataset_manifest.json"
    fields = (
        "capture_script_path",
        "capture_script_sha256",
        "capture_invocation",
        "transport_environment",
        "simulation_asset_inventory",
    )
    present = [field in manifest for field in fields]
    if not any(present):
        if config.allow_legacy_diagnostic_provenance:
            return LEGACY_CAPTURE_PROVENANCE
        raise DatasetMergeError(
            f"{context} lacks capture-script and simulation-asset provenance; "
            "training-grade merge is refused. For an immutable smoke audit only, "
            "use --allow-legacy-diagnostic-provenance; that output is explicitly "
            "marked non-training-grade."
        )
    if not all(present):
        missing = [field for field, exists in zip(fields, present) if not exists]
        raise DatasetMergeError(
            f"{context} has partial capture provenance and cannot be trusted; "
            f"missing={missing}"
        )

    if str(manifest.get("status", "")).strip() != "complete":
        raise DatasetMergeError(
            f"{context}.status must be 'complete' for a training-grade merge"
        )

    _require_text(manifest, "capture_script_path", context)
    script_sha = _require_sha256(manifest, "capture_script_sha256", context)
    invocation = manifest.get("capture_invocation")
    if not isinstance(invocation, dict):
        raise DatasetMergeError(f"{context}.capture_invocation must be a mapping")
    if _require_text(invocation, "schema_version", f"{context}.capture_invocation") != (
        CAPTURE_INVOCATION_SCHEMA_VERSION
    ):
        raise DatasetMergeError(
            f"{context}.capture_invocation has unsupported schema_version"
        )
    _require_text(invocation, "python_executable", f"{context}.capture_invocation")
    _require_text(invocation, "working_directory", f"{context}.capture_invocation")
    argv = invocation.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(value, str) or not value for value in argv)
    ):
        raise DatasetMergeError(
            f"{context}.capture_invocation.argv must be a non-empty string list"
        )
    resolved_arguments = invocation.get("resolved_arguments")
    if not isinstance(resolved_arguments, dict) or not resolved_arguments:
        raise DatasetMergeError(
            f"{context}.capture_invocation.resolved_arguments must be a non-empty mapping"
        )
    expected_arguments = {
        "world": manifest.get("world"),
        "camera_model": manifest.get("camera_model"),
        "image_topic": manifest.get("image_topic"),
        "labels_topic": manifest.get("labels_topic"),
    }
    for key, expected in expected_arguments.items():
        if str(resolved_arguments.get(key, "")).strip() != str(expected).strip():
            raise DatasetMergeError(
                f"{context}.capture_invocation.resolved_arguments.{key} does not "
                "match the manifest contract"
            )

    transport = manifest.get("transport_environment")
    transport_context = f"{context}.transport_environment"
    if not isinstance(transport, dict):
        raise DatasetMergeError(f"{transport_context} must be a mapping")
    if _require_text(transport, "schema_version", transport_context) != (
        CAPTURE_TRANSPORT_SCHEMA_VERSION
    ):
        raise DatasetMergeError(
            f"{transport_context} has unsupported schema_version"
        )
    required_values = transport.get("required_values")
    if required_values != TRAINING_CAPTURE_TRANSPORT_VALUES:
        raise DatasetMergeError(
            f"{transport_context}.required_values must exactly match the training transport contract"
        )
    observed_values = transport.get("observed_values")
    if not isinstance(observed_values, dict) or set(observed_values) != set(
        CAPTURE_TRANSPORT_VARIABLES
    ):
        raise DatasetMergeError(
            f"{transport_context}.observed_values must contain exactly "
            f"{list(CAPTURE_TRANSPORT_VARIABLES)!r}"
        )
    for key, expected in TRAINING_CAPTURE_TRANSPORT_VALUES.items():
        actual = observed_values.get(key)
        if not isinstance(actual, str) or actual != expected:
            raise DatasetMergeError(
                f"{transport_context}.observed_values.{key} must be {expected!r}, got {actual!r}"
            )
    domain_value = observed_values.get("ROS_DOMAIN_ID")
    try:
        domain_id = int(str(domain_value))
    except (TypeError, ValueError):
        domain_id = -1
    if not isinstance(domain_value, str) or not domain_value or not (0 <= domain_id <= 232):
        raise DatasetMergeError(
            f"{transport_context}.observed_values.ROS_DOMAIN_ID must be an explicit integer in 0..232"
        )
    partition = observed_values.get("IGN_PARTITION")
    if (
        not isinstance(partition, str)
        or not partition
        or any(character.isspace() for character in partition)
    ):
        raise DatasetMergeError(
            f"{transport_context}.observed_values.IGN_PARTITION must be a non-empty whitespace-free token"
        )
    if transport.get("isolation_verified") is not True:
        raise DatasetMergeError(f"{transport_context}.isolation_verified must be true")
    if transport.get("training_eligible") is not True:
        raise DatasetMergeError(f"{transport_context}.training_eligible must be true")
    if transport.get("diagnostic_override_used") is not False:
        raise DatasetMergeError(
            f"{transport_context}.diagnostic_override_used must be false for training"
        )
    violations = transport.get("violations")
    if not isinstance(violations, list) or violations:
        raise DatasetMergeError(f"{transport_context}.violations must be an empty list")

    inventory = manifest.get("simulation_asset_inventory")
    if not isinstance(inventory, dict):
        raise DatasetMergeError(
            f"{context}.simulation_asset_inventory must be a mapping"
        )
    inventory_context = f"{context}.simulation_asset_inventory"
    schema = _require_text(inventory, "schema_version", inventory_context)
    if schema != SIMULATION_ASSET_INVENTORY_SCHEMA_VERSION:
        raise DatasetMergeError(
            f"{inventory_context}.schema_version must be "
            f"{SIMULATION_ASSET_INVENTORY_SCHEMA_VERSION!r}, got {schema!r}"
        )
    declared_aggregate = _require_sha256(
        inventory, "aggregate_sha256", inventory_context
    )
    files = inventory.get("files")
    if not isinstance(files, list) or not files:
        raise DatasetMergeError(f"{inventory_context}.files must be a non-empty list")

    validated_entries: list[dict[str, Any]] = []
    logical_paths: set[str] = set()
    roles_present: set[str] = set()
    for index, raw_entry in enumerate(files):
        entry_context = f"{inventory_context}.files[{index}]"
        if not isinstance(raw_entry, dict):
            raise DatasetMergeError(f"{entry_context} must be a mapping")
        logical_path = _require_text(raw_entry, "logical_path", entry_context)
        _require_text(raw_entry, "path", entry_context)
        if logical_path in logical_paths:
            raise DatasetMergeError(
                f"{inventory_context} repeats logical_path {logical_path!r}"
            )
        logical_paths.add(logical_path)
        roles = raw_entry.get("roles")
        if (
            not isinstance(roles, list)
            or not roles
            or any(not isinstance(role, str) or not role.strip() for role in roles)
            or len(set(roles)) != len(roles)
        ):
            raise DatasetMergeError(
                f"{entry_context}.roles must be a non-empty unique string list"
            )
        sha256 = _require_sha256(raw_entry, "sha256", entry_context)
        size_bytes = _nonnegative_int(
            raw_entry.get("size_bytes"), f"{entry_context}.size_bytes"
        )
        roles_present.update(roles)
        validated_entries.append(
            {
                "logical_path": logical_path,
                "roles": list(roles),
                "sha256": sha256,
                "size_bytes": size_bytes,
            }
        )

    file_count = _nonnegative_int(
        inventory.get("file_count"), f"{inventory_context}.file_count"
    )
    total_size = _nonnegative_int(
        inventory.get("total_size_bytes"), f"{inventory_context}.total_size_bytes"
    )
    if file_count != len(validated_entries):
        raise DatasetMergeError(
            f"{inventory_context}.file_count {file_count} != {len(validated_entries)}"
        )
    actual_total = sum(int(entry["size_bytes"]) for entry in validated_entries)
    if total_size != actual_total:
        raise DatasetMergeError(
            f"{inventory_context}.total_size_bytes {total_size} != {actual_total}"
        )
    actual_aggregate = _canonical_inventory_sha256(validated_entries)
    if actual_aggregate != declared_aggregate:
        raise DatasetMergeError(
            f"{inventory_context}.aggregate_sha256 does not match its file inventory"
        )

    required_roles = {
        "capture_script",
        "world",
        "world_profiles",
        "route_exclusion_config",
        "sim_launch",
        "robot_description",
    }
    missing_roles = sorted(required_roles - roles_present)
    if missing_roles:
        raise DatasetMergeError(
            f"{inventory_context} lacks required asset roles: {missing_roles}"
        )
    model_names = inventory.get("referenced_model_names")
    model_uris = inventory.get("referenced_model_uris")
    if (
        not isinstance(model_names, list)
        or not model_names
        or any(not isinstance(value, str) or not value for value in model_names)
        or model_names != sorted(set(model_names))
    ):
        raise DatasetMergeError(
            f"{inventory_context}.referenced_model_names must be a non-empty sorted unique list"
        )
    if (
        not isinstance(model_uris, list)
        or not model_uris
        or any(not isinstance(value, str) or not value.startswith("model://") for value in model_uris)
        or model_uris != sorted(set(model_uris))
    ):
        raise DatasetMergeError(
            f"{inventory_context}.referenced_model_uris must be a non-empty sorted unique model:// list"
        )
    missing_models = [
        model_name
        for model_name in model_names
        if f"model_asset:{model_name}" not in roles_present
    ]
    if missing_models:
        raise DatasetMergeError(
            f"{inventory_context} lacks files for referenced models: {missing_models}"
        )

    role_hashes: dict[str, set[str]] = defaultdict(set)
    for entry in validated_entries:
        for role in entry["roles"]:
            role_hashes[role].add(str(entry["sha256"]))
    if script_sha not in role_hashes["capture_script"]:
        raise DatasetMergeError(
            f"{context}.capture_script_sha256 is absent from the asset inventory"
        )
    world_sha = _require_sha256(manifest, "world_sha256", context)
    profiles_sha = _require_sha256(manifest, "world_profiles_sha256", context)
    if world_sha not in role_hashes["world"]:
        raise DatasetMergeError(f"{context}.world_sha256 is absent from the asset inventory")
    if profiles_sha not in role_hashes["world_profiles"]:
        raise DatasetMergeError(
            f"{context}.world_profiles_sha256 is absent from the asset inventory"
        )
    return VERIFIED_CAPTURE_PROVENANCE


def _validate_capture_completion(
    path: Path,
    manifest_file: SourceFile,
    camera_id: str,
) -> SourceFile:
    context = f"{camera_id} .complete"
    completion_file = _source_file(path, manifest_file.path.parent)
    payload = _load_json(path)
    if _require_text(payload, "schema_version", context) != CAPTURE_COMPLETION_SCHEMA_VERSION:
        raise DatasetMergeError(f"{context} has unsupported schema_version")
    if _require_text(payload, "status", context) != "complete":
        raise DatasetMergeError(f"{context}.status must be 'complete'")
    if str(payload.get("training_eligible", "")).strip().lower() != "true":
        raise DatasetMergeError(f"{context}.training_eligible must be true")
    if _require_text(payload, "camera_id", context) != camera_id:
        raise DatasetMergeError(f"{context}.camera_id does not match {camera_id}")
    if _require_text(payload, "dataset_manifest", context) != "dataset_manifest.json":
        raise DatasetMergeError(
            f"{context}.dataset_manifest must be 'dataset_manifest.json'"
        )
    declared_sha = _require_sha256(payload, "dataset_manifest_sha256", context)
    if declared_sha != manifest_file.sha256:
        raise DatasetMergeError(
            f"{context}.dataset_manifest_sha256 does not match the source manifest"
        )
    _require_text(payload, "completed_at", context)
    return completion_file


def _parse_bool(value: Any, context: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise DatasetMergeError(f"{context} has invalid Boolean value {value!r}")


def _parse_accepted(value: Any, context: str) -> bool:
    return _parse_bool(value, f"{context}.accepted")


def _optional_finite_float(value: Any, context: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DatasetMergeError(f"{context} must be numeric or NaN, got {value!r}") from exc
    return number if math.isfinite(number) else None


def _relative_contract_path(value: Any, role: str, split: str, context: str) -> str:
    text = str(value).strip().replace("\\", "/")
    if not text:
        raise DatasetMergeError(f"{context} has an empty {role} path")
    candidate = PurePosixPath(text)
    if candidate.is_absolute() or ".." in candidate.parts or "." in candidate.parts:
        raise DatasetMergeError(f"{context} has unsafe {role} path {text!r}")
    if len(candidate.parts) != 3 or candidate.parts[:2] != (role, split):
        raise DatasetMergeError(
            f"{context} {role} path must be {role}/{split}/<file>, got {text!r}"
        )
    return candidate.as_posix()


def _enumerate_flat_files(
    root: Path,
    role: str,
    split: str,
    suffixes: frozenset[str],
    *,
    required_dir: bool,
) -> dict[str, Path]:
    directory = root / role / split
    if not directory.exists() and not required_dir:
        return {}
    if directory.is_symlink() or not directory.is_dir():
        raise DatasetMergeError(f"Expected a regular directory: {directory}")
    result: dict[str, Path] = {}
    try:
        children = sorted(directory.iterdir(), key=lambda item: item.name)
    except OSError as exc:
        raise DatasetMergeError(f"Cannot enumerate {directory}: {exc}") from exc
    for path in children:
        if path.is_symlink() or not path.is_file():
            raise DatasetMergeError(f"Nested entries and symlinks are forbidden in {directory}: {path}")
        suffix = path.suffix.lower()
        if suffix not in suffixes:
            raise DatasetMergeError(f"Unexpected file type in {directory}: {path.name}")
        stem = path.stem
        if not SAFE_STEM.fullmatch(stem):
            raise DatasetMergeError(f"Unsafe source filename stem in {path}: {stem!r}")
        if stem in result:
            raise DatasetMergeError(f"Duplicate stem {stem!r} in {directory}")
        result[stem] = path
    return result


def _format_edge(value: float) -> str:
    return format(float(value), ".15g").replace(".", "p")


def _range_bin_names(edges: Sequence[float]) -> tuple[str, ...]:
    names = [f"lt_{_format_edge(edges[0])}m"]
    for index, (left, right) in enumerate(zip(edges, edges[1:])):
        if index == len(edges) - 2:
            names.append(f"{_format_edge(left)}_to_{_format_edge(right)}m")
        else:
            names.append(f"{_format_edge(left)}_to_lt_{_format_edge(right)}m")
    names.append(f"gt_{_format_edge(edges[-1])}m")
    return tuple(names)


def _range_bin(range_m: float, edges: Sequence[float]) -> str:
    names = _range_bin_names(edges)
    if range_m < edges[0]:
        return names[0]
    for index in range(len(edges) - 1):
        upper = edges[index + 1]
        if range_m < upper or (index == len(edges) - 2 and range_m <= upper):
            return names[index + 1]
    return names[-1]


def _capture_range_bin(range_m: float) -> str:
    if range_m < 8.0:
        return "lt_8m"
    if range_m < 12.0:
        return "8_to_12m"
    if range_m <= 16.0:
        return "12_to_16m"
    return "gt_16m"


def _pose_group(row: Mapping[str, Any], x: float, y: float) -> str:
    for key in ("pose_group_id", "split_group", "pose_group"):
        value = str(row.get(key, "")).strip()
        if value:
            return f"declared:{value}"
    # Capture grids are emitted as binary floats.  Micrometre quantisation
    # absorbs harmless CSV round-off while keeping distinct commanded poses
    # separate.  Yaw is deliberately excluded: every view of one background
    # location must remain on one side of the train/validation boundary.
    qx = round(x, 6)
    qy = round(y, 6)
    if qx == 0.0:
        qx = 0.0
    if qy == 0.0:
        qy = 0.0
    return f"xy:{qx:.6f}:{qy:.6f}"


def _validate_yolo_seg_label(path: Path) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise DatasetMergeError(f"Cannot read YOLO label {path}: {exc}") from exc
    if not lines:
        raise DatasetMergeError(f"Accepted sample has an empty YOLO segmentation label: {path}")
    for line_number, raw_line in enumerate(lines, start=1):
        tokens = raw_line.split()
        context = f"{path}:{line_number}"
        if len(tokens) < 7 or (len(tokens) - 1) % 2:
            raise DatasetMergeError(
                f"{context} must contain class 0 and at least three x/y polygon points"
            )
        try:
            class_value = float(tokens[0])
            coords = [float(token) for token in tokens[1:]]
        except ValueError as exc:
            raise DatasetMergeError(f"{context} contains a non-numeric token") from exc
        if class_value != 0.0 or not class_value.is_integer():
            raise DatasetMergeError(f"{context} must use the single merged class ID 0")
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in coords):
            raise DatasetMergeError(f"{context} polygon coordinates must be finite and normalized")
        points = list(zip(coords[0::2], coords[1::2]))
        area_twice = abs(
            sum(
                x0 * y1 - x1 * y0
                for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1])
            )
        )
        if area_twice <= 1e-12:
            raise DatasetMergeError(f"{context} contains a degenerate polygon")


def _validate_yolo_negative_label(path: Path) -> None:
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise DatasetMergeError(f"Cannot read YOLO negative label {path}: {exc}") from exc
    if content.strip():
        raise DatasetMergeError(
            f"Negative sample must have an exactly empty YOLO label: {path}"
        )


def _validate_data_yaml(path: Path, camera_id: str) -> None:
    payload = _load_yaml(path)
    if str(payload.get("train", "")) != "images/train":
        raise DatasetMergeError(f"{camera_id} data.yaml train must be 'images/train'")
    if str(payload.get("val", "")) != "images/val":
        raise DatasetMergeError(f"{camera_id} data.yaml val must be 'images/val'")
    if str(payload.get("task", "")) != "segment":
        raise DatasetMergeError(f"{camera_id} data.yaml task must be 'segment'")
    names = payload.get("names")
    valid_names = names == {0: "robot"} or names == {"0": "robot"} or names == ["robot"]
    if not valid_names:
        raise DatasetMergeError(f"{camera_id} data.yaml must declare only class 0: robot")


def _validate_manifest(
    manifest: Mapping[str, Any],
    camera_id: str,
    config: MergeConfig,
) -> tuple[float, float, int, int, str]:
    context = f"{camera_id} dataset_manifest.json"
    contract = CAMERA_CONTRACTS[camera_id]
    if _require_text(manifest, "camera_id", context) != camera_id:
        raise DatasetMergeError(f"{context} camera_id does not match input key {camera_id}")
    for key, expected in contract.items():
        actual = _require_text(manifest, key, context)
        if actual != expected:
            raise DatasetMergeError(f"{context}.{key} must be {expected!r}, got {actual!r}")
    world = _require_text(manifest, "world", context)
    if Path(world).name != Path(config.expected_world).name:
        raise DatasetMergeError(
            f"{context}.world must name {Path(config.expected_world).name!r}, got {world!r}"
        )
    if _require_text(manifest, "source", context) != "gazebo_semantic_segmentation":
        raise DatasetMergeError(f"{context}.source must be 'gazebo_semantic_segmentation'")
    _require_sha256(manifest, "world_sha256", context)
    _require_sha256(manifest, "world_profiles_sha256", context)
    robot_label = _nonnegative_int(manifest.get("robot_label"), f"{context}.robot_label")
    pose = manifest.get("camera_pose_xyz_rpy")
    if not isinstance(pose, list) or len(pose) != 6:
        raise DatasetMergeError(f"{context}.camera_pose_xyz_rpy must contain six values")
    pose_values = [_finite_float(value, f"{context}.camera_pose_xyz_rpy") for value in pose]
    intrinsics = manifest.get("camera_intrinsics")
    if not isinstance(intrinsics, dict):
        raise DatasetMergeError(f"{context}.camera_intrinsics must be a mapping")
    width = _nonnegative_int(intrinsics.get("img_width"), f"{context}.camera_intrinsics.img_width")
    height = _nonnegative_int(intrinsics.get("img_height"), f"{context}.camera_intrinsics.img_height")
    if width <= 0 or height <= 0:
        raise DatasetMergeError(f"{context} image dimensions must be positive")
    _require_text(manifest, "split_mode", context)
    provenance_status = _validate_capture_provenance(manifest, camera_id, config)
    return pose_values[0], pose_values[1], width, height, provenance_status


def _capture_policy(manifest: Mapping[str, Any], camera_id: str) -> CapturePolicy:
    context = f"{camera_id} dataset_manifest.json.capture_thresholds"
    thresholds = manifest.get("capture_thresholds")
    if not isinstance(thresholds, dict):
        raise DatasetMergeError(f"{context} must be a mapping")
    policy = CapturePolicy(
        min_mask_area_px=_finite_float(thresholds.get("min_mask_area"), f"{context}.min_mask_area"),
        far_range_start_m=_finite_float(
            thresholds.get("far_range_start_m"), f"{context}.far_range_start_m"
        ),
        far_min_mask_area_px=_finite_float(
            thresholds.get("far_min_mask_area"), f"{context}.far_min_mask_area"
        ),
        min_visible_height_fraction=_finite_float(
            thresholds.get("min_visible_height_fraction"),
            f"{context}.min_visible_height_fraction",
        ),
        max_bottom_occlusion_px=_finite_float(
            thresholds.get("max_bottom_occlusion_px"),
            f"{context}.max_bottom_occlusion_px",
        ),
        occlusion_policy=str(thresholds.get("occlusion_policy", "")).strip(),
    )
    if (
        policy.min_mask_area_px < 0.0
        or policy.far_range_start_m < 0.0
        or policy.far_min_mask_area_px < 0.0
        or policy.max_bottom_occlusion_px < 0.0
    ):
        raise DatasetMergeError(f"{context} mask/range/occlusion thresholds must be non-negative")
    if not 0.0 <= policy.min_visible_height_fraction <= 1.0:
        raise DatasetMergeError(f"{context}.min_visible_height_fraction must be in [0, 1]")
    if policy.occlusion_policy not in {"reject", "visible-mask-positive"}:
        raise DatasetMergeError(
            f"{context}.occlusion_policy must be 'reject' or 'visible-mask-positive'"
        )
    return policy


def _validate_sample_qualification(
    *,
    camera_id: str,
    image_rel: str,
    row: Mapping[str, Any],
    camera_range_m: float,
    policy: CapturePolicy,
) -> dict[str, Any]:
    context = f"{camera_id} accepted row {image_rel}"
    localization_qualified = _parse_bool(
        row.get("localization_qualified"), f"{context}.localization_qualified"
    )
    occlusion_state = str(row.get("occlusion_state", "")).strip()
    allowed_states = {"clear", "low_visible_height", "bottom_hidden"}
    if occlusion_state not in allowed_states:
        raise DatasetMergeError(
            f"{context}.occlusion_state must be one of {sorted(allowed_states)}, "
            f"got {occlusion_state!r}"
        )
    if localization_qualified != (occlusion_state == "clear"):
        raise DatasetMergeError(
            f"{context} qualification/state contradiction: "
            f"localization_qualified={int(localization_qualified)}, "
            f"occlusion_state={occlusion_state!r}"
        )
    if not localization_qualified and policy.occlusion_policy != "visible-mask-positive":
        raise DatasetMergeError(
            f"{context} is an occluded detector positive but manifest policy is "
            f"{policy.occlusion_policy!r}"
        )

    applied = _finite_float(
        row.get("min_mask_area_applied_px"), f"{context}.min_mask_area_applied_px"
    )
    mask_area = _finite_float(row.get("mask_area_px"), f"{context}.mask_area_px")
    expected_applied = policy.min_mask_area_for_range(camera_range_m)
    if not math.isclose(applied, expected_applied, rel_tol=1e-9, abs_tol=1e-9):
        raise DatasetMergeError(
            f"{context} min-mask policy mismatch: applied {applied:g}, "
            f"expected {expected_applied:g} at range {camera_range_m:g} m"
        )
    if mask_area < applied:
        raise DatasetMergeError(
            f"{context} accepted mask area {mask_area:g}px is below its applied "
            f"minimum {applied:g}px"
        )

    visible_height = _optional_finite_float(
        row.get("visible_height_fraction"), f"{context}.visible_height_fraction"
    )
    bottom_occlusion = _optional_finite_float(
        row.get("bottom_occlusion_px"), f"{context}.bottom_occlusion_px"
    )
    if occlusion_state == "low_visible_height":
        if visible_height is None or visible_height >= policy.min_visible_height_fraction:
            raise DatasetMergeError(
                f"{context} low_visible_height is inconsistent with "
                f"visible_height_fraction={visible_height!r} and threshold "
                f"{policy.min_visible_height_fraction:g}"
            )
    elif occlusion_state == "bottom_hidden":
        if bottom_occlusion is None or bottom_occlusion <= policy.max_bottom_occlusion_px:
            raise DatasetMergeError(
                f"{context} bottom_hidden is inconsistent with "
                f"bottom_occlusion_px={bottom_occlusion!r} and threshold "
                f"{policy.max_bottom_occlusion_px:g}"
            )
    else:
        if (
            visible_height is not None
            and visible_height < policy.min_visible_height_fraction
        ):
            raise DatasetMergeError(
                f"{context} is marked clear below the visible-height threshold"
            )
        if (
            bottom_occlusion is not None
            and bottom_occlusion > policy.max_bottom_occlusion_px
        ):
            raise DatasetMergeError(
                f"{context} is marked clear above the bottom-occlusion threshold"
            )
    return {
        "min_mask_area_applied_px": applied,
        "mask_area_px": mask_area,
        "mask_area_policy": policy.mask_area_policy_for_range(camera_range_m),
        "localization_qualified": localization_qualified,
        "occlusion_state": occlusion_state,
        "visible_height_fraction": visible_height,
        "bottom_occlusion_px": bottom_occlusion,
        "geometry_certified_no_opportunity": False,
        "no_opportunity_reason": "",
        "semantic_robot_pixels": max(int(round(mask_area)), 1),
    }


def _validate_negative_qualification(
    *,
    camera_id: str,
    image_rel: str,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    context = f"{camera_id} negative row {image_rel}"
    geometry_certified = _parse_bool(
        row.get("geometry_certified_no_opportunity"),
        f"{context}.geometry_certified_no_opportunity",
    )
    if not geometry_certified:
        raise DatasetMergeError(
            f"{context} is not geometry-certified as a no-opportunity frame"
        )
    reason = str(row.get("no_opportunity_reason", "")).strip()
    allowed_reasons = {"projection_behind_camera", "projection_outside_image"}
    if reason not in allowed_reasons:
        raise DatasetMergeError(
            f"{context}.no_opportunity_reason must be one of {sorted(allowed_reasons)}, "
            f"got {reason!r}; occlusion is never a negative certification"
        )
    semantic_robot_pixels = _nonnegative_int(
        row.get("semantic_robot_pixels"), f"{context}.semantic_robot_pixels"
    )
    if semantic_robot_pixels != 0:
        raise DatasetMergeError(
            f"{context} contains {semantic_robot_pixels} semantic robot pixels"
        )
    localization_qualified = _parse_bool(
        row.get("localization_qualified"), f"{context}.localization_qualified"
    )
    if localization_qualified:
        raise DatasetMergeError(f"{context} cannot be localization-qualified")
    if str(row.get("occlusion_state", "")).strip() != "not_applicable":
        raise DatasetMergeError(
            f"{context}.occlusion_state must be 'not_applicable'; an occluded robot "
            "cannot be relabelled as a negative"
        )
    applied = _finite_float(
        row.get("min_mask_area_applied_px"), f"{context}.min_mask_area_applied_px"
    )
    mask_area = _finite_float(row.get("mask_area_px"), f"{context}.mask_area_px")
    if applied != 0.0 or mask_area != 0.0:
        raise DatasetMergeError(
            f"{context} must declare zero applied mask gate and zero mask area"
        )
    return {
        "min_mask_area_applied_px": 0.0,
        "mask_area_px": 0.0,
        "mask_area_policy": "not_applicable",
        "localization_qualified": False,
        "occlusion_state": "not_applicable",
        "visible_height_fraction": None,
        "bottom_occlusion_px": None,
        "geometry_certified_no_opportunity": True,
        "no_opportunity_reason": reason,
        "semantic_robot_pixels": 0,
    }


def _validate_capture_audit(
    manifest: Mapping[str, Any],
    camera_id: str,
    samples: Sequence[AcceptedSample],
) -> None:
    context = f"{camera_id} dataset_manifest.json"
    audit = manifest.get("audit")
    thresholds = manifest.get("capture_thresholds")
    if not isinstance(audit, dict) or not isinstance(thresholds, dict):
        raise DatasetMergeError(f"{context} must include audit and capture_thresholds mappings")
    accepted = _nonnegative_int(audit.get("accepted_samples"), f"{context}.audit.accepted_samples")
    if accepted != len(samples):
        raise DatasetMergeError(
            f"{context} declares {accepted} accepted samples but files/diagnostics contain {len(samples)}"
        )
    positive_samples = [sample for sample in samples if sample.sample_kind == "positive"]
    negative_samples = [sample for sample in samples if sample.sample_kind == "negative"]
    if provenance_status := str(manifest.get("capture_script_sha256", "")).strip():
        declared_kinds = audit.get("accepted_by_sample_kind")
        if not isinstance(declared_kinds, dict):
            raise DatasetMergeError(
                f"{context}.audit.accepted_by_sample_kind must be a mapping"
            )
        actual_kinds = Counter(sample.sample_kind for sample in samples)
        normalized_kinds = {
            str(key): _nonnegative_int(
                value, f"{context}.audit.accepted_by_sample_kind.{key}"
            )
            for key, value in declared_kinds.items()
            if _nonnegative_int(
                value, f"{context}.audit.accepted_by_sample_kind.{key}"
            ) > 0
        }
        if normalized_kinds != dict(actual_kinds):
            raise DatasetMergeError(
                f"{context} accepted_by_sample_kind mismatch: "
                f"declared={normalized_kinds}, actual={dict(actual_kinds)}"
            )
        declared_positive = _nonnegative_int(
            audit.get("positive_accepted_samples"),
            f"{context}.audit.positive_accepted_samples",
        )
        declared_negative = _nonnegative_int(
            audit.get("negative_accepted_samples"),
            f"{context}.audit.negative_accepted_samples",
        )
        if declared_positive != len(positive_samples) or declared_negative != len(negative_samples):
            raise DatasetMergeError(
                f"{context} positive/negative accepted counts do not match payload"
            )
    expected_splits = Counter(sample.split for sample in samples)
    split_counts = audit.get("split_counts")
    if not isinstance(split_counts, dict):
        raise DatasetMergeError(f"{context}.audit.split_counts must be a mapping")
    declared_splits = {
        str(split): _nonnegative_int(
            value, f"{context}.audit.split_counts.{split}"
        )
        for split, value in split_counts.items()
    }
    expected_split_dict = {split: expected_splits[split] for split in SPLITS}
    if declared_splits != expected_split_dict:
        raise DatasetMergeError(
            f"{context} split_counts mismatch: declared={declared_splits}, "
            f"actual={expected_split_dict}"
        )
    declared_ranges = audit.get("accepted_by_range_bin")
    if not isinstance(declared_ranges, dict):
        raise DatasetMergeError(f"{context}.audit.accepted_by_range_bin must be a mapping")
    actual_ranges = Counter(_capture_range_bin(sample.camera_range_m) for sample in samples)
    declared_normalized: dict[str, int] = {}
    for key, value in declared_ranges.items():
        parsed = _nonnegative_int(value, f"{context}.audit.accepted_by_range_bin.{key}")
        if parsed:
            declared_normalized[str(key)] = parsed
    if declared_normalized != dict(actual_ranges):
        raise DatasetMergeError(
            f"{context} accepted_by_range_bin mismatch: declared={declared_normalized}, "
            f"actual={dict(actual_ranges)}"
        )
    declared_occlusion = audit.get("accepted_by_occlusion_state")
    if not isinstance(declared_occlusion, dict):
        raise DatasetMergeError(f"{context}.audit.accepted_by_occlusion_state must be a mapping")
    declared_occlusion_normalized: dict[str, int] = {}
    for key, value in declared_occlusion.items():
        parsed = _nonnegative_int(
            value, f"{context}.audit.accepted_by_occlusion_state.{key}"
        )
        if parsed:
            declared_occlusion_normalized[str(key)] = parsed
    actual_occlusion = Counter(sample.occlusion_state for sample in samples)
    if declared_occlusion_normalized != dict(actual_occlusion):
        raise DatasetMergeError(
            f"{context} accepted_by_occlusion_state mismatch: "
            f"declared={declared_occlusion_normalized}, actual={dict(actual_occlusion)}"
        )
    declared_localization = _nonnegative_int(
        audit.get("localization_qualified_samples"),
        f"{context}.audit.localization_qualified_samples",
    )
    actual_localization = sum(sample.localization_qualified for sample in samples)
    if declared_localization != actual_localization:
        raise DatasetMergeError(
            f"{context} localization_qualified_samples mismatch: "
            f"declared={declared_localization}, actual={actual_localization}"
        )

    accepted_fraction = _finite_float(
        audit.get("accepted_fraction"), f"{context}.audit.accepted_fraction"
    )
    duplicate_fraction = _finite_float(
        audit.get("duplicate_accepted_fraction"),
        f"{context}.audit.duplicate_accepted_fraction",
    )
    planned = _nonnegative_int(
        audit.get("positive_planned_samples", audit.get("planned_samples")),
        f"{context}.audit.positive_planned_samples",
    )
    duplicate_images = _nonnegative_int(
        audit.get("duplicate_accepted_images"),
        f"{context}.audit.duplicate_accepted_images",
    )
    min_accepted = _nonnegative_int(
        thresholds.get("min_accepted_samples"),
        f"{context}.capture_thresholds.min_accepted_samples",
    )
    min_fraction = _finite_float(
        thresholds.get("min_accept_fraction"),
        f"{context}.capture_thresholds.min_accept_fraction",
    )
    max_duplicates = _finite_float(
        thresholds.get("max_final_duplicate_fraction"),
        f"{context}.capture_thresholds.max_final_duplicate_fraction",
    )
    failures: list[str] = []
    if planned < len(positive_samples):
        failures.append(
            f"positive_planned_samples {planned} < positive samples {len(positive_samples)}"
        )
    expected_accepted_fraction = len(positive_samples) / float(max(planned, 1))
    if not math.isclose(
        accepted_fraction, expected_accepted_fraction, rel_tol=1e-9, abs_tol=1e-9
    ):
        failures.append(
            f"accepted_fraction {accepted_fraction:g} != derived {expected_accepted_fraction:g}"
        )
    expected_duplicate_fraction = duplicate_images / float(max(accepted, 1))
    if not math.isclose(
        duplicate_fraction, expected_duplicate_fraction, rel_tol=1e-9, abs_tol=1e-9
    ):
        failures.append(
            f"duplicate fraction {duplicate_fraction:g} != derived "
            f"{expected_duplicate_fraction:g}"
        )
    for name, value in (
        ("accepted_fraction", accepted_fraction),
        ("duplicate_accepted_fraction", duplicate_fraction),
        ("min_accept_fraction", min_fraction),
        ("max_final_duplicate_fraction", max_duplicates),
    ):
        if not 0.0 <= value <= 1.0:
            failures.append(f"{name} {value:g} is outside [0, 1]")
    if len(positive_samples) < min_accepted:
        failures.append(
            f"positive samples {len(positive_samples)} < capture minimum {min_accepted}"
        )
    if accepted_fraction < min_fraction:
        failures.append(f"accepted_fraction {accepted_fraction:g} < {min_fraction:g}")
    if duplicate_fraction > max_duplicates:
        failures.append(f"duplicate fraction {duplicate_fraction:g} > {max_duplicates:g}")
    if failures:
        raise DatasetMergeError(f"{camera_id} capture did not pass its declared gates: " + "; ".join(failures))


def _read_diagnostics(path: Path, camera_id: str) -> list[dict[str, str]]:
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise DatasetMergeError(f"Diagnostics CSV has no header: {path}")
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise DatasetMergeError(f"Diagnostics CSV has duplicate columns: {path}")
            required = {
                "camera_id",
                "camera_model",
                "camera_range_m",
                "camera_range_bin",
                "min_mask_area_applied_px",
                "split",
                "accepted",
                "image",
                "label",
                "mask",
                "robot_x",
                "robot_y",
                "robot_yaw",
                "mask_area_px",
                "visible_height_fraction",
                "bottom_occlusion_px",
                "localization_qualified",
                "occlusion_state",
            }
            missing = sorted(required - set(reader.fieldnames))
            if missing:
                raise DatasetMergeError(f"{camera_id} diagnostics is missing columns: {missing}")
            return [dict(row) for row in reader]
    except DatasetMergeError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DatasetMergeError(f"Cannot read diagnostics CSV {path}: {exc}") from exc


def _audit_camera(camera_id: str, source_dir: Path, config: MergeConfig) -> CameraAudit:
    if source_dir.is_symlink() or not source_dir.is_dir():
        raise DatasetMergeError(f"{camera_id} source is not a regular directory: {source_dir}")
    manifest_path = source_dir / "dataset_manifest.json"
    diagnostics_path = source_dir / "label_diagnostics.csv"
    data_yaml_path = source_dir / "data.yaml"
    manifest_file = _source_file(manifest_path, source_dir)
    diagnostics_file = _source_file(diagnostics_path, source_dir)
    data_yaml_file = _source_file(data_yaml_path, source_dir)
    manifest = _load_json(manifest_path)
    camera_x, camera_y, expected_width, expected_height, provenance_status = _validate_manifest(
        manifest, camera_id, config
    )
    _validate_data_yaml(data_yaml_path, camera_id)
    completion_file = (
        _validate_capture_completion(source_dir / ".complete", manifest_file, camera_id)
        if provenance_status == VERIFIED_CAPTURE_PROVENANCE
        else None
    )
    capture_policy = _capture_policy(manifest, camera_id)

    pairs: dict[str, tuple[Path, Path, Path | None]] = {}
    for split in SPLITS:
        images = _enumerate_flat_files(
            source_dir, "images", split, IMAGE_SUFFIXES, required_dir=True
        )
        labels = _enumerate_flat_files(
            source_dir, "labels", split, frozenset({".txt"}), required_dir=True
        )
        masks = _enumerate_flat_files(
            source_dir, "masks", split, IMAGE_SUFFIXES, required_dir=False
        )
        if set(images) != set(labels):
            missing_labels = sorted(set(images) - set(labels))
            orphan_labels = sorted(set(labels) - set(images))
            raise DatasetMergeError(
                f"{camera_id}/{split} image-label pairing mismatch: "
                f"missing_labels={missing_labels}, orphan_labels={orphan_labels}"
            )
        if masks and set(masks) != set(images):
            missing_masks = sorted(set(images) - set(masks))
            orphan_masks = sorted(set(masks) - set(images))
            raise DatasetMergeError(
                f"{camera_id}/{split} masks must cover the whole split when present: "
                f"missing_masks={missing_masks}, orphan_masks={orphan_masks}"
            )
        for stem, image_path in images.items():
            image_rel = image_path.relative_to(source_dir).as_posix()
            pairs[image_rel] = (image_path, labels[stem], masks.get(stem))

    diagnostic_rows = _read_diagnostics(diagnostics_path, camera_id)
    accepted_rows: dict[str, dict[str, str]] = {}
    contract = CAMERA_CONTRACTS[camera_id]
    for index, row in enumerate(diagnostic_rows, start=2):
        context = f"{diagnostics_path}:{index}"
        accepted = _parse_accepted(row.get("accepted"), context)
        if not accepted:
            if str(row.get("image", "")).strip() or str(row.get("label", "")).strip():
                raise DatasetMergeError(f"{context} rejected row unexpectedly references dataset payload")
            continue
        split = str(row.get("split", "")).strip()
        if split not in SPLITS:
            raise DatasetMergeError(f"{context} has invalid split {split!r}")
        if str(row.get("camera_id", "")).strip() != camera_id:
            raise DatasetMergeError(f"{context} camera_id does not match {camera_id}")
        if str(row.get("camera_model", "")).strip() != contract["camera_model"]:
            raise DatasetMergeError(f"{context} camera_model does not match the camera contract")
        sample_kind = str(row.get("sample_kind", "")).strip()
        if not sample_kind and provenance_status == LEGACY_CAPTURE_PROVENANCE:
            sample_kind = "positive"
        if sample_kind not in {"positive", "negative"}:
            raise DatasetMergeError(
                f"{context}.sample_kind must be explicit 'positive' or 'negative', "
                f"got {sample_kind!r}"
            )
        row["_validated_sample_kind"] = sample_kind
        image_rel = _relative_contract_path(row.get("image"), "images", split, context)
        label_rel = _relative_contract_path(row.get("label"), "labels", split, context)
        if PurePosixPath(image_rel).stem != PurePosixPath(label_rel).stem:
            raise DatasetMergeError(f"{context} image and label stems do not match")
        mask_text = str(row.get("mask", "")).strip()
        if mask_text:
            mask_rel = _relative_contract_path(mask_text, "masks", split, context)
            if PurePosixPath(image_rel).stem != PurePosixPath(mask_rel).stem:
                raise DatasetMergeError(f"{context} image and mask stems do not match")
        if image_rel in accepted_rows:
            raise DatasetMergeError(f"{context} repeats accepted image {image_rel}")
        accepted_rows[image_rel] = row

    if set(accepted_rows) != set(pairs):
        missing_diagnostics = sorted(set(pairs) - set(accepted_rows))
        missing_payload = sorted(set(accepted_rows) - set(pairs))
        raise DatasetMergeError(
            f"{camera_id} diagnostics/payload mismatch: "
            f"files_without_accepted_row={missing_diagnostics}, "
            f"accepted_rows_without_files={missing_payload}"
        )

    samples: list[AcceptedSample] = []
    for image_rel in sorted(pairs):
        image_path, label_path, mask_path = pairs[image_rel]
        row = accepted_rows[image_rel]
        split = str(row["split"]).strip()
        declared_label = _relative_contract_path(
            row["label"], "labels", split, f"accepted row for {image_rel}"
        )
        if declared_label != label_path.relative_to(source_dir).as_posix():
            raise DatasetMergeError(f"Accepted row for {image_rel} references the wrong label")
        declared_mask = str(row.get("mask", "")).strip().replace("\\", "/")
        actual_mask = mask_path.relative_to(source_dir).as_posix() if mask_path else ""
        if declared_mask != actual_mask:
            raise DatasetMergeError(
                f"Accepted row for {image_rel} mask mismatch: {declared_mask!r} != {actual_mask!r}"
            )
        x = _finite_float(row["robot_x"], f"accepted row {image_rel}.robot_x")
        y = _finite_float(row["robot_y"], f"accepted row {image_rel}.robot_y")
        yaw = _finite_float(row["robot_yaw"], f"accepted row {image_rel}.robot_yaw")
        camera_range = math.hypot(x - camera_x, y - camera_y)
        declared_range = _finite_float(
            row["camera_range_m"], f"accepted row {image_rel}.camera_range_m"
        )
        if not math.isclose(camera_range, declared_range, rel_tol=1e-9, abs_tol=1e-6):
            raise DatasetMergeError(
                f"Accepted row {image_rel} range mismatch: declared {declared_range:g}, "
                f"computed {camera_range:g}"
            )
        declared_capture_bin = str(row["camera_range_bin"]).strip()
        if declared_capture_bin != _capture_range_bin(camera_range):
            raise DatasetMergeError(
                f"Accepted row {image_rel} camera_range_bin mismatch: "
                f"{declared_capture_bin!r} != {_capture_range_bin(camera_range)!r}"
            )
        sample_kind = str(row["_validated_sample_kind"])
        if sample_kind == "negative":
            if split != "train":
                raise DatasetMergeError(
                    f"Negative sample {image_rel} must be train-only in the fixed-scene protocol"
                )
            qualification = _validate_negative_qualification(
                camera_id=camera_id,
                image_rel=image_rel,
                row=row,
            )
        else:
            qualification = _validate_sample_qualification(
                camera_id=camera_id,
                image_rel=image_rel,
                row=row,
                camera_range_m=camera_range,
                policy=capture_policy,
            )
            if provenance_status == VERIFIED_CAPTURE_PROVENANCE:
                if _parse_bool(
                    row.get("geometry_certified_no_opportunity"),
                    f"accepted row {image_rel}.geometry_certified_no_opportunity",
                ):
                    raise DatasetMergeError(
                        f"Positive sample {image_rel} cannot be no-opportunity certified"
                    )
                semantic_pixels = _nonnegative_int(
                    row.get("semantic_robot_pixels"),
                    f"accepted row {image_rel}.semantic_robot_pixels",
                )
                if semantic_pixels <= 0:
                    raise DatasetMergeError(
                        f"Positive sample {image_rel} must contain semantic robot pixels"
                    )
                qualification["semantic_robot_pixels"] = semantic_pixels
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise DatasetMergeError(f"Image is not decodable: {image_path}")
        height, width = image.shape[:2]
        if (width, height) != (expected_width, expected_height):
            raise DatasetMergeError(
                f"{image_path} has {width}x{height}, manifest declares "
                f"{expected_width}x{expected_height}"
            )
        if sample_kind == "negative":
            _validate_yolo_negative_label(label_path)
        else:
            _validate_yolo_seg_label(label_path)
        if mask_path is not None:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.size == 0:
                raise DatasetMergeError(f"Mask is not decodable: {mask_path}")
            if mask.shape[:2] != (expected_height, expected_width):
                raise DatasetMergeError(
                    f"{mask_path} has {mask.shape[1]}x{mask.shape[0]}, manifest declares "
                    f"{expected_width}x{expected_height}"
                )
            if sample_kind == "negative" and int(cv2.countNonZero(mask)) != 0:
                raise DatasetMergeError(
                    f"Negative sample mask must contain exactly zero pixels: {mask_path}"
                )
        samples.append(
            AcceptedSample(
                camera_id=camera_id,
                sample_kind=sample_kind,
                split=split,
                image=_source_file(image_path, source_dir),
                label=_source_file(label_path, source_dir),
                mask=_source_file(mask_path, source_dir) if mask_path else None,
                robot_x=x,
                robot_y=y,
                robot_yaw=yaw,
                pose_group=_pose_group(row, x, y),
                camera_range_m=camera_range,
                range_bin=_range_bin(camera_range, config.range_edges_m),
                min_mask_area_applied_px=qualification["min_mask_area_applied_px"],
                mask_area_px=qualification["mask_area_px"],
                mask_area_policy=qualification["mask_area_policy"],
                localization_qualified=qualification["localization_qualified"],
                occlusion_state=qualification["occlusion_state"],
                visible_height_fraction=qualification["visible_height_fraction"],
                bottom_occlusion_px=qualification["bottom_occlusion_px"],
                geometry_certified_no_opportunity=qualification[
                    "geometry_certified_no_opportunity"
                ],
                no_opportunity_reason=qualification["no_opportunity_reason"],
                semantic_robot_pixels=qualification["semantic_robot_pixels"],
                image_width=width,
                image_height=height,
            )
        )
    _validate_capture_audit(manifest, camera_id, samples)
    return CameraAudit(
        camera_id=camera_id,
        source_dir=source_dir,
        manifest=dict(manifest),
        manifest_file=manifest_file,
        diagnostics_file=diagnostics_file,
        data_yaml_file=data_yaml_file,
        completion_file=completion_file,
        capture_policy=capture_policy,
        provenance_status=provenance_status,
        samples=tuple(samples),
    )


def _validate_cross_camera_contract(audits: Sequence[CameraAudit]) -> None:
    for key in ("world_sha256", "world_profiles_sha256", "robot_label"):
        values = {str(audit.manifest.get(key)) for audit in audits}
        if len(values) != 1:
            raise DatasetMergeError(f"Source manifests disagree on {key}: {sorted(values)}")
    policies = {audit.capture_policy for audit in audits}
    if len(policies) != 1:
        details = {
            audit.camera_id: audit.capture_policy.as_dict() for audit in audits
        }
        raise DatasetMergeError(
            "Source manifests disagree on the detector-label/localization policy: "
            f"{details}"
        )
    if all(audit.provenance_status == VERIFIED_CAPTURE_PROVENANCE for audit in audits):
        script_hashes = {
            str(audit.manifest.get("capture_script_sha256")) for audit in audits
        }
        asset_hashes = {
            str(audit.manifest["simulation_asset_inventory"]["aggregate_sha256"])
            for audit in audits
        }
        if len(script_hashes) != 1:
            raise DatasetMergeError(
                f"Source manifests disagree on capture_script_sha256: {sorted(script_hashes)}"
            )
        if len(asset_hashes) != 1:
            raise DatasetMergeError(
                "Source manifests disagree on simulation asset inventory: "
                f"{sorted(asset_hashes)}"
            )
        partitions = [
            str(audit.manifest["transport_environment"]["observed_values"]["IGN_PARTITION"])
            for audit in audits
        ]
        if len(set(partitions)) != len(partitions):
            raise DatasetMergeError(
                "Source manifests reuse IGN_PARTITION values; each source capture must "
                f"have an isolated transport partition, got {partitions}"
            )

    group_splits: dict[str, set[str]] = defaultdict(set)
    group_rows: dict[str, list[str]] = defaultdict(list)
    for audit in audits:
        for sample in audit.samples:
            group_splits[sample.pose_group].add(sample.split)
            group_rows[sample.pose_group].append(
                f"{sample.camera_id}:{sample.split}:{sample.image.relative_path}"
            )
    conflicts = {
        group: sorted(group_rows[group])
        for group, splits in group_splits.items()
        if len(splits) > 1
    }
    if conflicts:
        first_groups = dict(list(sorted(conflicts.items()))[:10])
        raise DatasetMergeError(
            "Train/validation pose-group leakage detected; every commanded XY pose "
            f"must use one split across all cameras: {first_groups}"
        )


def _duplicate_audit(audits: Sequence[CameraAudit]) -> dict[str, Any]:
    by_hash: dict[str, list[AcceptedSample]] = defaultdict(list)
    for audit in audits:
        for sample in audit.samples:
            by_hash[sample.image.sha256].append(sample)
    duplicate_groups: list[dict[str, Any]] = []
    for digest, samples in sorted(by_hash.items()):
        if len(samples) <= 1:
            continue
        camera_ids = sorted({sample.camera_id for sample in samples})
        splits = sorted({sample.split for sample in samples})
        duplicate_groups.append(
            {
                "sha256": digest,
                "camera_ids": camera_ids,
                "splits": splits,
                "files": [
                    {
                        "camera_id": sample.camera_id,
                        "path": sample.image.relative_path,
                    }
                    for sample in sorted(
                        samples, key=lambda item: (item.camera_id, item.image.relative_path)
                    )
                ],
            }
        )
    return {
        "cross_input_duplicate_hash_count": sum(
            len(group["camera_ids"]) > 1 for group in duplicate_groups
        ),
        "cross_input_duplicate_file_count": sum(
            len(group["files"])
            for group in duplicate_groups
            if len(group["camera_ids"]) > 1
        ),
        "within_camera_duplicate_hash_count": sum(
            len(group["camera_ids"]) == 1 for group in duplicate_groups
        ),
        "cross_split_duplicate_hash_count": sum(
            len(group["splits"]) > 1 for group in duplicate_groups
        ),
        "groups": duplicate_groups,
    }


def _qualification_counts(samples: Iterable[AcceptedSample]) -> dict[str, Any]:
    materialized = list(samples)
    qualified = sum(sample.localization_qualified for sample in materialized)
    occlusion = Counter(sample.occlusion_state for sample in materialized)
    mask_policy = Counter(sample.mask_area_policy for sample in materialized)
    min_mask_values = Counter(
        format(sample.min_mask_area_applied_px, ".15g") for sample in materialized
    )
    return {
        "localization_qualified": qualified,
        "localization_excluded": len(materialized) - qualified,
        "by_occlusion_state": {
            name: occlusion[name]
            for name in ("clear", "low_visible_height", "bottom_hidden")
        },
        "by_mask_area_policy": {
            name: mask_policy[name] for name in ("near", "far")
        },
        "by_min_mask_area_applied_px": dict(sorted(min_mask_values.items())),
        "by_sample_kind": dict(Counter(sample.sample_kind for sample in materialized)),
    }


def _count_samples(audits: Sequence[CameraAudit], config: MergeConfig) -> dict[str, Any]:
    bins = _range_bin_names(config.range_edges_m)
    by_camera: dict[str, Any] = {}
    overall_split = Counter()
    overall_range = Counter()
    overall_samples: list[AcceptedSample] = []
    for audit in audits:
        split_payload: dict[str, Any] = {}
        for split in SPLITS:
            selected = [sample for sample in audit.samples if sample.split == split]
            positives = [sample for sample in selected if sample.sample_kind == "positive"]
            negatives = [sample for sample in selected if sample.sample_kind == "negative"]
            range_counts = Counter(sample.range_bin for sample in positives)
            split_payload[split] = {
                "total": len(selected),
                "positive": len(positives),
                "negative": len(negatives),
                "by_range_bin": {name: range_counts[name] for name in bins},
                "qualification": _qualification_counts(selected),
            }
            overall_split[split] += len(selected)
            overall_range.update(sample.range_bin for sample in positives)
        by_camera[audit.camera_id] = {
            "total": len(audit.samples),
            "qualification": _qualification_counts(audit.samples),
            "by_split": split_payload,
        }
        overall_samples.extend(audit.samples)
    return {
        "total": sum(len(audit.samples) for audit in audits),
        "by_split": {split: overall_split[split] for split in SPLITS},
        "by_range_bin": {name: overall_range[name] for name in bins},
        "qualification": _qualification_counts(overall_samples),
        "by_camera": by_camera,
    }


def _gate_payload(config: MergeConfig) -> dict[str, int]:
    return {
        "min_train_per_camera": config.min_train_per_camera,
        "min_val_per_camera": config.min_val_per_camera,
        "min_train_per_core_range_bin": config.min_train_per_core_range_bin,
        "min_val_per_core_range_bin": config.min_val_per_core_range_bin,
        "max_cross_input_duplicate_hashes": config.max_cross_input_duplicate_hashes,
        "min_negative_train_per_camera": config.min_negative_train_per_camera,
    }


def _enforce_merge_gates(
    counts: Mapping[str, Any],
    duplicates: Mapping[str, Any],
    config: MergeConfig,
    *,
    require_training_negative_floor: bool,
) -> None:
    failures: list[str] = []
    core_bins = _range_bin_names(config.range_edges_m)[1:-1]
    for camera_id in CAMERAS:
        split_counts = counts["by_camera"][camera_id]["by_split"]
        for split, minimum in (
            ("train", config.min_train_per_camera),
            ("val", config.min_val_per_camera),
        ):
            actual = int(split_counts[split]["positive"])
            if actual < minimum:
                failures.append(f"{camera_id}/{split} positive {actual} < {minimum}")
        for split, minimum in (
            ("train", config.min_train_per_core_range_bin),
            ("val", config.min_val_per_core_range_bin),
        ):
            for range_name in core_bins:
                actual = int(split_counts[split]["by_range_bin"][range_name])
                if actual < minimum:
                    failures.append(
                        f"{camera_id}/{split}/{range_name} {actual} < {minimum}"
                    )
        if require_training_negative_floor:
            actual_negatives = int(split_counts["train"]["negative"])
            if actual_negatives < config.min_negative_train_per_camera:
                failures.append(
                    f"{camera_id}/train negative {actual_negatives} < "
                    f"{config.min_negative_train_per_camera}"
                )
    duplicate_count = int(duplicates["cross_input_duplicate_hash_count"])
    if duplicate_count > config.max_cross_input_duplicate_hashes:
        examples = [
            {
                "sha256": group["sha256"],
                "files": group["files"],
            }
            for group in duplicates.get("groups", [])[:3]
        ]
        failures.append(
            f"cross-input duplicate image hashes {duplicate_count} > "
            f"{config.max_cross_input_duplicate_hashes}; examples={examples}"
        )
    if int(duplicates["within_camera_duplicate_hash_count"]) > 0:
        failures.append("within-camera duplicate image hashes must be zero")
    if int(duplicates["cross_split_duplicate_hash_count"]) > 0:
        failures.append("train/validation duplicate image hashes must be zero")
    if failures:
        shown = failures[:40]
        suffix = f"; plus {len(failures) - len(shown)} more" if len(failures) > len(shown) else ""
        raise DatasetMergeError("Merged dataset failed quality gates: " + "; ".join(shown) + suffix)


def _copy_verified(source: SourceFile, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise DatasetMergeError(f"Internal destination collision: {destination}")
    try:
        shutil.copyfile(source.path, destination)
    except OSError as exc:
        raise DatasetMergeError(f"Cannot copy {source.path} to {destination}: {exc}") from exc
    destination_hash = _sha256(destination)
    destination_size = int(destination.stat().st_size)
    if destination_hash != source.sha256 or destination_size != source.size_bytes:
        raise DatasetMergeError(
            f"Source changed or copy verification failed for {source.path}: "
            f"expected {source.sha256}/{source.size_bytes}, got "
            f"{destination_hash}/{destination_size}"
        )
    return {"sha256": destination_hash, "size_bytes": destination_size}


def _source_summary(audit: CameraAudit) -> dict[str, Any]:
    manifest = audit.manifest
    return {
        "camera_id": audit.camera_id,
        "source_dir": str(audit.source_dir),
        "camera_model": manifest["camera_model"],
        "image_topic": manifest["image_topic"],
        "labels_topic": manifest["labels_topic"],
        "world": manifest["world"],
        "world_sha256": manifest["world_sha256"],
        "world_profiles_sha256": manifest["world_profiles_sha256"],
        "capture_provenance_status": audit.provenance_status,
        "capture_script_sha256": manifest.get("capture_script_sha256"),
        "simulation_asset_inventory_sha256": (
            manifest.get("simulation_asset_inventory", {}).get("aggregate_sha256")
            if isinstance(manifest.get("simulation_asset_inventory"), dict)
            else None
        ),
        "transport_environment": (
            manifest.get("transport_environment")
            if isinstance(manifest.get("transport_environment"), dict)
            else None
        ),
        "camera_pose_xyz_rpy": manifest["camera_pose_xyz_rpy"],
        "split_mode": manifest["split_mode"],
        "capture_policy": audit.capture_policy.as_dict(),
        "sample_count": len(audit.samples),
        "qualification": _qualification_counts(audit.samples),
        "source_files": {
            "dataset_manifest.json": audit.manifest_file.sha256,
            "label_diagnostics.csv": audit.diagnostics_file.sha256,
            "data.yaml": audit.data_yaml_file.sha256,
            ".complete": (
                audit.completion_file.sha256 if audit.completion_file is not None else None
            ),
        },
    }


def _merged_provenance(
    audits: Sequence[CameraAudit], config: MergeConfig
) -> dict[str, Any]:
    source_status = {
        audit.camera_id: audit.provenance_status for audit in audits
    }
    training_eligible = all(
        status == VERIFIED_CAPTURE_PROVENANCE for status in source_status.values()
    )
    return {
        "grade": (
            "training_grade" if training_eligible else "diagnostic_legacy_non_training"
        ),
        "training_eligible": training_eligible,
        "source_status": source_status,
        "legacy_override_enabled": config.allow_legacy_diagnostic_provenance,
        "legacy_override_used": not training_eligible,
        "rule": (
            "Every source must carry a verified capture-script hash, exact invocation, "
            "and canonical simulation/input asset inventory for training eligibility."
        ),
    }


def _dataset_card(
    output_dir: Path,
    audits: Sequence[CameraAudit],
    counts: Mapping[str, Any],
    duplicates: Mapping[str, Any],
    config: MergeConfig,
    created_at: str,
) -> dict[str, Any]:
    capture_policy = audits[0].capture_policy
    provenance = _merged_provenance(audits, config)
    intended_use = (
        "Fine-tuning and validating the single-class YOLO segmentation detector "
        f"for the frozen {len(CAMERAS)}-camera {Path(config.expected_world).name} "
        "commissioning study."
        if provenance["training_eligible"]
        else (
            "Diagnostic smoke auditing only. This legacy merge lacks complete capture "
            "code/asset provenance and MUST NOT be used for detector training, model "
            "selection, calibration, or evidence."
        )
    )
    return {
        "schema_version": CARD_SCHEMA_VERSION,
        "dataset_id": output_dir.name,
        "created_at_utc": created_at,
        "intended_use": intended_use,
        "provenance": provenance,
        "out_of_scope": [
            "Claiming detector-confidence calibration without a separate held-out calibration set.",
            "Using validation images for optimization or model selection beyond the declared protocol.",
            f"Generalizing performance claims beyond {Path(config.expected_world).name} "
            "without new evaluation data.",
        ],
        "contract": {
            "world": Path(config.expected_world).name,
            "camera_ids": list(CAMERAS),
            "class_names": {"0": "robot"},
            "task": "segment",
            "pose_group_rule": (
                "Declared pose group when present; otherwise robot_x/robot_y rounded "
                "to 1e-6 m. Yaw is excluded. One group may occur in only one split."
            ),
            "range_edges_m": list(config.range_edges_m),
            "range_bins": list(_range_bin_names(config.range_edges_m)),
            "detector_label_policy": capture_policy.as_dict(),
            "localization_calibration": {
                "eligible_when": "localization_qualified == 1 and occlusion_state == 'clear'",
                "all_detector_samples": "sample_qualifications.csv",
                "eligible_only_index": "localization_calibration_index.csv",
                "prohibition": (
                    "Never use a visible-mask-positive row with localization_qualified=0 "
                    "to calibrate or validate the projected bottom point."
                ),
            },
        },
        "quality_gates": _gate_payload(config),
        "counts": counts,
        "duplicate_audit": duplicates,
        "sources": [_source_summary(audit) for audit in audits],
        "limitations": [
            "Training labels originate from simulator semantic segmentation.",
            "Visible-mask-positive partial silhouettes are valid detector labels but not bottom-point labels.",
            "Camera views can be correlated even when exact image hashes differ.",
            "Range balance does not by itself guarantee pose, yaw, lighting, or occlusion balance.",
        ],
    }


def _markdown_card(card: Mapping[str, Any]) -> str:
    bins = card["contract"]["range_bins"]
    lines = [
        f"# Dataset card: {card['dataset_id']}",
        "",
        str(card["intended_use"]),
        "",
        f"- Schema: `{card['schema_version']}`",
        f"- Created: `{card['created_at_utc']}`",
        f"- World: `{card['contract']['world']}`",
        f"- Provenance grade: `{card['provenance']['grade']}`",
        f"- Training eligible: **{str(card['provenance']['training_eligible']).lower()}**",
        f"- Samples: **{card['counts']['total']}** "
        f"(train {card['counts']['by_split']['train']}, "
        f"validation {card['counts']['by_split']['val']})",
        f"- Bottom-point localization eligible: "
        f"**{card['counts']['qualification']['localization_qualified']}**; excluded: "
        f"**{card['counts']['qualification']['localization_excluded']}**",
        f"- Occlusion policy: `{card['contract']['detector_label_policy']['occlusion_policy']}`",
        "- Bottom-point calibration input: `localization_calibration_index.csv` "
        "(already filtered to localization-qualified rows)",
        "",
        "## Per-camera split and range counts",
        "",
        "| Camera | Split | Total | Loc. eligible | Loc. excluded | "
        + " | ".join(bins)
        + " |",
        "|---|---:|---:|---:|---:|" + "---:|" * len(bins),
    ]
    for camera_id in CAMERAS:
        for split in SPLITS:
            payload = card["counts"]["by_camera"][camera_id]["by_split"][split]
            values = [str(payload["by_range_bin"][name]) for name in bins]
            lines.append(
                f"| {camera_id} | {split} | {payload['total']} | "
                f"{payload['qualification']['localization_qualified']} | "
                f"{payload['qualification']['localization_excluded']} | "
                + " | ".join(values)
                + " |"
            )
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            "| Camera | Status | Capture manifest SHA-256 | Diagnostics SHA-256 |",
            "|---|---|---|---|",
        ]
    )
    for source in card["sources"]:
        lines.append(
            f"| {source['camera_id']} | `{source['capture_provenance_status']}` "
            f"| `{source['source_files']['dataset_manifest.json']}` "
            f"| `{source['source_files']['label_diagnostics.csv']}` |"
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in card["limitations"])
    lines.extend(["", "## Out of scope", ""])
    lines.extend(f"- {item}" for item in card["out_of_scope"])
    return "\n".join(lines) + "\n"


def _sample_record(
    sample: AcceptedSample,
    policy: CapturePolicy,
    *,
    image_output: str,
    label_output: str,
    mask_output: str,
) -> dict[str, Any]:
    return {
        "sample_id": f"{sample.camera_id}__{PurePosixPath(sample.image.relative_path).stem}",
        "camera_id": sample.camera_id,
        "sample_kind": sample.sample_kind,
        "split": sample.split,
        "image": image_output,
        "label": label_output,
        "mask": mask_output,
        "robot_x": sample.robot_x,
        "robot_y": sample.robot_y,
        "robot_yaw": sample.robot_yaw,
        "camera_range_m": sample.camera_range_m,
        "range_bin": sample.range_bin,
        "mask_area_policy": sample.mask_area_policy,
        "min_mask_area_applied_px": sample.min_mask_area_applied_px,
        "mask_area_px": sample.mask_area_px,
        "localization_qualified": sample.localization_qualified,
        "occlusion_state": sample.occlusion_state,
        "occlusion_policy": policy.occlusion_policy,
        "visible_height_fraction": sample.visible_height_fraction,
        "bottom_occlusion_px": sample.bottom_occlusion_px,
        "geometry_certified_no_opportunity": sample.geometry_certified_no_opportunity,
        "no_opportunity_reason": sample.no_opportunity_reason,
        "semantic_robot_pixels": sample.semantic_robot_pixels,
    }


def _write_qualification_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    try:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(QUALIFICATION_FIELDS))
            writer.writeheader()
            for record in records:
                row = {
                    field: (
                        int(record[field])
                        if field in {"localization_qualified", "geometry_certified_no_opportunity"}
                        else "" if record[field] is None else record[field]
                    )
                    for field in QUALIFICATION_FIELDS
                }
                writer.writerow(row)
    except (OSError, csv.Error, KeyError) as exc:
        raise DatasetMergeError(f"Cannot write qualification CSV {path}: {exc}") from exc


def _metadata_inventory(staging: Path, relative_paths: Iterable[str]) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for relative in sorted(relative_paths):
        path = staging / relative
        inventory.append(
            {
                "path": relative,
                "sha256": _sha256(path),
                "size_bytes": int(path.stat().st_size),
            }
        )
    return inventory


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def merge_fourcam_datasets(
    camera_dirs: Mapping[str, Path | str],
    output_dir: Path | str,
    config: MergeConfig | None = None,
) -> dict[str, Any]:
    """Validate four captures and atomically create one immutable merge."""

    config = config or MergeConfig()
    provided = set(camera_dirs)
    expected = set(CAMERAS)
    if provided != expected:
        raise DatasetMergeError(
            f"camera_dirs must contain exactly {list(CAMERAS)}; "
            f"missing={sorted(expected - provided)}, extra={sorted(provided - expected)}"
        )
    resolved_dirs = {
        camera_id: Path(camera_dirs[camera_id]).expanduser().resolve()
        for camera_id in CAMERAS
    }
    if len(set(resolved_dirs.values())) != len(CAMERAS):
        raise DatasetMergeError("Each camera must use a distinct source directory")
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise DatasetMergeError(f"Output already exists and will not be replaced: {output}")
    for camera_id, source in resolved_dirs.items():
        if _is_relative_to(output, source):
            raise DatasetMergeError(
                f"Output must not be inside source dataset {camera_id}: {output}"
            )

    audits = tuple(
        _audit_camera(camera_id, resolved_dirs[camera_id], config) for camera_id in CAMERAS
    )
    _validate_cross_camera_contract(audits)
    duplicates = _duplicate_audit(audits)
    counts = _count_samples(audits, config)
    _enforce_merge_gates(
        counts,
        duplicates,
        config,
        require_training_negative_floor=all(
            audit.provenance_status == VERIFIED_CAPTURE_PROVENANCE for audit in audits
        ),
    )

    created_at = _utc_now()
    card = _dataset_card(output, audits, counts, duplicates, config, created_at)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent))
        )
    except OSError as exc:
        raise DatasetMergeError(f"Cannot create staging directory beside {output}: {exc}") from exc

    try:
        payload_inventory: list[dict[str, Any]] = []
        qualification_records: list[dict[str, Any]] = []
        for audit in audits:
            for sample in audit.samples:
                stem = PurePosixPath(sample.image.relative_path).stem
                prefix = f"{sample.camera_id}__{stem}"
                image_output = (
                    f"images/{sample.split}/{prefix}{sample.image.path.suffix.lower()}"
                )
                label_output = f"labels/{sample.split}/{prefix}.txt"
                mask_output = (
                    f"masks/{sample.split}/{prefix}{sample.mask.path.suffix.lower()}"
                    if sample.mask is not None
                    else ""
                )
                qualification_records.append(
                    _sample_record(
                        sample,
                        audit.capture_policy,
                        image_output=image_output,
                        label_output=label_output,
                        mask_output=mask_output,
                    )
                )
                roles: list[tuple[str, SourceFile, str]] = [
                    ("image", sample.image, image_output),
                    ("label", sample.label, label_output),
                ]
                if sample.mask is not None:
                    roles.append(("mask", sample.mask, mask_output))
                for role, source_file, destination_rel in roles:
                    verified = _copy_verified(source_file, staging / destination_rel)
                    payload_inventory.append(
                        {
                            "role": role,
                            "camera_id": sample.camera_id,
                            "split": sample.split,
                            "source_path": source_file.relative_path,
                            "output_path": destination_rel,
                            **verified,
                        }
                    )

        qualification_records.sort(key=lambda row: str(row["sample_id"]))
        localization_records = [
            row for row in qualification_records if bool(row["localization_qualified"])
        ]
        if any(row["occlusion_state"] != "clear" for row in localization_records):
            raise DatasetMergeError(
                "Internal qualification filter admitted a non-clear localization row"
            )
        _write_qualification_csv(
            staging / "sample_qualifications.csv", qualification_records
        )
        _write_qualification_csv(
            staging / "localization_calibration_index.csv", localization_records
        )

        provenance_paths: list[str] = []
        for audit in audits:
            source_files = [
                audit.manifest_file,
                audit.diagnostics_file,
                audit.data_yaml_file,
            ]
            if audit.completion_file is not None:
                source_files.append(audit.completion_file)
            for source_file in source_files:
                destination_rel = f"provenance/{audit.camera_id}/{source_file.relative_path}"
                _copy_verified(source_file, staging / destination_rel)
                provenance_paths.append(destination_rel)

        data_yaml = {
            "path": str(output),
            "train": "images/train",
            "val": "images/val",
            "names": {0: "robot"},
            "task": "segment",
        }
        (staging / "data.yaml").write_text(
            yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8"
        )
        _write_json(staging / "dataset_card.json", card)
        (staging / "DATASET_CARD.md").write_text(_markdown_card(card), encoding="utf-8")

        metadata_paths = [
            "data.yaml",
            "dataset_card.json",
            "DATASET_CARD.md",
            "sample_qualifications.csv",
            "localization_calibration_index.csv",
            *provenance_paths,
        ]
        metadata_inventory = _metadata_inventory(staging, metadata_paths)
        metadata_by_path = {
            str(entry["path"]): entry for entry in metadata_inventory
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "created_at_utc": created_at,
            "output_dir": str(output),
            "tool": str(Path(__file__).resolve()),
            "tool_sha256": _sha256(Path(__file__).resolve()),
            "provenance": card["provenance"],
            "contract": card["contract"],
            "quality_gates": card["quality_gates"],
            "counts": counts,
            "duplicate_audit": duplicates,
            "sources": card["sources"],
            "sample_records": qualification_records,
            "localization_calibration": {
                "selection": "localization_qualified == 1 and occlusion_state == 'clear'",
                "all_detector_samples": {
                    **metadata_by_path["sample_qualifications.csv"],
                    "row_count": len(qualification_records),
                },
                "eligible_only_index": {
                    **metadata_by_path["localization_calibration_index.csv"],
                    "row_count": len(localization_records),
                },
                "excluded_visible_mask_positive_rows": (
                    len(qualification_records) - len(localization_records)
                ),
            },
            "payload_files": sorted(
                payload_inventory, key=lambda item: str(item["output_path"])
            ),
            "metadata_files": metadata_inventory,
        }
        _write_json(staging / "dataset_manifest.json", manifest)
        manifest_sha = _sha256(staging / "dataset_manifest.json")
        (staging / "dataset_manifest.sha256").write_text(
            f"{manifest_sha}  dataset_manifest.json\n", encoding="ascii"
        )
        (staging / ".complete").write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "dataset_manifest_sha256": manifest_sha,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        # Recheck source contract metadata immediately before publication.  A
        # capture being edited concurrently must never yield a completed merge.
        for audit in audits:
            source_files = [
                audit.manifest_file,
                audit.diagnostics_file,
                audit.data_yaml_file,
            ]
            if audit.completion_file is not None:
                source_files.append(audit.completion_file)
            for source_file in source_files:
                if _sha256(source_file.path) != source_file.sha256:
                    raise DatasetMergeError(f"Source changed during merge: {source_file.path}")
        if output.exists():
            raise DatasetMergeError(f"Output appeared during merge and will not be replaced: {output}")
        os.replace(staging, output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "output_dir": str(output),
        "dataset_manifest": str(output / "dataset_manifest.json"),
        "dataset_manifest_sha256": manifest_sha,
        "counts": counts,
        "quality_gates": _gate_payload(config),
        "provenance": card["provenance"],
    }


def _parse_camera_dirs(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise DatasetMergeError(
                f"--camera-dir must use CAMERA_ID=PATH syntax, got {value!r}"
            )
        camera_id, raw_path = value.split("=", 1)
        camera_id = camera_id.strip()
        raw_path = raw_path.strip()
        if camera_id in result:
            raise DatasetMergeError(f"Duplicate --camera-dir for {camera_id}")
        if not raw_path:
            raise DatasetMergeError(f"Empty --camera-dir path for {camera_id}")
        result[camera_id] = Path(raw_path)
    return result


def _parse_range_edges(value: str) -> tuple[float, ...]:
    try:
        return tuple(float(token.strip()) for token in value.split(",") if token.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Range edges must be comma-separated numbers, got {value!r}"
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and immutably merge four warehouse YOLO segmentation captures."
    )
    parser.add_argument(
        "--camera-dir",
        action="append",
        required=True,
        metavar="CAMERA_ID=PATH",
        help="Repeat exactly once for camera_A, camera_B, camera_C, and camera_D.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-world", default="warehouse_full_4cam.world.sdf")
    parser.add_argument(
        "--range-edges-m",
        type=_parse_range_edges,
        default=(5.0, 8.0, 12.0, 16.0),
        help="Comma-separated range edges; interior bins receive minimum-count gates.",
    )
    parser.add_argument("--min-train-per-camera", type=int, default=1)
    parser.add_argument("--min-val-per-camera", type=int, default=1)
    parser.add_argument("--min-train-per-core-range-bin", type=int, default=1)
    parser.add_argument("--min-val-per-core-range-bin", type=int, default=1)
    parser.add_argument("--max-cross-input-duplicate-hashes", type=int, default=0)
    parser.add_argument("--min-negative-train-per-camera", type=int, default=1)
    parser.add_argument(
        "--allow-legacy-diagnostic-provenance",
        action="store_true",
        help=(
            "Admit captures made before the capture-script/simulation-asset provenance "
            "contract. The merged dataset is marked diagnostic_legacy_non_training and "
            "must not be used for training, calibration, model selection, or evidence."
        ),
    )
    args = parser.parse_args()
    try:
        camera_dirs = _parse_camera_dirs(args.camera_dir)
        config = MergeConfig(
            expected_world=args.expected_world,
            range_edges_m=tuple(args.range_edges_m),
            min_train_per_camera=args.min_train_per_camera,
            min_val_per_camera=args.min_val_per_camera,
            min_train_per_core_range_bin=args.min_train_per_core_range_bin,
            min_val_per_core_range_bin=args.min_val_per_core_range_bin,
            max_cross_input_duplicate_hashes=args.max_cross_input_duplicate_hashes,
            min_negative_train_per_camera=args.min_negative_train_per_camera,
            allow_legacy_diagnostic_provenance=args.allow_legacy_diagnostic_provenance,
        )
        result = merge_fourcam_datasets(camera_dirs, args.output_dir, config)
    except DatasetMergeError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
