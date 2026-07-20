from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = (
    REPO_ROOT
    / "experiments"
    / "multicamera_commissioning_bigwarehouse"
    / "tools"
    / "merge_fourcam_yolo_dataset.py"
)
SPEC = importlib.util.spec_from_file_location("merge_fourcam_yolo_dataset", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
merge_tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = merge_tool
SPEC.loader.exec_module(merge_tool)


def _capture_range_bin(range_m: float) -> str:
    if range_m < 8.0:
        return "lt_8m"
    if range_m < 12.0:
        return "8_to_12m"
    if range_m <= 16.0:
        return "12_to_16m"
    return "gt_16m"


def _write_capture(root: Path, camera_id: str, camera_index: int) -> None:
    contract = merge_tool.CAMERA_CONTRACTS[camera_id]
    rows: list[dict[str, object]] = []
    accepted_ranges: list[str] = []
    for sample_index, (split, robot_x, group) in enumerate(
        (
            ("train", 6.0, "shared_train_pose"),
            ("val", 10.0, "shared_val_pose"),
        )
    ):
        stem = f"sample_{sample_index:06d}"
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        mask_dir = root / "masks" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        image = np.full(
            (12, 16, 3),
            10 + camera_index * 40 + sample_index * 7,
            dtype=np.uint8,
        )
        image[2:10, 4:12, :] = (camera_index * 20, sample_index * 80, 240)
        assert cv2.imwrite(str(image_dir / f"{stem}.png"), image)
        (label_dir / f"{stem}.txt").write_text(
            "0 0.25 0.20 0.75 0.20 0.75 0.80 0.25 0.80\n",
            encoding="utf-8",
        )
        mask = np.zeros((12, 16), dtype=np.uint8)
        mask[2:10, 4:12] = 255
        assert cv2.imwrite(str(mask_dir / f"{stem}.png"), mask)
        range_m = abs(robot_x)
        range_name = _capture_range_bin(range_m)
        accepted_ranges.append(range_name)
        rows.append(
            {
                "camera_id": camera_id,
                "camera_model": contract["camera_model"],
                "sample_kind": "positive",
                "camera_range_m": range_m,
                "camera_range_bin": range_name,
                "min_mask_area_applied_px": 20.0 if range_m < 8.0 else 10.0,
                "sample_index": sample_index,
                "attempt": 0,
                "split": split,
                "accepted": 1,
                "rejection_reason": "",
                "image": f"images/{split}/{stem}.png",
                "label": f"labels/{split}/{stem}.txt",
                "mask": f"masks/{split}/{stem}.png",
                "robot_x": robot_x,
                "robot_y": 0.0,
                "robot_yaw": 0.5 * sample_index,
                "mask_area_px": 64.0,
                "visible_height_fraction": 0.9,
                "bottom_occlusion_px": 0.0,
                "localization_qualified": 1,
                "occlusion_state": "clear",
                "geometry_certified_no_opportunity": 0,
                "no_opportunity_reason": "",
                "semantic_robot_pixels": 64,
                "pose_group_id": group,
            }
        )

    diagnostics_path = root / "label_diagnostics.csv"
    with diagnostics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    range_counts = {key: accepted_ranges.count(key) for key in sorted(set(accepted_ranges))}
    inventory_entries = [
        {
            "logical_path": logical_path,
            "path": f"/frozen/{logical_path}",
            "roles": [role],
            "sha256": digest,
            "size_bytes": size,
        }
        for logical_path, role, digest, size in (
            ("scripts/perception/capture_yolo_dataset.py", "capture_script", "c" * 64, 101),
            ("src/sim/gazebo_worlds/worlds/warehouse_full_4cam.world.sdf", "world", "a" * 64, 102),
            ("src/experiments/config/world_profiles.yaml", "world_profiles", "b" * 64, 103),
            ("experiments/multicamera_commissioning_bigwarehouse/config/study.yaml", "route_exclusion_config", "d" * 64, 104),
            ("src/sim/launch/bringup_sim.launch.py", "sim_launch", "e" * 64, 105),
            ("src/sim/robot_description/urdf/turtlebot3.urdf.xacro", "robot_description", "f" * 64, 106),
            ("src/sim/models/external_camera/model.sdf", "model_asset:external_camera", "1" * 64, 107),
        )
    ]
    simulation_asset_inventory = {
        "schema_version": merge_tool.SIMULATION_ASSET_INVENTORY_SCHEMA_VERSION,
        "aggregate_sha256": merge_tool._canonical_inventory_sha256(inventory_entries),
        "file_count": len(inventory_entries),
        "total_size_bytes": sum(entry["size_bytes"] for entry in inventory_entries),
        "referenced_model_uris": ["model://external_camera"],
        "referenced_model_names": ["external_camera"],
        "files": inventory_entries,
    }
    manifest = {
        "status": "complete",
        "world": "warehouse_full_4cam.world.sdf",
        "source": "gazebo_semantic_segmentation",
        "camera_id": camera_id,
        **contract,
        "world_profiles_path": "/frozen/world_profiles.yaml",
        "world_profiles_sha256": "b" * 64,
        "world_path": "/frozen/warehouse_full_4cam.world.sdf",
        "world_sha256": "a" * 64,
        "capture_script_path": "/frozen/capture_yolo_dataset.py",
        "capture_script_sha256": "c" * 64,
        "capture_invocation": {
            "schema_version": merge_tool.CAPTURE_INVOCATION_SCHEMA_VERSION,
            "python_executable": "/usr/bin/python3",
            "argv": ["scripts/perception/capture_yolo_dataset.py", "--world", "warehouse_full_4cam.world.sdf"],
            "working_directory": "/frozen/repository",
            "resolved_arguments": {
                "world": "warehouse_full_4cam.world.sdf",
                "camera_model": contract["camera_model"],
                "image_topic": contract["image_topic"],
                "labels_topic": contract["labels_topic"],
            },
        },
        "transport_environment": {
            "schema_version": merge_tool.CAPTURE_TRANSPORT_SCHEMA_VERSION,
            "required_values": dict(merge_tool.TRAINING_CAPTURE_TRANSPORT_VALUES),
            "observed_values": {
                "ROS_LOCALHOST_ONLY": "1",
                "IGN_IP": "127.0.0.1",
                "GZ_IP": "127.0.0.1",
                "ROS_DOMAIN_ID": "79",
                "IGN_PARTITION": f"fourcam_capture_{camera_id}",
            },
            "isolation_verified": True,
            "training_eligible": True,
            "diagnostic_override_used": False,
            "violations": [],
        },
        "simulation_asset_inventory": simulation_asset_inventory,
        "robot_label": 23,
        "camera_pose_xyz_rpy": [0.0, 0.0, 6.1, 0.0, 0.92, 0.0],
        "camera_intrinsics": {
            "img_width": 16,
            "img_height": 12,
            "fov_h_rad": 1.5708,
        },
        "split_mode": "spatial_cell",
        "capture_thresholds": {
            "min_mask_area": 20.0,
            "far_range_start_m": 8.0,
            "far_min_mask_area": 10.0,
            "min_visible_height_fraction": 0.55,
            "max_bottom_occlusion_px": 20.0,
            "occlusion_policy": "reject",
            "min_accepted_samples": 2,
            "min_accept_fraction": 0.5,
            "max_final_duplicate_fraction": 0.0,
        },
        "audit": {
            "planned_samples": 2,
            "accepted_samples": 2,
            "accepted_by_sample_kind": {"positive": 2},
            "positive_accepted_samples": 2,
            "negative_accepted_samples": 0,
            "accepted_fraction": 1.0,
            "split_counts": {"train": 1, "val": 1},
            "accepted_by_range_bin": range_counts,
            "accepted_by_occlusion_state": {"clear": 2},
            "localization_qualified_samples": 2,
            "duplicate_accepted_images": 0,
            "duplicate_accepted_fraction": 0.0,
        },
    }
    manifest_path = root / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (root / ".complete").write_text(
        json.dumps(
            {
                "schema_version": merge_tool.CAPTURE_COMPLETION_SCHEMA_VERSION,
                "status": "complete",
                "training_eligible": True,
                "camera_id": camera_id,
                "dataset_manifest": "dataset_manifest.json",
                "dataset_manifest_sha256": hashlib.sha256(
                    manifest_path.read_bytes()
                ).hexdigest(),
                "completed_at": "2026-07-17T12:00:00",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "robot"},
                "task": "segment",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


@pytest.fixture
def capture_dirs(tmp_path: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for index, camera_id in enumerate(merge_tool.CAMERAS):
        root = tmp_path / camera_id
        _write_capture(root, camera_id, index)
        result[camera_id] = root
    return result


def _loose_config(**overrides: object) -> object:
    kwargs = {
        "min_train_per_camera": 1,
        "min_val_per_camera": 1,
        "min_train_per_core_range_bin": 0,
        "min_val_per_core_range_bin": 0,
        "max_cross_input_duplicate_hashes": 0,
        "min_negative_train_per_camera": 0,
    }
    kwargs.update(overrides)
    return merge_tool.MergeConfig(**kwargs)


def _rewrite_json(path: Path, update: object) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    update(payload)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    completion_path = path.parent / ".complete"
    if path.name == "dataset_manifest.json" and completion_path.is_file():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        completion["dataset_manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        completion_path.write_text(
            json.dumps(completion, indent=2) + "\n", encoding="utf-8"
        )


def _rewrite_diagnostics(path: Path, update: object) -> None:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    update(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_merge_writes_prefixed_atomic_dataset_card_and_sha_provenance(
    tmp_path: Path, capture_dirs: dict[str, Path]
) -> None:
    output = tmp_path / "merged"

    result = merge_tool.merge_fourcam_datasets(capture_dirs, output, _loose_config())

    assert result["counts"]["total"] == 8
    assert result["counts"]["by_split"] == {"train": 4, "val": 4}
    assert sorted(path.name for path in (output / "images" / "train").iterdir()) == [
        f"{camera_id}__sample_000000.png" for camera_id in merge_tool.CAMERAS
    ]
    assert sorted(path.name for path in (output / "labels" / "val").iterdir()) == [
        f"{camera_id}__sample_000001.txt" for camera_id in merge_tool.CAMERAS
    ]
    assert (output / "masks" / "train" / "camera_A__sample_000000.png").is_file()
    assert (output / "provenance" / "camera_D" / "label_diagnostics.csv").is_file()
    with (output / "sample_qualifications.csv").open(
        "r", newline="", encoding="utf-8"
    ) as handle:
        qualifications = list(csv.DictReader(handle))
    assert len(qualifications) == 8
    assert {row["localization_qualified"] for row in qualifications} == {"1"}
    with (output / "localization_calibration_index.csv").open(
        "r", newline="", encoding="utf-8"
    ) as handle:
        localization_rows = list(csv.DictReader(handle))
    assert len(localization_rows) == 8
    data = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    assert data["path"] == str(output.resolve())
    assert data["names"] == {0: "robot"}
    manifest_path = output / "dataset_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    expected_sha = hashlib.sha256(manifest_bytes).hexdigest()
    assert result["dataset_manifest_sha256"] == expected_sha
    assert (output / "dataset_manifest.sha256").read_text(encoding="ascii") == (
        f"{expected_sha}  dataset_manifest.json\n"
    )
    manifest = json.loads(manifest_bytes)
    assert manifest["status"] == "complete"
    assert manifest["provenance"]["grade"] == "training_grade"
    assert manifest["provenance"]["training_eligible"] is True
    assert len(manifest["payload_files"]) == 24
    assert len(manifest["sample_records"]) == 8
    assert manifest["localization_calibration"]["eligible_only_index"]["row_count"] == 8
    assert {entry["camera_id"] for entry in manifest["payload_files"]} == set(
        merge_tool.CAMERAS
    )
    assert json.loads((output / "dataset_card.json").read_text(encoding="utf-8"))[
        "counts"
    ]["total"] == 8
    assert (output / ".complete").is_file()

    with pytest.raises(merge_tool.DatasetMergeError, match="will not be replaced"):
        merge_tool.merge_fourcam_datasets(capture_dirs, output, _loose_config())


def test_legacy_capture_requires_explicit_diagnostic_non_training_override(
    tmp_path: Path, capture_dirs: dict[str, Path]
) -> None:
    provenance_fields = (
        "capture_script_path",
        "capture_script_sha256",
        "capture_invocation",
        "transport_environment",
        "simulation_asset_inventory",
    )
    for camera_id in merge_tool.CAMERAS:
        _rewrite_json(
            capture_dirs[camera_id] / "dataset_manifest.json",
            lambda payload: [payload.pop(field) for field in provenance_fields],
        )

    refused = tmp_path / "legacy_refused"
    with pytest.raises(
        merge_tool.DatasetMergeError,
        match="training-grade merge is refused",
    ):
        merge_tool.merge_fourcam_datasets(capture_dirs, refused, _loose_config())
    assert not refused.exists()

    diagnostic = tmp_path / "legacy_diagnostic"
    result = merge_tool.merge_fourcam_datasets(
        capture_dirs,
        diagnostic,
        _loose_config(allow_legacy_diagnostic_provenance=True),
    )
    assert result["provenance"]["grade"] == "diagnostic_legacy_non_training"
    assert result["provenance"]["training_eligible"] is False
    manifest = json.loads(
        (diagnostic / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["provenance"]["legacy_override_used"] is True
    assert "MUST NOT be used" in json.loads(
        (diagnostic / "dataset_card.json").read_text(encoding="utf-8")
    )["intended_use"]


def test_merge_rejects_tampered_simulation_asset_inventory(
    tmp_path: Path, capture_dirs: dict[str, Path]
) -> None:
    manifest_path = capture_dirs["camera_A"] / "dataset_manifest.json"

    def corrupt_asset_hash(payload: dict[str, object]) -> None:
        payload["simulation_asset_inventory"]["files"][0]["sha256"] = "9" * 64

    _rewrite_json(manifest_path, corrupt_asset_hash)
    output = tmp_path / "tampered_provenance"
    with pytest.raises(
        merge_tool.DatasetMergeError,
        match="aggregate_sha256 does not match",
    ):
        merge_tool.merge_fourcam_datasets(capture_dirs, output, _loose_config())
    assert not output.exists()


def test_merge_rejects_nonisolated_capture_transport(
    tmp_path: Path, capture_dirs: dict[str, Path]
) -> None:
    manifest_path = capture_dirs["camera_A"] / "dataset_manifest.json"

    def corrupt_transport(payload: dict[str, object]) -> None:
        payload["transport_environment"]["observed_values"]["IGN_IP"] = "0.0.0.0"

    _rewrite_json(manifest_path, corrupt_transport)
    completion_path = capture_dirs["camera_A"] / ".complete"
    _rewrite_json(
        completion_path,
        lambda payload: payload.update(
            {"dataset_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
        ),
    )
    output = tmp_path / "unisolated_transport"

    with pytest.raises(
        merge_tool.DatasetMergeError,
        match=r"transport_environment\.observed_values\.IGN_IP",
    ):
        merge_tool.merge_fourcam_datasets(capture_dirs, output, _loose_config())
    assert not output.exists()


def test_merge_rejects_reused_capture_transport_partition(
    tmp_path: Path, capture_dirs: dict[str, Path]
) -> None:
    manifest_path = capture_dirs["camera_B"] / "dataset_manifest.json"

    def reuse_partition(payload: dict[str, object]) -> None:
        payload["transport_environment"]["observed_values"]["IGN_PARTITION"] = (
            "fourcam_capture_camera_A"
        )

    _rewrite_json(manifest_path, reuse_partition)
    completion_path = capture_dirs["camera_B"] / ".complete"
    _rewrite_json(
        completion_path,
        lambda payload: payload.update(
            {"dataset_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
        ),
    )
    output = tmp_path / "reused_transport_partition"

    with pytest.raises(merge_tool.DatasetMergeError, match="reuse IGN_PARTITION"):
        merge_tool.merge_fourcam_datasets(capture_dirs, output, _loose_config())
    assert not output.exists()


def test_merge_rejects_manifest_topic_mismatch_without_partial_output(
    tmp_path: Path, capture_dirs: dict[str, Path]
) -> None:
    manifest_path = capture_dirs["camera_C"] / "dataset_manifest.json"
    _rewrite_json(
        manifest_path,
        lambda payload: payload.update({"image_topic": "/external_camera_b/image_raw"}),
    )
    output = tmp_path / "bad_topic"

    with pytest.raises(merge_tool.DatasetMergeError, match="camera_C.*image_topic"):
        merge_tool.merge_fourcam_datasets(capture_dirs, output, _loose_config())

    assert not output.exists()


def test_merge_rejects_cross_camera_pose_group_split_leakage(
    tmp_path: Path, capture_dirs: dict[str, Path]
) -> None:
    diagnostics = capture_dirs["camera_B"] / "label_diagnostics.csv"

    def reuse_train_group_for_validation(rows: list[dict[str, str]]) -> None:
        next(row for row in rows if row["split"] == "val")[
            "pose_group_id"
        ] = "shared_train_pose"

    _rewrite_diagnostics(diagnostics, reuse_train_group_for_validation)
    output = tmp_path / "leaky"

    with pytest.raises(merge_tool.DatasetMergeError, match="pose-group leakage"):
        merge_tool.merge_fourcam_datasets(capture_dirs, output, _loose_config())

    assert not output.exists()


def test_merge_rejects_orphan_label(
    tmp_path: Path, capture_dirs: dict[str, Path]
) -> None:
    (capture_dirs["camera_D"] / "labels" / "train" / "orphan.txt").write_text(
        "0 0.1 0.1 0.2 0.1 0.2 0.2\n", encoding="utf-8"
    )
    output = tmp_path / "orphaned"

    with pytest.raises(merge_tool.DatasetMergeError, match="pairing mismatch"):
        merge_tool.merge_fourcam_datasets(capture_dirs, output, _loose_config())

    assert not output.exists()


def test_merge_rejects_cross_input_image_hash_duplicate(
    tmp_path: Path, capture_dirs: dict[str, Path]
) -> None:
    shutil.copyfile(
        capture_dirs["camera_A"] / "images" / "train" / "sample_000000.png",
        capture_dirs["camera_B"] / "images" / "train" / "sample_000000.png",
    )
    output = tmp_path / "duplicate"

    with pytest.raises(merge_tool.DatasetMergeError, match="duplicate image hashes 1"):
        merge_tool.merge_fourcam_datasets(capture_dirs, output, _loose_config())

    assert not output.exists()


def test_merge_enforces_per_camera_split_range_minimums(
    tmp_path: Path, capture_dirs: dict[str, Path]
) -> None:
    output = tmp_path / "under_range_gate"
    config = _loose_config(min_train_per_core_range_bin=1)

    with pytest.raises(
        merge_tool.DatasetMergeError,
        match=r"camera_A/train/8_to_lt_12m 0 < 1",
    ):
        merge_tool.merge_fourcam_datasets(capture_dirs, output, config)

    assert not output.exists()


def test_visible_mask_positive_stays_in_detector_data_but_out_of_localization_index(
    tmp_path: Path, capture_dirs: dict[str, Path]
) -> None:
    for camera_id in merge_tool.CAMERAS:
        manifest_path = capture_dirs[camera_id] / "dataset_manifest.json"

        def enable_visible_mask_positive(payload: dict[str, object]) -> None:
            payload["capture_thresholds"]["occlusion_policy"] = "visible-mask-positive"

        _rewrite_json(manifest_path, enable_visible_mask_positive)

    diagnostics = capture_dirs["camera_B"] / "label_diagnostics.csv"

    def mark_bottom_hidden(rows: list[dict[str, str]]) -> None:
        row = next(item for item in rows if item["split"] == "val")
        row["localization_qualified"] = "0"
        row["occlusion_state"] = "bottom_hidden"
        row["bottom_occlusion_px"] = "35.0"

    _rewrite_diagnostics(diagnostics, mark_bottom_hidden)

    def update_audit(payload: dict[str, object]) -> None:
        payload["audit"]["accepted_by_occlusion_state"] = {
            "clear": 1,
            "bottom_hidden": 1,
        }
        payload["audit"]["localization_qualified_samples"] = 1

    _rewrite_json(capture_dirs["camera_B"] / "dataset_manifest.json", update_audit)
    output = tmp_path / "visible_mask_positive"

    merge_tool.merge_fourcam_datasets(capture_dirs, output, _loose_config())

    with (output / "sample_qualifications.csv").open(
        "r", newline="", encoding="utf-8"
    ) as handle:
        all_detector_rows = list(csv.DictReader(handle))
    with (output / "localization_calibration_index.csv").open(
        "r", newline="", encoding="utf-8"
    ) as handle:
        localization_rows = list(csv.DictReader(handle))
    excluded = [row for row in all_detector_rows if row["localization_qualified"] == "0"]
    assert len(all_detector_rows) == 8
    assert len(excluded) == 1
    assert excluded[0]["occlusion_state"] == "bottom_hidden"
    assert excluded[0]["image"].endswith("camera_B__sample_000001.png")
    assert len(localization_rows) == 7
    assert excluded[0]["sample_id"] not in {row["sample_id"] for row in localization_rows}
    manifest = json.loads((output / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["qualification"]["localization_excluded"] == 1
    assert manifest["localization_calibration"]["eligible_only_index"]["row_count"] == 7
    assert manifest["contract"]["detector_label_policy"]["occlusion_policy"] == (
        "visible-mask-positive"
    )


def test_merge_rejects_row_that_violates_declared_range_mask_policy(
    tmp_path: Path, capture_dirs: dict[str, Path]
) -> None:
    diagnostics = capture_dirs["camera_A"] / "label_diagnostics.csv"

    def corrupt_applied_gate(rows: list[dict[str, str]]) -> None:
        next(row for row in rows if row["split"] == "train")[
            "min_mask_area_applied_px"
        ] = "10.0"

    _rewrite_diagnostics(diagnostics, corrupt_applied_gate)
    output = tmp_path / "wrong_mask_policy"

    with pytest.raises(merge_tool.DatasetMergeError, match="min-mask policy mismatch"):
        merge_tool.merge_fourcam_datasets(capture_dirs, output, _loose_config())

    assert not output.exists()
