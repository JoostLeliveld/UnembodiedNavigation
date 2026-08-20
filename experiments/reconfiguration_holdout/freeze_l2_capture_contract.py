#!/usr/bin/env python3
"""Freeze and enforce exact L2 capture membership.

The contract is generated after the public L2 layout draw and world generation but
before capture. It commits to every design input by SHA-256 and to the exact 15,072
camera-position-heading keys inherited from L0. Validation is deliberately set
equality, not an intersection: a duplicate, missing, or extra row fails the study.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONTRACT = HERE / "frozen/L2_capture_contract.json"
PREREG = HERE / "SECOND_RECONFIGURATION_PREREGISTRATION.md"
ELIGIBILITY = HERE / "layouts/L2_eligibility.json"
LAYOUT = HERE / "layouts/L2_layout.json"
BASE_WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_full_4cam.world.sdf"
L2_WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_full_4cam_recfg2.world.sdf"
L2_WORLD_INSTALL = (
    REPO / "install/sim/share/sim/gazebo_worlds/worlds/warehouse_full_4cam_recfg2.world.sdf"
)
PROFILES = HERE / "world_profiles_variants.yaml"
DETECTOR = REPO / "logs/perception_models/warehouse_yolo_detector_4cam_v3_960/model.pt"
L0_CAPTURE = REPO / "logs/visibility_comparison/commissioning_grid_20260807"
L2_CAPTURE = REPO / "logs/visibility_comparison/recfg_holdout_L2"

WORLD_FILE = "warehouse_full_4cam_recfg2.world.sdf"
WORLD_NAME = "warehouse_full_4cam_recfg2"
CAMERAS = (
    "external_camera",
    "external_camera_b",
    "external_camera_c",
    "external_camera_d",
)
HEADINGS = (0.0, 1.5707963267948966, 3.141592653589793, 4.71238898038469)
THETA_TOL = 1e-6
EXPECTED_POSITIONS = 942
EXPECTED_COUNT = EXPECTED_POSITIONS * len(HEADINGS) * len(CAMERAS)

HASHED_INPUTS = {
    "second_preregistration": PREREG,
    "eligibility_artifact": ELIGIBILITY,
    "precommitted_beacon_request": HERE / "layouts/L2_BEACON_REQUEST.md",
    "selected_layout": LAYOUT,
    "base_world": BASE_WORLD,
    "generated_L2_world": L2_WORLD,
    "variant_world_profiles": PROFILES,
    "detector_weights": DETECTOR,
    "L0_capture_manifest": L0_CAPTURE / "capture_manifest.json",
    "L0_detector_manifest": L0_CAPTURE / "manifest.json",
    "L0_samples": L0_CAPTURE / "samples.csv",
    "layout_generator": HERE / "choose_second_layout.py",
    "world_generator": HERE / "make_variant_worlds.py",
    "capture_driver": HERE / "capture_environment.sh",
    "contract_generator": HERE / "freeze_l2_capture_contract.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_key(row: dict[str, str]) -> str:
    try:
        camera = str(row["camera_frame"]).strip()
        x = f"{float(row['x']):.8f}"
        y = f"{float(row['y']):.8f}"
        theta = f"{float(row['theta']):.8f}"
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid membership row: {exc}") from exc
    return "|".join((camera, x, y, theta))


def membership_digest(keys: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for key in sorted(keys):
        digest.update((key + "\n").encode("utf-8"))
    return digest.hexdigest()


def load_membership(path: Path, *, filter_headings: bool) -> tuple[list[str], list[dict]]:
    if not path.is_file():
        raise ValueError(f"membership table does not exist: {path}")
    rows: list[dict] = []
    keys: list[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"camera_frame", "x", "y", "theta"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} lacks membership columns {sorted(missing)}")
        for row in reader:
            theta = float(row["theta"])
            if filter_headings and not any(abs(theta - value) <= THETA_TOL for value in HEADINGS):
                continue
            rows.append(row)
            keys.append(canonical_key(row))
    return keys, rows


def exact_membership(keys: list[str], expected_keys: list[str], *, label: str) -> None:
    duplicates = sorted(key for key, count in Counter(keys).items() if count != 1)
    if duplicates:
        raise ValueError(f"{label} has {len(duplicates)} duplicate membership keys; first={duplicates[0]}")
    observed = set(keys)
    expected = set(expected_keys)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra or len(keys) != len(expected_keys):
        detail = (
            f"{label} membership mismatch: rows={len(keys)}, expected={len(expected_keys)}, "
            f"missing={len(missing)}, extra={len(extra)}"
        )
        if missing:
            detail += f", first_missing={missing[0]}"
        if extra:
            detail += f", first_extra={extra[0]}"
        raise ValueError(detail)


def l0_expected_membership() -> tuple[list[str], dict]:
    keys, rows = load_membership(L0_CAPTURE / "samples.csv", filter_headings=True)
    if len(keys) != EXPECTED_COUNT:
        raise ValueError(f"L0 four-heading membership has {len(keys)} rows, expected {EXPECTED_COUNT}")
    duplicates = [key for key, count in Counter(keys).items() if count != 1]
    if duplicates:
        raise ValueError(f"L0 comparison source has {len(duplicates)} duplicate keys")
    positions = sorted({
        (f"{float(row['x']):.8f}", f"{float(row['y']):.8f}")
        for row in rows
    })
    cameras = sorted({str(row["camera_frame"]).strip() for row in rows})
    headings = sorted({f"{float(row['theta']):.8f}" for row in rows})
    if len(positions) != EXPECTED_POSITIONS:
        raise ValueError(f"L0 source has {len(positions)} positions, expected {EXPECTED_POSITIONS}")
    if cameras != sorted(CAMERAS):
        raise ValueError(f"L0 source camera set differs: {cameras}")
    expected_heading_strings = sorted(f"{value:.8f}" for value in HEADINGS)
    if headings != expected_heading_strings:
        raise ValueError(f"L0 source heading set differs: {headings}")
    return keys, {
        "key_columns": ["camera_frame", "x", "y", "theta"],
        "float_canonicalisation": "parse as binary64 and format with .8f",
        "key_separator": "|",
        "sort": "UTF-8 lexicographic canonical key",
        "row_count": len(keys),
        "unique_position_count": len(positions),
        "position_membership_sha256": membership_digest("|".join(value) for value in positions),
        "membership_sha256": membership_digest(keys),
    }


def build_contract() -> dict:
    if L2_CAPTURE.exists() and any(L2_CAPTURE.iterdir()):
        raise ValueError(
            f"cannot freeze after L2 capture material exists: {L2_CAPTURE}"
        )
    for label, path in HASHED_INPUTS.items():
        if not path.is_file():
            raise ValueError(f"cannot freeze: {label} is missing at {path}")
    layout = json.loads(LAYOUT.read_text(encoding="utf-8"))
    selection = layout.get("selection", {})
    randomness = selection.get("external_randomness", {})
    if layout.get("environment_key") != "L2" or layout.get("world_name") != WORLD_NAME:
        raise ValueError("selected layout is not the canonical L2 design")
    if selection.get("outcome_blind") is not True:
        raise ValueError("L2 layout is not marked outcome-blind")
    if randomness.get("source_type") != "NIST Randomness Beacon 2.0":
        raise ValueError("canonical L2 layout was not selected by a persisted NIST pulse")
    if sha256(L2_WORLD) != sha256(L2_WORLD_INSTALL):
        raise ValueError("source and installed L2 worlds are not byte-identical")

    expected_keys, membership = l0_expected_membership()
    if len(expected_keys) != EXPECTED_COUNT:
        raise AssertionError("internal expected-membership count drift")
    return {
        "schema_version": 1,
        "status": "frozen_before_L2_capture",
        "environment_key": "L2",
        "world_file": WORLD_FILE,
        "world_name": WORLD_NAME,
        "hash_algorithm": "SHA-256",
        "inputs": {
            label: {
                "path": str(path.relative_to(REPO)),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for label, path in HASHED_INPUTS.items()
        },
        "installed_world": {
            "path": str(L2_WORLD_INSTALL.relative_to(REPO)),
            "must_equal_input": "generated_L2_world",
        },
        "capture": {
            "method": "grid_teleport",
            "bounds": {
                "xmin": -11.7,
                "xmax": 11.7,
                "ymin": -9.0,
                "ymax": 9.0,
                "nx": 46,
                "ny": 36,
                "wall_margin_m": 0.45,
            },
            "cameras": list(CAMERAS),
            "headings_rad": list(HEADINGS),
            "position_count": EXPECTED_POSITIONS,
            "expected_sample_count": EXPECTED_COUNT,
            "required_n_stale_views": 0,
            "position_list_json": None,
        },
        "detector": {
            "weights_input": "detector_weights",
            "device": "0",
            "imgsz": 640,
            "conf_threshold_at_extraction": 0.01,
            "iou_threshold": 0.45,
            "target_class": "robot",
            "class_id": 0,
            "use_masks": False,
            "primary_rethreshold": 0.25,
        },
        "membership_contract": {
            "source_input": "L0_samples",
            "selected_headings_tolerance_rad": THETA_TOL,
            **membership,
            "comparison_rule": "exact set equality with multiplicity one; no intersection",
            "image_rule": "every samples.csv image_path exists, is nonempty, unique, and remains inside capture_dir",
            "failure_rule": "abort before analysis; repeat full capture with same frozen layout",
        },
    }


def render(value: dict) -> str:
    return json.dumps(value, indent=2, sort_keys=False) + "\n"


def freeze_contract(path: Path, *, check: bool) -> None:
    expected = render(build_contract())
    if check:
        if not path.is_file() or path.read_text(encoding="utf-8") != expected:
            raise ValueError(f"frozen contract differs from current inputs: {path}")
        print(f"[L2 contract] PASS: {path}")
        return
    if path.exists():
        if path.read_text(encoding="utf-8") != expected:
            raise ValueError(f"refusing to replace a different frozen contract: {path}")
        print(f"[L2 contract] already frozen and identical: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(expected, encoding="utf-8")
    print(f"[L2 contract] froze {path}")


def verify_inputs(contract: dict) -> None:
    for label, expected in contract["inputs"].items():
        path = REPO / expected["path"]
        if not path.is_file():
            raise ValueError(f"frozen input {label} is missing: {path}")
        observed = sha256(path)
        if observed != expected["sha256"]:
            raise ValueError(f"frozen input {label} hash drift: {observed} != {expected['sha256']}")
    world_input = contract["inputs"][contract["installed_world"]["must_equal_input"]]
    installed = REPO / contract["installed_world"]["path"]
    if not installed.is_file() or sha256(installed) != world_input["sha256"]:
        raise ValueError("installed L2 world differs from the frozen generated world")
    keys, _ = l0_expected_membership()
    observed_membership = membership_digest(keys)
    expected_membership = contract["membership_contract"]["membership_sha256"]
    if observed_membership != expected_membership:
        raise ValueError("L0 expected membership no longer matches the frozen contract")


def _equal_number(observed, expected, *, tolerance: float = 1e-12) -> bool:
    try:
        return abs(float(observed) - float(expected)) <= tolerance
    except (TypeError, ValueError):
        return False


def validate_manifest(manifest: dict, contract: dict) -> None:
    capture = contract["capture"]
    problems = []
    if manifest.get("world") != contract["world_file"]:
        problems.append(f"world={manifest.get('world')!r}")
    if manifest.get("world_name") != contract["world_name"]:
        problems.append(f"world_name={manifest.get('world_name')!r}")
    if manifest.get("capture_method") != capture["method"]:
        problems.append(f"capture_method={manifest.get('capture_method')!r}")
    cameras = [manifest.get("camera_frame"), *(manifest.get("extra_camera_frames") or [])]
    if cameras != capture["cameras"]:
        problems.append(f"cameras={cameras!r}")
    if int(manifest.get("n_stale_views", -1)) != capture["required_n_stale_views"]:
        problems.append(f"n_stale_views={manifest.get('n_stale_views')!r}")
    if manifest.get("stale_views") not in ([], None):
        problems.append(f"stale_views={manifest.get('stale_views')!r}")
    if int(manifest.get("sample_count", -1)) != capture["expected_sample_count"]:
        problems.append(f"sample_count={manifest.get('sample_count')!r}")
    heading = manifest.get("heading_sampling") or {}
    if int(heading.get("position_count", -1)) != capture["position_count"]:
        problems.append(f"position_count={heading.get('position_count')!r}")
    if int(heading.get("yaw_samples", -1)) != len(capture["headings_rad"]):
        problems.append(f"yaw_samples={heading.get('yaw_samples')!r}")
    yaws = heading.get("yaw_list_rad") or []
    if len(yaws) != len(capture["headings_rad"]) or any(
        not _equal_number(observed, expected, tolerance=THETA_TOL)
        for observed, expected in zip(yaws, capture["headings_rad"])
    ):
        problems.append(f"yaw_list_rad={yaws!r}")
    bounds = manifest.get("visibility_bounds") or {}
    for key, expected in capture["bounds"].items():
        if not _equal_number(bounds.get(key), expected):
            problems.append(f"visibility_bounds.{key}={bounds.get(key)!r}")
    world_path = Path(str(manifest.get("world_path", "")))
    world_hash = contract["inputs"]["generated_L2_world"]["sha256"]
    if not world_path.is_file() or sha256(world_path) != world_hash:
        problems.append("world_path missing or hash differs from generated_L2_world")
    profiles_path = Path(str(manifest.get("world_profiles_path", "")))
    profile_hash = contract["inputs"]["variant_world_profiles"]["sha256"]
    if not profiles_path.is_file() or sha256(profiles_path) != profile_hash:
        problems.append("world_profiles_path missing or hash differs from frozen profile")
    if problems:
        raise ValueError("capture manifest violates frozen L2 contract: " + "; ".join(problems))


def validate_images(capture_dir: Path, rows: list[dict]) -> None:
    root = capture_dir.resolve()
    paths = []
    for row_number, row in enumerate(rows, start=2):
        relative = Path(str(row.get("image_path", "")))
        image = (capture_dir / relative).resolve()
        if not image.is_relative_to(root):
            raise ValueError(f"samples.csv row {row_number} image escapes capture_dir: {relative}")
        if not image.is_file() or image.stat().st_size <= 0:
            raise ValueError(f"samples.csv row {row_number} image missing/empty: {relative}")
        paths.append(str(relative))
    duplicates = [value for value, count in Counter(paths).items() if count != 1]
    if duplicates:
        raise ValueError(f"samples.csv repeats {len(duplicates)} image paths; first={duplicates[0]}")


def validate_detector_manifest(capture_dir: Path, contract: dict) -> None:
    path = capture_dir / "manifest.json"
    if not path.is_file():
        raise ValueError(f"detector manifest is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = contract["detector"]
    checks = {
        "yolo_device": expected["device"],
        "yolo_imgsz": expected["imgsz"],
        "yolo_conf_threshold": expected["conf_threshold_at_extraction"],
        "yolo_iou_threshold": expected["iou_threshold"],
        "yolo_target_class": expected["target_class"],
        "yolo_class_id": expected["class_id"],
        "yolo_use_masks": expected["use_masks"],
    }
    problems = []
    for key, value in checks.items():
        observed = manifest.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not _equal_number(observed, value):
                problems.append(f"{key}={observed!r}")
        elif observed != value:
            problems.append(f"{key}={observed!r}")
    model = Path(str(manifest.get("yolo_model", "")))
    if not model.is_file() or sha256(model) != contract["inputs"]["detector_weights"]["sha256"]:
        problems.append("yolo_model missing or hash mismatch")
    summary = manifest.get("yolo_summary") or {}
    if int(summary.get("sample_count", -1)) != contract["capture"]["expected_sample_count"]:
        problems.append(f"yolo_summary.sample_count={summary.get('sample_count')!r}")
    if problems:
        raise ValueError("detector manifest violates frozen L2 contract: " + "; ".join(problems))


def validate_capture(contract_path: Path, capture_dir: Path, *, stage: str) -> dict:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    verify_inputs(contract)
    manifest_path = capture_dir / "capture_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"capture manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest, contract)
    expected_keys, _ = l0_expected_membership()
    samples_keys, samples_rows = load_membership(capture_dir / "samples.csv", filter_headings=False)
    exact_membership(samples_keys, expected_keys, label="samples.csv")
    wrong_world = sorted({
        str(row.get("world", "")) for row in samples_rows
        if str(row.get("world", "")) != contract["world_file"]
    })
    if wrong_world:
        raise ValueError(f"samples.csv contains wrong world value(s): {wrong_world}")
    validate_images(capture_dir, samples_rows)

    result = {
        "status": "pass",
        "stage": stage,
        "contract_path": str(contract_path.relative_to(REPO)),
        "contract_sha256": sha256(contract_path),
        "capture_manifest_sha256": sha256(manifest_path),
        "samples_sha256": sha256(capture_dir / "samples.csv"),
        "membership_sha256": membership_digest(samples_keys),
        "sample_count": len(samples_keys),
    }
    if stage == "perception":
        target_path = capture_dir / "perception_targets.csv"
        target_keys, target_rows = load_membership(target_path, filter_headings=False)
        exact_membership(target_keys, expected_keys, label="perception_targets.csv")
        sample_images = {
            canonical_key(row): str(row.get("image_path", "")) for row in samples_rows
        }
        target_images = {
            canonical_key(row): str(row.get("image_path", "")) for row in target_rows
        }
        if sample_images != target_images:
            different = sorted(
                key for key in sample_images if sample_images.get(key) != target_images.get(key)
            )
            raise ValueError(
                "perception_targets.csv image mapping differs from samples.csv; "
                f"first={different[0] if different else 'unknown'}"
            )
        validate_detector_manifest(capture_dir, contract)
        result["perception_targets_sha256"] = sha256(target_path)
        result["detector_manifest_sha256"] = sha256(capture_dir / "manifest.json")
    report = capture_dir / f"L2_contract_validation_{stage}.json"
    report.write_text(render(result), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--contract", type=Path, default=CONTRACT)
    check = sub.add_parser("check-inputs")
    check.add_argument("--contract", type=Path, default=CONTRACT)
    validate = sub.add_parser("validate-capture")
    validate.add_argument("--contract", type=Path, default=CONTRACT)
    validate.add_argument("--capture-dir", type=Path, required=True)
    validate.add_argument("--stage", choices=("samples", "perception"), required=True)
    args = parser.parse_args(argv)

    try:
        if args.command == "freeze":
            freeze_contract(args.contract, check=False)
        elif args.command == "check-inputs":
            contract = json.loads(args.contract.read_text(encoding="utf-8"))
            verify_inputs(contract)
            print(f"[L2 contract] PASS: all frozen input hashes match {args.contract}")
        else:
            result = validate_capture(args.contract, args.capture_dir, stage=args.stage)
            print(
                f"[L2 contract] PASS {args.stage}: exact {result['sample_count']} rows, "
                f"membership {result['membership_sha256'][:16]}..."
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[L2 contract] FAIL: {exc}")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
