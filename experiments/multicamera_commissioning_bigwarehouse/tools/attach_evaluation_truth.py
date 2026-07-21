#!/usr/bin/env python3
"""Attach recorded simulation truth to per-camera CSVs, evaluation-only.

Reads the operational per-camera perception CSVs written by
``record_operational_logs.py`` plus the ``ground_truth.csv`` written by
``record_evaluation_truth.py`` and emits **copies** with ``true_x``, ``true_y``
and ``true_yaw`` columns (nearest-stamp join within a tolerance).  The
operational inputs are never modified; the output directory is an
evaluation-only artifact that ``reliability_tools export-multicamera`` turns
into populated ``evaluation_only/`` samples.

The tool also writes ``truth_alignment_summary.json`` with a per-camera
projection audit on detected rows (bias per axis, mean/median/p90 absolute
error).  This is the measurement that attributes a cross-camera disagreement
(e.g. the pilot's C-D +y offset) to a specific camera's calibration.

Example:

    python3 experiments/multicamera_commissioning_bigwarehouse/tools/attach_evaluation_truth.py \
      --raw-dir logs/multicamera_commissioning_bigwarehouse/run_001/raw \
      --truth-csv logs/multicamera_commissioning_bigwarehouse/run_001/evaluation_only/ground_truth.csv \
      --out-dir logs/multicamera_commissioning_bigwarehouse/run_001/evaluation_inputs

"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

REPO = Path(__file__).resolve().parents[3]
for relative in ("src/reliability", "src/unav_common"):
    location = str(REPO / relative)
    if location not in sys.path:
        sys.path.insert(0, location)

from reliability.projection import camera_model_from_world  # noqa: E402

TRUTH_COLUMNS = ("true_x", "true_y", "true_yaw")
DEFAULT_WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_full_4cam.world.sdf"
DEFAULT_MODEL_INCLUDES = {
    "camera_A": "external_camera",
    "camera_B": "external_camera_b",
    "camera_C": "external_camera_c",
    "camera_D": "external_camera_d",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{label} is missing or invalid: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must be a JSON object: {path}")
    return payload


def _source_contract(
    *,
    raw_dir: Path,
    truth_csv: Path,
    evidence_role: str,
) -> dict[str, object]:
    """Verify that finalized operational and truth streams belong to one run."""

    run_dir = raw_dir.parent
    evaluation_dir = truth_csv.parent
    if raw_dir.name != "raw" or evaluation_dir != run_dir / "evaluation_only":
        raise SystemExit(
            "--raw-dir and --truth-csv must be the raw/ and "
            "evaluation_only/ground_truth.csv artifacts of the same run"
        )
    if truth_csv.name != "ground_truth.csv":
        raise SystemExit("--truth-csv must name the finalized ground_truth.csv artifact")
    unfinished = sorted(
        str(path.relative_to(run_dir))
        for directory in (raw_dir, evaluation_dir)
        for pattern in ("*.part", "*.in_progress.json", "*.failed.json")
        for path in directory.glob(pattern)
        if path.is_file()
    )
    if unfinished:
        raise SystemExit("run contains partial or failed artifacts: " + ", ".join(unfinished))

    operational_path = raw_dir / "operational_recording_manifest.json"
    truth_manifest_path = evaluation_dir / "evaluation_truth_manifest.json"
    operational = _load_json_object(operational_path, label="operational manifest")
    truth = _load_json_object(truth_manifest_path, label="evaluation truth manifest")
    if operational.get("status") != "completed" or operational.get("contains_ground_truth") is not False:
        raise SystemExit("operational manifest is not a completed ground-truth-free recording")
    if (
        truth.get("status") != "completed"
        or truth.get("evaluation_only") is not True
        or truth.get("contains_ground_truth") is not True
    ):
        raise SystemExit("truth manifest is not a completed evaluation-only recording")
    identity_fields = ("run_id", "plan_row_id", "seed")
    mismatched = [
        field for field in identity_fields if operational.get(field) != truth.get(field)
    ]
    if mismatched:
        raise SystemExit("operational/truth identity mismatch: " + ", ".join(mismatched))
    operational_role = str(operational.get("evidence_role", ""))
    truth_role = str(truth.get("evidence_role", ""))
    if operational_role != evidence_role or truth_role != evidence_role:
        raise SystemExit(
            "--projection-role must match the role pre-declared by both recorders; "
            f"requested={evidence_role!r}, operational={operational_role!r}, truth={truth_role!r}"
        )
    operational_completion = operational.get("route_completion_manifest")
    truth_completion = truth.get("route_completion_manifest")
    if (
        not isinstance(operational_completion, dict)
        or not isinstance(truth_completion, dict)
        or operational_completion.get("sha256") != truth_completion.get("sha256")
    ):
        raise SystemExit("operational and truth recorders do not reference the same route completion")

    completion_path = run_dir / "completion_manifest.json"
    completion = _load_json_object(completion_path, label="campaign completion manifest")
    declared = completion.get("artifacts_sha256")
    if completion.get("schema_version") != 1 or not isinstance(declared, dict):
        raise SystemExit("campaign completion manifest lacks the immutable artifact contract")
    artifact_paths = {
        "raw/operational_recording_manifest.json": operational_path,
        "evaluation_only/evaluation_truth_manifest.json": truth_manifest_path,
        "evaluation_only/ground_truth.csv": truth_csv,
    }
    for source in sorted(raw_dir.glob("camera_*_perception.csv")):
        artifact_paths[f"raw/{source.name}"] = source
    for relative, path in artifact_paths.items():
        if not path.is_file() or declared.get(relative) != _sha256(path):
            raise SystemExit(f"campaign completion hash mismatch for {relative}")

    return {
        "run_id": operational.get("run_id"),
        "plan_row_id": operational.get("plan_row_id"),
        "seed": operational.get("seed"),
        "evidence_role": evidence_role,
        "row_tuple": completion.get("row_tuple"),
        "operational_manifest": {
            "path": str(operational_path),
            "sha256": _sha256(operational_path),
        },
        "truth_manifest": {
            "path": str(truth_manifest_path),
            "sha256": _sha256(truth_manifest_path),
        },
        "campaign_completion": {
            "path": str(completion_path),
            "sha256": _sha256(completion_path),
        },
        "detector_model_sha256": (operational.get("detector_runtime") or {}).get(
            "model_sha256"
        ) if isinstance(operational.get("detector_runtime"), dict) else None,
        "projection_calibration_sha256": (
            operational.get("projection_calibration") or {}
        ).get("sha256") if isinstance(operational.get("projection_calibration"), dict) else None,
    }


def _write_json_new(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_truth(path: Path) -> tuple[list[float], list[tuple[float, float, float]]]:
    stamps: list[float] = []
    poses: list[tuple[float, float, float]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                stamp = float(row["stamp"])
                pose = (float(row["gt_x"]), float(row["gt_y"]), float(row["gt_yaw"]))
            except (KeyError, TypeError, ValueError):
                continue
            stamps.append(stamp)
            poses.append(pose)
    paired = sorted(zip(stamps, poses))
    return [item[0] for item in paired], [item[1] for item in paired]


def _nearest(
    stamps: list[float],
    poses: list[tuple[float, float, float]],
    stamp: float,
    tolerance_s: float,
) -> tuple[float, float, float] | None:
    if not stamps:
        return None
    index = bisect.bisect_left(stamps, stamp)
    best: tuple[float, int] | None = None
    for candidate in (index - 1, index):
        if 0 <= candidate < len(stamps):
            delta = abs(stamps[candidate] - stamp)
            if best is None or delta < best[0]:
                best = (delta, candidate)
    if best is None or best[0] > tolerance_s:
        return None
    return poses[best[1]]


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[position]


def attach_camera_csv(
    source: Path,
    destination: Path,
    stamps: list[float],
    poses: list[tuple[float, float, float]],
    tolerance_s: float,
    camera_ground_xy: tuple[float, float] | None = None,
) -> dict[str, object]:
    matched = 0
    total = 0
    error_x: list[float] = []
    error_y: list[float] = []
    error_norm: list[float] = []
    error_along_bearing: list[float] = []
    error_cross_bearing: list[float] = []
    with source.open("r", newline="", encoding="utf-8") as in_handle:
        reader = csv.DictReader(in_handle)
        fieldnames = list(reader.fieldnames or [])
        for column in TRUTH_COLUMNS:
            if column not in fieldnames:
                fieldnames.append(column)
        with destination.open("w", newline="", encoding="utf-8") as out_handle:
            writer = csv.DictWriter(out_handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                total += 1
                truth = None
                try:
                    stamp = float(row["diag_stamp"])
                except (KeyError, TypeError, ValueError):
                    stamp = math.nan
                if math.isfinite(stamp):
                    truth = _nearest(stamps, poses, stamp, tolerance_s)
                if truth is None:
                    row.update({column: "" for column in TRUTH_COLUMNS})
                else:
                    matched += 1
                    row["true_x"] = f"{truth[0]:.9f}"
                    row["true_y"] = f"{truth[1]:.9f}"
                    row["true_yaw"] = f"{truth[2]:.9f}"
                    if row.get("detected") == "1" and row.get("pred_world_x") and row.get("pred_world_y"):
                        px = float(row["pred_world_x"])
                        py = float(row["pred_world_y"])
                        dx = px - truth[0]
                        dy = py - truth[1]
                        error_x.append(dx)
                        error_y.append(dy)
                        error_norm.append(math.hypot(dx, dy))
                        if camera_ground_xy is not None:
                            # Resolve residuals in the truth-referenced radial
                            # basis.  Using the prediction itself would rotate
                            # the basis by the very projection error being
                            # audited and can hide a systematic cross-bearing
                            # component.
                            bearing_x = truth[0] - camera_ground_xy[0]
                            bearing_y = truth[1] - camera_ground_xy[1]
                            norm = math.hypot(bearing_x, bearing_y)
                            if norm > 1.0e-9:
                                error_along_bearing.append(
                                    (dx * bearing_x + dy * bearing_y) / norm
                                )
                                error_cross_bearing.append(
                                    (dx * -bearing_y + dy * bearing_x) / norm
                                )
                writer.writerow(row)
    audit: dict[str, object] = {
        "rows": total,
        "rows_with_truth": matched,
        "detected_rows_audited": len(error_norm),
    }
    if error_norm:
        audit.update(
            {
                "bias_x_m": statistics.fmean(error_x),
                "bias_y_m": statistics.fmean(error_y),
                "mean_abs_error_m": statistics.fmean(error_norm),
                "median_abs_error_m": statistics.median(error_norm),
                "p90_abs_error_m": _percentile(error_norm, 0.90),
                "max_abs_error_m": max(error_norm),
            }
        )
    if error_along_bearing:
        audit["along_bearing_bias_m"] = statistics.fmean(error_along_bearing)
        audit["along_bearing_std_m"] = (
            statistics.stdev(error_along_bearing) if len(error_along_bearing) > 1 else 0.0
        )
    if error_cross_bearing:
        audit["cross_bearing_bias_m"] = statistics.fmean(error_cross_bearing)
        audit["cross_bearing_std_m"] = (
            statistics.stdev(error_cross_bearing) if len(error_cross_bearing) > 1 else 0.0
        )
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-time-delta-s", type=float, default=0.05)
    parser.add_argument(
        "--projection-role",
        choices=("fit", "qualification", "diagnostic"),
        default="diagnostic",
        help="Pre-declared role of this run; readiness uses qualification runs only.",
    )
    parser.add_argument(
        "--camera-glob",
        default="camera_*_perception.csv",
        help="Per-camera CSV pattern inside --raw-dir",
    )
    parser.add_argument("--world-sdf", type=Path, default=DEFAULT_WORLD)
    parser.add_argument(
        "--emit-projection-calibration",
        type=Path,
        default=None,
        help=(
            "Write per-camera along-bearing projection offsets (JSON) estimated "
            "from this run's truth-referenced bias. Commissioning-time "
            "calibration output; consumed via --projection-calibration on the "
            "operational recorder and the camera manager node."
        ),
    )
    args = parser.parse_args()
    raw_dir = args.raw_dir.expanduser().resolve()
    truth_csv = args.truth_csv.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    world_sdf = args.world_sdf.expanduser().resolve()
    if out_dir == raw_dir:
        raise SystemExit("--out-dir must differ from --raw-dir: operational inputs are never modified")
    if out_dir.exists():
        raise SystemExit(f"refusing to overwrite evaluation attachment output: {out_dir}")
    if not world_sdf.is_file():
        raise SystemExit(f"world SDF is missing: {world_sdf}")
    if args.emit_projection_calibration is not None and args.projection_role != "fit":
        raise SystemExit("projection calibration may be emitted from pre-declared fit runs only")
    calibration_output = (
        args.emit_projection_calibration.expanduser().resolve()
        if args.emit_projection_calibration is not None
        else None
    )
    if calibration_output is not None and calibration_output.exists():
        raise SystemExit(f"refusing to overwrite projection calibration: {calibration_output}")

    source_contract = _source_contract(
        raw_dir=raw_dir,
        truth_csv=truth_csv,
        evidence_role=str(args.projection_role),
    )

    stamps, poses = _load_truth(truth_csv)
    if not stamps:
        raise SystemExit(f"no usable truth rows in {truth_csv}")

    sources = sorted(raw_dir.glob(args.camera_glob))
    if not sources:
        raise SystemExit(f"no camera CSVs matching {args.camera_glob!r} in {raw_dir}")

    staging = out_dir.with_name(f".{out_dir.name}.{os.getpid()}.part")
    if staging.exists():
        raise SystemExit(f"stale attachment staging directory exists: {staging}")
    staging.mkdir(parents=True, exist_ok=False)
    audits: dict[str, dict[str, object]] = {}
    for source in sources:
        destination = staging / source.name
        camera_id = source.name.replace("_perception.csv", "")
        camera_ground_xy = None
        include = DEFAULT_MODEL_INCLUDES.get(camera_id)
        if include is not None and world_sdf.exists():
            model = camera_model_from_world(world_sdf, include_name=include)
            camera_ground_xy = (float(model.cam_pos[0]), float(model.cam_pos[1]))
        audits[camera_id] = attach_camera_csv(
            source,
            destination,
            stamps,
            poses,
            args.max_time_delta_s,
            camera_ground_xy=camera_ground_xy,
        )

    summary = {
        "evaluation_only": True,
        "contains_ground_truth": True,
        "truth_csv": str(truth_csv),
        "truth_csv_sha256": _sha256(truth_csv),
        "raw_dir": str(raw_dir),
        "max_time_delta_s": float(args.max_time_delta_s),
        "projection_role": str(args.projection_role),
        "role_predeclared_by_recorders": True,
        "source_contract": source_contract,
        "source_camera_csv_sha256": {
            source.name: _sha256(source) for source in sources
        },
        "world_sdf": {
            "path": str(world_sdf),
            "sha256": _sha256(world_sdf) if world_sdf.is_file() else None,
        },
        "truth_samples": len(stamps),
        "cameras": audits,
        "note": (
            "Copies of operational CSVs with true_x/true_y/true_yaw attached for "
            "evaluation_only exports and per-camera calibration audits. The "
            "per-camera bias_x/bias_y attribute cross-camera disagreement to a "
            "specific calibration. Never feed these columns to a model or manager."
        ),
    }
    summary_path = staging / "truth_alignment_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    attached_artifacts = {
        path.name: _sha256(path)
        for path in sorted(staging.glob("*.csv"))
    }
    attached_artifacts[summary_path.name] = _sha256(summary_path)
    attachment_manifest = {
        "schema_version": 1,
        "status": "completed",
        "evaluation_only": True,
        "contains_ground_truth": True,
        "projection_role": str(args.projection_role),
        "source_contract": source_contract,
        "artifacts_sha256": attached_artifacts,
    }
    (staging / "truth_attachment_manifest.json").write_text(
        json.dumps(
            attachment_manifest, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    os.rename(staging, out_dir)
    summary_path = out_dir / "truth_alignment_summary.json"

    if args.emit_projection_calibration is not None:
        cameras = {}
        for camera_id, audit in sorted(audits.items()):
            if "along_bearing_bias_m" not in audit:
                continue
            cameras[camera_id] = {
                # The measured bias is toward the camera (negative along the
                # bearing); the corrective offset pushes away from the camera.
                "along_bearing_offset_m": -float(audit["along_bearing_bias_m"]),
                "along_bearing_std_m": float(audit["along_bearing_std_m"]),
                "samples": int(audit["detected_rows_audited"]),
            }
        calibration = {
            "kind": "projection_along_bearing_offsets",
            "method": (
                "per-camera mean along-bearing projection error vs simulation "
                "truth (near-edge box-bottom pull); commissioning-time constant, "
                "never refit during deployment"
            ),
            "source_run": str(raw_dir),
            "source_contract": source_contract,
            "world_sdf": str(world_sdf),
            "cameras": cameras,
        }
        calibration_path = calibration_output
        assert calibration_path is not None
        _write_json_new(calibration_path, calibration)
        print(f"wrote projection calibration for {len(cameras)} cameras -> {calibration_path}")
    for camera_id, audit in sorted(audits.items()):
        bias = (
            f" bias=({audit.get('bias_x_m', math.nan):+.3f}, {audit.get('bias_y_m', math.nan):+.3f}) m"
            if "bias_x_m" in audit
            else ""
        )
        print(f"{camera_id}: {audit['rows_with_truth']}/{audit['rows']} rows matched{bias}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
