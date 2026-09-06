#!/usr/bin/env python3
"""Audit the transport/provenance/determinism gate for a characterization capture."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "perception"))
from capture_yolo_dataset import _sha1_array  # noqa: E402

CAMERAS = {"camera_A", "camera_B", "camera_C", "camera_D", "camera_E"}
BOX_FIELDS = ("detected", "n_candidates", "confidence", "x0", "y0", "x1", "y1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--max-batch-span-s", type=float, default=0.05)
    args = parser.parse_args()

    capture = args.capture.expanduser().resolve()
    paths = {
        name: capture / name
        for name in (
            "capture_manifest.json", "capture_index.csv", "bbox_detector_manifest.json",
            "bbox_observations.csv", "observation_interpretations.csv",
        )
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing P0 artifacts: {missing}")

    capture_manifest = json.loads(paths["capture_manifest.json"].read_text(encoding="utf-8"))
    detector_manifest = json.loads(
        paths["bbox_detector_manifest.json"].read_text(encoding="utf-8")
    )
    capture_rows = list(csv.DictReader(paths["capture_index.csv"].open(encoding="utf-8")))
    detector_rows = list(csv.DictReader(paths["bbox_observations.csv"].open(encoding="utf-8")))
    interpretation_rows = list(
        csv.DictReader(paths["observation_interpretations.csv"].open(encoding="utf-8"))
    )

    checks: dict[str, bool] = {}
    checks["capture_complete"] = capture_manifest.get("status") == "complete"
    checks["capture_schema_v2"] = capture_manifest.get("schema") == "bbox_characterization_capture.v2"
    checks["detector_complete"] = detector_manifest.get("status") == "complete"
    checks["detector_schema_v2"] = detector_manifest.get("schema") == "bbox_characterization_detector.v2"
    checks["transport_isolated"] = bool(capture_manifest.get("transport", {}).get("isolation_verified"))
    checks["no_failed_capture_rows"] = all(row["capture_status"] == "ok" for row in capture_rows)
    checks["planned_row_count"] = (
        len(capture_rows) == int(capture_manifest["plan"]["planned_rows"])
    )
    checks["detector_attempt_row_count"] = len(detector_rows) == len(capture_rows)
    checks["interpretation_attempt_row_count"] = len(interpretation_rows) == len(capture_rows)
    checks["detector_has_explicit_clipping_flag"] = bool(
        detector_rows and "detector_clipped" in detector_rows[0]
    )

    batches: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for row in capture_rows:
        batches[row.get("source_batch_id", "")].append(row)
    checks["source_batch_id_present"] = "" not in batches
    checks["exactly_five_cameras_per_batch"] = all(
        len(rows) == 5 and {row["camera_id"] for row in rows} == CAMERAS
        for rows in batches.values()
    )
    spans = [float(row["batch_image_span_s"]) for row in capture_rows]
    max_span = max(spans) if spans else math.inf
    checks["batch_span_within_limit"] = max_span <= float(args.max_batch_span_s)

    detector_by_key = {
        (row["source_batch_id"], row["camera_id"]): row for row in detector_rows
    }
    interpretation_by_key = {
        (row["source_batch_id"], row["camera_id"]): row for row in interpretation_rows
    }
    capture_keys = {(row["source_batch_id"], row["camera_id"]) for row in capture_rows}
    checks["source_batch_ids_preserved"] = (
        set(detector_by_key) == capture_keys and set(interpretation_by_key) == capture_keys
    )

    decoded_hash_mismatches = []
    for row in capture_rows:
        image_path = capture / row["image"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or _sha1_array(image) != row["image_sha1"]:
            decoded_hash_mismatches.append(str(image_path))
    checks["decoded_image_hashes_match"] = not decoded_hash_mismatches
    checks["capture_index_hash_matches_manifest"] = (
        sha256(paths["capture_index.csv"]) == capture_manifest.get("capture_index_sha256")
    )
    checks["detector_inputs_match_capture"] = (
        sha256(paths["capture_index.csv"]) == detector_manifest.get("capture_index_sha256")
        and sha256(paths["capture_manifest.json"])
        == detector_manifest.get("capture_manifest_sha256")
    )
    checks["detector_output_hash_matches_manifest"] = (
        sha256(paths["bbox_observations.csv"])
        == detector_manifest.get("bbox_observations_sha256")
    )

    capture_cells: dict[tuple[str, str, str], list[dict[str, str]]] = collections.defaultdict(list)
    detector_cells: dict[tuple[str, str, str], list[dict[str, str]]] = collections.defaultdict(list)
    for row in capture_rows:
        capture_cells[(row["position_id"], row["heading_id"], row["camera_id"])].append(row)
    for row in detector_rows:
        detector_cells[(row["position_id"], row["heading_id"], row["camera_id"])].append(row)
    image_repeat_identical = all(
        len({row["image_sha1"] for row in rows}) == 1 for rows in capture_cells.values()
    )
    detector_repeat_identical = all(
        len({tuple(row[field] for field in BOX_FIELDS) for row in rows}) == 1
        for rows in detector_cells.values()
    )
    checks["static_image_repeats_identical"] = image_repeat_identical
    checks["static_detector_repeats_identical"] = detector_repeat_identical

    report = {
        "experiment": "P0_capture_transport_and_determinism_gate",
        "capture": str(capture),
        "passed": bool(all(checks.values())),
        "checks": checks,
        "counts": {
            "source_batches": len(batches),
            "camera_opportunities": len(capture_rows),
            "same_state_camera_cells": len(capture_cells),
            "unique_decoded_images": len({row["image_sha1"] for row in capture_rows}),
            "detector_hits": sum(int(row["detected"]) for row in detector_rows),
            "detector_clipped_hits": sum(
                int(row["detector_clipped"]) for row in detector_rows
            ),
        },
        "timing": {
            "maximum_batch_span_s": max_span,
            "required_maximum_s": float(args.max_batch_span_s),
        },
        "determinism": {
            "interpretation": (
                "Exact same-state renders and detector boxes repeat identically. Static "
                "repetition therefore measures a zero repeat component in this simulator "
                "condition and must not be used to set deployed R."
            ),
            "R_repeat": 0.0 if image_repeat_identical and detector_repeat_identical else None,
        },
        "decoded_hash_mismatches": decoded_hash_mismatches,
        "provenance": {name: sha256(path) for name, path in paths.items()},
    }
    output = capture / "p0_audit.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
