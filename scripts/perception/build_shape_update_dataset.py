#!/usr/bin/env python3
"""Build paired analytic/learned update rows from frozen detections and perturbed priors."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _camera(record: dict[str, str]):
    from unav_common.camera_model import ObliqueCameraModel

    pose = [float(value) for value in json.loads(record["camera_pose_xyz_rpy"])]
    x, y, z, _roll, pitch, yaw = pose
    forward = (math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), -math.sin(pitch))
    scale = -z / forward[2]
    return ObliqueCameraModel(
        cam_pos=pose[:3], look_at=(x + scale * forward[0], y + scale * forward[1], 0.0),
        img_width=int(record["image_width"]), img_height=int(record["image_height"]),
        fov_h_rad=float(record["fov_h_rad"]),
    )


def _noise(record: dict[str, str], replicate: int, sigma_xy_m: float) -> tuple[float, float]:
    if replicate == 0:
        return 0.0, 0.0
    key = f"shape-update-v1|{record['camera']}|{record['sample_index']}|{replicate}".encode()
    seed = int(hashlib.sha256(key).hexdigest()[:16], 16) % (2 ** 32)
    rng = np.random.default_rng(seed)
    noise = np.clip(rng.normal(0.0, sigma_xy_m, size=2), -3 * sigma_xy_m, 3 * sigma_xy_m)
    return float(noise[0]), float(noise[1])


def build(source: Path, output: Path, *, replicates: int, sigma_xy_m: float,
          min_confidence: float) -> dict:
    from reliability.silhouette_observation import (
        equivalent_position_measurement, observation_jacobian, plausibility_reasons,
        predicted_ground,
    )
    from unav_common.robot_hull import VISUAL_HULL, silhouette_box

    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"output exists: {output}")
    staged = output.with_name(output.name + ".incomplete")
    if staged.exists():
        raise FileExistsError(f"staging output exists: {staged}")
    staged.mkdir(parents=True)
    try:
        with (source / "records.csv").open(newline="", encoding="utf-8") as handle:
            base_rows = [row for row in csv.DictReader(handle) if row["detected"] == "1"]
        output_rows = []
        for base in base_rows:
            camera = _camera(base)
            gt_x, gt_y, yaw = float(base["robot_x"]), float(base["robot_y"]), float(base["robot_yaw"])
            det_box = tuple(float(base[key]) for key in ("box_x1", "box_y1", "box_x2", "box_y2"))
            raw = camera.pixel_to_world_at_z(float(base["box_bottom_u"]), float(base["box_bottom_v"]), 0.0)
            detection_only = camera.pixel_to_world_at_z(
                float(base["target_floor_u"]) + float(base["detector_error_u"]),
                float(base["target_floor_v"]) + float(base["detector_error_v"]), 0.0,
            )
            if raw is None or detection_only is None:
                continue
            for replicate in range(replicates):
                dx, dy = _noise(base, replicate, sigma_xy_m)
                prior_x, prior_y = gt_x + dx, gt_y + dy
                hull = silhouette_box(camera, prior_x, prior_y, yaw, VISUAL_HULL)
                h_ground = predicted_ground(camera, prior_x, prior_y, yaw)
                jacobian = observation_jacobian(camera, prior_x, prior_y, yaw)
                if hull is None or h_ground is None or jacobian is None:
                    continue
                analytic = equivalent_position_measurement(
                    raw, ((1.0, 0.0), (0.0, 1.0)), camera,
                    (prior_x, prior_y), yaw,
                )
                reasons = plausibility_reasons(
                    det_box, camera, prior_x, prior_y, yaw,
                    image_size=(int(base["image_width"]), int(base["image_height"])),
                )
                confidence_pass = float(base["box_confidence"]) >= min_confidence
                gate_pass = confidence_pass and not reasons and analytic is not None
                row: dict[str, object] = dict(base)
                row.update({
                    "replicate": replicate, "exact_prior": int(replicate == 0),
                    "prior_x": f"{prior_x:.10f}", "prior_y": f"{prior_y:.10f}",
                    "prior_yaw": f"{yaw:.10f}",
                    "target_dx": f"{gt_x - prior_x:.10f}", "target_dy": f"{gt_y - prior_y:.10f}",
                    "raw_ground_x": f"{raw[0]:.10f}", "raw_ground_y": f"{raw[1]:.10f}",
                    "detection_only_x": f"{detection_only[0]:.10f}",
                    "detection_only_y": f"{detection_only[1]:.10f}",
                    "hull_u0": f"{hull[0]:.10f}", "hull_v0": f"{hull[1]:.10f}",
                    "hull_u1": f"{hull[2]:.10f}", "hull_v1": f"{hull[3]:.10f}",
                    "h_ground_x": f"{h_ground[0]:.10f}", "h_ground_y": f"{h_ground[1]:.10f}",
                    "dhx_dx": f"{jacobian[0][0]:.10f}", "dhx_dy": f"{jacobian[0][1]:.10f}",
                    "dhy_dx": f"{jacobian[1][0]:.10f}", "dhy_dy": f"{jacobian[1][1]:.10f}",
                    "confidence_pass": int(confidence_pass), "gate_pass": int(gate_pass),
                    "gate_reasons": "|".join(reasons),
                    "analytic_x": "" if analytic is None else f"{analytic[0][0]:.10f}",
                    "analytic_y": "" if analytic is None else f"{analytic[0][1]:.10f}",
                })
                output_rows.append(row)
        fields = sorted({key for row in output_rows for key in row})
        records = staged / "records.csv"
        with records.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(output_rows)
        counts = Counter((row["residual_split"], int(row["exact_prior"]), int(row["gate_pass"]))
                         for row in output_rows)
        payload = {
            "status": "complete_provisional_shape_update_comparison_dataset",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source": str(source), "source_manifest_sha256": _sha256(source / "dataset_manifest.json"),
            "experimental_unit": "frozen detection with deterministic candidate-prior perturbation",
            "prior_design": {
                "replicates_per_detection": replicates, "replicate_0": "exact commanded GT prior",
                "other_replicates": f"deterministic clipped Gaussian xy, sigma={sigma_xy_m} m",
                "heading": "exact commanded heading for every arm",
            },
            "admission": {
                "detector_confidence_min": min_confidence,
                "shape_gate": "runtime reliability.silhouette_observation.plausibility_reasons",
            },
            "online_inputs": ["detector box/confidence", "fixed camera", "candidate prior xy/heading", "known robot hull"],
            "evaluation_only_inputs": ["commanded GT xy", "semantic mask for detection-only floor"],
            "counts": {f"{s}_{'exact' if e else 'perturbed'}_{'kept' if g else 'rejected'}": n
                       for (s, e, g), n in sorted(counts.items())},
            "records_sha256": _sha256(records),
        }
        manifest = staged / "dataset_manifest.json"
        manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        (staged / ".complete").write_text(json.dumps({"manifest_sha256": _sha256(manifest)}) + "\n")
        os.replace(staged, output)
        return payload
    except BaseException:
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--sigma-xy-m", type=float, default=0.10)
    parser.add_argument("--min-confidence", type=float, default=0.25)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.out, replicates=args.replicates,
                           sigma_xy_m=args.sigma_xy_m, min_confidence=args.min_confidence),
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
