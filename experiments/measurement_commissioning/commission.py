"""Commission the camera measurement.  The driver: it runs the four parts and freezes the result.

    python3 experiments/measurement_commissioning/commission.py

Offline, a few minutes, reproducible from files already on disk.  The four parts live in
separate modules because they answer separate questions and must not leak into each other:

| module          | the one question it answers                                  | fitted    |
|-----------------|--------------------------------------------------------------|-----------|
| ``camera``      | where are the cameras and how do they project?               | nothing   |
| ``observation`` | the box is not the robot -- where should the box be?         | nothing   |
| ``admission``   | is this sighting usable?                                     | nothing   |
| ``detector``    | what does the frozen detector actually do?                   | nothing   |
| ``capture``     | what was collected, and which job may each part serve?       | nothing   |
| ``offset``      | what half-centimetre lean is left, and how is it removed?    | 6 numbers |
| ``uncertainty`` | how much should a sighting be trusted?                       | 1 number  |

Information flows one way.  The first five are inputs to ``offset`` and ``uncertainty``;
nothing measured in those two may ever change the others.  That is not pedantry -- the
detector was once retrained to make a residual smaller, and a label convention change then
looked like a bias result.

**Two things are called bias and only one is.**  The box-versus-centre problem is worth 30 cm
and lives in ``observation``, where nothing is fitted.  The leftover lean is worth half a
centimetre and lives in ``offset``.

Ground truth forms the residual and scores the outcome.  It never becomes something the
running robot consults.
"""
from __future__ import annotations

import collections
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import admission  # noqa: E402
import camera  # noqa: E402
import capture  # noqa: E402
import detector  # noqa: E402
import observation  # noqa: E402
import offset  # noqa: E402
import uncertainty  # noqa: E402


def build_sightings(cams, rows, edge, pos_id):
    """Every detection that survives the admission check, with its residual in pixels."""
    eo = (edge["edge_offset_u_px"], edge["edge_offset_v_px"])
    sightings, failed = [], collections.Counter()
    for r, d in rows:
        if d["detected"] != "1":
            continue
        cam = cams[r["camera_id"]]
        x, y, yaw = float(r["robot_x"]), float(r["robot_y"]), float(r["robot_yaw"])
        box = observation.predicted_box(cam, x, y, yaw)
        if box is None:
            continue
        det = (float(d["x0"]), float(d["y0"]), float(d["x1"]), float(d["y1"]))
        passed, reasons = admission.gate(box, det)
        if not passed:
            for reason in reasons:
                failed[reason] += 1
            continue
        predicted = observation.h(cam, x, y, yaw, eo)
        if predicted is None:
            continue
        hu, hv = predicted
        J = observation.jacobian(cam, x, y, yaw)
        if abs(np.linalg.det(J)) < 1e-9:
            continue
        cx, cy = cam.cam_pos[0], cam.cam_pos[1]
        cell = (round(x, 4), round(y, 4))
        sightings.append({
            "camera": r["camera_id"], "image": r["image"], "x": x, "y": y, "yaw": yaw,
            "cell": cell, "position_id": pos_id.get(cell, -1),
            "range_m": float(r["camera_range_m"]),
            "rel_heading_rad": (yaw - math.atan2(y - cy, x - cx) + math.pi) % (2 * math.pi) - math.pi,
            "du_px": 0.5 * (det[0] + det[2]) - hu, "dv_px": det[3] - hv,
            "Jinv": np.linalg.inv(J), "confidence": float(d["confidence"]),
            "visible_height": float(r.get("height_fraction") or "nan"),
            "mask_bottom_v": float(r.get("mask_bottom_v") or "nan"),
            "pred_hull_bottom_v": float(r.get("pred_hull_bottom_v") or "nan"),
            "det_bottom_v": det[3],
        })
    return sightings, dict(failed)


def main():
    capture.OUT.mkdir(parents=True, exist_ok=True)
    cams = camera.camera_models(capture.DATASET)
    rows = capture.load_capture()
    print(f"capture trials joined: {len(rows)}")

    verify = capture.verify_reconstruction(cams, rows)
    print(f"[observ.] camera reconstruction: median {verify['median_px']:.4f} px "
          f"-> {'PASS' if verify['passed'] else 'FAIL'}")
    if not verify["passed"]:
        print("    refusing to continue: rebuilt cameras disagree with the sealed predictions")
        return 1

    edge = capture.measure_edge_offset(cams, rows)
    print(f"[observ.] edge offset, detector not used: u {edge['edge_offset_u_px']:+.3f} "
          f"v {edge['edge_offset_v_px']:+.3f} px  from {edge['n']} masks")

    char = detector.characterize(rows)
    print(f"[capture]     detection rate {char['detection_rate']:.4f} over {char['trials']} "
          f"trials; multi-box {char['multi_box_fraction']*100:.2f}%")

    pos_id = capture.position_ids(rows)
    offset_positions, avail_positions = capture.split_positions(rows)
    print(f"[capture]     {len(offset_positions)} positions reserved for the correction, "
          f"{len(avail_positions)} for the availability map, none shared")

    sightings, failed = build_sightings(cams, rows, edge, pos_id)
    attempted = len(sightings) + sum(failed.values())
    print(f"[capture]     {len(sightings)} of {attempted} detections pass the admission check")

    # --- uncertainty: one number, measured in pixels ----------------------------------
    residuals = np.array([[s["du_px"], s["dv_px"]] for s in sightings])
    sigma_px = uncertainty.fit_sigma_px(residuals)
    per_cam = collections.defaultdict(list)
    for s in sightings:
        per_cam[s["camera"]].append([s["du_px"], s["dv_px"]])
    sigma_by_camera = uncertainty.sigma_px_by_camera(per_cam)
    print(f"[uncert.]  sigma {sigma_px:.3f} px  "
          f"(per camera {min(sigma_by_camera.values()):.2f}-{max(sigma_by_camera.values()):.2f})")

    # --- bias: six numbers, fitted on its own positions only --------------------------
    fit_set = [s for s in sightings if s["cell"] in offset_positions]
    held_out = [s for s in sightings if s["cell"] not in offset_positions]
    coeffs = offset.fit(fit_set)
    spread = float(np.mean([uncertainty.stated_spread_cm(s["Jinv"], sigma_px) for s in held_out]))
    before = offset.score(held_out, None, spread)
    after = offset.score(held_out, coeffs, spread)
    null = offset.worst_group_null(held_out, coeffs, after["worst_conditional_cm"])
    print(f"[offset]     fitted on {len({s['cell'] for s in fit_set})} positions "
          f"({len(fit_set)} sightings), checked on {len({s['cell'] for s in held_out})} "
          f"({len(held_out)})")
    print(f"[offset]     {before['pooled_mean_error_cm']:.3f} -> "
          f"{after['pooled_mean_error_cm']:.3f} cm against {spread:.2f} cm of random "
          f"scatter   ({before['r_b']:.2f} -> {after['r_b']:.2f})")

    capture.write_offset_positions(offset_positions, pos_id)
    capture.write_sightings(sightings)
    avail = capture.write_availability(rows, sightings, offset_positions, pos_id)
    print(f"[capture]     availability table: {avail['trials']} trials, "
          f"{avail['line_of_sight']} with a line of sight, {avail['usable']} usable")
    if avail["usable"] != len(sightings):
        print(f"    MISMATCH: {avail['usable']} usable in the table vs {len(sightings)} sightings")
        return 1

    frozen = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "detector": detector.DETECTOR,
        "inputs": {
            "capture": str(capture.DATASET.relative_to(capture.REPO)),
            "calibration_index_sha256": capture.sha256(
                capture.DATASET / "localization_calibration_index_hull.csv"),
            "observation_sha256": capture.sha256(HERE / "observation.py"),
            "capture_sha256": capture.sha256(HERE / "capture.py"),
            "offset_sha256": capture.sha256(HERE / "offset.py"),
            "uncertainty_sha256": capture.sha256(HERE / "uncertainty.py"),
            "commission_sha256": capture.sha256(Path(__file__)),
        },
        "camera_reconstruction_check": verify,
        "edge_offset": edge,
        "detector_characterization": char,
        "availability_table": avail,
        "calibration": {
            "mechanism": offset.mechanism(sightings),
            "gate": dict(admission.GATE),
            "gate_failures": failed,
            "gate_pass": len(sightings), "gate_attempted": attempted,
            "offset_positions": len(offset_positions), "bias_sightings": len(fit_set),
            "fit_cells": len({s["cell"] for s in fit_set}),
            "held_out_cells": len({s["cell"] for s in held_out}),
            "fit_sightings": len(fit_set), "held_out_sightings": len(held_out),
            "sigma_px": sigma_px, "sigma_px_by_camera": sigma_by_camera,
            "coefficients_du": coeffs[:, 0].tolist(),
            "coefficients_dv": coeffs[:, 1].tolist(),
            "design": offset.DESIGN,
            "before": before, "after": after,
            "worst_conditional_null": null,
        },
    }
    (capture.OUT / "calibration.json").write_text(json.dumps(frozen, indent=2, sort_keys=True))
    print(f"\nwrote calibration.json, sightings.csv, availability.csv, offset_positions.csv "
          f"to {capture.OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
