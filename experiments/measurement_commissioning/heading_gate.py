"""The gate that decides whether ``camera_xy_only`` is safe.

**The architecture, and why this gate exists.**  The heading in ``h(x, y, theta)`` comes from
the robot's own estimator -- odometry-driven -- never from ground truth.  The camera update
is then allowed to move ``x, y`` and nothing else.  That is not a preference: two places in
the runtime raise a ``RuntimeError`` if ``heading_update_mode`` is anything but
``camera_xy_only`` (``unicycle_planner_node.py``, ``visibility_launch_common.py``).

That architecture has one failure mode.  The camera sees a single residual, which contains
position error and heading error together.  Forbidden to move the heading, the filter must
explain all of it by moving ``x, y`` -- so **a wrong heading is absorbed as a position
correction**, silently.  It cannot be caught by the admission check either, because the
bottom-centre barely moves in pixels when the heading changes.

**The bottom-centre is good for x,y and weak for theta**, and both halves of that follow from
the same small sensitivity.  Weak sensitivity means a few degrees of heading error changes
the prediction by less than the detector's own noise -- reassuring.  It also means the
bottom-centre carries almost no information with which to correct a bad heading -- so
recovering theta from it should not be claimed.

**What this script computes.**  Half the gate, offline, on the commissioning capture: for a
heading error of a given size, how big is the induced pixel error, and how much position
error does ``camera_xy_only`` absorb as a result?  It is a lookup table.

**What it cannot compute.**  The other half: how far the operational heading estimate
actually drifts.  That needs a driven trajectory, logging the estimator's own heading, with
ground truth recorded separately and used only afterwards to score it.  When that exists,
read the answer off the table below.

    python3 experiments/measurement_commissioning/heading_gate.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import capture  # noqa: E402
from camera import camera_models  # noqa: E402
from observation import h, jacobian  # noqa: E402

# how wrong the estimator's heading might plausibly be, in degrees
LEVELS = (1.0, 2.0, 3.0, 5.0, 10.0, 20.0)


def main():
    cal = json.loads((capture.OUT / "calibration.json").read_text())["calibration"]
    sigma_px = cal["sigma_px"]
    cams = camera_models(capture.DATASET)

    rows = []
    for r in csv.DictReader(open(capture.OUT / "sightings.csv")):
        x, y, yaw = float(r["x"]), float(r["y"]), float(r["yaw"])
        cam = cams[r["camera"]]
        base = h(cam, x, y, yaw)
        if base is None:
            continue
        J = jacobian(cam, x, y, yaw)
        if abs(np.linalg.det(J)) < 1e-9:
            continue
        rows.append((cam, x, y, yaw, np.array(base), np.linalg.inv(J)))
    print(f"{len(rows)} usable sightings; the detector's own noise is {sigma_px:.3f} px\n")

    print("  if the estimator's heading is wrong by...")
    print("      degrees   the prediction moves   ...as a fraction of      the position error")
    print("                    (pixels, median)     detector noise      camera_xy_only absorbs")
    table = []
    for deg in LEVELS:
        d = math.radians(deg)
        px, cm = [], []
        for cam, x, y, yaw, base, Jinv in rows:
            moved = h(cam, x, y, yaw + d)
            if moved is None:
                continue
            dz = np.array(moved) - base
            px.append(float(np.linalg.norm(dz)))
            # forbidden to move theta, the filter nulls the residual by moving x,y instead
            cm.append(float(np.linalg.norm(Jinv @ dz) * 100.0))
        px, cm = np.array(px), np.array(cm)
        table.append({"heading_error_deg": deg,
                      "pixel_shift_median_px": float(np.median(px)),
                      "pixel_shift_p90_px": float(np.percentile(px, 90)),
                      "fraction_of_detector_noise": float(np.median(px) / sigma_px),
                      "absorbed_position_error_median_cm": float(np.median(cm)),
                      "absorbed_position_error_p90_cm": float(np.percentile(cm, 90))})
        print(f"      {deg:5.1f}      {np.median(px):10.3f} px      {np.median(px)/sigma_px:12.2f}"
              f"      {np.median(cm):12.2f} cm  (p90 {np.percentile(cm, 90):.2f})")

    # the heading error at which the induced pixel shift reaches the detector's own noise
    fr = np.array([t["fraction_of_detector_noise"] for t in table])
    dg = np.array([t["heading_error_deg"] for t in table])
    breakeven = float(np.interp(1.0, fr, dg)) if fr[-1] >= 1.0 else float("inf")
    print(f"\n  break-even: the induced pixel shift equals the detector's own noise at about "
          f"{breakeven:.0f} degrees of heading error.")
    print("  Below that, heading error is quieter than the noise already present.")

    verdict = (
        "PASS the gate if the operational heading error stays well under the break-even: "
        "camera_xy_only is then defensible and the paper says the operational estimator "
        "supplied heading while the cameras corrected translational drift. FAIL it if "
        "heading regularly drifts near or past the break-even: the filter would then be "
        "turning orientation error into position correction, and either theta must enter "
        "the camera update or a heading-sensitive feature such as the box width must be "
        "added. Decide from the measured drift, not from a guess.")
    out = {"sigma_px": sigma_px, "sightings": len(rows),
           "break_even_heading_error_deg": breakeven,
           "table": table,
           "architecture": ("h(x, y, theta) takes theta from the robot's own estimator; the "
                            "camera update moves x, y only. Enforced by a RuntimeError in "
                            "unicycle_planner_node.py and visibility_launch_common.py."),
           "what_is_missing": ("the operational heading error itself, which needs a driven "
                               "trajectory logging the estimator's heading with ground truth "
                               "recorded separately and used only to score it afterwards"),
           "verdict_rule": verdict}
    (capture.OUT / "heading_gate.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {capture.OUT / 'heading_gate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
