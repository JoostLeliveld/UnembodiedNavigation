"""Can the availability dataset be built WITHOUT ground truth?

The bias correction needs truth by definition -- the offset is the reading minus the true
position.  The availability map does not: "did a usable sighting arrive?" is a question the
running robot can answer for itself.  That is what would let the availability dataset be
collected during ordinary operation, and grow, while the bias dataset stays small and
deliberate.

There is one catch.  The check that decides whether a sighting is usable compares the
detection against a box predicted from the robot's pose -- and during commissioning that
pose is the true one, while at runtime it is the robot's own estimate.  So the *label*
changes when truth is taken away.  This measures by how much: it re-runs the admission
check with the pose deliberately corrupted, and reports how many labels flip.

    python3 experiments/measurement_commissioning/availability_robustness.py
"""
from __future__ import annotations

import collections
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from admission import gate  # noqa: E402
from camera import camera_models  # noqa: E402
from observation import predicted_box  # noqa: E402
from capture import DATASET, load_capture  # noqa: E402

# position error in metres, heading error in degrees -- the pairs a real belief might carry
LEVELS = [(0.00, 0.0), (0.05, 1.0), (0.10, 2.0), (0.25, 5.0), (0.50, 10.0)]
SEED = 0


def main():
    cams = camera_models(DATASET)
    rows = load_capture()
    rng = np.random.default_rng(SEED)

    truth = {}
    trials = []
    for r, d in rows:
        if d["detected"] != "1":
            continue
        cam = cams[r["camera_id"]]
        x, y, yaw = float(r["robot_x"]), float(r["robot_y"]), float(r["robot_yaw"])
        det = (float(d["x0"]), float(d["y0"]), float(d["x1"]), float(d["y1"]))
        box = predicted_box(cam, x, y, yaw)
        if box is None:
            continue
        ok, _ = gate(box, det)
        truth[(r["camera_id"], r["image"])] = ok
        trials.append((r["camera_id"], r["image"], x, y, yaw, det))

    print(f"{len(trials)} detections; {sum(truth.values())} usable when the pose is exact\n")
    print("  the robot's own pose is wrong by      usable   labels that flip   availability")
    print("  position   heading                      rate    wrongly wrongly       measured")
    print("                                                     kept  dropped")
    out = []
    for sp, sh in LEVELS:
        kept = flip_keep = flip_drop = 0
        for cid, img, x, y, yaw, det in trials:
            if sp > 0:
                dx, dy = rng.normal(0, sp, 2)
                dth = math.radians(rng.normal(0, sh))
            else:
                dx = dy = dth = 0.0
            box = predicted_box(cams[cid], x + dx, y + dy, yaw + dth)
            ok = False if box is None else gate(box, det)[0]
            kept += ok
            t = truth[(cid, img)]
            if ok and not t:
                flip_keep += 1
            if t and not ok:
                flip_drop += 1
        n = len(trials)
        out.append({"position_sigma_m": sp, "heading_sigma_deg": sh,
                    "usable": kept, "usable_rate": kept / n,
                    "wrongly_kept": flip_keep, "wrongly_dropped": flip_drop,
                    "label_flip_rate": (flip_keep + flip_drop) / n})
        print(f"   {sp:.2f} m     {sh:4.1f} deg                     {kept/n:.3f}    "
              f"{flip_keep/n*100:5.1f}%   {flip_drop/n*100:5.1f}%          "
              f"{kept/ len(rows):.3f}")

    (REPO / "logs/studies/measurement_commissioning/availability_robustness.json").write_text(
        json.dumps({"levels": out, "detections": len(trials),
                    "usable_at_truth": sum(truth.values()),
                    "note": ("The admission check compares a detection against a box predicted "
                             "from the robot's pose.  During commissioning that pose is exact; "
                             "at runtime it is the robot's own estimate.  These rows are the "
                             "same detections re-judged with a deliberately wrong pose, so they "
                             "bound how far a truth-free availability dataset would drift from "
                             "the commissioned one.")}, indent=2))
    print(f"\nwrote logs/studies/measurement_commissioning/availability_robustness.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
