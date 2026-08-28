"""If the reading has to be unbiased, what does the camera network look like?

Nothing here is synthetic and nothing is simulated. "An unbiased detector" is obtained by
*refusing* the sightings that carry the lean — the ones whose box is cut off by an
obstruction — on the real commissioning capture. Every rule on the ladder is computable
online from a detected box and a predicted one, so every row is a rule the runtime could
actually run.

The point is not that a stricter rule is more accurate. It is that accuracy and
availability trade against each other, and the trade is what the paper is about:

    a refused sighting is an ABSENT measurement, not a wrong one,
    so tightening the gate moves error out of R and into p_c(s).

    python3 experiments/unbiased_observation/gate_ladder.py

Writes ``logs/studies/unbiased_observation/gate_ladder.json``.
"""
from __future__ import annotations

import collections
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/measurement_commissioning"))
sys.path.insert(0, str(REPO / "src/unav_common"))
import capture                                        # noqa: E402
from camera import camera_models, IMG_W, IMG_H        # noqa: E402
from observation import h, jacobian, predicted_box    # noqa: E402

OUT = REPO / "logs/studies/unbiased_observation"
CAMS = camera_models(capture.DATASET)

#: the ladder. Each rule is (label, predicate on a sighting record). Every field the
#: predicate touches is available at runtime -- none of them is ground truth.
LADDER = (
    ("as deployed — box at least 85% of predicted height",
     lambda s: s["passes_deployed"]),
    ("+ at least 92%", lambda s: s["passes_deployed"] and s["height_ratio"] >= 0.92),
    ("+ at least 96%", lambda s: s["passes_deployed"] and s["height_ratio"] >= 0.96),
    ("+ at least 99%", lambda s: s["passes_deployed"] and s["height_ratio"] >= 0.99),
    ("as deployed, but nothing beyond 16 m",
     lambda s: s["passes_deployed"] and s["range_m"] < 16.0),
    ("at least 96% AND nothing beyond 16 m",
     lambda s: s["passes_deployed"] and s["height_ratio"] >= 0.96 and s["range_m"] < 16.0),
)


def sightings():
    """Every attempted sighting in the capture, with everything a gate could look at."""

    out = []
    for index_row, detection in capture.load_capture():
        if detection["detected"] != "1":
            continue
        cam_id = index_row["camera_id"]
        cam = CAMS[cam_id]
        x, y = float(index_row["robot_x"]), float(index_row["robot_y"])
        yaw = float(index_row["robot_yaw"])
        pred = predicted_box(cam, x, y, yaw)
        if pred is None:
            continue
        u0, v0, u1, v1 = pred
        pw, ph = u1 - u0, v1 - v0
        if pw <= 0.0 or ph <= 0.0:
            continue
        x0, y0, x1, y1 = (float(detection[k]) for k in ("x0", "y0", "x1", "y1"))
        dw, dh = x1 - x0, y1 - y0
        border = x0 <= 1 or y0 <= 1 or x1 >= IMG_W - 1 or y1 >= IMG_H - 1
        passes = (dh / ph >= 0.85 and abs(dw - pw) / pw <= 0.10
                  and (v1 - y1) <= 3.0 and not border)
        predicted_uv = h(cam, x, y, yaw)
        J = jacobian(cam, x, y, yaw)
        if predicted_uv is None or abs(np.linalg.det(J)) < 1.0e-12:
            continue
        measured_uv = np.array([0.5 * (x0 + x1), y1])
        error = np.linalg.solve(J, measured_uv - np.asarray(predicted_uv))  # metres
        ray = np.array([x, y]) - np.asarray(cam.cam_pos[:2])
        ray = ray / np.linalg.norm(ray)
        out.append(dict(
            camera=cam_id[-1], x=x, y=y, yaw=yaw, image=index_row["image"],
            range_m=float(index_row["camera_range_m"]), height_ratio=dh / ph,
            passes_deployed=bool(passes), error=error,
            along=float(error @ ray), across=float(error @ np.array([-ray[1], ray[0]]))))
    return out


def availability_trials():
    """How many sightings were ATTEMPTED per place — the denominator for p_c(s).

    Every position the capture visited, whether or not the detector found anything there.
    Without this the ladder would measure accuracy on what survives and quietly hide the
    cost, which is the whole thing this study exists to avoid.
    """

    trials = collections.Counter()
    for row in csv.DictReader(open(capture.OUT / "availability.csv")):
        trials[(round(float(row["x"]), 4), round(float(row["y"]), 4),
                row["camera"][-1])] += 1
    return trials


def field(kept, trials, cell_m=2.0):
    """The availability field p_c(s) under one rule, and how uneven it is.

    Uneven is the operative word: a field the same everywhere gives a route planner nothing
    to choose between, so the spread across places is the quantity that decides whether the
    planning experiment can exist at all.
    """

    usable = collections.Counter()
    for s in kept:
        usable[(round(s["x"], 4), round(s["y"], 4), s["camera"])] += 1
    places = collections.defaultdict(dict)
    for (x, y, cam), n_trials in trials.items():
        if n_trials:
            places[(x, y)][cam] = usable.get((x, y, cam), 0) / n_trials
    best, n_cameras, per_place = [], [], []
    for place, cams in places.items():
        values = list(cams.values())
        best.append(max(values) if values else 0.0)
        n_cameras.append(sum(1 for v in values if v >= 0.5))
        per_place.append((place, max(values) if values else 0.0))
    best = np.asarray(best)
    n_cameras = np.asarray(n_cameras)
    return dict(
        places=len(places),
        best_camera_p_median=float(np.median(best)),
        share_of_places_with_no_usable_camera=float(np.mean(best < 0.05)),
        share_of_places_below_half=float(np.mean(best < 0.5)),
        median_cameras_that_work=float(np.median(n_cameras)),
        share_of_places_with_one_camera_or_none=float(np.mean(n_cameras <= 1)),
        spread_across_places=float(np.std(best)),
    )


def main():
    all_sightings = sightings()
    trials = availability_trials()
    attempted = sum(trials.values())
    print(f"{len(all_sightings):,} detected sightings over {attempted:,} attempted "
          f"(capture: {len({(s['x'], s['y']) for s in all_sightings})} floor positions, "
          f"5 cameras)")

    report = {}
    print(f"\n{'rule':52} {'kept':>7} {'lean':>8} {'worst cam':>10} {'spread':>8} "
          f"{'miss':>8} {'dead places':>12} {'1 cam or none':>14}")
    for label, keep in LADDER:
        kept = [s for s in all_sightings if keep(s)]
        if not kept:
            continue
        along = np.array([s["along"] for s in kept]) * 100
        miss = np.array([np.linalg.norm(s["error"]) for s in kept]) * 100
        worst = max(abs(float(np.mean([s["along"] for s in kept if s["camera"] == c]) * 100))
                    for c in "ABCDE" if any(s["camera"] == c for s in kept))
        f = field(kept, trials)
        report[label] = dict(
            kept=len(kept), kept_share_of_attempted=len(kept) / attempted,
            lean_along_cm=float(along.mean()), worst_camera_lean_cm=worst,
            spread_along_cm=float(along.std()), median_miss_cm=float(np.median(miss)),
            p90_miss_cm=float(np.percentile(miss, 90)), field=f)
        print(f"{label:52} {len(kept) / attempted * 100:6.1f}% {along.mean():+6.2f} cm "
              f"{worst:7.2f} cm {along.std():5.2f} cm {np.median(miss):5.2f} cm "
              f"{f['share_of_places_with_no_usable_camera'] * 100:10.0f}% "
              f"{f['share_of_places_with_one_camera_or_none'] * 100:12.0f}%")

    print("\n'dead places' = floor positions no camera reads reliably; "
          "'1 cam or none' = positions with at most one camera that works "
          "at least half the time.")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gate_ladder.json").write_text(json.dumps(dict(
        detected_sightings=len(all_sightings), attempted_trials=attempted,
        capture=str(capture.DATASET.relative_to(REPO)), ladder=report), indent=1))
    print(f"\nwrote {OUT / 'gate_ladder.json'}")


if __name__ == "__main__":
    main()
