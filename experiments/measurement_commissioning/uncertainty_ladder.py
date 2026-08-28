"""Does anything beat one pixel-noise number pushed through the geometry?

The claim ``uncertainty.py`` rests on is that the detector's error belongs in **pixels** and
everything position-dependent should be left to the camera geometry.  This tests it against
the alternatives, on positions none of them were fitted on.

The rungs, in order of how much they are allowed to learn:

* **one covariance for everything, in ground units** -- the naive choice: measure the spread
  in centimetres and use it everywhere;
* **one per camera, in ground units** -- allows cameras to differ but not places;
* **one pixel number through the geometry** -- what is used now;
* **one pixel number per camera through the geometry**;
* **pixel noise by distance band through the geometry**;
* **a fitted map of pixel noise over the floor** -- the most a spatial model can do.

Scored three ways, because they disagree and each catches a different failure:

* **how surprised** -- the average negative log likelihood of the errors actually seen. The
  deciding column: inflating a covariance always improves coverage but does not improve this.
* **how often the truth is inside the stated 95 % ellipse** -- should be 95 %.
* **the normalised squared error** -- should average 2 for a two-dimensional estimate.
  Below 2 means the stated uncertainty is too large, above means too small.

    python3 experiments/measurement_commissioning/uncertainty_ladder.py
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
sys.path.insert(0, str(HERE))

import capture  # noqa: E402
import offset  # noqa: E402
from camera import camera_models  # noqa: E402
from observation import jacobian  # noqa: E402


def load():
    cal = json.loads((capture.OUT / "calibration.json").read_text())["calibration"]
    coeffs = np.array([cal["coefficients_du"], cal["coefficients_dv"]]).T
    cams = camera_models(capture.DATASET)
    rows = []
    for r in csv.DictReader(open(capture.OUT / "sightings.csv")):
        x, y, yaw = float(r["x"]), float(r["y"]), float(r["yaw"])
        rng = float(r["range_m"])
        J = jacobian(cams[r["camera"]], x, y, yaw)
        if abs(np.linalg.det(J)) < 1e-9:
            continue
        d = np.array([float(r["du_px"]), float(r["dv_px"])]) - offset.apply(coeffs, rng)
        rows.append({"pos": int(r["position_id"]), "camera": r["camera"], "x": x, "y": y,
                     "range_m": rng, "px": d, "Jinv": np.linalg.inv(J)})
    return rows


def score(rows, r_of):
    """Negative log likelihood, 95 % coverage and normalised squared error, in ground units."""
    nll, inside, nis = [], [], []
    for s in rows:
        R = r_of(s)
        e = s["Jinv"] @ s["px"] if R.shape == (2, 2) else None
        e = (s["Jinv"] @ s["px"])
        Ri = np.linalg.inv(R)
        m = float(e @ Ri @ e)
        nll.append(0.5 * (m + math.log(np.linalg.det(R)) + 2 * math.log(2 * math.pi)))
        inside.append(m <= 5.991)          # 95 % of a two-dimensional Gaussian
        nis.append(m)
    return {"negative_log_likelihood": float(np.mean(nll)),
            "coverage_95": float(np.mean(inside)),
            "normalised_squared_error": float(np.mean(nis))}


def main():
    rows = load()
    positions = sorted({s["pos"] for s in rows})
    fit_pos = set(positions[::2])
    fit = [s for s in rows if s["pos"] in fit_pos]
    held = [s for s in rows if s["pos"] not in fit_pos]
    print(f"{len(rows)} sightings; fitted on {len(fit_pos)} positions, "
          f"scored on {len(positions)-len(fit_pos)}\n")

    ground = lambda s: (s["Jinv"] @ s["px"])                    # noqa: E731
    cov = lambda es: np.cov(np.array(es).T) + 1e-12 * np.eye(2)  # noqa: E731

    Rall = cov([ground(s) for s in fit])
    Rcam = {c: cov([ground(s) for s in fit if s["camera"] == c])
            for c in {s["camera"] for s in fit}}
    px_all = float(np.array([s["px"] for s in fit]).std(axis=0).mean())
    px_cam = {c: float(np.array([s["px"] for s in fit if s["camera"] == c]).std(axis=0).mean())
              for c in {s["camera"] for s in fit}}
    band = lambda s: min(int(s["range_m"] // 4), 5)              # noqa: E731
    px_band = {}
    for b in range(6):
        sub = [s for s in fit if band(s) == b]
        px_band[b] = float(np.array([s["px"] for s in sub]).std(axis=0).mean()) if len(sub) > 20 else px_all
    grid = collections.defaultdict(list)
    for s in fit:
        grid[(int(s["x"] // 2), int(s["y"] // 2))].append(s["px"])
    px_grid = {k: float(np.array(v).std(axis=0).mean()) for k, v in grid.items() if len(v) > 12}

    def through(sig):
        return lambda s: s["Jinv"] @ (sig(s) ** 2 * np.eye(2)) @ s["Jinv"].T

    def nearest_grid(s):
        k = (int(s["x"] // 2), int(s["y"] // 2))
        if k in px_grid:
            return px_grid[k]
        if not px_grid:
            return px_all
        return px_grid[min(px_grid, key=lambda q: (q[0] - k[0]) ** 2 + (q[1] - k[1]) ** 2)]

    ladder = [
        ("one covariance for everything", 3, lambda s: Rall),
        ("one per camera", 15, lambda s: Rcam[s["camera"]]),
        ("one pixel number, through geometry", 1, through(lambda s: px_all)),
        ("one pixel number per camera", 5, through(lambda s: px_cam[s["camera"]])),
        ("pixel noise by distance band", 6, through(lambda s: px_band[band(s)])),
        (f"a fitted map over the floor", len(px_grid), through(nearest_grid)),
    ]
    print(f"  {'model':38s} {'numbers':>8} {'surprise':>10} {'inside 95%':>12} {'squared err':>12}")
    out = []
    for name, npar, r_of in ladder:
        sc = score(held, r_of)
        out.append({"model": name, "parameters": npar, **sc})
        print(f"  {name:38s} {npar:8d} {sc['negative_log_likelihood']:10.3f} "
              f"{sc['coverage_95']*100:11.1f}% {sc['normalised_squared_error']:12.2f}")
    print("\n  lower surprise is better; coverage should be 95%; squared error should be 2.0")

    # what a single ground covariance does across distance -- the failure a constant R makes
    print("\n  what one constant covariance does, by distance:")
    print(f"     {'distance':>10} {'n':>6} {'inside 95%':>12} {'squared err':>12}")
    by_band = []
    for b in range(6):
        sub = [s for s in held if band(s) == b]
        if len(sub) < 30:
            continue
        sc = score(sub, lambda s: Rall)
        sg = score(sub, through(lambda s: px_all))
        by_band.append({"band": f"{b*4}-{b*4+4} m", "n": len(sub),
                        "constant": sc, "through_geometry": sg})
        print(f"     {b*4:4d}-{b*4+4:<4d} m {len(sub):6d} {sc['coverage_95']*100:11.1f}% "
              f"{sc['normalised_squared_error']:12.2f}")
    (capture.OUT / "uncertainty_ladder.json").write_text(json.dumps(
        {"ladder": out, "constant_by_distance": by_band,
         "fit_positions": len(fit_pos), "held_positions": len(positions) - len(fit_pos),
         "note": ("Fitted on alternating floor positions and scored on the others. "
                  "Surprise is the deciding column: inflating a covariance always improves "
                  "coverage but does not improve the likelihood of the errors seen.")},
        indent=2))
    print(f"\nwrote {capture.OUT / 'uncertainty_ladder.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
