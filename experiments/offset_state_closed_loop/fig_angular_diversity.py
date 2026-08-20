#!/usr/bin/env python3
"""Is the shared contact-point bias identifiable from inter-camera disagreement?

THE CLAIM. All four cameras report the robot about 27 mm too close along their own
sightline. That looks like a common bias, but it is NOT the unobservable common world
shift, because "toward camera A" and "toward camera D" are different world directions.
So a shared radial bias `b` should make two cameras at the same instant disagree by

    z_c - z_d = b * (u_c - u_d)

where `u_c` is the unit vector from camera c's ground position toward its own projected
point. Everything on the right is known WITHOUT ground truth, so `b` should be
recoverable from disagreement alone -- and the more the two bearings differ, the better
conditioned that recovery should be.

THE TEST. Balanced set-pose grid: every site is a commanded pose held still, so several
cameras report the SAME position and the difference is pure measurement disagreement.

  1  estimate `b` by least squares on camera pairs, using no ground truth at all;
  2  compare with the truth-measured radial bias on the same rows;
  3  bin pairs by the angle between their bearings and show the estimate is useless when
     the cameras look from similar directions and tight when they oppose.

If (2) matches and (3) trends, angular diversity is a real identifiability resource and
belongs in camera placement and route choice. If not, the idea is dead.

Ground truth (the commanded pose) is used ONLY for the reference in step 2.

Outputs -> logs/studies/offset_state_closed_loop/angular_diversity/
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "scripts" / "shared"))
sys.path.insert(0, str(_HERE.parents[1] / "pixel_ground_path"))
from paths import repo_root  # noqa: E402

REPO = repo_root(_HERE)
sys.path.insert(0, str(REPO / "src/reliability"))
sys.path.insert(0, str(REPO / "src/unav_common"))

from dataset_paths import dataset_root  # noqa: E402
from reliability.projection import camera_model_from_world  # noqa: E402

WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_full_4cam.world.sdf"
DATASET = dataset_root(REPO)
DET_CACHE = (REPO / "logs/studies/pixel_ground_path/e2_detector_edge_characterisation"
             / "detector_boxes.csv")
OUT = REPO / "logs/studies/offset_state_closed_loop/angular_diversity"

CAMERA_INCLUDES = {
    "camera_A": "external_camera",
    "camera_B": "external_camera_b",
    "camera_C": "external_camera_c",
    "camera_D": "external_camera_d",
}
CAM_GROUND = {"camera_A": (-6.0, -10.0), "camera_B": (-6.0, 10.0),
              "camera_C": (6.0, -10.0), "camera_D": (6.0, 10.0)}

INK = "#1A1A1A"
FIT = "#0072B2"
REF = "#009E73"
BAD = "#D55E00"


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 140, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#666666", "text.color": INK,
        "xtick.color": "#555555", "ytick.color": "#555555",
        "axes.grid": True, "grid.color": "#E2E2E2", "grid.linewidth": 0.6,
    })


def load_sites(models):
    """Project every qualified clear detection; group by commanded pose."""
    boxes = {}
    with DET_CACHE.open(newline="", encoding="utf-8") as handle:
        for rec in csv.DictReader(handle):
            if str(rec["detected"]) == "1" and rec["pu0"] != "":
                boxes[rec["sample_id"]] = rec

    sites = defaultdict(dict)
    skipped_occluded = 0
    with (DATASET / "localization_calibration_index.csv").open(
            newline="", encoding="utf-8") as handle:
        for rec in csv.DictReader(handle):
            box = boxes.get(rec["sample_id"])
            if box is None:
                continue
            if rec["occlusion_state"] != "clear":
                skipped_occluded += 1
                continue
            camera = rec["camera_id"]
            model = models.get(camera)
            if model is None:
                continue
            u = 0.5 * (float(box["pu0"]) + float(box["pu1"]))
            v = max(float(box["pv0"]), float(box["pv1"]))     # box bottom
            world = model.pixel_to_world(u, v)
            if world is None:
                continue
            key = (round(float(rec["robot_x"]), 3),
                   round(float(rec["robot_y"]), 3),
                   round(float(rec["robot_yaw"]), 3))
            sites[key][camera] = np.array([float(world[0]), float(world[1])])
    return sites, skipped_occluded


def bearing(camera: str, point: np.ndarray) -> np.ndarray:
    """Unit vector from the camera's ground position toward the projected point.

    Uses the PROJECTED point, not the true pose -- this has to be computable online.
    A 27 mm error changes the bearing by ~0.2 deg at these ranges, so it is harmless.
    """
    cam = np.asarray(CAM_GROUND[camera], dtype=float)
    delta = point - cam
    norm = float(np.linalg.norm(delta))
    return delta / norm if norm > 1e-9 else np.zeros(2)


def build_pairs(sites):
    """One record per (site, camera pair): the disagreement and the bearing gap."""
    rows = []
    for (tx, ty, yaw), seen in sites.items():
        truth = np.array([tx, ty])
        cams = sorted(seen)
        for i in range(len(cams)):
            for j in range(i + 1, len(cams)):
                c, d = cams[i], cams[j]
                zc, zd = seen[c], seen[d]
                uc, ud = bearing(c, zc), bearing(d, zd)
                du = uc - ud
                if float(np.linalg.norm(du)) < 1e-9:
                    continue
                cosang = float(np.clip(np.dot(uc, ud), -1.0, 1.0))
                rows.append({
                    "dz": zc - zd, "du": du,
                    "sep_deg": math.degrees(math.acos(cosang)),
                    "truth": truth, "yaw": yaw, "pair": f"{c[-1]}{d[-1]}",
                })
    return rows


def estimate_b(rows) -> tuple[float, float, int]:
    """Least squares on dz = b * du, with a standard error. NO ground truth."""
    num = sum(float(np.dot(r["du"], r["dz"])) for r in rows)
    den = sum(float(np.dot(r["du"], r["du"])) for r in rows)
    if den <= 0.0:
        return math.nan, math.nan, len(rows)
    b = num / den
    resid, dof = 0.0, 0
    for r in rows:
        e = r["dz"] - b * r["du"]
        resid += float(np.dot(e, e))
        dof += 2
    sigma2 = resid / max(dof - 1, 1)
    return b, math.sqrt(sigma2 / den), len(rows)


def truth_radial_bias(sites) -> tuple[float, int]:
    """Reference: mean radial component of (projected - commanded). USES TRUTH."""
    values = []
    for (tx, ty, _yaw), seen in sites.items():
        truth = np.array([tx, ty])
        for camera, z in seen.items():
            u = bearing(camera, z)
            values.append(float(np.dot(z - truth, u)))
    return float(np.mean(values)), len(values)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    _style()

    models = {c: camera_model_from_world(WORLD, include_name=inc)
              for c, inc in CAMERA_INCLUDES.items()}
    sites, skipped = load_sites(models)
    multi = {k: v for k, v in sites.items() if len(v) >= 2}
    rows = build_pairs(multi)
    print(f"  sites total {len(sites)}, with 2+ cameras {len(multi)}, "
          f"pairs {len(rows)}, occluded rows skipped {skipped}")

    b, se, n = estimate_b(rows)
    ref, n_ref = truth_radial_bias(sites)
    print(f"  GT-FREE shared radial bias  = {1000 * b:+.1f} +/- {1000 * se:.1f} mm  "
          f"({n} pairs)")
    print(f"  truth-measured radial bias  = {1000 * ref:+.1f} mm  ({n_ref} detections)")

    edges = [0, 20, 40, 60, 80, 100, 140, 180]
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        subset = [r for r in rows if lo <= r["sep_deg"] < hi]
        if len(subset) < 20:
            continue
        bb, bse, bn = estimate_b(subset)
        bins.append({"lo": lo, "hi": hi, "mid": 0.5 * (lo + hi),
                     "b_mm": 1000 * bb, "se_mm": 1000 * bse, "n": bn,
                     "mean_du": float(np.mean([np.linalg.norm(r["du"])
                                               for r in subset]))})
        print(f"    {lo:3d}-{hi:3d} deg: b = {1000 * bb:+7.1f} +/- {1000 * bse:5.1f} mm  "
              f"n={bn:5d}  mean|du|={bins[-1]['mean_du']:.2f}")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.0))

    # ---- LEFT: the regression, and why small bearing gaps are useless -----------
    xs = np.array([float(np.linalg.norm(r["du"])) for r in rows])
    ys = np.array([float(np.dot(r["dz"], r["du"])) / float(np.linalg.norm(r["du"]))
                   for r in rows]) * 1000.0
    ax.scatter(xs, ys, s=5, alpha=0.16, color=FIT, edgecolors="none")
    grid = np.linspace(0, xs.max() * 1.02, 50)
    ax.plot(grid, 1000 * b * grid, color=INK, lw=2.4,
            label=f"best fit, no ground truth: {1000 * b:+.1f} mm")
    ax.plot(grid, 1000 * ref * grid, color=REF, lw=2.0, ls="--",
            label=f"truth-measured: {1000 * ref:+.1f} mm")
    ax.axhline(0.0, color="#999999", lw=1.0)
    ax.set_xlabel("how differently the two cameras look at the robot\n"
                  "|difference of the two bearing directions|  (0 = same, 2 = opposite)")
    ax.set_ylabel("how much the two cameras disagree, along that\n"
                  "direction (mm)")
    ax.set_title("One shared number explains the disagreement\n"
                 "and it is recovered without ground truth",
                 fontweight="bold", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="lower left")

    # ---- RIGHT: identifiability against angular separation ----------------------
    mids = [c["mid"] for c in bins]
    vals = [c["b_mm"] for c in bins]
    errs = [c["se_mm"] for c in bins]
    ax2.errorbar(mids, vals, yerr=errs, marker="o", ms=6, lw=2.0, capsize=4,
                 color=FIT, label="estimated from this bin alone")
    ax2.axhline(1000 * ref, color=REF, lw=2.0, ls="--",
                label=f"truth-measured: {1000 * ref:+.1f} mm")
    for c in bins:
        ax2.annotate(f"±{c['se_mm']:.0f}", xy=(c["mid"], c["b_mm"]),
                     xytext=(c["mid"], c["b_mm"] + max(errs) * 0.55),
                     ha="center", fontsize=8.5, color="#555555")
    # Mark the region where the answer is not merely imprecise but wrong-signed.
    # ref is in metres, b_mm in millimetres: only the SIGN of the product matters.
    wrong = [c for c in bins if c["b_mm"] * ref < 0.0]
    if wrong:
        edge = max(c["hi"] for c in wrong)
        ax2.axvspan(min(mids) - 12, edge, color=BAD, alpha=0.09)
        ax2.text(0.5 * (min(mids) - 12 + edge), max(vals) * 0.55,
                 "WRONG SIGN\nand confident about it", ha="center", va="center",
                 fontsize=9.5, fontweight="bold", color=BAD)
    ax2.axhline(0.0, color="#999999", lw=1.0)
    ax2.set_xlabel("angle between the two cameras' viewing directions (degrees)")
    ax2.set_ylabel("shared radial bias recovered (mm)")
    ax2.set_title("Cameras looking from OPPOSITE directions pin it down.\n"
                  "Similar directions give a confidently WRONG answer",
                  fontweight="bold", fontsize=10.5)
    ax2.legend(fontsize=8.5, loc="center right")

    fig.suptitle("Angular diversity, not redundancy, is what makes the shared bias "
                 "identifiable", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.923,
             f"{len(rows)} camera pairs at {len(multi)} commanded poses, balanced "
             "set-pose grid, current floor-plane projection, clear views only. The "
             "estimate uses camera disagreement only; ground truth appears solely as the "
             "reference line.",
             ha="center", va="top", fontsize=8.5, color="#444444")
    fig.tight_layout(rect=(0, 0, 1, 0.878))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_a1_angular_diversity.{ext}", bbox_inches="tight")
    plt.close(fig)

    (OUT / "summary.json").write_text(json.dumps({
        "sites_total": len(sites), "sites_multi_camera": len(multi),
        "pairs": len(rows), "occluded_rows_skipped": skipped,
        "gt_free_shared_radial_bias_mm": 1000 * b,
        "gt_free_standard_error_mm": 1000 * se,
        "truth_measured_radial_bias_mm": 1000 * ref,
        "truth_reference_detections": n_ref,
        "by_separation": bins,
    }, indent=2), encoding="utf-8")
    print(f"wrote -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
