#!/usr/bin/env python3
"""Is the per-camera error actually a constant world offset, or something else?

THE CRITICISM THIS ANSWERS. The offset-state filter carries a constant 2-D world
offset per camera. That is a very specific shape, and it is the shape the filter can
represent -- so fitting it well proves nothing unless a real calibration-style error
would ALSO have that shape. A drifted mounting angle does not: its ground error grows
with range and points along the camera's bearing.

So: score competing shapes against the SAME real residuals, held out.

  shape                      free numbers   what it says
  ------------------------------------------------------------------------------
  nothing                          0        the projection is unbiased
  shared radial                    1        every camera reads the same amount too
                                            near/far along its own sightline
  constant world offset            2/cam    what the filter assumes
  constant radial                  1/cam    reads short or long along its sightline
  radial growing with range        2/cam    the signature of a drifted mounting angle
  world offset + radial            3/cam    both together

Data: the balanced set-pose grid, 1844 real detections, 942 sites, four robot yaws,
current floor-plane IPM. Ground truth is the commanded pose; it scores only.

Held out by SPACE, not at random: fit on the west half of the floor, score on the east
half, and the reverse. A shape that only works where it was fitted is the failure mode
that killed the deployed correction (e7), so random splits would hide exactly the thing
worth knowing.

Outputs -> logs/studies/offset_state_closed_loop/error_shape/
"""

from __future__ import annotations

import csv
import json
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
OUT = REPO / "logs/studies/offset_state_closed_loop/error_shape"

CAMERAS = ["camera_A", "camera_B", "camera_C", "camera_D"]
INCLUDES = {"camera_A": "external_camera", "camera_B": "external_camera_b",
            "camera_C": "external_camera_c", "camera_D": "external_camera_d"}
CAM_GROUND = {"camera_A": (-6.0, -10.0), "camera_B": (-6.0, 10.0),
              "camera_C": (6.0, -10.0), "camera_D": (6.0, 10.0)}
CAM_COLOR = {"camera_A": "#5F6A73", "camera_B": "#56B4E9",
             "camera_C": "#D55E00", "camera_D": "#CC79A7"}
INK = "#1A1A1A"


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 140, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#666666", "text.color": INK,
        "xtick.color": "#555555", "ytick.color": "#555555",
        "axes.grid": True, "grid.color": "#E4E4E4", "grid.linewidth": 0.6,
    })


def load_rows():
    """Real detections, projected with the CURRENT IPM, joined to commanded poses."""
    models = {c: camera_model_from_world(WORLD, include_name=INCLUDES[c])
              for c in CAMERAS}
    boxes = {}
    with DET_CACHE.open(newline="", encoding="utf-8") as handle:
        for rec in csv.DictReader(handle):
            if str(rec["detected"]) == "1" and rec["pu0"] != "":
                boxes[rec["sample_id"]] = rec

    rows = []
    with (DATASET / "localization_calibration_index.csv").open(
            newline="", encoding="utf-8") as handle:
        for rec in csv.DictReader(handle):
            box = boxes.get(rec["sample_id"])
            if box is None or rec["occlusion_state"] != "clear":
                continue
            cam = rec["camera_id"]
            model = models.get(cam)
            if model is None:
                continue
            u = 0.5 * (float(box["pu0"]) + float(box["pu1"]))
            v = max(float(box["pv0"]), float(box["pv1"]))
            world = model.pixel_to_world(u, v)
            if world is None:
                continue
            truth = np.array([float(rec["robot_x"]), float(rec["robot_y"])])
            z = np.array([float(world[0]), float(world[1])])
            cam_xy = np.array(CAM_GROUND[cam])
            delta = z - cam_xy
            rng = float(np.linalg.norm(delta))
            u_hat = delta / max(rng, 1e-9)
            rows.append({"cam": cam, "truth": truth, "z": z, "res": z - truth,
                         "u": u_hat, "range": rng, "yaw": float(rec["robot_yaw"])})
    return rows


# --------------------------------------------------------------- candidate shapes
def basis(shape: str, row) -> np.ndarray:
    """Columns of the design matrix contributed by one detection, shape (2, k)."""
    u = row["u"]
    if shape == "none":
        return np.zeros((2, 0))
    if shape == "shared_radial":
        return u.reshape(2, 1)
    if shape == "world":
        return np.eye(2)
    if shape == "radial":
        return u.reshape(2, 1)
    if shape == "radial_range":
        return np.column_stack([u, u * row["range"]])
    if shape == "world_radial":
        return np.column_stack([np.eye(2), u])
    raise ValueError(shape)


PER_CAMERA = {"none": False, "shared_radial": False, "world": True,
              "radial": True, "radial_range": True, "world_radial": True}
LABEL = {
    "none": "nothing — assume the projection is unbiased",
    "shared_radial": "ONE shared radial number for all four cameras",
    "world": "a constant world offset per camera  ← what the filter assumes",
    "radial": "a constant radial offset per camera",
    "radial_range": "radial, growing with range, per camera  ← drifted-mounting signature",
    "world_radial": "world offset AND radial, per camera",
}


def fit_predict(shape: str, fit_rows, score_rows):
    """Least squares on the fit half, error on the score half. Returns mm."""
    if shape == "none":
        err = [float(np.linalg.norm(r["res"])) for r in score_rows]
        return 1000 * float(np.mean(err)), 0

    groups = CAMERAS if PER_CAMERA[shape] else ["ALL"]
    theta, n_params = {}, 0
    for g in groups:
        sel = [r for r in fit_rows if g == "ALL" or r["cam"] == g]
        if not sel:
            theta[g] = None
            continue
        A = np.vstack([basis(shape, r) for r in sel])
        b = np.concatenate([r["res"] for r in sel])
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        theta[g] = sol
        n_params += sol.size

    err = []
    for r in score_rows:
        g = r["cam"] if PER_CAMERA[shape] else "ALL"
        t = theta.get(g)
        pred = basis(shape, r) @ t if t is not None else np.zeros(2)
        err.append(float(np.linalg.norm(r["res"] - pred)))
    return 1000 * float(np.mean(err)), n_params


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    _style()
    rows = load_rows()
    west = [r for r in rows if r["truth"][0] < 0.0]
    east = [r for r in rows if r["truth"][0] >= 0.0]
    print(f"  {len(rows)} real detections — west half {len(west)}, east half {len(east)}")

    shapes = ["none", "shared_radial", "world", "radial", "radial_range", "world_radial"]
    results = []
    for shape in shapes:
        a, n = fit_predict(shape, west, east)      # fit west, score east
        b, _ = fit_predict(shape, east, west)      # and the reverse
        same, _ = fit_predict(shape, rows, rows)   # fitted and scored on everything
        results.append({"shape": shape, "held_out_mm": 0.5 * (a + b),
                        "w2e_mm": a, "e2w_mm": b, "in_sample_mm": same,
                        "n_params": n})
        print(f"    {LABEL[shape]:<62} held-out {0.5 * (a + b):6.1f} mm  "
              f"(in-sample {same:5.1f}, {n} params)")

    # ---------------------------------------------------------------- figure
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14.6, 5.4),
                                  gridspec_kw={"width_ratios": [1.25, 1.0]})

    order = list(range(len(results)))
    ypos = np.arange(len(order))
    held = [results[i]["held_out_mm"] for i in order]
    ins = [results[i]["in_sample_mm"] for i in order]
    colors = ["#0072B2" if results[i]["shape"] == "world" else "#9AA7B1" for i in order]
    ax.barh(ypos - 0.19, held, 0.36, color=colors, label="scored on floor it never saw")
    ax.barh(ypos + 0.19, ins, 0.36, color="#D8DEE3", label="scored where it was fitted")
    for y, (h, s) in enumerate(zip(held, ins)):
        ax.text(h + 0.6, y - 0.19, f"{h:.1f}", va="center", fontsize=9.5,
                fontweight="bold", color=INK)
        ax.text(s + 0.6, y + 0.19, f"{s:.1f}", va="center", fontsize=9, color="#6B7780")
    ax.set_yticks(ypos, [LABEL[results[i]["shape"]] for i in order], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("mean error left over (mm) — smaller is better")
    ax.set_xlim(0, max(held + ins) * 1.22)
    ax.set_title("Which shape actually fits the real error?\n"
                 "fit on one half of the floor, scored on the other",
                 fontweight="bold", fontsize=10.5, loc="left")
    ax.legend(fontsize=8.5, loc="upper center", ncol=2, frameon=False,
              bbox_to_anchor=(0.5, -0.16))

    # right: is the residual radial, and does it grow with range?
    for cam in CAMERAS:
        sel = [r for r in rows if r["cam"] == cam]
        rng = np.array([r["range"] for r in sel])
        rad = np.array([float(np.dot(r["res"], r["u"])) for r in sel]) * 1000
        bins = np.linspace(rng.min(), rng.max(), 9)
        idx = np.digitize(rng, bins) - 1
        xs, ys = [], []
        for b in range(len(bins) - 1):
            m = idx == b
            if m.sum() < 12:
                continue
            xs.append(0.5 * (bins[b] + bins[b + 1]))
            ys.append(float(np.mean(rad[m])))
        ax2.plot(xs, ys, marker="o", ms=5, lw=2.0, color=CAM_COLOR[cam],
                 label=f"{cam[-1]} ({len(sel)})")
    ax2.axhline(0.0, color="#999999", lw=1.0)
    ax2.set_xlabel("how far the robot is from that camera (m)")
    ax2.set_ylabel("error along the camera's own sightline (mm)\nnegative = reads too near")
    # Corrected 2026-08-11: an earlier version of this title claimed the radial error
    # GROWS with range. The measurement says the opposite -- it is largest close in and
    # fades toward zero by ~14 m, which rules OUT the drifted-mounting signature.
    ax2.set_title("The radial error is strongest CLOSE IN and fades with range\n"
                  "so it is not a drifted mounting angle, and not a constant either",
                  fontweight="bold", fontsize=10.5, loc="left")
    ax2.legend(fontsize=8.5, ncol=4, loc="upper center", frameon=False,
               bbox_to_anchor=(0.5, -0.16), title="camera (detections)",
               title_fontsize=8.5)

    fig.suptitle("What shape is the per-camera error, really?",
                 fontsize=13.5, fontweight="bold")
    fig.text(0.5, 0.925,
             f"{len(rows)} real detections, balanced set-pose grid, current floor-plane "
             "IPM, clear views only. Scored against commanded ground truth.",
             ha="center", va="top", fontsize=8.8, color="#444444")
    fig.tight_layout(rect=(0, 0, 1, 0.878))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_e1_error_shape.{ext}", bbox_inches="tight")
    plt.close(fig)

    (OUT / "summary.json").write_text(json.dumps(
        {"n_detections": len(rows), "west": len(west), "east": len(east),
         "results": results}, indent=2), encoding="utf-8")
    print(f"wrote -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
