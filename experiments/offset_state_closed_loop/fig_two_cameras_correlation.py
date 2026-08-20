#!/usr/bin/env python3
"""When is a second camera worth anything? Not when it is perpendicular -- opposite.

THE QUESTION THIS ANSWERS. Trustworthiness is carried as one number per camera in the
deployed fusion, but a camera watching a floor does not measure x and y equally well: a
pixel of detector error moves the reported position much further along the camera's own
line of sight than across it. So each `R_c` has an orientation, its x and y errors are
correlated, and the hope is that two cameras crossing at 90 degrees are complementary --
take this one's sideways coordinate and that one's.

THE TWO COMPETING LAWS, both pure geometry, no fitting.

  * if the error is RANDOM, fusing two 1.7:1 ellipses is best when they cross at 90
    degrees (each covers the other's vague axis) and buys only the usual 1/sqrt(2) when
    they are parallel OR head-on, because then both are vague about the same direction.
  * if the error is a repeatable LEAN along each camera's own sightline, the pair's
    remaining lean is `b * cos(half the crossing angle)`: unchanged at 0 degrees,
    0.71 at 90 degrees, and exactly cancelled head-on at 180 degrees.

The two laws disagree about where the sweet spot is, so measuring which one governs
tells you what a second camera is actually for.

DATA. The balanced set-pose grid: every site is a commanded pose held still, so when two
cameras both report it the difference is pure measurement disagreement -- no motion, no
clock skew. 1844 clear detections, current floor-plane IPM, 575 camera pairs. Measured
on the deployed projection AND on the same readings with the mesh-predicted silhouette
offset removed, because which law dominates turns out to depend on that.

GROUND TRUTH is the commanded pose and is used ONLY to score. The offset prediction is
anchored on the camera's own reading, not on the true position. The one pose quantity it
does need is the robot's heading, taken here from the commanded yaw; at run time that
comes from the odometry heading, which is why a heading-free version of this correction
does not exist.

CONFOUND, stated rather than hidden: the crossing angle is not freely assignable. The
two same-wall pairs (A-C, B-D) are the only narrow-crossing pairs this layout offers and
they occur at longer range, so a range-matched check is printed alongside.

Outputs -> logs/studies/offset_state_closed_loop/two_cameras/
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
from matplotlib.patches import Ellipse, Rectangle

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "scripts" / "shared"))
sys.path.insert(0, str(_HERE.parents[1] / "pixel_ground_path"))
sys.path.insert(0, str(_HERE.parents[1] / "filter_notebook"))
from paths import repo_root  # noqa: E402

REPO = repo_root(_HERE)
sys.path.insert(0, str(REPO / "src/reliability"))
sys.path.insert(0, str(REPO / "src/unav_common"))
sys.path.insert(0, str(REPO / "src/state"))

from dataset_paths import dataset_root  # noqa: E402
from reliability.projection import camera_model_from_world  # noqa: E402
import notebook_data as nd  # noqa: E402

WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_full_4cam.world.sdf"
DATASET = dataset_root(REPO)
DET_CACHE = (REPO / "logs/studies/pixel_ground_path/e2_detector_edge_characterisation"
             / "detector_boxes.csv")
OUT = REPO / "logs/studies/offset_state_closed_loop/two_cameras"

CAMERA_INCLUDES = {
    "camera_A": "external_camera", "camera_B": "external_camera_b",
    "camera_C": "external_camera_c", "camera_D": "external_camera_d",
}
CAMERAS = tuple(CAMERA_INCLUDES)
SHORT = {c: c[-1] for c in CAMERAS}
CAMERA_COLOUR = {"camera_A": "#0072B2", "camera_B": "#D55E00",
                 "camera_C": "#009E73", "camera_D": "#CC79A7"}

INK = "#1A1A1A"
LEAN = "#6A3D9A"      # the repeatable part of the error
SCATTER = "#2E8B8B"   # the random part
FIX = "#0072B2"
RAW = "#B9770E"
REF = "#B00020"

# 1 px of detector error as a standard deviation. It cancels out of every fusion weight
# here (both cameras get the same number); it only sets the ellipse sizes drawn.
SIGMA_PX = 1.0
ELLIPSE_MAGNIFY = 22.0
ANISOTROPY = 1.75     # measured median of the four cameras, used for the ideal curve
ANGLE_EDGES = (45.0, 75.0, 105.0, 135.0, 165.0, 180.01)
MIN_PAIRS_PER_BIN = 20


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9.5,
        "axes.titlesize": 10.5, "axes.labelsize": 9.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#666666", "text.color": INK,
        "xtick.color": "#555555", "ytick.color": "#555555",
        "axes.grid": True, "grid.color": "#E4E4E4", "grid.linewidth": 0.6,
        "legend.frameon": False,
    })


# --------------------------------------------------------------------------- data

def projection_jacobian(model, u, v, step_px=0.5):
    """d(world)/d(pixel), by the same central difference the runtime uses."""
    columns = []
    for axis in (0, 1):
        delta = (step_px if axis == 0 else 0.0, step_px if axis == 1 else 0.0)
        plus = model.pixel_to_world(u + delta[0], v + delta[1])
        minus = model.pixel_to_world(u - delta[0], v - delta[1])
        if plus is None or minus is None:
            return None
        columns.append((np.asarray(plus) - np.asarray(minus)) / (2.0 * step_px))
    return np.column_stack(columns)


def silhouette_bottom(model, x, y, yaw, points):
    """Where the bottom-centre of the robot's projected box lands on the floor.

    The observation function the pipeline should be using: shape from the URDF meshes,
    camera from the world file, no fitted parameters.
    """
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    world = points @ rotation.T + np.array([x, y, 0.0])
    in_camera = (world - model.cam_pos) @ model.R.T
    ahead = in_camera[:, 2] > 1e-6
    if not ahead.any():
        return None
    projected = (model.K @ in_camera[ahead].T).T
    uv = projected[:, :2] / projected[:, 2:3]
    return model.pixel_to_world(0.5 * (uv[:, 0].min() + uv[:, 0].max()), uv[:, 1].max())


def load_observations(models):
    """Every clear, qualified detection: pixel, world reading, its Jacobian, truth."""
    boxes = {}
    with DET_CACHE.open(newline="", encoding="utf-8") as handle:
        for rec in csv.DictReader(handle):
            if str(rec["detected"]) == "1" and rec["pu0"] != "":
                boxes[rec["sample_id"]] = rec

    points = nd.robot_point_cloud()
    observations, sites, skipped = [], defaultdict(dict), 0
    index = DATASET / "localization_calibration_index.csv"
    with index.open(newline="", encoding="utf-8") as handle:
        for rec in csv.DictReader(handle):
            box = boxes.get(rec["sample_id"])
            if box is None:
                continue
            if rec["occlusion_state"] != "clear":
                skipped += 1
                continue
            camera = rec["camera_id"]
            model = models.get(camera)
            if model is None:
                continue
            u = 0.5 * (float(box["pu0"]) + float(box["pu1"]))
            v = max(float(box["pv0"]), float(box["pv1"]))          # box bottom
            world = model.pixel_to_world(u, v)
            jacobian = projection_jacobian(model, u, v)
            if world is None or jacobian is None:
                continue
            truth = np.array([float(rec["robot_x"]), float(rec["robot_y"])])
            yaw = float(rec["robot_yaw"])
            reading = np.array([float(world[0]), float(world[1])])
            # Anchor the offset prediction on the camera's OWN reading, never on truth.
            # (Anchoring it on truth instead moves every corrected reading by 0.10 mm
            # median, 0.69 mm worst, and changes no number in this figure.)
            landing = silhouette_bottom(model, reading[0], reading[1], yaw, points)
            if landing is None:
                continue
            offset = np.array([landing[0] - reading[0], landing[1] - reading[1]])
            record = {
                "camera": camera, "uv": (u, v), "truth": truth, "yaw": yaw,
                "raw": reading,                       # deployed: box bottom through IPM
                "fixed": reading - offset,            # minus the mesh-predicted offset
                "R": (SIGMA_PX ** 2) * (jacobian @ jacobian.T),
            }
            observations.append(record)
            sites[(round(truth[0], 3), round(truth[1], 3), round(yaw, 3))][camera] = record
    return observations, sites, skipped


def bearing(models, camera, point):
    """Unit vector from the camera's ground position toward the reported point."""
    ground = np.asarray(models[camera].cam_pos[:2], dtype=float)
    delta = np.asarray(point) - ground
    return delta / float(np.linalg.norm(delta))


def fuse(z_a, R_a, z_b, R_b):
    info_a, info_b = np.linalg.inv(R_a), np.linalg.inv(R_b)
    P = np.linalg.inv(info_a + info_b)
    return P @ (info_a @ z_a + info_b @ z_b), P


def build_pairs(models, sites):
    """One record per (site, camera pair): crossing angle, both errors, both R."""
    rows = []
    for seen in sites.values():
        cameras = sorted(seen)
        for i in range(len(cameras)):
            for j in range(i + 1, len(cameras)):
                a, b = seen[cameras[i]], seen[cameras[j]]
                u_a = bearing(models, cameras[i], a["fixed"])
                u_b = bearing(models, cameras[j], b["fixed"])
                separation = math.degrees(math.acos(float(np.clip(u_a @ u_b, -1.0, 1.0))))
                truth = a["truth"]
                # Express everything in the PAIR's own frame (first camera's sightline
                # and its perpendicular) so that pooling different pairs is legitimate.
                frame = np.column_stack([u_a, np.array([-u_a[1], u_a[0]])])
                row = {
                    "pair": f"{SHORT[cameras[i]]}{SHORT[cameras[j]]}",
                    "sep_deg": separation, "truth": truth,
                    "range_m": float(np.mean([
                        np.linalg.norm(truth - np.asarray(models[c].cam_pos[:2]))
                        for c in (cameras[i], cameras[j])])),
                }
                for path in ("raw", "fixed"):
                    z_a, z_b = a[path], b[path]
                    weighted, P = fuse(z_a, a["R"], z_b, b["R"])
                    averaged = 0.5 * (z_a + z_b)
                    err_a = float(np.linalg.norm(z_a - truth))
                    err_b = float(np.linalg.norm(z_b - truth))
                    row[path] = {
                        "e_a": frame.T @ (z_a - truth),      # in the pair frame
                        "e_b": frame.T @ (z_b - truth),
                        "e_fused": frame.T @ (averaged - truth),
                        "e_weighted": frame.T @ (weighted - truth),
                        "one": 0.5 * (err_a + err_b),
                        "averaged": float(np.linalg.norm(averaged - truth)),
                        "weighted": float(np.linalg.norm(weighted - truth)),
                        # what the covariances THEMSELVES predict the gain should be
                        "predicted_ratio": (
                            math.sqrt(np.trace(P) / 2.0)
                            / (0.5 * (math.sqrt(np.trace(a["R"]) / 2.0)
                                      + math.sqrt(np.trace(b["R"]) / 2.0)))),
                    }
                rows.append(row)
    return rows


def bin_pairs(rows):
    """Group pairs by crossing angle and split each group's error into lean + scatter."""
    out = []
    for lo, hi in zip(ANGLE_EDGES[:-1], ANGLE_EDGES[1:]):
        selected = [r for r in rows if lo <= r["sep_deg"] < hi]
        if len(selected) < MIN_PAIRS_PER_BIN:
            continue
        entry = {"lo_deg": lo, "hi_deg": min(hi, 180.0), "n": len(selected),
                 "median_sep_deg": float(np.median([r["sep_deg"] for r in selected])),
                 "median_range_m": float(np.median([r["range_m"] for r in selected])),
                 "rows": selected}
        for path in ("raw", "fixed"):
            first = np.array([r[path]["e_a"] for r in selected])
            # The mechanism is measured on the PLAIN AVERAGE of the two readings, because
            # that is the case the cos law is exact geometry for. Weighting by R is
            # scored separately below; it never does worse.
            fused = np.array([r[path]["e_fused"] for r in selected])
            weighted = np.array([r[path]["e_weighted"] for r in selected])
            # lean = the part that repeats every time; scatter = the part that does not
            lean_one = float(np.linalg.norm(first.mean(axis=0)))
            lean_pair = float(np.linalg.norm(fused.mean(axis=0)))
            entry[path] = {
                "lean_one_mm": 1000 * lean_one,
                "lean_pair_mm": 1000 * lean_pair,
                "lean_ratio": lean_pair / lean_one,
                "lean_cos_law_mm": 1000 * lean_one * math.cos(
                    math.radians(entry["median_sep_deg"] / 2.0)),
                "lean_pair_weighted_mm": 1000 * float(
                    np.linalg.norm(weighted.mean(axis=0))),
                "scatter_one_mm": 1000 * math.sqrt(np.trace(np.cov(first.T)) / 2.0),
                "scatter_pair_mm": 1000 * math.sqrt(np.trace(np.cov(fused.T)) / 2.0),
                "scatter_pair_weighted_mm": 1000 * math.sqrt(
                    np.trace(np.cov(weighted.T)) / 2.0),
                "scatter_ratio": (math.sqrt(np.trace(np.cov(fused.T)))
                                  / math.sqrt(np.trace(np.cov(first.T)))),
                "ratio_weighted": float(np.median(
                    [r[path]["weighted"] / r[path]["one"] for r in selected])),
                "ratio_averaged": float(np.median(
                    [r[path]["averaged"] / r[path]["one"] for r in selected])),
                "ratio_predicted": float(np.median(
                    [r[path]["predicted_ratio"] for r in selected])),
                "error_one_cm": float(100 * np.median(
                    [r[path]["one"] for r in selected])),
                "error_fused_cm": float(100 * np.median(
                    [r[path]["weighted"] for r in selected])),
            }
        out.append(entry)
    return out


# ---------------------------------------------------------------------- panels

def panel_one_camera(ax, models, observations, camera="camera_A"):
    """The shape of one camera's error across the floor. Geometry only."""
    ax.add_patch(Rectangle((-11.0, -10.0), 22.0, 20.0, facecolor="#FAFAFA",
                           edgecolor="#D8D8D8", lw=1.0, zorder=0))
    mine = [o for o in observations if o["camera"] == camera]
    ground = np.asarray(models[camera].cam_pos[:2], dtype=float)

    chosen = []
    for gx in (-9.0, -6.0, -3.0, 0.0, 3.0, 6.0, 9.0):
        for gy in (-6.0, -1.0, 4.0, 8.5):
            target = np.array([gx, gy])
            near = min(mine, key=lambda o: np.linalg.norm(o["truth"] - target))
            if np.linalg.norm(near["truth"] - target) < 2.0:
                chosen.append(near)

    for other in CAMERAS:
        pos = np.asarray(models[other].cam_pos[:2], dtype=float)
        ax.plot(*pos, marker="s", ms=8, color=CAMERA_COLOUR[other], zorder=6,
                clip_on=False)
        ax.annotate(SHORT[other], pos, textcoords="offset points",
                    xytext=(9, 4) if other == camera else (0, 10 if pos[1] < 0 else -17),
                    ha="center", fontsize=9.5, color=CAMERA_COLOUR[other], weight="bold")

    quoted = None
    for record in chosen:
        centre = record["truth"]
        ax.plot([ground[0], centre[0]], [ground[1], centre[1]], lw=0.5,
                color="#CFCFCF", zorder=1)
        values, vectors = np.linalg.eigh(record["R"])
        order = np.argsort(values)[::-1]
        values, vectors = values[order], vectors[:, order]
        width, height = 2.0 * ELLIPSE_MAGNIFY * np.sqrt(np.maximum(values, 1e-12))
        ax.add_patch(Ellipse(centre, width, height,
                             angle=math.degrees(math.atan2(vectors[1, 0], vectors[0, 0])),
                             lw=1.2, edgecolor="#00517F", facecolor="#4C9BCF",
                             alpha=0.42, zorder=3))
        rho = record["R"][0, 1] / math.sqrt(record["R"][0, 0] * record["R"][1, 1])
        if centre[1] > 0.0 and (quoted is None or abs(rho) > abs(quoted[1])):
            quoted = (record, rho)

    if quoted is not None:
        record, rho = quoted
        R = record["R"] * 1e4
        ax.annotate(
            f"one pixel of detector error at this spot means\n"
            f"$R$ = [[{R[0, 0]:.1f}, {R[0, 1]:+.1f}], [{R[1, 0]:+.1f}, {R[1, 1]:.1f}]] cm$^2$\n"
            f"{math.sqrt(R[0, 0]):.1f} cm in x, {math.sqrt(R[1, 1]):.1f} cm in y, "
            f"moving together ({rho:+.2f})",
            record["truth"], textcoords="axes fraction", xytext=(0.015, 0.99),
            fontsize=8.2, color=INK, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#CCCCCC", lw=0.7),
            arrowprops=dict(arrowstyle="-", lw=0.9, color="#888888"))

    ax.set_title("A camera does not measure x and y equally well\n"
                 "each error ellipse is stretched along that camera's own sightline")
    ax.set_xlabel(f"x (m)   ·   ellipses are 1 px of detector error, drawn "
                  f"{ELLIPSE_MAGNIFY:.0f}$\\times$ oversize\n"
                  "geometry says 1.7–1.8$\\times$ longer than wide; "
                  "the real residuals measure 1.5–1.8$\\times$")
    ax.set_ylabel("y (m)")
    ax.set_xlim(-11.8, 11.8)
    ax.set_ylim(-11.0, 16.0)
    ax.set_aspect("equal")
    return quoted


def panel_two_laws(ax, bins):
    """The two competing predictions, with the measured points on top."""
    def rotate(t):
        c, s = math.cos(t), math.sin(t)
        return np.array([[c, -s], [s, c]])

    angles = np.linspace(0.0, 180.0, 181)
    single = math.sqrt((ANISOTROPY ** 2 + 1.0) / 2.0)
    random_curve = []
    for deg in angles:
        R_1 = np.diag([ANISOTROPY ** 2, 1.0])
        R_2 = rotate(math.radians(deg)) @ R_1 @ rotate(math.radians(deg)).T
        P = np.linalg.inv(np.linalg.inv(R_1) + np.linalg.inv(R_2))
        random_curve.append(math.sqrt(np.trace(P) / 2.0) / single)
    lean_curve = np.cos(np.radians(angles / 2.0))

    ax.plot(angles, random_curve, lw=2.0, color=SCATTER,
            label="predicted, if the error is random noise")
    ax.plot(angles, lean_curve, lw=2.0, color=LEAN,
            label="predicted, if it is a repeatable lean:  $\\cos(\\theta/2)$")

    centres = [e["median_sep_deg"] for e in bins]
    ax.plot(centres, [e["fixed"]["scatter_ratio"] for e in bins], marker="o", ms=7,
            ls="none", mfc="white", mew=2.0, mec=SCATTER,
            label="measured, the random part")
    ax.plot(centres, [e["fixed"]["lean_ratio"] for e in bins], marker="D", ms=7,
            ls="none", color=LEAN, label="measured, the repeatable part")

    ax.annotate("noise averaging is best\nwhen they cross at 90$\\degree$",
                (90, random_curve[90]), textcoords="offset points", xytext=(-30, -46),
                fontsize=8.2, color=SCATTER, ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.8, color=SCATTER))
    ax.annotate("a lean only cancels\nwhen they face each other",
                (168, lean_curve[168]), textcoords="offset points", xytext=(-46, 52),
                fontsize=8.2, color=LEAN, ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.8, color=LEAN))
    ax.set_title("Two laws, disagreeing about where the sweet spot is\n"
                 "the random part dips in the middle; the lean falls all the way to zero")
    ax.set_xlabel("angle between the two cameras' sightlines (deg)\n"
                  "measured points come from plainly averaging the two readings, "
                  "the case the $\\cos$ law is exact for")
    ax.set_ylabel("error of the pair $\\div$ error of one camera")
    ax.set_xticks(range(0, 181, 30))
    ax.set_ylim(-0.04, 1.12)
    ax.legend(fontsize=8.2, loc="lower left")


def panel_absolute(ax, bins):
    """The actual millimetres: what shrinks when you add the second camera."""
    labels = []
    for row, entry in enumerate(bins):
        labels.append(f"{entry['lo_deg']:.0f}$\\degree$–{entry['hi_deg']:.0f}$\\degree$\n"
                      f"{entry['n']} pairs")
        for offset, key, colour in ((-0.17, "lean", LEAN), (0.17, "scatter", SCATTER)):
            one = entry["fixed"][f"{key}_one_mm"]
            pair = entry["fixed"][f"{key}_pair_mm"]
            y = row + offset
            ax.annotate("", (pair, y), (one, y),
                        arrowprops=dict(arrowstyle="-|>", lw=1.8, color=colour,
                                        shrinkA=0, shrinkB=0, mutation_scale=11))
            ax.plot([one], [y], marker="o", ms=5.5, color=colour, mfc="white", mew=1.8)
            ax.annotate(f"{pair:.0f} mm", (pair, y), textcoords="offset points",
                        xytext=(-13, -3), fontsize=7.8, color=colour, ha="right",
                        va="center")
    ax.plot([], [], marker="o", ls="-", color=LEAN, mfc="white", mew=1.8,
            label="the repeatable lean")
    ax.plot([], [], marker="o", ls="-", color=SCATTER, mfc="white", mew=1.8,
            label="the random scatter")
    ax.set_yticks(range(len(bins)))
    ax.set_yticklabels(labels, fontsize=8.6)
    ax.invert_yaxis()
    ax.set_xlim(-6, 80)
    ax.set_xlabel("millimetres of error, on the floor\n"
                  "(circle = one camera alone, arrowhead = the two averaged)")
    ax.set_ylabel("angle between the\nsightlines")
    ax.set_title("The second camera cancels the lean, not the noise\n"
                 "and it can only cancel it if it looks from the other side")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=8.2, loc="upper right", bbox_to_anchor=(1.005, 0.40),
              handlelength=1.6)


def panel_what_you_get(ax, bins):
    """Delivered end to end: fused error over single-camera error, both paths."""
    labels = [f"{e['lo_deg']:.0f}$\\degree$–{e['hi_deg']:.0f}$\\degree$\n{e['n']} pairs"
              for e in bins]
    y = np.arange(len(bins))
    ax.barh(y - 0.19, [e["raw"]["ratio_weighted"] for e in bins], height=0.35,
            color=RAW, label="the projection as deployed today")
    ax.barh(y + 0.19, [e["fixed"]["ratio_weighted"] for e in bins], height=0.35,
            color=FIX, label="with the offset predicted from the robot's meshes")
    ax.plot([e["fixed"]["ratio_averaged"] for e in bins], y + 0.19, marker="D", ms=5.5,
            ls="none", color=INK,
            label="same readings, plainly averaged instead of weighted by $R$")
    ax.axvline(1.0, color="#555555", lw=1.0)
    ax.axvline(1.0 / math.sqrt(2.0), color=REF, ls="--", lw=1.3,
               label="what two independent, equally good cameras would give")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.6)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.16)
    ax.set_xlabel("error of the fused reading $\\div$ error of one camera\n"
                  "(median over pairs; below 1 means the pair helped)")
    ax.set_ylabel("angle between the\nsightlines")
    ax.set_title("What a second camera actually delivers\n"
                 "nothing at all until the lean is dealt with first")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=8.2, loc="upper center", bbox_to_anchor=(0.5, -0.245), ncol=2,
              handlelength=1.6, columnspacing=1.4)


# ------------------------------------------------------------------------- main

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    _style()

    models = {c: camera_model_from_world(WORLD, include_name=inc)
              for c, inc in CAMERA_INCLUDES.items()}
    observations, sites, skipped = load_observations(models)
    rows = build_pairs(models, sites)
    bins = bin_pairs(rows)
    multi = sum(1 for v in sites.values() if len(v) >= 2)
    print(f"  detections {len(observations)}, sites {len(sites)} ({multi} seen by 2+ "
          f"cameras), pairs {len(rows)}, occluded rows skipped {skipped}")

    summary = {"n_detections": len(observations), "n_sites": len(sites),
               "n_multi_camera_sites": multi, "n_pairs": len(rows),
               "sigma_px": SIGMA_PX, "per_camera": {}, "by_angle": [], "pairs": {}}

    print("\n  1. is the residual stretched along the sightline, the way J J^T says?")
    print(f"  {'cam':>5s}{'n':>6s}{'predicted':>11s}{'measured':>10s}"
          f"{'along cm':>10s}{'across cm':>11s}{'lean mm':>9s}")
    for camera in CAMERAS:
        mine = [o for o in observations if o["camera"] == camera]
        predicted = [math.sqrt(v.max() / v.min())
                     for v in (np.linalg.eigvalsh(o["R"]) for o in mine)]
        errors = np.array([o["fixed"] - o["truth"] for o in mine])
        centred = errors - errors.mean(axis=0)
        along_axes = np.array([bearing(models, camera, o["fixed"]) for o in mine])
        along = np.array([float(centred[i] @ along_axes[i]) for i in range(len(mine))])
        across = np.array([float(centred[i] @ np.array([-along_axes[i, 1],
                                                        along_axes[i, 0]]))
                           for i in range(len(mine))])
        entry = {"n": len(mine), "predicted_anisotropy": float(np.median(predicted)),
                 "measured_anisotropy": float(along.std() / across.std()),
                 "sd_along_cm": float(100 * along.std()),
                 "sd_across_cm": float(100 * across.std()),
                 "lean_mm": float(1000 * np.linalg.norm(errors.mean(axis=0)))}
        summary["per_camera"][camera] = entry
        print(f"  {SHORT[camera]:>5s}{entry['n']:>6d}{entry['predicted_anisotropy']:>11.2f}"
              f"{entry['measured_anisotropy']:>10.2f}{entry['sd_along_cm']:>10.2f}"
              f"{entry['sd_across_cm']:>11.2f}{entry['lean_mm']:>9.1f}")

    for path in ("fixed", "raw"):
        print(f"\n  2. {path}: does the pair's leftover lean follow b·cos(angle/2)?")
        print(f"  {'bin':>10s}{'n':>5s}{'angle':>7s}{'lean one':>10s}{'cos law':>9s}"
              f"{'averaged':>10s}{'R-weighted':>11s}{'scatter one':>13s}"
              f"{'scatter pair':>14s}{'total ratio':>13s}")
        for entry in bins:
            e = entry[path]
            label = f"{entry['lo_deg']:.0f}-{entry['hi_deg']:.0f}"
            print(f"  {label:>10s}"
                  f"{entry['n']:>5d}{entry['median_sep_deg']:>7.0f}"
                  f"{e['lean_one_mm']:>10.1f}{e['lean_cos_law_mm']:>9.1f}"
                  f"{e['lean_pair_mm']:>10.1f}{e['lean_pair_weighted_mm']:>11.1f}"
                  f"{e['scatter_one_mm']:>13.1f}"
                  f"{e['scatter_pair_mm']:>14.1f}{e['ratio_weighted']:>13.2f}")

    for entry in bins:
        summary["by_angle"].append({k: v for k, v in entry.items() if k != "rows"})

    print("\n  3. which pairs this warehouse even offers")
    for pair in sorted({r["pair"] for r in rows}):
        seps = [r["sep_deg"] for r in rows if r["pair"] == pair]
        summary["pairs"][pair] = {"n": len(seps),
                                  "median_sep_deg": float(np.median(seps)),
                                  "p5_deg": float(np.percentile(seps, 5)),
                                  "p95_deg": float(np.percentile(seps, 95))}
        print(f"    {pair}: n={len(seps):3d}  crossing "
              f"{np.percentile(seps, 5):3.0f}-{np.percentile(seps, 95):3.0f}deg "
              f"(median {np.median(seps):3.0f}deg)")

    print("\n  4. range-matched check: both cameras 6-11 m away, so only the angle moves")
    band = [r for r in rows if 6.0 <= r["range_m"] <= 11.0]
    summary["range_matched"] = []
    for lo, hi in ((45.0, 90.0), (90.0, 135.0), (135.0, 180.01)):
        sel = [r for r in band if lo <= r["sep_deg"] < hi]
        if len(sel) < MIN_PAIRS_PER_BIN:
            continue
        entry = {"lo_deg": lo, "hi_deg": min(hi, 180.0), "n": len(sel),
                 "median_range_m": float(np.median([r["range_m"] for r in sel])),
                 "error_one_cm": float(100 * np.median([r["fixed"]["one"] for r in sel])),
                 "error_fused_cm": float(100 * np.median(
                     [r["fixed"]["weighted"] for r in sel])),
                 "ratio": float(np.median(
                     [r["fixed"]["weighted"] / r["fixed"]["one"] for r in sel]))}
        summary["range_matched"].append(entry)
        print(f"    {lo:3.0f}-{min(hi, 180.0):3.0f}deg  n={entry['n']:3d}  range "
              f"{entry['median_range_m']:.1f} m   one camera "
              f"{entry['error_one_cm']:.2f} cm -> pair {entry['error_fused_cm']:.2f} cm "
              f"(x{entry['ratio']:.2f})")

    figure = plt.figure(figsize=(14.0, 10.0))
    grid = figure.add_gridspec(2, 2, hspace=0.52, wspace=0.30,
                               left=0.062, right=0.985, top=0.875, bottom=0.175)
    panel_one_camera(figure.add_subplot(grid[0, 0]), models, observations)
    panel_two_laws(figure.add_subplot(grid[0, 1]), bins)
    panel_absolute(figure.add_subplot(grid[1, 0]), bins)
    panel_what_you_get(figure.add_subplot(grid[1, 1]), bins)

    figure.suptitle("A second camera helps in proportion to how differently it looks at "
                    "the robot", fontsize=15, y=0.972)
    figure.text(0.5, 0.918,
                "Real captures throughout: the balanced set-pose grid, "
                f"{len(observations)} clear detections, {len(rows)} camera pairs that see "
                "the same held pose, current floor-plane projection. The error shapes are "
                "pure geometry — one pixel-noise number, nothing fitted. Ground truth is "
                "the commanded pose and only ever scores.",
                ha="center", fontsize=9, color="#555555")
    figure.text(0.5, 0.014,
                "The crossing angle is not free: the two same-wall pairs (A–C, B–D) are "
                "the only narrow-crossing ones this layout offers, and they sit at longer "
                "range. Repeating the comparison at matched range (~10 m) gives the same "
                "answer — narrow pairs ×1.03, head-on pairs ×0.37. The random-part points "
                "sit above the ideal curve because the two cameras' random errors are not "
                "fully independent either.",
                ha="center", fontsize=8.3, color="#666666")

    for extension in ("png", "pdf"):
        figure.savefig(OUT / f"fig_two_cameras_correlation.{extension}",
                       bbox_inches="tight", facecolor="white")
    plt.close(figure)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  wrote {OUT}/fig_two_cameras_correlation.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
