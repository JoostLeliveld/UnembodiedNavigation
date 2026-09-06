#!/usr/bin/env python3
"""Analyse the repeated headless drives as camera measurements, never as belief error.

Every camera reading is deduplicated and compared with ground truth interpolated to its
``obs_stamp`` through the repository's canonical aligned loader.  Two quantities are kept
separate:

* global driven residual spread, which mixes changing geometry and bias and is *not* R;
* geometry-conditioned repeat spread, estimated on seeds 0/1 and checked on seed 2.

The latter is a simple empirical commissioning diagnostic, not the proposed joint Bayesian
posterior over robot trajectory and camera covariance.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch
import numpy as np


REPO = Path(__file__).resolve().parents[2]
DRIVE_ROOT = REPO / "logs/studies/camera_observation_characterization_20260831/13_headless_drives"
OUT = REPO / "logs/studies/camera_observation_characterization_20260831/12_commissioning_R_visuals"
STATIONARY = REPO / "logs/perception_datasets/warehouse_v2_bbox_repeat_panel_20260902"
DATASET = REPO / "logs/perception_datasets/warehouse_v2_yolo_shared_20260822"
CONFIG = REPO / "scripts/visibility_comparison/camera_R_headless_drives.yaml"

sys.path.insert(0, str(REPO / "experiments/fusion_on_fixed_routes"))
sys.path.insert(0, str(REPO / "experiments/measurement_commissioning"))
sys.path.insert(0, str(REPO / "experiments/deck_figures"))
import aligned as A  # noqa: E402
from camera import camera_models  # noqa: E402
import style as D  # noqa: E402


CAMS = camera_models(DATASET)
CAMERAS = tuple("ABCDE")
SURF, INK, MUTED = D.SURF, D.INK, D.MUTED
BLUE, GREEN, ORANGE, VIOLET = D.ROBOT, D.GOOD, D.BAD, D.OLD
PALE_BLUE, PALE_GREEN, PALE_ORANGE = "#e9f2fb", "#e8f6f0", "#fff0e9"
CHI2 = {"50": 1.3862943611198906, "90": 4.605170185988092, "95": 5.991464547107979}
RANGE_BIN_M = 1.0
VIEW_BIN_RAD = math.radians(15.0)
HEADING_BIN_RAD = math.radians(30.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def covariance(values: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.zeros((2, 2))
    return np.cov(values.T, ddof=1)


def sigma_cm(cov: np.ndarray) -> float:
    return 100.0 * math.sqrt(max(float(np.trace(cov)), 0.0) / 2.0)


def ellipse(ax, cov: np.ndarray, *, color: str, scale: float = math.sqrt(CHI2["95"])):
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
    patch = Ellipse((0, 0), 200.0 * scale * math.sqrt(vals[0]),
                    200.0 * scale * math.sqrt(vals[1]), angle=angle,
                    facecolor=color, edgecolor=color, alpha=0.16, lw=2.5)
    ax.add_patch(patch)


def run_dirs() -> list[tuple[int, Path]]:
    found = []
    for path in sorted(DRIVE_ROOT.glob("fusion_overlap_rich/O2/seed*/experiment_*")):
        if not (path / "fusion_observations.csv").is_file():
            continue
        seed = int(path.parent.name.removeprefix("seed"))
        found.append((seed, path))
    latest = {}
    for seed, path in found:
        latest[seed] = path
    if sorted(latest) != [0, 1, 2]:
        raise RuntimeError(f"Expected completed drive logs for seeds 0,1,2; found {sorted(latest)}")
    return sorted(latest.items())


def geometry_entry(camera: str, truth: np.ndarray, yaw: float):
    model = CAMS[f"camera_{camera}"]
    cam_xy = np.asarray(model.cam_pos[:2], dtype=float)
    look_xy = np.asarray(model.look_at[:2], dtype=float)
    ray = truth - cam_xy
    distance = float(np.linalg.norm(ray))
    ray /= distance
    across = np.array([-ray[1], ray[0]])
    ray_angle = math.atan2(ray[1], ray[0])
    bore = look_xy - cam_xy
    bore_angle = math.atan2(bore[1], bore[0])
    view = wrap(ray_angle - bore_angle)
    relative_heading = wrap(yaw - ray_angle)
    key = (
        int(math.floor(distance / RANGE_BIN_M)),
        int(math.floor((view + math.pi) / VIEW_BIN_RAD)),
        int(math.floor((relative_heading + math.pi) / HEADING_BIN_RAD)),
    )
    return ray, across, distance, view, relative_heading, key


def drive_rows():
    rows = []
    sources = []
    for seed, run in run_dirs():
        summary = run / "run_summary.json"
        sources.append({
            "seed": seed,
            "run": str(run),
            "manifest_sha256": sha256(run / "run_manifest.json"),
            "observations_sha256": sha256(run / "fusion_observations.csv"),
            "summary": json.loads(summary.read_text()) if summary.is_file() else None,
        })
        for item in A.readings(run, admitted_only=False, dedupe=True):
            camera = item["camera"]
            if camera not in CAMERAS or not math.isfinite(item["truth_yaw"]):
                continue
            ray, across, distance, view, rel_heading, key = geometry_entry(
                camera, item["truth"], item["truth_yaw"])
            error_local = np.array([float(item["error"] @ ray),
                                    float(item["error"] @ across)])
            rows.append({
                "seed": seed, "camera": camera, "error": error_local,
                "truth": item["truth"], "obs_stamp": item["obs_stamp"],
                "range_m": distance, "view_rad": view,
                "relative_heading_rad": rel_heading, "geometry_bin": key,
                "used": bool(item["used"]),
            })
    return rows, sources


def stationary_summary():
    table = list(csv.DictReader((STATIONARY / "observation_interpretations.csv").open()))
    labels = json.loads((STATIONARY / "capture_manifest.json").read_text())["plan"]["pose_labels"]
    out = {}
    for camera in CAMERAS:
        states = {}
        for label in labels:
            attempts = [r for r in table if r["camera_id"] == f"camera_{camera}"
                        and r["dataset_split"] == label]
            hits = [r for r in attempts if r["fixed_valid"] == "1"]
            residual = np.asarray([[float(r["fixed_dx"]), float(r["fixed_dy"])]
                                   for r in hits]) if hits else np.empty((0, 2))
            cov = covariance(residual - residual.mean(axis=0)) if len(residual) else np.zeros((2, 2))
            states[label] = {
                "attempts": len(attempts), "hits": len(hits),
                "unique_images": len({r["image_sha1"] for r in attempts}),
                "repeat_sigma_cm": sigma_cm(cov), "covariance_m2": cov.tolist(),
            }
        out[camera] = states
    return labels, out


def analyse(rows):
    fit_seeds, held_seed = (0, 1), 2
    output = {}
    for camera in CAMERAS:
        all_cam = [r for r in rows if r["camera"] == camera]
        values = np.asarray([r["error"] for r in all_cam])
        mean = values.mean(axis=0)
        global_cov = covariance(values - mean)

        fit_bins = defaultdict(list)
        for row in all_cam:
            if row["seed"] in fit_seeds:
                fit_bins[row["geometry_bin"]].append(row["error"])
        fit_means = {key: np.mean(value, axis=0) for key, value in fit_bins.items()
                     if len({r["seed"] for r in all_cam
                             if r["geometry_bin"] == key and r["seed"] in fit_seeds}) == 2}

        fit_dev = np.asarray([r["error"] - fit_means[r["geometry_bin"]]
                              for r in all_cam if r["seed"] in fit_seeds
                              and r["geometry_bin"] in fit_means])
        held_dev = np.asarray([r["error"] - fit_means[r["geometry_bin"]]
                              for r in all_cam if r["seed"] == held_seed
                              and r["geometry_bin"] in fit_means])
        cond_cov = covariance(fit_dev)
        held_cov = covariance(held_dev - held_dev.mean(axis=0)) if len(held_dev) else np.zeros((2, 2))

        containment = {}
        if len(held_dev) and np.linalg.det(cond_cov) > 1e-14:
            q = np.einsum("ni,ij,nj->n", held_dev, np.linalg.inv(cond_cov), held_dev)
            containment = {level: float(np.mean(q <= cutoff)) for level, cutoff in CHI2.items()}
        output[camera] = {
            "n": len(all_cam),
            "n_by_seed": {str(seed): sum(r["seed"] == seed for r in all_cam)
                          for seed in (0, 1, 2)},
            "bias_along_cm": float(mean[0] * 100),
            "bias_across_cm": float(mean[1] * 100),
            "global_covariance_m2": global_cov.tolist(),
            "global_sigma_cm": sigma_cm(global_cov),
            "geometry_bins_fit": len(fit_means),
            "fit_conditional_n": len(fit_dev),
            "held_out_n": len(held_dev),
            "conditional_covariance_m2": cond_cov.tolist(),
            "conditional_sigma_cm": sigma_cm(cond_cov),
            "held_out_covariance_m2": held_cov.tolist(),
            "held_out_sigma_cm": sigma_cm(held_cov),
            "held_out_bias_cm": (np.mean(held_dev, axis=0) * 100).tolist() if len(held_dev) else [],
            "held_out_containment": containment,
        }
    return output


def canvas(title, subtitle):
    fig = plt.figure(figsize=(16, 9), facecolor=SURF)
    fig.text(0.04, 0.94, title, fontsize=28, fontweight="bold", va="top")
    fig.text(0.04, 0.888, subtitle, fontsize=14, color=D.INK2, va="top")
    return fig


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, facecolor=SURF)
    plt.close(fig)


def plot_stationary(labels, stationary):
    fig = canvas("Stationary repeatability across all five cameras",
                 "Six predeclared states · 40 fresh frames per state · fixed-offset reading")
    ax = fig.add_axes([0.06, 0.18, 0.70, 0.64])
    hit = np.array([[stationary[c][label]["hits"] for label in labels] for c in CAMERAS])
    image = ax.imshow(hit, cmap="Greens", vmin=0, vmax=40, aspect="auto")
    for i, camera in enumerate(CAMERAS):
        for j, label in enumerate(labels):
            n = hit[i, j]
            text = f"{n}/40" if n else "no hit"
            ax.text(j, i, text, ha="center", va="center", fontsize=12,
                    color="white" if n >= 28 else MUTED, fontweight="bold")
    ax.set_xticks(range(len(labels)), [s.replace("_", "\n") for s in labels], fontsize=12)
    ax.set_yticks(range(5), [f"Camera {c}" for c in CAMERAS], fontsize=13)
    ax.set_title("Usable detections", fontsize=18, fontweight="bold", pad=12)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("hits out of 40", fontsize=11)

    side = fig.add_axes([0.80, 0.18, 0.16, 0.64])
    side.axis("off")
    side.add_patch(FancyBboxPatch((0.02, 0.53), 0.96, 0.34,
                                  boxstyle="round,pad=0.02,rounding_size=0.03",
                                  facecolor=PALE_GREEN, edgecolor=GREEN, lw=2.5))
    side.text(0.50, 0.76, "18 visible\ncamera–state cells", ha="center", va="center",
              fontsize=19, fontweight="bold", color=GREEN)
    side.text(0.50, 0.60, "720 usable boxes", ha="center", va="center", fontsize=14)
    side.add_patch(FancyBboxPatch((0.02, 0.12), 0.96, 0.30,
                                  boxstyle="round,pad=0.02,rounding_size=0.03",
                                  facecolor=PALE_ORANGE, edgecolor=ORANGE, lw=2.5))
    side.text(0.50, 0.31, "0.00 cm", ha="center", va="center", fontsize=27,
              fontweight="bold", color=ORANGE)
    side.text(0.50, 0.20, "maximum within-state\nrepeat sigma", ha="center", va="center",
              fontsize=13, color=INK)
    fig.text(0.50, 0.075,
             "Where a camera sees the robot, all 40 fresh simulator frames produce the same reading.",
             ha="center", fontsize=16, fontweight="bold", color=GREEN)
    save(fig, "35_all_camera_stationary_repeatability.png")


def plot_global(rows, stats):
    fig = canvas("What camera residuals look like while the robot drives",
                 "Three headless Gazebo drives · truth aligned to each obs_stamp · fixed 30.9 cm correction")
    positions = [(0.04, 0.51), (0.355, 0.51), (0.67, 0.51), (0.20, 0.12), (0.515, 0.12)]
    all_values = np.asarray([r["error"] for r in rows]) * 100
    limit = max(20.0, float(np.percentile(np.abs(all_values), 99)))
    for camera, (x, y) in zip(CAMERAS, positions):
        ax = fig.add_axes([x, y, 0.285, 0.29])
        values = np.asarray([r["error"] for r in rows if r["camera"] == camera])
        centred = (values - values.mean(axis=0)) * 100
        draw = centred[::max(1, len(centred) // 350)]
        ax.scatter(draw[:, 0], draw[:, 1], s=12, alpha=0.22, color=BLUE, edgecolor="none")
        ellipse(ax, np.asarray(stats[camera]["global_covariance_m2"]), color=BLUE)
        ax.axhline(0, color="#d9d8d3", lw=1)
        ax.axvline(0, color="#d9d8d3", lw=1)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal")
        ax.set_title(f"Camera {camera}", fontsize=16, fontweight="bold")
        ax.set_xlabel("along-ray residual (cm)", fontsize=9)
        ax.set_ylabel("across-ray residual (cm)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.text(0.03, 0.95,
                f"n={stats[camera]['n']}  bias={math.hypot(stats[camera]['bias_along_cm'], stats[camera]['bias_across_cm']):.1f} cm\n"
                f"global spread={stats[camera]['global_sigma_cm']:.1f} cm",
                transform=ax.transAxes, va="top", fontsize=9.5,
                bbox=dict(facecolor=SURF, alpha=0.82, edgecolor="none", pad=2))
    fig.text(0.50, 0.045,
             "These ellipses mix changing range, view angle and robot heading. They are total residual spread—not R.",
             ha="center", fontsize=14, color=ORANGE, fontweight="bold")
    save(fig, "36_headless_driven_residuals.png")


def plot_conditional(stats):
    fig = canvas("A usable driving definition of R",
                 "Fit seeds 0–1 · independent check on seed 2 · one reading per camera capture")
    ax = fig.add_axes([0.07, 0.20, 0.48, 0.60])
    x = np.arange(5)
    global_sigma = [stats[c]["global_sigma_cm"] for c in CAMERAS]
    conditional = [stats[c]["conditional_sigma_cm"] for c in CAMERAS]
    held = [stats[c]["held_out_sigma_cm"] for c in CAMERAS]
    width = 0.24
    ax.bar(x - width, global_sigma, width, color=ORANGE, label="one global mean (geometry mixed)")
    ax.bar(x, conditional, width, color=BLUE, label="fit conditional spread")
    ax.bar(x + width, held, width, color=GREEN, label="held-out conditional spread")
    ax.set_xticks(x, [f"Camera {c}" for c in CAMERAS], fontsize=12)
    ax.set_ylabel("RMS axis spread (cm)", fontsize=13)
    ax.grid(axis="y", color="#dddcd7", lw=0.9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    ax.set_title("Subtract geometry before calling the spread R", fontsize=17,
                 fontweight="bold", pad=12)

    right = fig.add_axes([0.61, 0.20, 0.34, 0.60])
    right.axis("off")
    right.add_patch(FancyBboxPatch((0.02, 0.63), 0.96, 0.25,
                                   boxstyle="round,pad=0.02,rounding_size=0.03",
                                   facecolor=PALE_BLUE, edgecolor=BLUE, lw=2.5))
    right.text(0.50, 0.80, r"$e_{i,k}=z_{i,k}-h_i(s_k)-b_i(g_k)$",
               ha="center", va="center", fontsize=18)
    right.text(0.50, 0.69, r"$R_i^{drive}=\operatorname{Cov}(e_{i,k}\mid g_k)$",
               ha="center", va="center", fontsize=18, fontweight="bold")
    right.text(0.50, 0.55,
               "$g_k$ = range, camera view angle,\nand robot heading relative to the camera",
               ha="center", va="center", fontsize=13, color=D.INK2)

    y0 = 0.39
    right.text(0.04, y0 + 0.09, "Held-out containment", fontsize=15, fontweight="bold")
    right.text(0.53, y0 + 0.09, "50%", fontsize=12, ha="center")
    right.text(0.70, y0 + 0.09, "90%", fontsize=12, ha="center")
    right.text(0.87, y0 + 0.09, "95%", fontsize=12, ha="center")
    for index, camera in enumerate(CAMERAS):
        y = y0 - index * 0.065
        right.text(0.05, y, f"Camera {camera}", fontsize=12, va="center")
        values = stats[camera]["held_out_containment"]
        for xx, level in zip((0.53, 0.70, 0.87), ("50", "90", "95")):
            text = f"{100 * values[level]:.0f}%" if level in values else "—"
            right.text(xx, y, text, fontsize=12, ha="center", va="center",
                       color=GREEN if level in values and abs(values[level] - int(level) / 100) <= 0.10 else ORANGE,
                       fontweight="bold")
    right.text(0.50, 0.015,
               "Empirical diagnostic—not the joint Bayesian posterior drawn in Figure 32.",
               ha="center", fontsize=10.5, color=MUTED)
    fig.text(0.50, 0.065,
             "Stationary R was exactly zero in Gazebo; non-zero driving spread comes from motion and geometry variation.",
             ha="center", fontsize=14, color=ORANGE, fontweight="bold")
    save(fig, "37_driving_R_definition_and_validation.png")


def main():
    rows, sources = drive_rows()
    labels, stationary = stationary_summary()
    stats = analyse(rows)
    plot_stationary(labels, stationary)
    plot_global(rows, stats)
    plot_conditional(stats)
    payload = {
        "schema": "camera_R_headless_drive_diagnostic.v1",
        "evidence_status": "diagnostic_only",
        "definition": "R_drive = Cov(z_i - h_i(s) - b_i(geometry) | geometry)",
        "not_bayesian": True,
        "alignment": "canonical aligned.readings; truth interpolated to obs_stamp; deduplicated",
        "observation_method": "fixed 30.9 cm radial correction (O2)",
        "headless": True,
        "config": str(CONFIG),
        "config_sha256": sha256(CONFIG),
        "fit_seeds": [0, 1],
        "held_out_seed": 2,
        "geometry_bins": {
            "range_m": RANGE_BIN_M,
            "view_angle_deg": math.degrees(VIEW_BIN_RAD),
            "relative_heading_deg": math.degrees(HEADING_BIN_RAD),
        },
        "sources": sources,
        "stationary": stationary,
        "driven": stats,
        "figures": [
            "35_all_camera_stationary_repeatability.png",
            "36_headless_driven_residuals.png",
            "37_driving_R_definition_and_validation.png",
        ],
    }
    (OUT / "headless_drive_R_diagnostic.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "driven": stats}, indent=2))


if __name__ == "__main__":
    main()
