#!/usr/bin/env python3
"""The failure cases as the camera actually saw them.

Every other figure in this study reduces a reading to a number or an arrow. This one shows the
photograph the detector was given, the box it returned, and the box the analytic robot hull
says it should have returned. The gap between those two boxes, multiplied by how much a pixel
is worth at that range, is the whole error.

Cases are chosen by the cause diagnosed in figure 10, and each row is the reading closest to
that category's median error, so nothing is cherry-picked for drama:

    clear sightline, large box   the camera working normally
    clear sightline, small box   the robot far away and only a few pixels tall
    a rack in the way            the detector boxing the visible sliver
    box against the image edge   the robot half out of frame

The yellow box is what YOLO returned and the yellow dot is the bottom-centre pixel that becomes
the reading. The green box is the analytic hull projected from the TRUE pose -- where the box
should have been. Both the pixel gap and the floor error it produces are printed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
for rel in ("experiments/deck_figures", "experiments/camera_observation_characterization",
            "experiments/measurement_commissioning", "src/experiments", "src/unav_common"):
    value = str((REPO / rel).resolve())
    if value not in sys.path:
        sys.path.insert(0, value)

import style as D  # noqa: E402
from _sightline import blocked, camera_positions, focal_pixels, metres_per_pixel, rack_footprints  # noqa: E402
from experiments.core.world_profiles import compute_look_at_from_pose  # noqa: E402
from observation import predicted_box  # noqa: E402
from unav_common.camera_model import ObliqueCameraModel  # noqa: E402

DEFAULT_CAPTURE = REPO / "logs/perception_datasets/warehouse_v2_bbox_characterization_20260831"
DEFAULT_OUT = REPO / "logs/studies/camera_observation_characterization_20260831"
FOLDER = "04_why_readings_fail"
SMALL_BOX_PX = 25.0
EDGE_MARGIN_PX = 8.0
CROP_PAD_PX = 95.0

CASES = (
    ("clear_large", "Working normally", "clear sightline, box over 25 px", "#1baf7a"),
    ("clear_small", "Too far away", "clear sightline, box under 25 px", "#eb6834"),
    ("occluded", "A rack in the way", "sightline crosses a rack footprint", "#c63131"),
    ("edge", "Half out of frame", "box within 8 px of an image edge", "#4a3aa7"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def camera_models(capture_manifest: dict, profiles: dict) -> dict:
    models = {}
    for item in capture_manifest["cameras"]:
        pose = [float(value) for value in item["pose_xyz_rpy"]]
        models[item["camera_id"]] = ObliqueCameraModel(
            cam_pos=pose[:3], look_at=compute_look_at_from_pose(pose[:3], *pose[3:]),
            img_width=int(item["image_width"]), img_height=int(item["image_height"]),
            fov_h_rad=float(profiles["camera_intrinsics"]["fov_h_rad"]))
    return models


def classify(row: dict, width: float, height: float, racks, cameras) -> str:
    x0, y0, x1, y1 = (float(row[key]) for key in ("x0", "y0", "x1", "y1"))
    if min(x0, y0, width - x1, height - y1) < EDGE_MARGIN_PX:
        return "edge"
    if blocked(row["camera_id"][-1], float(row["robot_x"]), float(row["robot_y"]),
               racks=racks, cameras=cameras):
        return "occluded"
    return "clear_small" if (y1 - y0) < SMALL_BOX_PX else "clear_large"


def pick(rows: list[dict], camera_id: str, width: float, height: float) -> dict:
    """One representative reading per cause: the median-error member of each group."""
    racks, cameras = rack_footprints(), camera_positions()
    groups: dict[str, list[dict]] = {key: [] for key, _t, _s, _c in CASES}
    for row in rows:
        if row["camera_id"] != camera_id or row["raw_valid"] != "1" or row["hull_valid"] != "1":
            continue
        groups[classify(row, width, height, racks, cameras)].append(row)
    chosen = {}
    for key, group in groups.items():
        if not group:
            continue
        errors = np.asarray([float(r["hull_error_m"]) for r in group])
        chosen[key] = {"row": group[int(np.argsort(errors)[len(errors) // 2])],
                       "n": len(group),
                       "median_error_m": float(np.median(errors))}
    return chosen


def draw(chosen: dict, capture: Path, models: dict, focal: float, camera_id: str,
         out: Path) -> dict:
    shown = [(key, title, subtitle, colour) for key, title, subtitle, colour in CASES
             if key in chosen]
    fig, axes = plt.subplots(2, len(shown), figsize=(6.2 * len(shown), 12.4),
                             gridspec_kw={"height_ratios": (1.42, 1.0)},
                             constrained_layout=True)
    if len(shown) == 1:
        axes = axes.reshape(2, 1)

    report = {}
    for column, (key, title, subtitle, colour) in enumerate(shown):
        entry = chosen[key]
        row = entry["row"]
        image = cv2.imread(str(capture / row["image"]))
        if image is None:
            raise RuntimeError(f"Missing capture image: {row['image']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        x0, y0, x1, y1 = (float(row[key2]) for key2 in ("x0", "y0", "x1", "y1"))
        u, v = float(row["u_bbox_bottom"]), float(row["v_bbox_bottom"])
        cam = models[camera_id]
        truth_box = predicted_box(cam, float(row["robot_x"]), float(row["robot_y"]),
                                  float(row["robot_yaw"]))
        range_m = float(row["camera_range_m"])
        cm_per_px = 100 * metres_per_pixel(range_m, focal)

        # ---- top: the photograph, cropped around the robot -----------------------------
        ax = axes[0, column]
        ax.imshow(image)
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#ffdd00",
                                   lw=2.6, zorder=5))
        ax.scatter([u], [v], s=90, c="#ffdd00", ec=D.INK, lw=1.3, zorder=7)
        dv = math.nan
        if truth_box is not None:
            tx0, ty0, tx1, ty1 = truth_box
            ax.add_patch(plt.Rectangle((tx0, ty0), tx1 - tx0, ty1 - ty0, fill=False,
                                       ec="#1baf7a", lw=2.2, linestyle="--", zorder=6))
            ax.scatter([0.5 * (tx0 + tx1)], [ty1], s=70, c="#1baf7a", ec="white", lw=1.0,
                       zorder=7)
            dv = v - ty1
            ax.annotate("", xy=(u, v), xytext=(0.5 * (tx0 + tx1), ty1),
                        arrowprops=dict(arrowstyle="-|>", color="#c63131", lw=2.4), zorder=8)
        centre_x, centre_y = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        span = max(CROP_PAD_PX, 0.9 * max(x1 - x0, y1 - y0) + CROP_PAD_PX)
        ax.set_xlim(max(0, centre_x - span * 1.35), min(image.shape[1], centre_x + span * 1.35))
        ax.set_ylim(min(image.shape[0], centre_y + span), max(0, centre_y - span))
        ax.set_axis_off()
        ax.set_title(f"{title}\n{subtitle}", fontsize=14.5, fontweight="bold", color=colour)

        # ---- bottom: what that pixel gap becomes on the floor --------------------------
        ax = axes[1, column]
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")
        hull_cm = 100 * float(row["hull_error_m"])
        raw_cm = 100 * float(row["raw_error_m"])
        nn_cm = 100 * float(row["nn_error_m"]) if row["nn_valid"] == "1" else math.nan
        lines = [
            ("box size", f"{x1 - x0:.0f} x {y1 - y0:.0f} px"),
            ("range to camera", f"{range_m:.1f} m"),
            ("one pixel is worth", f"{cm_per_px:.1f} cm on the floor"),
            ("box bottom off by", f"{dv:+.1f} px" if math.isfinite(dv) else "n/a"),
            ("YOLO confidence", f"{float(row['confidence']):.2f}"),
            ("", ""),
            ("hull reading misses by", f"{hull_cm:.1f} cm"),
            ("raw reading misses by", f"{raw_cm:.1f} cm"),
            ("neural reading misses by",
             f"{nn_cm:.1f} cm" if math.isfinite(nn_cm) else "n/a"),
        ]
        ax.text(0.0, 1.0, f"{entry['n']} readings like this on camera {camera_id[-1]}\n"
                          f"median error for the group: {100 * entry['median_error_m']:.1f} cm",
                transform=ax.transAxes, fontsize=12.5, fontweight="bold", va="top",
                color=colour)
        y = 0.80
        for label, value in lines:
            if label:
                ax.text(0.0, y, label, transform=ax.transAxes, fontsize=12, va="top",
                        color=D.INK2)
                ax.text(1.0, y, value, transform=ax.transAxes, fontsize=12, va="top",
                        ha="right", fontweight="bold" if "misses" in label else "normal")
            y -= 0.083
        for line_y in (0.845, 0.325):
            ax.plot([0, 1], [line_y, line_y], color="#d0cec7", lw=1.0,
                    transform=ax.transAxes, clip_on=False)

        report[key] = {
            "pose_id": row["pose_id"], "image": row["image"],
            "group_size": entry["n"],
            "group_median_hull_cm": 100 * entry["median_error_m"],
            "box_px": [x1 - x0, y1 - y0], "range_m": range_m,
            "cm_per_pixel": cm_per_px,
            "box_bottom_offset_px": None if not math.isfinite(dv) else dv,
            "confidence": float(row["confidence"]),
            "hull_cm": hull_cm, "raw_cm": raw_cm,
            "nn_cm": None if not math.isfinite(nn_cm) else nn_cm,
        }

    handles = [
        Line2D([0], [0], color="#ffdd00", lw=3, label="what YOLO returned"),
        Line2D([0], [0], color="#1baf7a", lw=3, linestyle="--",
               label="where the analytic hull says the box should be, from the TRUE pose"),
        Line2D([0], [0], color="#c63131", lw=3, label="the gap that becomes the reading error"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=12.5, frameon=True,
               bbox_to_anchor=(0.5, -0.022))
    fig.suptitle(
        f"Camera {camera_id[-1]} — the failure cases as the camera saw them\n"
        "One reading per cause, each the median-error member of its group, so none is "
        "picked for drama\n"
        "A few pixels of box error become decimetres of floor error once range and tilt "
        "amplify them",
        fontsize=19, fontweight="bold")
    path = out / FOLDER / f"11_failure_cases_in_the_image_{camera_id}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--camera", action="append",
                        help="Camera id to draw; repeat. Defaults to D (worst) and B (best).")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    capture = args.capture.expanduser().resolve()
    out = args.out.expanduser().resolve()
    wanted = args.camera or ["camera_D", "camera_B"]

    capture_manifest = json.loads((capture / "capture_manifest.json").read_text(encoding="utf-8"))
    profiles = yaml.safe_load(Path(capture_manifest["world_profiles_path"]).read_text(encoding="utf-8"))
    models = camera_models(capture_manifest, profiles)
    focal = focal_pixels(capture_manifest, profiles)
    width = float(capture_manifest["cameras"][0]["image_width"])
    height = float(capture_manifest["cameras"][0]["image_height"])

    table = capture / "bias_update_interpretations.csv"
    rows = list(csv.DictReader(table.open(encoding="utf-8")))

    report = {}
    for camera_id in wanted:
        path = out / FOLDER / f"11_failure_cases_in_the_image_{camera_id}.png"
        if path.exists() and not args.overwrite:
            raise RuntimeError(f"Output already exists: {path}; pass --overwrite")
        chosen = pick(rows, camera_id, width, height)
        if not chosen:
            raise RuntimeError(f"No usable readings for {camera_id}")
        report[camera_id] = draw(chosen, capture, models, focal, camera_id, out)

    manifest = {
        "status": "complete",
        "schema": "camera_failure_gallery.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_table_sha256": sha256(table),
        "selection_rule": (
            "readings are grouped by diagnosed cause, and each panel shows the group's "
            "median-error member, so no panel is chosen for effect"
        ),
        "case_definitions": {
            "edge": f"box within {EDGE_MARGIN_PX:.0f} px of an image edge",
            "occluded": "sightline from camera to true position crosses a rack footprint",
            "clear_small": f"clear sightline, box height under {SMALL_BOX_PX:.0f} px",
            "clear_large": f"clear sightline, box height at least {SMALL_BOX_PX:.0f} px",
        },
        "reference_box": (
            "the analytic hull projected from the TRUE pose; an oracle, shown to expose what "
            "the detector's box got wrong, never available at runtime"
        ),
        "cameras": report,
        "figures": [f"11_failure_cases_in_the_image_{camera}.png" for camera in wanted],
    }
    (out / FOLDER / "failure_gallery_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out / FOLDER), "cameras": wanted}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
