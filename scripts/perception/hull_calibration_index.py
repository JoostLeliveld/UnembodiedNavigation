"""Rebuild the localization-calibration index from the robot's VISUAL HULL.

Why the old index cannot be used
--------------------------------
`capture_yolo_dataset` judges occlusion against a single bounding PRISM. It was given
the AMR's new dimensions (0.80 x 0.55 x 0.35, robot_z 0) but kept the prism shape, and
a prism is not this robot: the real visual hull reaches 0.400 m at the rear cabinet,
spans x in [-0.406, +0.407] at the bumper and sensor bar, and its lowest contact is the
wheel and caster tangent at z = 0. Judging a silhouette against the wrong solid gives
`bottom_occlusion_px` a floor of about +3 px and caps `visible_height_fraction` below 1
even with nothing in the way, so genuinely occluded sightings pass as clean.

Measured consequence on warehouse_v2_yolo_20260821: of 2692 rows marked clear and
localization-qualified, 272 have their bottom row more than 3 px above the true contact
row and 346 show under 90% of the hull height. Those rows are exactly the ones a bottom
-point projection must not be calibrated on.

What this writes
----------------
`localization_calibration_index_hull.csv`, one row per accepted sample, carrying the
hull-referenced quantities and a single `calibration_eligible` verdict. The gate:

* visible height  >= 85% of the projected hull height
* bottom row      <= 3 px above the projected ground-contact row
* no contact with the image border (a truncated robot has no valid bottom point)
* visible area    >= a declared fraction of the projected hull area, which replaces the
  Burger-scale absolute pixel floors -- a fraction is range- and scale-invariant

The hull is built from the URDF, not typed in, so it follows the robot description.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/unav_common"))
sys.path.insert(0, str(REPO / "experiments/warehouse_v2_sketches"))

from unav_common.camera_model import ObliqueCameraModel  # noqa: E402

# THE hull lives in the observation-model study, which is the registered owner of the
# shape question (EXP-OBS-MODEL). This file does not define a second one -- two
# definitions of one robot is how the Burger dimensions survived into an AMR capture in
# the first place. `hull_points_from_urdf` below re-derives the same solid straight from
# the xacro and exists only to prove the transcription has not drifted; the two agree to
# 0.000 mm on every bounding-box extreme and to 0.013% in convex-hull volume.
from unav_common.robot_hull import VISUAL_HULL  # noqa: E402

XACRO = REPO / "src/sim/robot_description/urdf/warehouse_amr.urdf.xacro"
MIN_HEIGHT_FRACTION = 0.85
MAX_BOTTOM_GAP_PX = 3.0
MIN_AREA_FRACTION = 0.30


def _rpy(rpy: str | None) -> np.ndarray:
    r, p, y = (float(v) for v in (rpy or "0 0 0").split())
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def hull_points_from_urdf() -> np.ndarray:
    """Re-derive the hull from the xacro. A drift guard, not the production path."""
    urdf = subprocess.run(["xacro", str(XACRO)], capture_output=True, text=True, check=True).stdout
    root = ET.fromstring(urdf)
    joints = {}
    for j in root.findall("joint"):
        origin = j.find("origin")
        joints[j.find("child").get("link")] = (
            j.find("parent").get("link"),
            np.array([float(v) for v in (origin.get("xyz") if origin is not None else "0 0 0").split()]),
            _rpy(origin.get("rpy") if origin is not None else None),
        )

    def to_root(link):
        translation, rotation = np.zeros(3), np.eye(3)
        while link in joints:
            parent, xyz, rot = joints[link]
            translation = xyz + rot @ translation
            rotation = rot @ rotation
            link = parent
        return translation, rotation

    points = []
    for link in root.findall("link"):
        name = link.get("name")
        t_link, r_link = to_root(name)
        for vis in link.findall("visual"):
            origin = vis.find("origin")
            t_v = np.array([float(v) for v in (origin.get("xyz") if origin is not None else "0 0 0").split()])
            r_v = _rpy(origin.get("rpy") if origin is not None else None)
            geo = list(vis.find("geometry"))[0]
            local = []
            if geo.tag == "box":
                sx, sy, sz = (float(v) / 2.0 for v in geo.get("size").split())
                local = [np.array([i * sx, j * sy, k * sz])
                         for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]
            elif geo.tag == "cylinder":
                rad, half = float(geo.get("radius")), float(geo.get("length")) / 2.0
                for a in np.linspace(0, 2 * math.pi, 16, endpoint=False):
                    for h in (-half, half):
                        local.append(np.array([rad * math.cos(a), rad * math.sin(a), h]))
            elif geo.tag == "sphere":
                rad = float(geo.get("radius"))
                for a in np.linspace(0, 2 * math.pi, 12, endpoint=False):
                    for e in np.linspace(-math.pi / 2, math.pi / 2, 5):
                        local.append(rad * np.array([math.cos(e) * math.cos(a),
                                                     math.cos(e) * math.sin(a), math.sin(e)]))
            for p in local:
                points.append(t_link + r_link @ (t_v + r_v @ p))
    return np.asarray(points)


def project(camera, points_world):
    cam = (points_world - camera.cam_pos) @ camera.R.T
    ahead = cam[:, 2] > 1e-6
    if not ahead.any():
        return None
    uv = (camera.K @ cam[ahead].T).T
    return uv[:, :2] / uv[:, 2:3]


def camera_models(width, height):
    from warehouse_v2 import build
    layout = build()
    models = {}
    for cam in layout.cameras:
        yaw, pitch = math.radians(cam.yaw_deg), math.radians(cam.pitch_deg)
        d = 10.0
        look = (cam.x + d * math.cos(pitch) * math.cos(yaw),
                cam.y + d * math.cos(pitch) * math.sin(yaw),
                cam.z - d * math.sin(pitch))
        models[f"camera_{cam.name}"] = ObliqueCameraModel(
            cam_pos=(cam.x, cam.y, cam.z), look_at=look,
            img_width=width, img_height=height)
    return models


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--min-height-fraction", type=float, default=MIN_HEIGHT_FRACTION)
    ap.add_argument("--max-bottom-gap-px", type=float, default=MAX_BOTTOM_GAP_PX)
    ap.add_argument("--min-area-fraction", type=float, default=MIN_AREA_FRACTION)
    args = ap.parse_args()

    root = Path(args.dataset).expanduser().resolve()
    hull = VISUAL_HULL
    print(f"visual hull: {len(hull)} sampled points, "
          f"x [{hull[:,0].min():.3f}, {hull[:,0].max():.3f}] "
          f"y [{hull[:,1].min():.3f}, {hull[:,1].max():.3f}] "
          f"z [{hull[:,2].min():.3f}, {hull[:,2].max():.3f}] m")

    models, rows = None, []
    for cam_dir in sorted(root.glob("camera_*")):
        diag = cam_dir / "label_diagnostics.csv"
        if not diag.exists():
            continue
        cam_id = cam_dir.name
        for rec in csv.DictReader(diag.open()):
            if rec["accepted"] != "1" or rec.get("sample_kind") == "negative":
                continue
            mask_rel = rec.get("mask") or ""
            mask_path = cam_dir / mask_rel if mask_rel else None
            if mask_path is None or not mask_path.exists():
                mask_path = next(iter(cam_dir.glob(f"masks/*/{Path(rec['image']).stem}.png")), None)
            if mask_path is None or not mask_path.exists():
                continue
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            h, w = mask.shape[:2]
            if models is None:
                models = camera_models(w, h)
            camera = models.get(cam_id)
            if camera is None:
                continue
            x, y, yaw = float(rec["robot_x"]), float(rec["robot_y"]), float(rec["robot_yaw"])
            c, s = math.cos(yaw), math.sin(yaw)
            rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            world = hull @ rot.T + np.array([x, y, 0.0])
            uv = project(camera, world)
            if uv is None or len(uv) < 3:
                continue
            pred_bottom = float(uv[:, 1].max())
            pred_top = float(uv[:, 1].min())
            pred_height = pred_bottom - pred_top
            pred_area = float(cv2.contourArea(cv2.convexHull(uv.astype(np.float32))))
            ys, xs = np.nonzero(mask)
            if len(ys) == 0:
                continue
            mask_bottom, mask_top = float(ys.max()), float(ys.min())
            mask_height = mask_bottom - mask_top
            mask_area = float(len(ys))
            border = bool(xs.min() <= 0 or ys.min() <= 0 or xs.max() >= w - 1 or ys.max() >= h - 1)
            gap = pred_bottom - mask_bottom
            hf = mask_height / pred_height if pred_height > 1e-6 else float("nan")
            af = mask_area / pred_area if pred_area > 1e-6 else float("nan")
            eligible = (
                hf >= args.min_height_fraction
                and gap <= args.max_bottom_gap_px
                and not border
                and af >= args.min_area_fraction
            )
            rows.append({
                "camera_id": cam_id, "image": rec["image"], "split": rec["split"],
                "robot_x": x, "robot_y": y, "robot_yaw": yaw,
                "camera_range_m": rec.get("camera_range_m", ""),
                "pred_hull_bottom_v": round(pred_bottom, 3),
                "pred_hull_top_v": round(pred_top, 3),
                "pred_hull_height_px": round(pred_height, 3),
                "pred_hull_area_px": round(pred_area, 1),
                "mask_bottom_v": mask_bottom, "mask_top_v": mask_top,
                "mask_height_px": mask_height, "mask_area_px": mask_area,
                "bottom_gap_px": round(gap, 3),
                "height_fraction": round(hf, 4),
                "area_fraction": round(af, 4),
                "border_contact": int(border),
                "old_localization_qualified": rec.get("localization_qualified", ""),
                "calibration_eligible": int(eligible),
            })

    if not rows:
        print("no rows built")
        return 1
    out = root / "localization_calibration_index_hull.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def q(key, sub=None):
        vals = sorted(r[key] for r in (sub or rows) if isinstance(r[key], float) and r[key] == r[key])
        if not vals:
            return "n/a"
        return (f"median {vals[len(vals)//2]:7.3f}  p90 {vals[int(0.9*(len(vals)-1))]:7.3f}  "
                f"max {vals[-1]:8.3f}")
    ok = [r for r in rows if r["calibration_eligible"]]
    was_clean = [r for r in rows if str(r["old_localization_qualified"]) in ("1", "True", "true")]
    print(f"\n{len(rows)} accepted positives scored against the hull")
    print(f"  bottom gap px, all rows : {q('bottom_gap_px')}")
    print(f"  bottom gap px, eligible : {q('bottom_gap_px', ok)}")
    print(f"  height fraction, all    : {q('height_fraction')}")
    print(f"  area fraction, all      : {q('area_fraction')}")
    print(f"  border contact          : {sum(r['border_contact'] for r in rows)}")
    print(f"\n  previously localization_qualified : {len(was_clean)}")
    print(f"  hull-eligible                     : {len(ok)}")
    dropped = [r for r in was_clean if not r["calibration_eligible"]]
    gained = [r for r in ok if str(r["old_localization_qualified"]) not in ("1", "True", "true")]
    print(f"  dropped by the hull gate          : {len(dropped)}")
    print(f"  admitted that the prism rejected   : {len(gained)}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
