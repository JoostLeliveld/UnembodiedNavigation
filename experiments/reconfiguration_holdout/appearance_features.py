#!/usr/bin/env python3
"""Appearance statistics of each captured frame, as a deployed camera could compute them.

The geometric availability field answers "is there a clear sight-line to this
position".  It cannot answer "and will the detector fire on what it sees", because
brightness, contrast, glare and shadow do not move any geometry.  This script emits
the features a *conditional detection* model is allowed to use, and the boundary it
respects is what makes it usable at deployment:

ALLOWED, and used here
  * the camera's own frame -- one image, no history, no ground truth
  * the camera's calibration, to project the CANDIDATE position into the image

FORBIDDEN, and not read
  * ``oracle_bottom_u`` / ``oracle_bottom_v`` in samples.csv.  Those pixels are
    computed from the true robot pose and are evaluation-only.  The projected pixel
    used here is recomputed from the commanded candidate ``(x, y)`` and the
    calibration, which is exactly what a planner evaluating a candidate pose has.
    Numerically the two agree; the point is that the *input* is a candidate, not a
    measurement, so nothing here would be unavailable in a real deployment.
  * the detector's output, the oracle visibility flag, and the world geometry.

Two feature groups.  Frame-level statistics describe the illumination the camera is
working under at all.  Patch-level statistics, in a window centred on the projected
candidate pixel, describe the local contrast the detector would actually have to work
with there -- a robot standing in a hard shadow or inside a glare bloom is a different
detection problem from the same robot on evenly lit floor, at the same sight-line.

    python3 experiments/reconfiguration_holdout/appearance_features.py \
        --capture-dir logs/studies/reconfiguration_holdout/captures/L0_lit

Writes ``appearance_features.csv`` beside the capture's ``samples.csv``, one row per
sample, keyed by ``sample_id`` and ``camera_frame``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts/shared"))
sys.path.insert(0, str(HERE.parents[1] / "src/unav_common"))
sys.path.insert(0, str(HERE.parents[1] / "src/experiments"))
from paths import repo_root  # noqa: E402
from unav_common.camera_model import ObliqueCameraModel  # noqa: E402
from experiments.core.world_profiles import compute_look_at_from_pose  # noqa: E402

REPO = repo_root(HERE)

#: Half-width of the local window, in pixels.  64 px at 1280x720 is a few times the
#: robot's own apparent size across the useful range, so the window sees the robot's
#: immediate surroundings rather than only the robot.
PATCH_HALF_PX = 64

#: Saturation and darkness cut-offs on the 0-255 grey level.  A pixel above the first
#: carries no usable gradient because the sensor has clipped; below the second the
#: noise floor dominates.
SATURATED_LEVEL = 250
DARK_LEVEL = 20

#: The robot marker height the whole study projects at, matching the oracle's
#: target height.
TARGET_HEIGHT_M = 0.35

COLUMNS = (
    "sample_id", "camera_frame", "x", "y", "theta",
    "proj_u", "proj_v", "proj_in_image",
    "frame_mean", "frame_std", "frame_sat_frac", "frame_dark_frac", "frame_grad_mean",
    "patch_mean", "patch_std", "patch_sat_frac", "patch_dark_frac", "patch_grad_mean",
    "patch_contrast_ratio",
)


def cameras_from_manifest(manifest: dict) -> dict:
    """One projection model per captured camera, built from the capture's own record."""
    intr = {
        "img_width": int(manifest["img_width"]),
        "img_height": int(manifest["img_height"]),
        "fov_h_rad": float(manifest["fov_h_rad"]),
    }
    out = {
        str(manifest["camera_frame"]): ObliqueCameraModel(
            cam_pos=tuple(manifest["camera_pos"]), look_at=tuple(manifest["look_at"]), **intr)
    }
    mounts = manifest.get("extra_camera_mounts") or {}
    for frame, pose in mounts.items():
        pos = tuple(float(v) for v in pose[:3])
        out[str(frame)] = ObliqueCameraModel(
            cam_pos=pos,
            look_at=tuple(compute_look_at_from_pose(list(pos), float(pose[3]), float(pose[4]), float(pose[5]))),
            **intr,
        )
    return out


def frame_stats(grey: np.ndarray) -> dict:
    gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    return {
        "mean": float(grey.mean()),
        "std": float(grey.std()),
        "sat_frac": float((grey >= SATURATED_LEVEL).mean()),
        "dark_frac": float((grey <= DARK_LEVEL).mean()),
        "grad_mean": float(grad.mean()),
    }


def patch_stats(grey: np.ndarray, u: float, v: float) -> dict:
    h, w = grey.shape
    u0, u1 = int(max(0, u - PATCH_HALF_PX)), int(min(w, u + PATCH_HALF_PX))
    v0, v1 = int(max(0, v - PATCH_HALF_PX)), int(min(h, v + PATCH_HALF_PX))
    if u1 - u0 < 4 or v1 - v0 < 4:
        return {k: float("nan") for k in ("mean", "std", "sat_frac", "dark_frac", "grad_mean")}
    return frame_stats(grey[v0:v1, u0:u1])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture-dir", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    cap = Path(args.capture_dir)
    manifest = json.loads((cap / "capture_manifest.json").read_text(encoding="utf-8"))
    cams = cameras_from_manifest(manifest)
    rows = list(csv.DictReader((cap / "samples.csv").open(encoding="utf-8")))
    print(f"[appearance] {cap.name}: {len(rows)} samples, {len(cams)} cameras")

    out_path = Path(args.out) if args.out else cap / "appearance_features.csv"
    written = 0
    missing = 0
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(COLUMNS))
        writer.writeheader()
        for row in rows:
            frame = str(row.get("camera_frame") or manifest["camera_frame"])
            img_path = cap / str(row["image_path"])
            if not img_path.exists():
                missing += 1
                continue
            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if bgr is None:
                missing += 1
                continue
            grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            x, y = float(row["x"]), float(row["y"])
            cam = cams[frame]
            u, v, in_image = cam.world_to_pixel(x, y, TARGET_HEIGHT_M)
            f = frame_stats(grey)
            p = patch_stats(grey, u, v) if in_image else {
                k: float("nan") for k in ("mean", "std", "sat_frac", "dark_frac", "grad_mean")}
            # Local contrast relative to the frame: a patch can be well lit in a dim
            # frame or crushed in a bright one, and the detector cares about the ratio
            # rather than either level on its own.
            ratio = (p["std"] / f["std"]) if (f["std"] > 1e-6 and np.isfinite(p["std"])) else float("nan")
            writer.writerow({
                "sample_id": row.get("sample_id", ""), "camera_frame": frame,
                "x": f"{x:.8f}", "y": f"{y:.8f}", "theta": row.get("theta", ""),
                "proj_u": f"{u:.3f}", "proj_v": f"{v:.3f}",
                "proj_in_image": int(bool(in_image)),
                "frame_mean": f"{f['mean']:.4f}", "frame_std": f"{f['std']:.4f}",
                "frame_sat_frac": f"{f['sat_frac']:.6f}",
                "frame_dark_frac": f"{f['dark_frac']:.6f}",
                "frame_grad_mean": f"{f['grad_mean']:.4f}",
                "patch_mean": f"{p['mean']:.4f}", "patch_std": f"{p['std']:.4f}",
                "patch_sat_frac": f"{p['sat_frac']:.6f}",
                "patch_dark_frac": f"{p['dark_frac']:.6f}",
                "patch_grad_mean": f"{p['grad_mean']:.4f}",
                "patch_contrast_ratio": f"{ratio:.6f}",
            })
            written += 1
    print(f"[appearance] wrote {written} rows to {out_path.relative_to(REPO)}"
          + (f" ({missing} frames missing)" if missing else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
