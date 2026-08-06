#!/usr/bin/env python3
"""e6 -- does the CAD object model predict the pixels in the DEPLOYED logs?

e1-e5 were scored on a purpose-built capture (1849 set-pose samples, semantic-mask labels).
This experiment holds that model up against a completely independent dataset: the 1424
truth-matched detections in ``external_camera_bias_model/exp1_residual_characterization``,
recorded during real closed-loop driving through the deployed runtime, with the deployed
detector and the deployed nearest-stamp truth join.

Those logs record only ``obs_u/obs_v`` -- the box BOTTOM-CENTRE pixel -- and no box, so the
box-centre estimator itself cannot be scored here.  What CAN be scored, and is the thing the
whole path rests on, is the forward model:

    predicted pixel = bottom-centre of the visual-mesh silhouette at (true_x, true_y, yaw)

If that predicts the recorded pixel, then the per-camera pixel offsets that the 2-line
residual set showed (du = -0.07/-0.09/-4.01/+1.74 px) are geometry, not detector, and no
per-camera constant is warranted.  If it does not, the object model is wrong off-manifold
and e1-e5 do not transfer.

The captures carry exactly two headings -- smoke1 = +90.0 deg on 2475/2475 GT rows,
smoke2 = 0.0 deg on 1140/1140 -- which is itself the reason no correction fitted here ever
generalised.  fusion_handover has no gt_yaw column at all; its heading is derived from the
ground-truth path tangent and it is reported as a separate, weaker stratum.

Ground truth is evaluation-only: it scores, and also supplies the pose the forward model is
evaluated at.  Nothing is fitted.

Outputs -> logs/studies/pixel_ground_path/e6_external_log_validation/
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent / "scripts" / "shared"))
from paths import repo_root  # noqa: E402

REPO = repo_root(_HERE)
sys.path.insert(0, str(REPO / "src" / "reliability"))
sys.path.insert(0, str(REPO / "src" / "unav_common"))
sys.path.insert(0, str(_HERE.parent))

from reliability.projection import camera_model_from_world  # noqa: E402
import robot_silhouette_model as RSM  # noqa: E402

WORLD_SDF = REPO / "src/sim/gazebo_worlds/worlds/warehouse_full_4cam.world.sdf"
RESIDUALS = (
    REPO
    / "logs/studies/external_camera_bias_model/exp1_residual_characterization/residuals.csv"
)
OUT = REPO / "logs/studies/pixel_ground_path/e6_external_log_validation"

CAMERAS = ("camera_A", "camera_B", "camera_C", "camera_D")
MODEL_INCLUDES = {
    "camera_A": "external_camera",
    "camera_B": "external_camera_b",
    "camera_C": "external_camera_c",
    "camera_D": "external_camera_d",
}
DEPLOYED_CONTACT_Z_M = 0.05

# Verified in this experiment (see `yaw_provenance` in summary.json): every ground-truth row
# of these two captures carries one heading, to the recorded precision.
FIXED_YAW_RAD = {
    "smoke1_20260716": 0.5 * math.pi,
    "smoke2_20260716": 0.0,
}
TANGENT_YAW_CAPTURES = {
    "fusion_handover_20260721": REPO
    / "logs/studies/multicamera_fusion_extension/fusion_handover_real_20260721/data"
    / "evaluation_only/ground_truth.csv",
}
GT_FILES = {
    "smoke1_20260716": REPO
    / "logs/studies/multicamera_commissioning_bigwarehouse/gt_validation_smoke_20260716"
    / "evaluation_only/ground_truth.csv",
    "smoke2_20260716": REPO
    / "logs/studies/multicamera_commissioning_bigwarehouse/gt_validation_smoke2_20260716"
    / "evaluation_only/ground_truth.csv",
}
TANGENT_MIN_STEP_M = 0.02  # below this the path tangent is noise, not a heading
GATE_BIAS_OVER_SIGMA = 1.2  # exp2's per-camera cross-bearing resolvability gate


# --------------------------------------------------------------------------- yaw provenance


def verify_fixed_yaw():
    """Confirm the single-heading claim rather than inheriting it."""
    out = {}
    for capture, path in GT_FILES.items():
        yaws = []
        for row in csv.DictReader(path.open(newline="", encoding="utf-8")):
            raw = row.get("gt_yaw")
            if raw in (None, ""):
                continue
            yaws.append(float(raw))
        arr = np.asarray(yaws, dtype=float)
        expected = FIXED_YAW_RAD[capture]
        wrapped = np.abs(np.arctan2(np.sin(arr - expected), np.cos(arr - expected)))
        out[capture] = {
            "n_rows": int(arr.size),
            "assumed_yaw_deg": math.degrees(expected),
            "max_abs_dev_deg": float(np.degrees(wrapped.max())) if arr.size else None,
            "frac_within_1deg": float(np.mean(wrapped < math.radians(1.0))) if arr.size else None,
        }
    return out


def tangent_yaw_table(path: Path):
    """(stamps, yaws) from the ground-truth path tangent, for captures with no gt_yaw."""
    stamps, xs, ys = [], [], []
    for row in csv.DictReader(path.open(newline="", encoding="utf-8")):
        try:
            stamps.append(float(row["stamp"]))
            xs.append(float(row["gt_x"]))
            ys.append(float(row["gt_y"]))
        except (KeyError, TypeError, ValueError):
            continue
    order = np.argsort(np.asarray(stamps))
    s = np.asarray(stamps)[order]
    x = np.asarray(xs)[order]
    y = np.asarray(ys)[order]
    yaw = np.full(s.shape, np.nan)
    for i in range(s.size):
        # walk outward until the robot has actually moved far enough to define a heading
        lo, hi = i, i
        while hi < s.size - 1 and math.hypot(x[hi] - x[lo], y[hi] - y[lo]) < TANGENT_MIN_STEP_M:
            hi += 1
            if lo > 0 and math.hypot(x[hi] - x[lo], y[hi] - y[lo]) < TANGENT_MIN_STEP_M:
                lo -= 1
        dx, dy = x[hi] - x[lo], y[hi] - y[lo]
        if math.hypot(dx, dy) >= TANGENT_MIN_STEP_M:
            yaw[i] = math.atan2(dy, dx)
    return s, yaw


def nearest(table_stamps, values, stamp):
    idx = int(np.argmin(np.abs(table_stamps - stamp)))
    return float(values[idx]), float(abs(table_stamps[idx] - stamp))


# --------------------------------------------------------------------------------- scoring


def bearing_frame(camera, x, y):
    bx = x - float(camera.cam_pos[0])
    by = y - float(camera.cam_pos[1])
    n = math.hypot(bx, by)
    return (bx / n, by / n) if n > 1e-9 else (1.0, 0.0)


def decompose(camera, true_x, true_y, ex, ey):
    """Split a metre error into along-bearing (away from camera) and cross-bearing (left)."""
    ux, uy = bearing_frame(camera, true_x, true_y)
    return ex * ux + ey * uy, -ex * uy + ey * ux


def stats(values):
    a = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "median": float(np.median(a)),
        "mean_abs": float(np.abs(a).mean()),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    yaw_provenance = verify_fixed_yaw()
    for capture, info in yaw_provenance.items():
        if info["max_abs_dev_deg"] is None or info["max_abs_dev_deg"] > 1.0:
            raise SystemExit(f"{capture}: gt_yaw is not the single heading assumed -- {info}")

    tangent_tables = {c: tangent_yaw_table(p) for c, p in TANGENT_YAW_CAPTURES.items()}
    cams = {c: camera_model_from_world(WORLD_SDF, include_name=MODEL_INCLUDES[c]) for c in CAMERAS}

    rows = []
    for row in csv.DictReader(RESIDUALS.open(newline="", encoding="utf-8")):
        capture = row["capture"]
        camera_id = row["camera"]
        cam = cams[camera_id]
        stamp = float(row["stamp"])
        tx, ty = float(row["true_x"]), float(row["true_y"])
        u, v = float(row["u"]), float(row["v"])

        if capture in FIXED_YAW_RAD:
            yaw, yaw_source, yaw_dt = FIXED_YAW_RAD[capture], "gt_yaw_constant", 0.0
        elif capture in tangent_tables:
            s, yy = tangent_tables[capture]
            yaw, yaw_dt = nearest(s, yy, stamp)
            yaw_source = "path_tangent"
        else:
            raise SystemExit(f"no heading source for capture {capture!r}")
        if not math.isfinite(yaw):
            continue

        # --- reference pixel A: the true ground-contact point (what the deployed path assumes)
        cu, cv, _ = cam.world_to_pixel(tx, ty, 0.0)
        # --- reference pixel B: bottom-centre of the CAD visual-mesh silhouette at this pose
        box = RSM.mesh_silhouette_bbox(cam, tx, ty, yaw)
        if box is None:
            continue
        mu = 0.5 * (box[0] + box[2])
        mv = box[3]

        # --- metre space, through the deployed contact plane, for the observed and the
        #     CAD-predicted pixel.  Same projection for both, so the difference is the model.
        obs_xy = cam.pixel_to_world_at_z(u, v, DEPLOYED_CONTACT_Z_M)
        cad_xy = cam.pixel_to_world_at_z(mu, mv, DEPLOYED_CONTACT_Z_M)
        if obs_xy is None or cad_xy is None:
            continue
        obs_along, obs_cross = decompose(cam, tx, ty, obs_xy[0] - tx, obs_xy[1] - ty)
        cad_along, cad_cross = decompose(cam, tx, ty, cad_xy[0] - tx, cad_xy[1] - ty)

        rows.append(
            {
                "capture": capture,
                "camera": camera_id,
                "yaw_source": yaw_source,
                "yaw_deg": math.degrees(yaw),
                "yaw_join_dt_s": yaw_dt,
                "range_m": float(row["range_m"]),
                "du_contact": u - cu,
                "dv_contact": v - cv,
                "du_mesh": u - mu,
                "dv_mesh": v - mv,
                "obs_along_m": obs_along,
                "obs_cross_m": obs_cross,
                "cad_along_m": cad_along,
                "cad_cross_m": cad_cross,
                "res_along_m": obs_along - cad_along,
                "res_cross_m": obs_cross - cad_cross,
                "obs_norm_m": math.hypot(obs_xy[0] - tx, obs_xy[1] - ty),
                "res_norm_m": math.hypot(obs_xy[0] - cad_xy[0], obs_xy[1] - cad_xy[1]),
            }
        )

    strong = [r for r in rows if r["yaw_source"] == "gt_yaw_constant"]

    def block(subset):
        out = {"n": len(subset)}
        for key in ("du_contact", "dv_contact", "du_mesh", "dv_mesh"):
            out[key] = stats([r[key] for r in subset])
        for key in ("obs_along_m", "obs_cross_m", "res_along_m", "res_cross_m",
                    "obs_norm_m", "res_norm_m"):
            out[key] = stats([r[key] for r in subset])
        return out

    summary = {
        "inputs": {
            "residuals_csv": str(RESIDUALS.relative_to(REPO)),
            "world_sdf": str(WORLD_SDF.relative_to(REPO)),
            "deployed_contact_z_m": DEPLOYED_CONTACT_Z_M,
            "object_model": "visual meshes (robot_silhouette_model.MESH_LOCAL)",
        },
        "yaw_provenance": yaw_provenance,
        "n_rows_scored": len(rows),
        "all_captures": block(rows),
        "gt_yaw_only": block(strong),
        "per_camera_gt_yaw_only": {},
        "per_camera_all": {},
        "per_capture": {},
        "cross_bearing_gate": {},
    }
    for cam_id in CAMERAS:
        sub = [r for r in strong if r["camera"] == cam_id]
        if sub:
            summary["per_camera_gt_yaw_only"][cam_id] = block(sub)
        sub_all = [r for r in rows if r["camera"] == cam_id]
        if sub_all:
            summary["per_camera_all"][cam_id] = block(sub_all)
    for capture in sorted({r["capture"] for r in rows}):
        summary["per_capture"][capture] = block([r for r in rows if r["capture"] == capture])

    # Does exp2's gated per-camera cross-bearing constant survive once the object model is
    # accounted for?  Same gate, applied before and after.
    for cam_id in CAMERAS:
        sub = [r for r in rows if r["camera"] == cam_id]
        if len(sub) < 2:
            continue
        entry = {}
        for tag, key in (("raw", "obs_cross_m"), ("after_cad_model", "res_cross_m")):
            a = np.asarray([r[key] for r in sub], dtype=float)
            bias, sd = float(a.mean()), float(a.std(ddof=1))
            ratio = abs(bias) / sd if sd > 0 else float("inf")
            entry[tag] = {
                "bias_m": bias,
                "sd_m": sd,
                "bias_over_sigma": ratio,
                "passes_gate": bool(ratio >= GATE_BIAS_OVER_SIGMA),
            }
        summary["cross_bearing_gate"][cam_id] = entry

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    field_names = list(rows[0].keys())
    with (OUT / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)

    # ------------------------------------------------------------------ console report
    print(f"scored {len(rows)} rows ({len(strong)} with a recorded constant gt_yaw)\n")
    print("yaw provenance:")
    for capture, info in yaw_provenance.items():
        print(f"  {capture}: {info['n_rows']} rows, all within "
              f"{info['max_abs_dev_deg']:.4f} deg of {info['assumed_yaw_deg']:.1f} deg")
    print("\npixel residual, recorded obs_uv minus reference pixel (gt_yaw captures only)")
    print(f"{'camera':>9} {'n':>5} | {'du vs contact':>14} {'dv vs contact':>14} "
          f"| {'du vs mesh':>13} {'dv vs mesh':>13}")
    for cam_id in CAMERAS:
        b = summary["per_camera_gt_yaw_only"].get(cam_id)
        if not b:
            continue
        print(f"{cam_id:>9} {b['n']:>5} | "
              f"{b['du_contact']['mean']:>+7.2f}+-{b['du_contact']['sd']:<5.2f} "
              f"{b['dv_contact']['mean']:>+7.2f}+-{b['dv_contact']['sd']:<5.2f} | "
              f"{b['du_mesh']['mean']:>+6.2f}+-{b['du_mesh']['sd']:<5.2f} "
              f"{b['dv_mesh']['mean']:>+6.2f}+-{b['dv_mesh']['sd']:<5.2f}")
    print("\nmetre space at the deployed contact plane (all captures)")
    b = summary["all_captures"]
    print(f"  observed error vs truth        : mean |e| {b['obs_norm_m']['mean']*1000:6.1f} mm, "
          f"along {b['obs_along_m']['mean']*1000:+7.1f} mm, cross {b['obs_cross_m']['mean']*1000:+7.1f} mm")
    print(f"  observed minus CAD prediction  : mean |e| {b['res_norm_m']['mean']*1000:6.1f} mm, "
          f"along {b['res_along_m']['mean']*1000:+7.1f} mm, cross {b['res_cross_m']['mean']*1000:+7.1f} mm")
    print("\nexp2 cross-bearing gate (|bias|/sigma >= 1.2), before and after the object model")
    for cam_id, entry in summary["cross_bearing_gate"].items():
        r, c = entry["raw"], entry["after_cad_model"]
        print(f"  {cam_id}: raw {r['bias_m']*1000:+7.1f} mm ratio {r['bias_over_sigma']:.2f} "
              f"{'PASS' if r['passes_gate'] else 'fail'}  ->  after model "
              f"{c['bias_m']*1000:+7.1f} mm ratio {c['bias_over_sigma']:.2f} "
              f"{'PASS' if c['passes_gate'] else 'fail'}")
    print(f"\nwrote {OUT.relative_to(REPO)}/summary.json and per_sample.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
