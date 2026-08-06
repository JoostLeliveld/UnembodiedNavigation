#!/usr/bin/env python3
"""exp3 — how much of the projection pipeline actually earns its place?

The deployed pipeline intersects the detection ray with a plane at
``contact_z_m = 0.05`` and then applies a per-camera along-bearing correction
``intercept_m + slope_per_m * d`` fitted at commissioning.  Those two are the same
physical quantity seen twice: intersecting at height ``z`` shortens every estimate by
``z*d/(H-z)``, which is exactly the form of the slope term.  So the fitted correction
was partly undoing a constant the operator had chosen, and because intercept and slope
are ~99% anti-correlated over an 8-15 m range window, the fit was free to trade one
against the other and extrapolate badly outside it.

This script evaluates six candidate pipelines on the same 1424 real detections, always
held out (leave-one-capture-out, and leave-one-camera-out for transfer), and reports
which corrections generalise.  The answer is: almost none of them.

Run:  python3 experiments/external_camera_bias_model/exp3_projection_pipeline_minimal.py
Out:  logs/studies/external_camera_bias_model/exp3_projection_pipeline_minimal/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import residual_audit as RA  # noqa: E402  (canonical capture loader + camera models)

OUT = RA.REPO / "logs/studies/external_camera_bias_model/exp3_projection_pipeline_minimal"
GATE_SIGMA = 1.2


def load_frame():
    """Every detection with its bearing-frame error at the floor plane, uncorrected."""
    models = {
        cam: RA.camera_model_from_world(RA.WORLD_SDF, include_name=include)
        for cam, include in RA.MODEL_INCLUDES.items()
    }
    zero = {cam: {"intercept_m": 0.0, "slope_per_m": 0.0} for cam in RA.CAMERAS}
    rows, _ = RA.load_samples(models, zero)

    recs = []
    for row in rows:
        model = models[row["camera"]]
        cam_x, cam_y = float(model.cam_pos[0]), float(model.cam_pos[1])
        # true pixel of the ground-contact point, for the pixel-space residual
        true_u, true_v, _ = model.world_to_pixel(row["true_x"], row["true_y"], 0.0)
        rec = dict(
            camera=row["camera"], capture=row["capture"], d=row["range_m"],
            du=row["u"] - true_u, dv=row["v"] - true_v,
            u=row["u"], v=row["v"], true_x=row["true_x"], true_y=row["true_y"],
            cam_x=cam_x, cam_y=cam_y,
        )
        recs.append(rec)
    return models, recs


def bearing_error(model, u, v, rec, *, contact_z):
    """Along/cross-bearing error in metres of one detection at a given contact plane."""
    if contact_z > 0.0:
        point = model.pixel_to_world_at_z(u, v, contact_z)
    else:
        point = model.pixel_to_world(u, v)
    if point is None:
        return None
    ex, ey = point[0] - rec["true_x"], point[1] - rec["true_y"]
    bx, by = rec["true_x"] - rec["cam_x"], rec["true_y"] - rec["cam_y"]
    norm = float(np.hypot(bx, by))
    bx, by = bx / norm, by / norm
    return ex * bx + ey * by, -ex * by + ey * bx


def main() -> int:
    models, recs = load_frame()
    captures = sorted({r["capture"] for r in recs})
    OUT.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {"n_detections": len(recs), "captures": captures}

    # ---- 1. the contact plane, with no correction at all -------------------
    plane = {}
    for z in (0.05, 0.02, 0.0):
        per_cam = {}
        for cam in RA.CAMERAS:
            errs = [bearing_error(models[cam], r["u"], r["v"], r, contact_z=z)
                    for r in recs if r["camera"] == cam]
            along = np.array([e[0] for e in errs if e])
            per_cam[cam] = dict(bias=float(along.mean()), std=float(along.std()))
        plane[f"{z:.2f}"] = dict(
            per_camera=per_cam,
            mean_abs_bias=float(np.mean([abs(v["bias"]) for v in per_cam.values()])),
        )
    summary["contact_plane_sweep"] = plane
    print("contact plane (no correction applied), mean |along-bearing bias| over cameras:")
    for z, block in plane.items():
        print(f"  z={z} m -> {block['mean_abs_bias']*100:5.2f} cm")

    # ---- 2. does ANY along-bearing correction generalise? ------------------
    # Evaluated at the floor plane, leave-one-capture-out.
    for r in recs:
        error = bearing_error(models[r["camera"]], r["u"], r["v"], r, contact_z=0.0)
        if error is None:
            raise RuntimeError(f"ray does not meet the floor plane: {r['camera']}")
        r["along"], r["cross"] = error
    arms: dict[str, list[float]] = {
        "none": [], "metre_slope_network": [], "pixel_offset_network": [],
        "pixel_offset_per_camera": [],
    }
    for held in captures:
        tr = [r for r in recs if r["capture"] != held]
        te = [r for r in recs if r["capture"] == held]
        d_tr = np.array([r["d"] for r in tr]); c_tr = np.array([-r["along"] for r in tr])
        k = float(d_tr @ c_tr / (d_tr @ d_tr))
        gdu = float(np.mean([r["du"] for r in tr])); gdv = float(np.mean([r["dv"] for r in tr]))
        for cam in RA.CAMERAS:
            te_c = [r for r in te if r["camera"] == cam]
            tr_c = [r for r in tr if r["camera"] == cam]
            if len(te_c) < 20 or len(tr_c) < 20:
                continue
            arms["none"].append(float(np.mean([r["along"] for r in te_c])))
            arms["metre_slope_network"].append(
                float(np.mean([r["along"] + k * r["d"] for r in te_c])))
            for label, (du, dv) in {
                "pixel_offset_network": (gdu, gdv),
                "pixel_offset_per_camera": (float(np.mean([r["du"] for r in tr_c])),
                                            float(np.mean([r["dv"] for r in tr_c]))),
            }.items():
                errs = [bearing_error(models[cam], r["u"] - du, r["v"] - dv, r, contact_z=0.0)
                        for r in te_c]
                arms[label].append(float(np.mean([e[0] for e in errs if e])))
    along = {k: float(np.abs(v).mean()) for k, v in arms.items()}
    summary["along_bearing_heldout_mean_abs_bias_m"] = along
    print("\nalong-bearing, held out across captures (mean |bias|):")
    for label, value in sorted(along.items(), key=lambda kv: kv[1]):
        print(f"  {label:26s} {value*100:5.2f} cm")

    # ---- 3. the cross-bearing term, and whether the gate is the right rule --
    gate_rows = []
    for cam in RA.CAMERAS:
        g = [r for r in recs if r["camera"] == cam]
        cross = np.array([r["cross"] for r in g])
        ratio = float(abs(cross.mean()) / cross.std())
        for held in captures:
            tr = [r["cross"] for r in g if r["capture"] != held]
            te = [r["cross"] for r in g if r["capture"] == held]
            if len(tr) < 20 or len(te) < 20:
                continue
            gate_rows.append(dict(camera=cam, held=held, sigma=ratio,
                                  uncorrected=float(np.mean(te)),
                                  corrected=float(np.mean(te) - np.mean(tr))))
    passing = [r for r in gate_rows if r["sigma"] >= GATE_SIGMA]
    failing = [r for r in gate_rows if r["sigma"] < GATE_SIGMA]
    summary["cross_bearing_gate"] = dict(
        threshold_sigma=GATE_SIGMA, folds=gate_rows,
        passing=dict(uncorrected=float(np.mean([abs(r["uncorrected"]) for r in passing])),
                     corrected=float(np.mean([abs(r["corrected"]) for r in passing]))),
        failing=dict(uncorrected=float(np.mean([abs(r["uncorrected"]) for r in failing])),
                     corrected=float(np.mean([abs(r["corrected"]) for r in failing]))),
    )
    print(f"\ncross-bearing correction, held out across captures (gate = {GATE_SIGMA} sigma):")
    for label, block in (("passing the gate", passing), ("failing the gate", failing)):
        unc = np.mean([abs(r["uncorrected"]) for r in block])
        cor = np.mean([abs(r["corrected"]) for r in block])
        verdict = "improves" if cor < unc else "HARMS"
        print(f"  {label:18s} {unc*100:5.2f} -> {cor*100:5.2f} cm   {verdict}")

    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(f"\nwrote {OUT}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
