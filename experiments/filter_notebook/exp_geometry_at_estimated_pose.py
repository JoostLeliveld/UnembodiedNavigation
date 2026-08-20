#!/usr/bin/env python3
"""Does the geometry correction survive being applied at the ESTIMATED pose?

THE QUESTION THIS DECIDES. Every "geometry fixes the bias" number in this project is
measured by correcting at the TRUE pose. A deployed filter has no true pose -- it has to
evaluate the correction at its own estimate, and the correction is a function of that
estimate, so its error feeds back. That makes it an implicit measurement, and the size of
the resulting loss decides whether an ordinary box detector plus known robot geometry is
enough, or whether the perception side has to change.

Arms, all scored on the same detections:

  1 nothing                     the baseline
  2 geometry, TRUE pose + TRUE heading      oracle upper bound
  3 geometry, TRUE pose + ODOMETRY heading  how much heading error alone costs
  4 geometry, ESTIMATED pose + odom heading DEPLOYABLE -- the number that decides it
  5 arm 4 + a lean state                    does estimating the remainder recover it
  6 lean state alone, no geometry           the alternative that needs no robot model

An earlier version of this experiment reported arms 2 and 4 as byte-identical, which is
impossible; the correction was silently not applied. Every arm here asserts that its
observations actually moved before it is scored.

    python3 experiments/filter_notebook/exp_geometry_at_estimated_pose.py
"""

from __future__ import annotations

import math

import numpy as np

import notebook_data as nd
import notebook_model as nm
import story_model as sm

DRIVES = ["aws_aisle_east_north", "aws_apron_west_to_east", "aws_aisle_west_north",
          "aws_mid_cross_east", "aws_apron_diagonal_ne", "aws_apron_diagonal_sw",
          "aws_apron_corner_left", "aws_apron_arc_left", "aws_apron_reverse_spin",
          "aws_graze_aisle_north", "aws_graze_aisle_south",
          "aws_cross_aisle_full_east", "aws_cross_aisle_full_west"]


def corrected_at_truth(drive, camera, *, use_true_heading):
    """Shape correction evaluated at the TRUE pose. CHEATING -- an upper bound only."""
    base = drive["seq"]
    out = nm.Sequence.__new__(nm.Sequence)
    out.__dict__.update({k: (v.copy() if isinstance(v, np.ndarray)
                             else list(v) if isinstance(v, list) else v)
                         for k, v in base.__dict__.items()})
    by_step = {r["step"]: r for r in drive["rows"]}
    moved = 0
    for k in range(out.n_steps):
        if out.camera[k] is None or k not in by_step:
            continue
        row = by_step[k]
        yaw = row["true_yaw"] if use_true_heading else row["odom_yaw"]
        if not np.isfinite(yaw):
            continue
        offset = nm.predicted_offset(camera, float(row["truth"][0]),
                                     float(row["truth"][1]), float(yaw))
        if offset is None:
            continue
        out.y[k] = out.y[k] - offset
        moved += 1
    setattr(out, "n_corrected", moved)
    return out


def residual_noise(drives, camera_of):
    """Covariance of what is left after the shape model, with each drive's mean removed."""
    stack = []
    for d in drives:
        v = []
        for r in d["rows"]:
            if not np.isfinite(r["odom_yaw"]):
                continue
            landing = nm.silhouette_bottom(camera_of(d), float(r["truth"][0]),
                                           float(r["truth"][1]), float(r["odom_yaw"]))
            if landing is not None:
                v.append(r["observed"] - np.asarray(landing))
        v = np.asarray(v)
        if len(v) > 5:
            stack.append(v - v.mean(axis=0))
    return np.cov(np.vstack(stack).T)


def main() -> None:
    nd.use_world(nd.AWS_SINGLE)
    models = nd.camera_models()
    drives = [sm.drive(t, models) for t in DRIVES]
    R_raw = sm.oracle_R(drives)
    R_shape = residual_noise(drives, lambda d: d["camera"])
    print(f"{len(drives)} recorded drives, {sum(len(d['rows']) for d in drives)} detections")
    print(f"noise with every lean removed : ({100 * math.sqrt(R_raw[0, 0]):.2f}, "
          f"{100 * math.sqrt(R_raw[1, 1]):.2f}) cm")
    print(f"noise after the shape model   : ({100 * math.sqrt(R_shape[0, 0]):.2f}, "
          f"{100 * math.sqrt(R_shape[1, 1]):.2f}) cm\n")

    arms = []

    def add(label, build, R, frame, s_lean=0.10, s0=0.02):
        rows, checks = [], []
        for d in drives:
            seq = build(d)
            checks.append(float(np.nanmax(np.abs(
                np.nan_to_num(seq.y - d["seq"].y, nan=0.0)))))
            result = sm.lean_filter(seq, d["heading"], R, d["camera"],
                                    frame=frame, sigma_lean_prior=s_lean, initial_sigma=s0)
            rows.append(sm.score(result, seq, label))
        moved = float(np.mean(checks))
        arms.append({
            "label": label, "moved_cm": 100 * moved,
            "error": float(np.mean([r["median_error_cm"] for r in rows])),
            "stated": float(np.mean([r["stated_sigma_cm"] for r in rows])),
            "cover": float(np.mean([r["coverage_95"] for r in rows])),
            "nees": float(np.median([r["median_nees"] for r in rows])),
            "per_drive": rows,
        })

    add("1. nothing", lambda d: d["seq"], R_raw, "none")
    add("2. geometry, TRUE pose + TRUE heading",
        lambda d: corrected_at_truth(d, d["camera"], use_true_heading=True), R_shape, "none")
    add("3. geometry, TRUE pose + odometry heading",
        lambda d: corrected_at_truth(d, d["camera"], use_true_heading=False), R_shape, "none")
    add("4. geometry, ESTIMATED pose (deployable)",
        lambda d: nm.GeometryCorrected(d["seq"], models, {"camera_A": R_shape}, d["heading"]),
        R_shape, "none")
    add("5. geometry at estimated pose + lean state",
        lambda d: nm.GeometryCorrected(d["seq"], models, {"camera_A": R_shape}, d["heading"]),
        R_shape, "world", s_lean=0.03)
    add("6. lean state alone, no robot model", lambda d: d["seq"], R_raw, "world")

    # every arm that claims to correct something must actually have moved the data
    for a in arms[1:5]:
        assert a["moved_cm"] > 0.5, f"{a['label']} did not change the observations"
    assert abs(arms[1]["error"] - arms[3]["error"]) > 1e-6, "arms 2 and 4 identical again"

    print("=" * 96)
    print("DOES THE GEOMETRY CORRECTION SURVIVE THE ESTIMATED POSE?")
    print("=" * 96)
    print(f"{'arm':<44}{'error':>9}{'stated':>9}{'in 95%':>9}{'NEES':>8}{'moved':>9}")
    for a in arms:
        print(f"{a['label']:<44}{a['error']:>7.2f}cm{a['stated']:>7.2f}cm"
              f"{a['cover']:>8.0f}%{a['nees']:>8.2f}{a['moved_cm']:>7.1f}cm")

    true_pose, est_pose = arms[2]["error"], arms[3]["error"]
    print(f"\nCOST OF NOT KNOWING THE POSE: {true_pose:.2f} -> {est_pose:.2f} cm "
          f"({est_pose / true_pose:.2f}x)")
    print(f"HEADING COST (true vs odometry): {arms[1]['error']:.2f} -> {arms[2]['error']:.2f} cm")
    print(f"\nBEST DEPLOYABLE ARM: "
          f"{min(arms[3:], key=lambda a: a['error'])['label'].strip()} "
          f"at {min(a['error'] for a in arms[3:]):.2f} cm")

    print("\n" + "=" * 96)
    print("PER DRIVE, deployable geometry vs lean state  (does occlusion split them?)")
    print("=" * 96)
    print(f"{'drive':<28}{'nothing':>9}{'geometry@est':>14}{'lean state':>12}{'geom+lean':>11}")
    for i, d in enumerate(drives):
        print(f"{d['tag'].replace('aws_', ''):<28}"
              f"{arms[0]['per_drive'][i]['median_error_cm']:>9.2f}"
              f"{arms[3]['per_drive'][i]['median_error_cm']:>14.2f}"
              f"{arms[5]['per_drive'][i]['median_error_cm']:>12.2f}"
              f"{arms[4]['per_drive'][i]['median_error_cm']:>11.2f}")


if __name__ == "__main__":
    main()
