#!/usr/bin/env python3
"""Model the geometry in the projection; carry only what is left in the filter.

`exp1` and `exp2` showed the per-camera offset is not a constant in any frame -- it
is a function of viewing geometry, so a filter state cannot learn it. The repo
already has that geometry as a MODEL with zero fitted parameters: the CAD visual
meshes, which `pixel_ground_path/e6` used to take the mean along-bearing error on
these same logs from -133.0 mm to -0.0 mm.

This experiment composes the two. Each detection's world point is recomputed by
inverting the OBJECT MODEL instead of the ground-plane homography:

    find (x, y) such that the bottom-centre pixel of the robot's rendered
    silhouette at pose (x, y, theta) equals the observed pixel,

with theta from the ODOMETRY path tangent -- never truth. Then the same filter
arms run on both measurement paths, and the same transfer test is repeated.

Three questions, each with a prediction that can fail:

  Q1  does the raw filter become honest once the geometry is modelled, with no
      floor and no offset state at all?
  Q2  do the offsets A5 estimates collapse toward zero?
  Q3  does whatever residual remains now TRANSFER across a heading change, where
      the deployed-path offset did not?

Ground truth is EVALUATION-ONLY: it scores, and it measures commissioning
quantities. It never enters a filter and never supplies the heading.

Outputs -> logs/studies/offset_state_model/exp3_model_the_geometry/
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "scripts" / "shared"))
from paths import repo_root  # noqa: E402

REPO = repo_root(_HERE)
for _relative in ("src/reliability", "src/unav_common", "src/state",
                  "experiments/operational_residual_rcond",
                  "experiments/pixel_ground_path",
                  "experiments/bayesian_filter_showcase",
                  "experiments/offset_state_model"):
    sys.path.insert(0, str(REPO / _relative))

import rcond_common as rc  # noqa: E402
import robot_silhouette_model as RSM  # noqa: E402
import exp1_graceful_vs_trusting as f1  # noqa: E402
import demo_how_the_filter_works as d1  # noqa: E402
import demo_state_space_model as m1  # noqa: E402
import exp2_offset_in_the_robot_frame as b1  # noqa: E402

OUT = REPO / "logs/studies/offset_state_model/exp3_model_the_geometry"

CONTACT_Z_M = rc.CONTACT_Z_M
SIGMA_PRIOR_M = 0.05
SIGMA_WALK = 0.0016

PAIRS = [
    ("smoke1_20260716", "fusion_handover_20260721", "same route, same heading"),
    ("smoke1_20260716", "smoke2_20260716", "DIFFERENT heading"),
    ("smoke2_20260716", "smoke1_20260716", "DIFFERENT heading"),
]

C_DEPLOYED = "#D55E00"
C_OBJECT = "#009E73"
C_FLOOR = "#0072B2"
C_TRUTH = "#111111"


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{suffix}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


# ------------------------------------------------- inverting the object model


def _modelled_bottom_centre(camera, x, y, yaw):
    box = RSM.mesh_silhouette_bbox(camera, x, y, yaw)
    if box is None:
        return None
    return 0.5 * (box[0] + box[2]), box[3]


def solve_object_model(camera, u_obs, v_obs, yaw, *, iters=8, tol=1e-3, h=0.01):
    """Gauss-Newton on (x, y) so the modelled silhouette bottom edge hits (u, v).

    The same solver shape as `pixel_ground_path/e5.solve_yaw_aware`, but matching
    the BOTTOM-CENTRE pixel, because the operational logs record only that -- they
    carry `obs_u`/`obs_v` and no box, which is the limit `e6` documented.

    Seeded with the deployed contact-plane inversion, which is within a few cm, so
    this converges in two or three steps.
    """
    seed = camera.pixel_to_world_at_z(u_obs, v_obs, CONTACT_Z_M)
    if seed is None:
        return None
    x, y = float(seed[0]), float(seed[1])
    for _ in range(iters):
        modelled = _modelled_bottom_centre(camera, x, y, yaw)
        if modelled is None:
            return None
        residual = np.array([u_obs - modelled[0], v_obs - modelled[1]])
        if float(np.abs(residual).max()) < tol:
            break
        jacobian = np.zeros((2, 2))
        for k, (dx, dy) in enumerate(((h, 0.0), (0.0, h))):
            bumped = _modelled_bottom_centre(camera, x + dx, y + dy, yaw)
            if bumped is None:
                return None
            jacobian[0, k] = (bumped[0] - modelled[0]) / h
            jacobian[1, k] = (bumped[1] - modelled[1]) / h
        try:
            step = np.linalg.solve(jacobian, residual)
        except np.linalg.LinAlgError:
            return None
        x += float(step[0])
        y += float(step[1])
    return x, y


def reproject_capture(capture, models):
    """A copy of the capture whose detection world points come from the object model.

    Heading is the odometry path tangent, so nothing here consumes truth.
    """
    heading = b1.odometry_heading(capture)
    stamps = np.asarray(capture.stamps, dtype=float)
    rebuilt, failed = {}, 0
    for camera_id in rc.CAMERAS:
        camera = models[camera_id]
        out = []
        for detection in capture.detections[camera_id]:
            index = int(np.searchsorted(stamps, detection.stamp))
            index = min(max(index, 0), len(stamps) - 1)
            solved = solve_object_model(camera, detection.u, detection.v,
                                        float(heading[index]))
            if solved is None:
                failed += 1
                continue
            out.append(dataclasses.replace(detection, world=(solved[0], solved[1])))
        rebuilt[camera_id] = out
    return dataclasses.replace(capture, detections=rebuilt), failed


# --------------------------------------------------------------- measurements


def commissioning_bias(capture, lookup) -> dict:
    """Per-camera mean residual against truth -- the commissioning quantity A4 floors with.

    EVALUATION-ONLY input, used exactly as assumption A1 permits: measured once,
    at commissioning, against truth.
    """
    found = {}
    for camera_id in rc.CAMERAS:
        residuals = []
        for detection in capture.detections[camera_id]:
            truth = lookup(detection.stamp)
            if truth is None:
                continue
            residuals.append(np.asarray(detection.world, dtype=float)
                             - np.asarray(truth[:2], dtype=float))
        if residuals:
            mean = np.mean(np.asarray(residuals), axis=0)
            found[camera_id] = {"bias_mm": float(1000 * np.hypot(*mean)),
                                "bias_xy_mm": (1000 * mean).tolist(),
                                "n": len(residuals)}
        else:
            found[camera_id] = {"bias_mm": float("nan"), "bias_xy_mm": [None, None],
                                "n": 0}
    return found


def score(records) -> dict:
    summary = f1.summarize(records)
    summary.update(m1.logarithmic_score(records))
    return summary


def arms_on(capture, lookup, *, frozen=None) -> dict:
    """The three arms that matter here, on whichever measurement path is supplied."""
    zero = {c: np.zeros(2) for c in rc.CAMERAS}
    out = {
        "raw sharp R": score(m1.run_as_ladder_arm(
            capture, lookup, sigma_bias_prior=SIGMA_PRIOR_M,
            sigma_bias_walk_per_sqrt_s=SIGMA_WALK, frozen_offsets=zero)),
        "A4 floor": score(d1.trace_arm(capture, "A4_correlation_floor", lookup)),
        "A5 offsets": score(m1.run_as_ladder_arm(
            capture, lookup, sigma_bias_prior=SIGMA_PRIOR_M,
            sigma_bias_walk_per_sqrt_s=SIGMA_WALK)),
    }
    if frozen is not None:
        out["A5 frozen"] = score(m1.run_as_ladder_arm(
            capture, lookup, sigma_bias_prior=SIGMA_PRIOR_M,
            sigma_bias_walk_per_sqrt_s=SIGMA_WALK, frozen_offsets=frozen))
    return out


def fitted_offsets(capture, lookup) -> dict:
    records = m1.run_as_ladder_arm(capture, lookup, sigma_bias_prior=SIGMA_PRIOR_M,
                                  sigma_bias_walk_per_sqrt_s=SIGMA_WALK)
    if not records:
        return {c: np.zeros(2) for c in rc.CAMERAS}
    return {c: np.asarray(v, dtype=float)
            for c, v in records[-1]["offset_xy"].items()}


# ------------------------------------------------------------------- figures


def fig_g1_residual_collapse(deployed_bias, object_bias, counts) -> dict:
    cameras = [c for c in rc.CAMERAS if counts[c] > 0]
    positions = np.arange(len(cameras))
    before = [deployed_bias[c]["bias_mm"] for c in cameras]
    after = [object_bias[c]["bias_mm"] for c in cameras]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.7))
    width = 0.38
    ax.bar(positions - width / 2, before, width, color=C_DEPLOYED,
           label="deployed path: ground-plane homography")
    ax.bar(positions + width / 2, after, width, color=C_OBJECT,
           label="object-model path: CAD silhouette, zero fitted parameters")
    for i, (a, b) in enumerate(zip(before, after)):
        ax.annotate(f"{a:.0f}", xy=(i - width / 2, a), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8.5)
        ax.annotate(f"{b:.0f}", xy=(i + width / 2, b), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8.5)
    ax.set_xticks(positions, [c.replace("camera_", "") for c in cameras])
    ax.set_ylabel("systematic residual against truth (mm)")
    ax.set_xlabel("camera")
    ax.set_title("Q2: what the filter would have to absorb", loc="left")
    ax.legend(frameon=False, fontsize=8.5)

    pooled_before = float(np.mean(before))
    pooled_after = float(np.mean(after))
    ax2.bar([0, 1], [pooled_before, pooled_after], 0.5,
            color=[C_DEPLOYED, C_OBJECT])
    for i, value in enumerate([pooled_before, pooled_after]):
        ax2.annotate(f"{value:.1f} mm", xy=(i, value), xytext=(0, 4),
                     textcoords="offset points", ha="center", fontsize=10,
                     fontweight="bold")
    ax2.set_xticks([0, 1], ["deployed\nhomography", "object model\n(0 parameters)"])
    ax2.set_ylabel("mean systematic residual (mm)")
    ax2.set_ylim(0, max(pooled_before, pooled_after) * 1.3)
    ax2.set_title(f"{pooled_before / max(pooled_after, 1e-9):.1f}x smaller, "
                  "nothing fitted", loc="left")

    fig.suptitle("Modelling the robot's geometry removes most of what looked like "
                 "per-camera bias", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig_g1_residual_collapse")
    return {"deployed_mean_mm": pooled_before, "object_mean_mm": pooled_after,
            "ratio": pooled_before / max(pooled_after, 1e-9)}


def fig_g2_arms(deployed_arms, object_arms) -> dict:
    arms = ["raw sharp R", "A4 floor", "A5 offsets"]
    positions = np.arange(len(arms))
    fig, axes = plt.subplots(1, 4, figsize=(17.6, 4.9))
    width = 0.38

    panels = [
        ("RMSE (cm)  — accuracy", lambda s: 100 * s["rmse_m"], None, None),
        ("median NEES  — honesty", lambda s: s["median_nees"], 1.386, "honest = 1.39"),
        ("truth outside 95 % (%)",
         lambda s: 100 * s["unearned_confidence_fraction"], 5.0, "nominal 5 %"),
        ("mean $-\\log p(\\mathrm{truth})$", lambda s: s["mean_nlpd"], None, None),
    ]
    for ax, (label, getter, reference, note) in zip(axes, panels):
        before = [getter(deployed_arms[a]) for a in arms]
        after = [getter(object_arms[a]) for a in arms]
        ax.bar(positions - width / 2, before, width, color=C_DEPLOYED,
               label="deployed homography")
        ax.bar(positions + width / 2, after, width, color=C_OBJECT,
               label="object model")
        for i, (a, b) in enumerate(zip(before, after)):
            ax.annotate(f"{a:.2f}" if abs(a) < 100 else f"{a:.0f}",
                        xy=(i - width / 2, a),
                        xytext=(0, 4 if a >= 0 else -12), textcoords="offset points",
                        ha="center", fontsize=8)
            ax.annotate(f"{b:.2f}" if abs(b) < 100 else f"{b:.0f}",
                        xy=(i + width / 2, b),
                        xytext=(0, 4 if b >= 0 else -12), textcoords="offset points",
                        ha="center", fontsize=8, fontweight="bold")
        low, high = ax.get_ylim()
        ax.set_ylim(low, high + 0.18 * (high - low))
        if reference is not None:
            ax.axhline(reference, color=C_TRUTH, lw=1.4, ls="--")
            ax.annotate(note, xy=(0.60, 0.055), xycoords="axes fraction", fontsize=8)
        else:
            ax.axhline(0, color=C_TRUTH, lw=0.8)
        ax.set_xticks(positions, arms, fontsize=8.5)
        ax.set_ylabel(label)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper right")
    axes[0].set_title("Geometry buys accuracy", loc="left", fontsize=10)
    axes[1].set_title("Q1: the raw filter is still NOT honest", loc="left", fontsize=10)

    fig.suptitle("The same three arms on both measurement paths, pooled over three "
                 "captures — the two levers act on different axes",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig_g2_arms")
    return {"deployed": {a: deployed_arms[a] for a in arms},
            "object_model": {a: object_arms[a] for a in arms}}


def fig_g3_transfer(deployed_transfer, object_transfer) -> dict:
    labels = [f"{test.replace('_20260716', '').replace('_20260721', '')}\n"
              f"from {train.replace('_20260716', '').replace('_20260721', '')}\n"
              f"({note})" for train, test, note in PAIRS]
    positions = np.arange(len(PAIRS))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.1))
    width = 0.38

    for offset, (name, table, colour) in enumerate((
            ("deployed homography", deployed_transfer, C_DEPLOYED),
            ("object model", object_transfer, C_OBJECT))):
        nlpd = [table[(t, s)]["A5 frozen"]["mean_nlpd"] for t, s, _ in PAIRS]
        unearned = [100 * table[(t, s)]["A5 frozen"]["unearned_confidence_fraction"]
                    for t, s, _ in PAIRS]
        ax.bar(positions + (offset - 0.5) * width, nlpd, width * 0.9, color=colour,
               label=name)
        ax2.bar(positions + (offset - 0.5) * width, unearned, width * 0.9,
                color=colour, label=name)
        for i, (a, b) in enumerate(zip(nlpd, unearned)):
            ax.annotate(f"{a:.1f}", xy=(positions[i] + (offset - 0.5) * width, a),
                        xytext=(0, 4 if a >= 0 else -12), textcoords="offset points",
                        ha="center", fontsize=8, fontweight="bold")
            ax2.annotate(f"{b:.1f}", xy=(positions[i] + (offset - 0.5) * width, b),
                         xytext=(0, 4), textcoords="offset points", ha="center",
                         fontsize=8, fontweight="bold")

    ax.axhline(0, color=C_TRUTH, lw=0.9)
    ax.set_xticks(positions, labels, fontsize=8)
    ax.set_ylabel("mean $-\\log p(\\mathrm{truth})$   (lower = better)")
    ax.set_title("Q3: frozen offsets on a capture they were not fitted on", loc="left")
    ax.legend(frameon=False, fontsize=8.5)

    ax2.axhline(5.0, color=C_TRUTH, lw=1.4, ls="--")
    ax2.annotate("nominal 5 %", xy=(0.02, 0.94), xycoords="axes fraction", fontsize=8.5)
    ax2.set_xticks(positions, labels, fontsize=8)
    ax2.set_ylabel("truth outside the stated 95 % ellipse (%)")
    ax2.set_title("Unearned confidence on the held-out capture", loc="left")
    ax2.legend(frameon=False, fontsize=8.5)

    fig.suptitle("Does the residual transfer once the geometry is modelled?",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig_g3_transfer")
    return {}


def main() -> int:
    d1._style()
    OUT.mkdir(parents=True, exist_ok=True)
    models = rc.camera_models()
    calib = rc.deployed_calibration()

    deployed, obj, lookups = {}, {}, {}
    print("re-projecting every detection through the object model "
          "(heading from odometry, never truth)")
    for name in rc.CAPTURES:
        capture = rc.load_operational_capture(name, models=models, calib=calib)
        table = rc.load_truth_table(name)                 # EVALUATION ONLY

        def lookup(stamp, _table=table):
            return rc.truth_at(_table, stamp)

        rebuilt, failed = reproject_capture(capture, models)
        deployed[name], obj[name], lookups[name] = capture, rebuilt, lookup
        before = sum(len(v) for v in capture.detections.values())
        after = sum(len(v) for v in rebuilt.detections.values())
        print(f"  {name:<26} {before} detections -> {after} solved "
              f"({failed} failed)")

    # ---- Q2: does the systematic residual collapse?
    print("\nQ2 — systematic residual against truth, per camera (mm)")
    deployed_bias, object_bias, counts = {}, {}, {}
    for camera_id in rc.CAMERAS:
        counts[camera_id] = 0
    for name in rc.CAPTURES:
        for camera_id in rc.CAMERAS:
            counts[camera_id] += len(deployed[name].detections[camera_id])
    pooled = {}
    for label, table in (("deployed", deployed), ("object", obj)):
        merged = {c: [] for c in rc.CAMERAS}
        for name in rc.CAPTURES:
            per = commissioning_bias(table[name], lookups[name])
            for camera_id in rc.CAMERAS:
                if per[camera_id]["n"]:
                    merged[camera_id].append(
                        (per[camera_id]["n"], per[camera_id]["bias_xy_mm"]))
        out = {}
        for camera_id, entries in merged.items():
            if not entries:
                out[camera_id] = {"bias_mm": float("nan"), "n": 0}
                continue
            weight = sum(n for n, _ in entries)
            vector = sum(n * np.asarray(v, dtype=float) for n, v in entries) / weight
            out[camera_id] = {"bias_mm": float(np.hypot(*vector)),
                              "bias_xy_mm": vector.tolist(), "n": weight}
        pooled[label] = out
    deployed_bias, object_bias = pooled["deployed"], pooled["object"]
    for camera_id in rc.CAMERAS:
        if not counts[camera_id]:
            continue
        print(f"  {camera_id.replace('camera_', ''):<3} "
              f"deployed {deployed_bias[camera_id]['bias_mm']:6.1f}  ->  "
              f"object model {object_bias[camera_id]['bias_mm']:6.1f}")
    collapse = fig_g1_residual_collapse(deployed_bias, object_bias, counts)
    print(f"  pooled {collapse['deployed_mean_mm']:.1f} -> "
          f"{collapse['object_mean_mm']:.1f} mm  ({collapse['ratio']:.1f}x smaller)")

    # ---- Q1: are the arms honest on each path?
    print("\nQ1 — the same arms on both paths, pooled over three captures")
    pooled_arms = {}
    for label, table in (("deployed", deployed), ("object", obj)):
        merged = {}
        for arm in ("raw sharp R", "A4 floor", "A5 offsets"):
            records = []
            for name in rc.CAPTURES:
                if arm == "A4 floor":
                    records.extend(d1.trace_arm(table[name], "A4_correlation_floor",
                                                lookups[name]))
                else:
                    zero = ({c: np.zeros(2) for c in rc.CAMERAS}
                            if arm == "raw sharp R" else None)
                    records.extend(m1.run_as_ladder_arm(
                        table[name], lookups[name], sigma_bias_prior=SIGMA_PRIOR_M,
                        sigma_bias_walk_per_sqrt_s=SIGMA_WALK, frozen_offsets=zero))
            merged[arm] = score(records)
        pooled_arms[label] = merged
        print(f"  {label} path")
        print(f"    {'arm':<14}{'NLPD':>9}{'medNEES':>9}{'unearned%':>11}"
              f"{'RMSE cm':>9}{'stated cm':>11}")
        for arm, summary in merged.items():
            print(f"    {arm:<14}{summary['mean_nlpd']:>9.2f}"
                  f"{summary['median_nees']:>9.2f}"
                  f"{100 * summary['unearned_confidence_fraction']:>11.1f}"
                  f"{100 * summary['rmse_m']:>9.1f}"
                  f"{100 * summary['mean_stated_sigma_m']:>11.1f}")
    arms_chart = fig_g2_arms(pooled_arms["deployed"], pooled_arms["object"])

    # ---- Q3: does the residual transfer now?
    print("\nQ3 — transfer of the frozen offsets, both paths")
    transfer = {}
    for label, table in (("deployed", deployed), ("object", obj)):
        fitted = {name: fitted_offsets(table[name], lookups[name])
                  for name in rc.CAPTURES}
        results = {}
        for train, test, note in PAIRS:
            results[(train, test)] = arms_on(table[test], lookups[test],
                                             frozen=fitted[train])
        transfer[label] = results
        print(f"  {label} path")
        for train, test, note in PAIRS:
            row = results[(train, test)]
            print(f"    {train[:6]} -> {test[:6]:<8} ({note[:22]:<22}) "
                  f"frozen NLPD {row['A5 frozen']['mean_nlpd']:8.2f}  "
                  f"raw {row['raw sharp R']['mean_nlpd']:7.2f}  "
                  f"unearned {100 * row['A5 frozen']['unearned_confidence_fraction']:5.1f} %")
        transfer[label + "_offsets"] = {
            n: {c: (1000 * np.asarray(v)).tolist() for c, v in fitted[n].items()}
            for n in fitted}
    fig_g3_transfer(transfer["deployed"], transfer["object"])

    cross = [p for p in PAIRS if "DIFFERENT" in p[2]]
    now_transfers = all(
        transfer["object"][(t, s)]["A5 frozen"]["mean_nlpd"]
        < transfer["object"][(t, s)]["raw sharp R"]["mean_nlpd"] for t, s, _ in cross)
    raw_is_honest = (pooled_arms["object"]["raw sharp R"]["median_nees"] < 4.0)
    print(f"\nQ1 raw filter honest on the object-model path: {raw_is_honest} "
          f"(median NEES {pooled_arms['object']['raw sharp R']['median_nees']:.2f} "
          f"vs {pooled_arms['deployed']['raw sharp R']['median_nees']:.2f} deployed)")
    print(f"Q2 residual collapse: {collapse['ratio']:.1f}x, "
          f"{collapse['deployed_mean_mm']:.1f} -> {collapse['object_mean_mm']:.1f} mm")
    print(f"Q3 residual transfers across a heading change: {now_transfers}")

    rc.write_json(OUT / "summary.json", {
        "question": "does modelling the geometry in the projection remove the need "
                    "for a filter-side offset?",
        "heading_source": "odometry path tangent, never truth",
        "solver": "Gauss-Newton on (x,y) matching the mesh silhouette bottom-centre "
                  "pixel; seeded with the deployed contact-plane inversion",
        "q2_residual_collapse": collapse,
        "q2_per_camera_bias_mm": {
            "deployed": {c: deployed_bias[c].get("bias_mm") for c in rc.CAMERAS},
            "object_model": {c: object_bias[c].get("bias_mm") for c in rc.CAMERAS}},
        "q1_arms": arms_chart,
        "q3_transfer": {label: {f"{t}->{s}": {a: v for a, v in row.items()}
                                for (t, s), row in table.items()}
                        for label, table in transfer.items()
                        if not label.endswith("_offsets")},
        "q3_fitted_offsets_mm": {k: v for k, v in transfer.items()
                                 if k.endswith("_offsets")},
        "q1_raw_filter_honest_on_object_path": bool(raw_is_honest),
        "q3_transfers_across_heading": bool(now_transfers),
    })
    print(f"\nwrote -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
