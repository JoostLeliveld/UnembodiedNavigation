#!/usr/bin/env python3
"""exp2: is the honest-belief result real, or is the floor fitted in-sample?

exp1's headline rests on one number per camera -- the residual bias budget that
floors the posterior covariance. Those numbers were measured on the same three
captures exp1 evaluates. That is exactly the kind of thing that produces a
beautiful calibration curve and no transferable method, so it has to be checked
before the result is quoted anywhere.

Three checks:

  C1  LEAVE-ONE-CAPTURE-OUT. Re-measure the per-camera bias from two captures and
      evaluate on the third, which contributed nothing to it. If the honesty
      survives, the floor is a camera property. If it collapses, it was a fit.

  C2  HOW WRONG CAN THE FLOOR BE? Scale every floor by 0.25x .. 4x and watch the
      calibration degrade. A mechanism that only works at exactly the measured
      value is a tuned constant wearing a mechanism's clothes.

  C3  IS THE BIAS EVEN STABLE ACROSS CAPTURES? Report the per-camera bias measured
      separately per capture. This is the assumption the whole floor depends on,
      and it is cheap to state plainly.

Ground truth is EVALUATION ONLY: it measures the bias budget (a commissioning
quantity, assumption A1 of the register) and scores the arms.

Outputs -> logs/studies/bayesian_filter_showcase/exp2_does_it_generalize/
"""

from __future__ import annotations

import json
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
                  "experiments/bayesian_filter_showcase"):
    sys.path.insert(0, str(REPO / _relative))

import rcond_common as rc  # noqa: E402
import exp1_graceful_vs_trusting as EXP1  # noqa: E402

OUT = REPO / "logs/studies/bayesian_filter_showcase/exp2_does_it_generalize"
ARMS_CHECKED = ("A0_trust_everything", "A4_correlation_floor")
SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)


def measured_bias_per_capture(models, calib) -> dict[str, dict[str, float]]:
    """Per-capture, per-camera mean projection error magnitude. EVALUATION ONLY."""

    out: dict[str, dict[str, float]] = {}
    for name in rc.CAPTURES:
        capture = rc.load_operational_capture(name, models=models, calib=calib)
        table = rc.load_truth_table(name)
        per_camera: dict[str, float] = {}
        for camera in rc.CAMERAS:
            errors = []
            for detection in capture.detections[camera]:
                truth = rc.truth_at(table, detection.stamp)
                if truth is None:
                    continue
                errors.append(np.asarray(detection.world, dtype=float)
                              - np.asarray(truth[:2], dtype=float))
            if len(errors) >= 5:
                per_camera[camera] = float(np.hypot(*np.mean(errors, axis=0)))
        out[name] = per_camera
    return out


def evaluate(models, calib, bias_budget: dict[str, float], captures, arms):
    """Run the listed arms over the listed captures under a given bias budget."""

    original = dict(EXP1.RESIDUAL_BIAS_M)
    EXP1.RESIDUAL_BIAS_M.update(bias_budget)
    try:
        pooled: dict[str, list] = {arm: [] for arm in arms}
        for name in captures:
            capture = rc.load_operational_capture(name, models=models, calib=calib)
            table = rc.load_truth_table(name)

            def truth(stamp: float, _table=table):
                return rc.truth_at(_table, stamp)

            for arm in arms:
                pooled[arm].extend(EXP1.run_arm(capture, arm, truth)["records"])
        return {arm: EXP1.summarize(pooled[arm]) for arm in arms}
    finally:
        EXP1.RESIDUAL_BIAS_M.clear()
        EXP1.RESIDUAL_BIAS_M.update(original)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    EXP1._style()
    models, calib = rc.camera_models(), rc.deployed_calibration()
    per_capture_bias = measured_bias_per_capture(models, calib)

    # ---- C3: is the bias stable across captures at all? --------------------
    stability = {}
    for camera in rc.CAMERAS:
        values = [per_capture_bias[n][camera] for n in per_capture_bias
                  if camera in per_capture_bias[n]]
        if values:
            stability[camera] = {
                "per_capture_m": {n: per_capture_bias[n].get(camera)
                                  for n in per_capture_bias},
                "mean_m": float(np.mean(values)),
                "spread_ratio": float(max(values) / max(min(values), 1e-6)),
            }

    # ---- C1: leave-one-capture-out ----------------------------------------
    loo = {}
    for held_out in rc.CAPTURES:
        others = [n for n in rc.CAPTURES if n != held_out]
        budget = {}
        for camera in rc.CAMERAS:
            values = [per_capture_bias[n][camera] for n in others
                      if camera in per_capture_bias[n]]
            budget[camera] = float(np.mean(values)) if values else 0.02
        loo[held_out] = {
            "budget_from_others_m": budget,
            "results": evaluate(models, calib, budget, [held_out], ARMS_CHECKED),
        }

    # in-sample reference on the same single captures, for a like-for-like read
    in_sample = {
        name: evaluate(models, calib, dict(EXP1.RESIDUAL_BIAS_M), [name], ARMS_CHECKED)
        for name in rc.CAPTURES
    }

    # ---- C2: sensitivity to a wrong floor ---------------------------------
    sensitivity = {}
    for scale in SCALES:
        budget = {c: v * scale for c, v in EXP1.RESIDUAL_BIAS_M.items()}
        sensitivity[str(scale)] = evaluate(
            models, calib, budget, list(rc.CAPTURES), ("A4_correlation_floor",)
        )["A4_correlation_floor"]

    payload = {"bias_stability": stability, "leave_one_capture_out": loo,
               "in_sample_per_capture": in_sample, "floor_scale_sensitivity": sensitivity}
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------- figure
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.4))
    names = list(rc.CAPTURES)
    width = 0.35
    positions = np.arange(len(names))
    ax.bar(positions - width / 2,
           [100 * loo[n]["results"]["A0_trust_everything"]["unearned_confidence_fraction"]
            for n in names], width, color=EXP1.ARM_COLOR["A0_trust_everything"],
           label="A0 trust everything")
    ax.bar(positions + width / 2,
           [100 * loo[n]["results"]["A4_correlation_floor"]["unearned_confidence_fraction"]
            for n in names], width, color=EXP1.ARM_COLOR["A4_correlation_floor"],
           label="A4, floor from the OTHER captures")
    ax.axhline(5.0, color="#009E73", lw=2.0, ls="--", label="nominal 5 %")
    ax.set_xticks(positions, [n.replace("_2026", "\n2026") for n in names], fontsize=7.5)
    ax.set_ylabel("% of updates outside the stated 95 % ellipse")
    ax.set_title("Leave-one-capture-out:\nthe floor never saw this capture",
                 fontweight="bold", fontsize=10)
    ax.legend(fontsize=7.5)

    scales = [float(s) for s in sensitivity]
    ax2.plot(scales, [100 * sensitivity[str(s)]["unearned_confidence_fraction"]
                      for s in scales], "o-", color=EXP1.ARM_COLOR["A4_correlation_floor"],
             lw=2.0)
    ax2.axhline(5.0, color="#009E73", lw=2.0, ls="--", label="nominal 5 %")
    ax2.axvline(1.0, color="#666666", lw=1.2, ls=":", label="measured value")
    ax2.set_xscale("log")
    ax2.set_xlabel("floor scaled by")
    ax2.set_ylabel("% outside the stated 95 % ellipse")
    ax2.set_title("How wrong may the floor be?\n(flat = mechanism, spike = tuned constant)",
                  fontweight="bold", fontsize=10)
    ax2.legend(fontsize=7.5)
    fig.suptitle("Does the honest-belief result survive contact with data it was not fitted on?",
                 fontsize=12.0, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_v1_generalization.{ext}", bbox_inches="tight")
    plt.close(fig)

    print("C3  per-camera bias measured separately per capture (m):")
    for camera, entry in stability.items():
        cells = "  ".join(f"{n.split('_')[0]}={v:.3f}" if v else f"{n.split('_')[0]}=--"
                          for n, v in entry["per_capture_m"].items())
        print(f"    {camera}: {cells}   spread {entry['spread_ratio']:.1f}x")
    print("\nC1  leave-one-capture-out (floor fitted on the OTHER two):")
    print(f"    {'held-out capture':<28}{'arm':<24}{'medNEES':>9}{'unearned%':>11}{'stated cm':>11}")
    for name in names:
        for arm in ARMS_CHECKED:
            s = loo[name]["results"][arm]
            print(f"    {name:<28}{arm:<24}{s['median_nees']:>9.2f}"
                  f"{100 * s['unearned_confidence_fraction']:>11.1f}"
                  f"{100 * s['mean_stated_sigma_m']:>11.1f}")
    print("\nC2  floor scale sensitivity (A4):")
    for scale in SCALES:
        s = sensitivity[str(scale)]
        print(f"    x{scale:<6} medNEES {s['median_nees']:>6.2f}   "
              f"unearned {100 * s['unearned_confidence_fraction']:>5.1f}%   "
              f"stated {100 * s['mean_stated_sigma_m']:>5.1f} cm   "
              f"RMSE {100 * s['rmse_m']:>5.1f} cm")
    print("\nwrote ->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
