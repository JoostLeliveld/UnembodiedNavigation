#!/usr/bin/env python3
"""exp1: can the calibration gate be run in a real warehouse?

The bias fix that passed the belief-calibration gate
(`logs/studies/operational_residual_rcond/exp3_two_dof_rcond`) was selected with
Gazebo ground truth and evaluated on four cameras that each saw hundreds of
detections. A real warehouse has neither. Two questions decide whether any of it
deploys, and both are answerable from the captures already recorded:

  Q1  Can the gate |b_cross| / sigma_cross be COMPUTED WITHOUT TRUTH?
      If it needs a surveyed reference trajectory, commissioning a 100-camera
      network means surveying 100 camera footprints. If odometry plus the rest of
      the network suffices, it is a background task.

  Q2  How many detections does a camera need before the gate is trustworthy?
      This is the scaling question in disguise. In a large network most cameras
      are in camera A's regime -- few passes, thin coverage -- not camera C's.
      A gate that needs 500 detections per camera is not a gate, it is a survey.

Three estimators of the same quantity are compared:

    oracle       residual against Gazebo truth            (EVALUATION ONLY)
    operational  residual against the smoothed belief, with that camera HELD OUT
                 of its own reference (GT-free, deployable)
    op-corrected operational, minus the belief's own uncertainty projected onto
                 the cross axis: sigma^2_cond = sigma^2_op - E[u' H P^s H' u]

The state correction matters here for a specific reason: the operational residual
carries the robot's own position uncertainty, which inflates sigma_cross and so
DEFLATES the ratio. An uncorrected operational gate is therefore biased toward
"do not calibrate" -- the safe direction, but in this retired-v2 study it would leave
camera C's historical 76.9 mm signed lateral bias in place.

Sampling honesty: subsamples are CONTIGUOUS windows, not random draws. Detections
0.2 s apart are not independent, and "this camera has seen N detections" means one
or a few passes, not N independent looks. Random subsampling would overstate
sample efficiency by exactly the autocorrelation factor.

Outputs -> logs/studies/network_commissioning_realism/exp1_gate_without_truth/
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
                  "experiments/operational_residual_rcond"):
    sys.path.insert(0, str(REPO / _relative))

import estimate_rcond as ER  # noqa: E402  (owns collect / collect_oracle)
import rcond_common as rc  # noqa: E402

OUT = REPO / "logs/studies/network_commissioning_realism/exp1_gate_without_truth"

#: The gate deployed in fit_projection_calibration.py.
GATE = 1.2
#: Contiguous window sizes, in detections.
WINDOWS = (20, 40, 80, 160, 320)
DRAWS = 400
RNG = np.random.default_rng(20260804)
SIGMA_FLOOR_M = 1.0e-4

C_ORACLE = "#000000"
C_OP = "#E69F00"
C_OPC = "#0072B2"


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150,
        "axes.grid": True, "grid.color": "#CCCCCC",
        "grid.alpha": 0.3, "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 11, "font.size": 9,
    })


# ------------------------------------------------------- bearing-frame extraction


def cross_axis_series(records, camera_model) -> dict[str, np.ndarray]:
    """Cross-bearing residual component and its state-uncertainty share, per record.

    The bearing basis is built from the camera to the MEASURED point, consistently
    for every estimator, so the oracle and operational numbers are comparable. The
    projected-point-versus-raw-point distinction shifts the bearing direction by
    under a degree at these ranges and cancels in the comparison.
    """

    cam_x, cam_y = float(camera_model.cam_pos[0]), float(camera_model.cam_pos[1])
    cross, state_var, distance = [], [], []
    for record in records:
        bx = record.measured[0] - cam_x
        by = record.measured[1] - cam_y
        norm = math.hypot(bx, by)
        if norm <= 1.0e-9:
            continue
        unit_cross = np.array([-by / norm, bx / norm])
        residual = np.asarray(record.residual, dtype=float)
        cross.append(float(residual @ unit_cross))
        projection = np.asarray(record.state_projection, dtype=float)
        state_var.append(float(unit_cross @ projection @ unit_cross))
        distance.append(norm)
    return {
        "cross": np.asarray(cross),
        "state_var": np.asarray(state_var),
        "distance": np.asarray(distance),
    }


def gate_ratio(cross: np.ndarray, state_var: np.ndarray | None = None) -> dict:
    """|mean| / sigma on the cross axis, optionally state-corrected."""

    if cross.size < 2:
        return {"n": int(cross.size), "bias_m": math.nan, "sigma_m": math.nan,
                "ratio": math.nan, "state_corrected": state_var is not None,
                "floored": False}
    bias = float(cross.mean())
    variance = float(cross.var(ddof=1))
    floored = False
    if state_var is not None:
        variance -= float(state_var.mean())
        if variance <= SIGMA_FLOOR_M**2:
            variance = SIGMA_FLOOR_M**2
            floored = True
    sigma = math.sqrt(max(variance, SIGMA_FLOOR_M**2))
    return {"n": int(cross.size), "bias_m": bias, "sigma_m": sigma,
            "ratio": abs(bias) / sigma, "state_corrected": state_var is not None,
            "floored": floored}


# -------------------------------------------------------------- Q2: how many samples


def contiguous_windows(n: int, size: int, draws: int) -> list[np.ndarray]:
    """Random contiguous windows of length ``size`` -- one or a few robot passes."""

    if n < size:
        return []
    starts = RNG.integers(0, n - size + 1, draws)
    return [np.arange(start, start + size) for start in starts]


def sample_efficiency(series: dict, truth_decision: bool) -> dict:
    """P(the gate decides correctly) versus detections available, per estimator."""

    out: dict[str, dict] = {}
    for label, use_state in (("operational", False), ("op_corrected", True)):
        per_window = {}
        for size in WINDOWS:
            windows = contiguous_windows(series["cross"].size, size, DRAWS)
            if not windows:
                continue
            agree, ratios = [], []
            for index in windows:
                estimate = gate_ratio(
                    series["cross"][index],
                    series["state_var"][index] if use_state else None,
                )
                if not math.isfinite(estimate["ratio"]):
                    continue
                agree.append((estimate["ratio"] >= GATE) == truth_decision)
                ratios.append(estimate["ratio"])
            if not agree:
                continue
            per_window[size] = {
                "p_correct": float(np.mean(agree)),
                "ratio_median": float(np.median(ratios)),
                "ratio_p10": float(np.percentile(ratios, 10)),
                "ratio_p90": float(np.percentile(ratios, 90)),
                "windows": len(agree),
            }
        out[label] = per_window
    return out


# --------------------------------------------------------------------- figures


def fig_n1(table: dict) -> None:
    cameras = [c for c in rc.CAMERAS if table[c]["oracle"]["n"] >= 2]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    width = 0.26
    positions = np.arange(len(cameras))
    for offset, key, color, label in (
        (-width, "oracle", C_ORACLE, "oracle (ground truth) — EVAL ONLY"),
        (0.0, "operational", C_OP, "operational (GT-free)"),
        (width, "op_corrected", C_OPC, "operational, state-corrected"),
    ):
        values = [table[c][key]["ratio"] for c in cameras]
        bars = ax.bar(positions + offset, values, width, color=color, label=label)
        for bar, camera in zip(bars, cameras):
            if table[camera][key].get("floored"):
                ax.annotate(
                    f"{table[camera][key]['ratio']:.0f}\nDEGENERATE",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=6.5, color="#C1121F", fontweight="bold",
                )
    ax.axhline(GATE, color="#D55E00", lw=1.8, ls="--",
               label=f"deployed gate = {GATE}")
    ax.set_yscale("log")
    ax.set_xticks(positions, [c.replace("camera_", "") for c in cameras])
    ax.set_ylabel(r"$|b_{\rm cross}|\ /\ \sigma_{\rm cross}$   (log scale)")
    ax.set_title("The gate is computable without ground truth — but the textbook state\n"
                 "correction is degenerate, and the marginal camera flips the wrong way",
                 fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_n1_gate_without_truth.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig_n2(efficiency: dict, table: dict) -> None:
    cameras = [c for c in rc.CAMERAS if efficiency.get(c)]
    fig, axes = plt.subplots(1, len(cameras), figsize=(3.6 * len(cameras), 3.9),
                             sharey=True)
    for ax, camera in zip(np.atleast_1d(axes), cameras):
        for label, color in (("operational", C_OP), ("op_corrected", C_OPC)):
            per_window = efficiency[camera][label]
            sizes = sorted(per_window)
            if not sizes:
                continue
            ax.plot(sizes, [100 * per_window[s]["p_correct"] for s in sizes],
                    "o-", color=color, lw=1.9, ms=5, label=label)
        ax.axhline(90.0, color="#666666", lw=1.0, ls=":")
        ax.set_xscale("log")
        ax.set_ylim(0, 103)
        ax.set_xlabel("detections available\n(contiguous)")
        decision = "CALIBRATE" if table[camera]["oracle"]["ratio"] >= GATE else "leave raw"
        ax.set_title(f"{camera.replace('camera_', 'camera ')}\n"
                     f"correct answer: {decision}", fontweight="bold", fontsize=9.5)
    np.atleast_1d(axes)[0].set_ylabel("P(gate decides correctly)  [%]")
    np.atleast_1d(axes)[0].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("How much data before the gate can be trusted — the scaling question "
                 "for a large network", fontsize=12.0, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_n2_sample_efficiency.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig_n3(table: dict) -> dict:
    """Threshold sensitivity: where does each estimator put each camera?"""

    cameras = [c for c in rc.CAMERAS if table[c]["oracle"]["n"] >= 2]
    thresholds = np.linspace(0.4, 3.0, 53)
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    agreement = []
    for threshold in thresholds:
        oracle = {c: table[c]["oracle"]["ratio"] >= threshold for c in cameras}
        corrected = {c: table[c]["op_corrected"]["ratio"] >= threshold for c in cameras}
        agreement.append(sum(oracle[c] == corrected[c] for c in cameras) / len(cameras))
    ax.plot(thresholds, 100 * np.asarray(agreement), color=C_OPC, lw=2.2)
    ax.axvline(GATE, color="#D55E00", lw=1.8, ls="--", label=f"deployed gate = {GATE}")
    ax.set_ylim(0, 103)
    ax.set_xlabel("gate threshold")
    ax.set_ylabel("cameras where the GT-free gate\nagrees with the oracle  [%]")
    ax.set_title("Threshold sensitivity of the deployable gate", fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_n3_threshold_agreement.{ext}", bbox_inches="tight")
    plt.close(fig)
    perfect = [float(t) for t, a in zip(thresholds, agreement) if a == 1.0]
    return {
        "agreement_at_deployed_gate": float(
            agreement[int(np.argmin(np.abs(thresholds - GATE)))]
        ),
        "thresholds_with_full_agreement": [min(perfect), max(perfect)] if perfect else [],
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    _style()
    models = rc.camera_models()
    _trajectories, _anchored, held_out = ER.collect(ER.ANCHOR_STD_M)
    oracle = ER.collect_oracle(ER.ANCHOR_STD_M)

    table: dict[str, dict] = {}
    efficiency: dict[str, dict] = {}
    duty: dict[str, dict] = {}
    for camera in rc.CAMERAS:
        oracle_series = cross_axis_series(oracle[camera], models[camera])
        op_series = cross_axis_series(held_out[camera], models[camera])
        table[camera] = {
            # Ground truth reference carries no state uncertainty, so no correction.
            "oracle": gate_ratio(oracle_series["cross"]),
            "operational": gate_ratio(op_series["cross"]),
            "op_corrected": gate_ratio(op_series["cross"], op_series["state_var"]),
        }
        if oracle_series["cross"].size >= 2:
            truth_decision = table[camera]["oracle"]["ratio"] >= GATE
            efficiency[camera] = sample_efficiency(op_series, truth_decision)
            # Detections per second of robot time, so window sizes convert into
            # something an operator can schedule.
            stamps = [r.index for r in held_out[camera]]
            duty[camera] = {"detections": len(stamps)}

    payload = {
        "config": {"gate": GATE, "windows": list(WINDOWS), "draws": DRAWS,
                   "anchor_std_m": ER.ANCHOR_STD_M,
                   "calibration": str(rc.calibration_path().relative_to(REPO))},
        "gate_table": table,
        "sample_efficiency": efficiency,
        "duty": duty,
    }
    fig_n1(table)
    fig_n2(efficiency, table)
    payload["threshold_sensitivity"] = fig_n3(table)
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"gate = {GATE}, calibration in force = {payload['config']['calibration']}\n")
    header = f"{'camera':<10}{'n':>5}  " + "".join(f"{k:>22}" for k in
                                                   ("oracle", "operational", "op_corrected"))
    print(header)
    for camera in rc.CAMERAS:
        entry = table[camera]
        cells = ""
        for key in ("oracle", "operational", "op_corrected"):
            item = entry[key]
            if not math.isfinite(item["ratio"]):
                cells += f"{'--':>22}"
                continue
            mark = "*" if item.get("floored") else " "
            verdict = "CAL" if item["ratio"] >= GATE else "raw"
            cells += f"{item['ratio']:>13.2f}{mark} {verdict:>7}"
        print(f"{camera:<10}{entry['oracle']['n']:>5}  {cells}")
    print("\nsample efficiency, P(correct decision) by detections available:")
    for camera, per_label in efficiency.items():
        for label, per_window in per_label.items():
            sizes = sorted(per_window)
            if not sizes:
                continue
            trail = "  ".join(f"{s}:{100 * per_window[s]['p_correct']:.0f}%" for s in sizes)
            print(f"  {camera} {label:>13}: {trail}")
    print(f"\nthreshold sensitivity: {payload['threshold_sensitivity']}")
    print("wrote ->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
