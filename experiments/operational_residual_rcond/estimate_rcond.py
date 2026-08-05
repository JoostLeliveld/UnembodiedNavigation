#!/usr/bin/env python3
"""R1-R4 — operational R_cond from smoothed beliefs, scored against the oracle.

Pipeline per capture (operational streams only):

  odometry increments + per-camera world fixes
      -> KF forward + RTS backward            (R1, state.core.trajectory_smoother)
      -> r_t = z_t - h(mu_t^s), C_t = H P^s H^T + R_cond   (R2)
      -> per camera, twice: anchored by itself vs held out  (R3)
      -> compared against the GT-referenced oracle          (R4, EVAL ONLY)

Ground truth is opened only inside the functions marked EVALUATION ONLY, all of
which run after every covariance has been estimated.

Outputs -> logs/studies/operational_residual_rcond/exp2_operational_rcond/
"""

from __future__ import annotations

import numpy as np

import rcond_common as rc  # noqa: F401  (imported first: it puts src/* on sys.path)

from reliability.conditional_covariance import chi2_coverage, matrix_nll, sharpness  # noqa: E402
from reliability.operational_residual import (  # noqa: E402
    OperationalResidual,
    build_operational_residuals,
    circularity_factor,
    pooled_target,
    shrink_summary,
    summarize_residuals,
    total_covariances,
)
from state.core import trajectory_smoother as ts  # noqa: E402


#: The 2-DOF arm writes to its own directory so the exp2 artifact fitted against the
#: deployed v2 calibration is never overwritten by a different projection.
OUT = rc.OUT_ROOT / (
    "exp2_operational_rcond"
    if rc.calibration_path() == rc.DEPLOYED_CALIB
    else "exp3_two_dof_rcond"
)

#: Anchor covariance the SMOOTHER assumes (per-axis std, m). Frozen input, not
#: derived from the R_cond being estimated -- reusing the estimate here would close
#: the feedback loop the one-pass method is not entitled to. Swept for sensitivity.
ANCHOR_STD_M = 0.05
ANCHOR_SWEEP_M = (0.02, 0.05, 0.10, 0.20)

#: Rate-invariant odometry noise. These captures log at 50 Hz, so exp5's per-step
#: form would accumulate ~5x its intended drift variance (it adds q_base^2 per
#: *step*). Values are an odometry specification, not a fit: sigma_per_sqrt_m =
#: 0.04 puts ~0.15 m of 1-sigma drift over the ~14 m these routes cover, matching
#: the measured odometry-vs-truth displacement spread, and the time term adds
#: ~0.06 m over a 135 s capture.
PROCESS = ts.SmootherConfig(
    process_model=ts.PROCESS_RATE_INVARIANT,
    sigma_per_sqrt_s=0.005,
    sigma_per_sqrt_m=0.040,
    initial_position_std_m=0.30,
)

#: The estimator subtracts ``H P^s H^T``, so its accuracy is bounded by how well the
#: odometry drift is specified. Swept to expose that coupling, NOT to choose the Q
#: that makes R_cond look best.
PROCESS_SWEEP_SIGMA_PER_SQRT_M = (0.020, 0.040, 0.080, 0.160)


def _per_axis_sigma(matrix) -> float:
    return float(np.sqrt(max(np.trace(np.asarray(matrix, dtype=float)) / 2.0, 0.0)))


# --------------------------------------------------------------------------- #
# R1 / R2 / R3
# --------------------------------------------------------------------------- #


def smooth_capture(cap, anchor_std: float, hold_out: str | None = None, process=None):
    """Smooth one capture, optionally holding one camera out of the anchor set."""
    measurements, dropped = rc.measurements_for(cap, anchor_std)
    if hold_out is not None:
        measurements = ts.without_source(measurements, hold_out)
    if not measurements:
        return None, dropped
    increments = ts.increments_from_track(cap.odom)
    dt = np.zeros(cap.n_steps)
    dt[1:] = np.diff(cap.stamps)
    traj = ts.smooth_trajectory(
        increments, measurements, initial_mean=cap.odom[0],
        config=process if process is not None else PROCESS, dt=dt,
    )
    return traj, dropped


def residuals_for_camera(cap, traj, camera: str, anchor_std: float):
    """Operational residual records for one camera against a given trajectory."""
    measurements, _ = rc.measurements_for(cap, anchor_std)
    own = [m for m in measurements if m.source == camera]
    if not own:
        return []
    return build_operational_residuals(
        smoothed_mean=traj.smoothed_mean,
        smoothed_cov=traj.smoothed_cov,
        measurements=own,
        camera_id=camera,
        frame="xy",
        anchored_by=traj.sources,
    )


def collect(anchor_std: float, process=None):
    """Anchored and held-out residual records per camera, pooled over captures."""
    models, calib = rc.camera_models(), rc.deployed_calibration()
    anchored: dict[str, list[OperationalResidual]] = {c: [] for c in rc.CAMERAS}
    held_out: dict[str, list[OperationalResidual]] = {c: [] for c in rc.CAMERAS}
    trajectories = {}

    for name in rc.CAPTURES:
        cap = rc.load_operational_capture(name, models=models, calib=calib)
        full, _ = smooth_capture(cap, anchor_std, process=process)
        if full is None:
            continue
        trajectories[name] = (cap, full)
        for cam in rc.CAMERAS:
            if not cap.detections[cam]:
                continue
            anchored[cam].extend(residuals_for_camera(cap, full, cam, anchor_std))
            loo, _ = smooth_capture(cap, anchor_std, hold_out=cam, process=process)
            if loo is None:
                continue
            held_out[cam].extend(residuals_for_camera(cap, loo, cam, anchor_std))

    return trajectories, anchored, held_out


# --------------------------------------------------------------------------- #
# R4 and Gate R1 scoring — EVALUATION ONLY below this line
# --------------------------------------------------------------------------- #


def oracle_records(cap, camera: str, anchor_std: float) -> list[OperationalResidual]:
    """**EVALUATION ONLY.** Residuals referenced to ground truth.

    Built directly rather than through ``build_operational_residuals`` so that no
    truth array is ever passed to an inference function. ``state_projection`` is
    zero: truth has no uncertainty, which is exactly why the oracle needs no
    state correction and the operational estimator does.
    """
    table = rc.load_truth_table(cap.name)
    measurements, _ = rc.measurements_for(cap, anchor_std)
    out = []
    for meas in measurements:
        if meas.source != camera:
            continue
        # Recover the detection stamp for the truth join via the odometry index.
        stamp = float(cap.stamps[meas.index])
        truth = rc.truth_at(table, stamp)
        if truth is None:
            continue
        z = np.asarray(meas.z, dtype=float)
        predicted = (float(truth[0]), float(truth[1]))
        out.append(
            OperationalResidual(
                camera_id=camera,
                index=int(meas.index),
                frame="xy",
                measured=(float(z[0]), float(z[1])),
                predicted=predicted,
                residual=(float(z[0] - predicted[0]), float(z[1] - predicted[1])),
                state_projection=((0.0, 0.0), (0.0, 0.0)),
                anchored_by=(),
                held_out=True,
            )
        )
    return out


def collect_oracle(anchor_std: float) -> dict[str, list[OperationalResidual]]:
    """**EVALUATION ONLY.**"""
    models, calib = rc.camera_models(), rc.deployed_calibration()
    out: dict[str, list[OperationalResidual]] = {c: [] for c in rc.CAMERAS}
    for name in rc.CAPTURES:
        cap = rc.load_operational_capture(name, models=models, calib=calib)
        for cam in rc.CAMERAS:
            if cap.detections[cam]:
                out[cam].extend(oracle_records(cap, cam, anchor_std))
    return out


def gate_r1_calibration(trajectories) -> list[dict]:
    """**EVALUATION ONLY.** Filtered vs smoothed covariance calibration (Gate R1).

    Gate R1 is deliberately about NEES, not point error: exp5 already measured
    that the offline smoother does not beat a camera-anchored mean. What must
    improve is covariance honesty, because the smoothed belief is consumed as a
    training distribution.
    """
    rows = []
    for name, (cap, traj) in trajectories.items():
        table = rc.load_truth_table(name)
        truth = np.array(
            [
                (lambda p: (p[0], p[1]) if p else (np.nan, np.nan))(rc.truth_at(table, float(t)))
                for t in cap.stamps
            ],
            dtype=float,
        )
        nees_f = ts.nees(traj.filtered_mean, traj.filtered_cov, truth)
        nees_s = ts.nees(traj.smoothed_mean, traj.smoothed_cov, truth)
        err_f = np.linalg.norm(traj.filtered_mean - truth, axis=1)
        err_s = np.linalg.norm(traj.smoothed_mean - truth, axis=1)
        err_o = np.linalg.norm(cap.odom - truth, axis=1)

        # The residuals only ever live at detection instants, where the belief is
        # camera-anchored. A whole-track NEES is dominated by the long unanchored
        # stretches between fixes, so it is the wrong number for judging whether
        # ``H P^s H^T`` is the right size to subtract.
        measurements, _ = rc.measurements_for(cap, ANCHOR_STD_M)
        det_idx = np.array(sorted({m.index for m in measurements}), dtype=int)
        at_det = nees_s[det_idx] if det_idx.size else np.array([np.nan])

        rows.append(
            {
                "capture": name,
                "steps": cap.n_steps,
                "detection_steps": int(det_idx.size),
                "nees_median_filtered": round(float(np.nanmedian(nees_f)), 3),
                "nees_median_smoothed": round(float(np.nanmedian(nees_s)), 3),
                "nees_median_smoothed_at_detections": round(float(np.nanmedian(at_det)), 3),
                "nees_mean_filtered": round(float(np.nanmean(nees_f)), 3),
                "nees_mean_smoothed": round(float(np.nanmean(nees_s)), 3),
                "err_median_odometry_m": round(float(np.nanmedian(err_o)), 4),
                "err_median_filtered_m": round(float(np.nanmedian(err_f)), 4),
                "err_median_smoothed_m": round(float(np.nanmedian(err_s)), 4),
                "err_p95_filtered_m": round(float(np.nanpercentile(err_f, 95)), 4),
                "err_p95_smoothed_m": round(float(np.nanpercentile(err_s, 95)), 4),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


def summarize(anchored, held_out, oracle):
    """Per-camera table: operational (anchored + held out) vs oracle."""
    rows = []
    for cam in rc.CAMERAS:
        entry: dict[str, object] = {"camera": cam}
        summaries = {}
        for tag, records in (("anchored", anchored[cam]), ("held_out", held_out[cam]), ("oracle", oracle[cam])):
            if len(records) < 2:
                entry[f"{tag}_n"] = len(records)
                continue
            summary = summarize_residuals(records)
            summaries[tag] = summary
            entry[f"{tag}_n"] = summary.sample_count
            entry[f"{tag}_sigma_m"] = round(_per_axis_sigma(summary.state_corrected), 4)
            entry[f"{tag}_bias_m"] = round(summary.bias_norm, 4)
            entry[f"{tag}_bias_xy_m"] = [round(v, 4) for v in summary.mean_residual]
            entry[f"{tag}_raw_sigma_m"] = round(_per_axis_sigma(summary.raw_second_moment), 4)
            entry[f"{tag}_state_sigma_m"] = round(_per_axis_sigma(summary.mean_state_projection), 4)
            entry[f"{tag}_psd_projected"] = summary.psd_projection_applied
        if "held_out" in summaries and "anchored" in summaries:
            # A floored estimate carries no scale, so a ratio against it is
            # meaningless -- report the reason instead of a fake number.
            floored = [
                tag for tag in ("held_out", "anchored") if summaries[tag].psd_projection_applied
            ]
            if floored:
                entry["circularity_factor"] = None
                entry["circularity_undefined_because"] = f"psd-floored: {'+'.join(floored)}"
            else:
                entry["circularity_factor"] = round(
                    circularity_factor(summaries["held_out"], summaries["anchored"]), 3
                )
        if "held_out" in summaries and "oracle" in summaries:
            op = _per_axis_sigma(summaries["held_out"].state_corrected)
            orc = _per_axis_sigma(summaries["oracle"].state_corrected)
            entry["operational_over_oracle_sigma"] = (
                round(op / orc, 3)
                if orc > 0 and not summaries["held_out"].psd_projection_applied
                else None
            )
            if summaries["held_out"].psd_projection_applied:
                entry["operational_unresolved"] = (
                    "state uncertainty exceeds the residual second moment: this camera is "
                    "sharper than the odometry-anchored trajectory can resolve"
                )
            entry["bias_recovery_m"] = round(
                summaries["held_out"].bias_norm - summaries["oracle"].bias_norm, 4
            )
        rows.append(entry)
        entry["_summaries"] = summaries
    return rows


def score_predictive(records, r_cond) -> dict:
    """MNLL / coverage / sharpness under ``C_t = H P^s H^T + R_cond``."""
    residuals = [rec.residual for rec in records]
    covariances = total_covariances(records, r_cond)
    return {
        "mnll": round(matrix_nll(residuals, covariances), 4),
        "coverage_95": round(chi2_coverage(residuals, covariances, q=0.95), 4),
        "coverage_50": round(chi2_coverage(residuals, covariances, q=0.50), 4),
        "sharpness_log_det": round(sharpness(covariances), 4),
    }


def main() -> None:
    trajectories, anchored, held_out = collect(ANCHOR_STD_M)
    oracle = collect_oracle(ANCHOR_STD_M)
    rows = summarize(anchored, held_out, oracle)
    calibration = gate_r1_calibration(trajectories)

    # Held-out summaries feed the pooled shrinkage target and the predictive score.
    held_summaries = [r["_summaries"]["held_out"] for r in rows if "held_out" in r["_summaries"]]
    target = pooled_target(held_summaries) if held_summaries else None
    per_camera_r = {}
    scores = {}
    if target is not None:
        for row in rows:
            summaries = row["_summaries"]
            if "held_out" not in summaries:
                continue
            cam = row["camera"]
            estimate = shrink_summary(summaries["held_out"], target)
            per_camera_r[cam] = estimate.covariance
            row["shrunk_sigma_m"] = round(_per_axis_sigma(estimate.covariance), 4)
            row["shrinkage_lambda"] = round(estimate.shrinkage_lambda, 3)

        pooled_records = [rec for cam in rc.CAMERAS for rec in held_out[cam]]
        pooled_records = [r for r in pooled_records if r.camera_id in per_camera_r]
        constant = pooled_target(held_summaries)
        scores = {
            "per_camera_R_cond": score_predictive(pooled_records, per_camera_r),
            "constant_pooled_R_cond": score_predictive(pooled_records, constant),
            "n_scored": len(pooled_records),
        }

    # Sensitivity: does the estimate survive a different assumed anchor?
    sensitivity = []
    for std in ANCHOR_SWEEP_M:
        _, _, held = collect(std)
        entry: dict[str, object] = {"anchor_std_m": std}
        for cam in rc.CAMERAS:
            if len(held[cam]) >= 2:
                summary = summarize_residuals(held[cam])
                entry[cam] = round(_per_axis_sigma(summary.state_corrected), 4)
                entry[f"{cam}_floored"] = summary.psd_projection_applied
        sensitivity.append(entry)

    # Sensitivity to the assumed odometry drift. The estimator subtracts H P^s H^T,
    # so its accuracy is bounded by how well Q is specified; this quantifies that
    # coupling rather than picking the Q that flatters the result.
    process_sensitivity = []
    for sigma_m in PROCESS_SWEEP_SIGMA_PER_SQRT_M:
        process = ts.SmootherConfig(
            process_model=ts.PROCESS_RATE_INVARIANT,
            sigma_per_sqrt_s=PROCESS.sigma_per_sqrt_s,
            sigma_per_sqrt_m=sigma_m,
            initial_position_std_m=PROCESS.initial_position_std_m,
        )
        trajs, _, held = collect(ANCHOR_STD_M, process=process)
        calib = gate_r1_calibration(trajs)
        entry = {
            "sigma_per_sqrt_m": sigma_m,
            "nees_median_smoothed": [round(r["nees_median_smoothed"], 2) for r in calib],
            "nees_median_at_detections": [
                round(r["nees_median_smoothed_at_detections"], 2) for r in calib
            ],
        }
        for cam in rc.CAMERAS:
            if len(held[cam]) >= 2:
                summary = summarize_residuals(held[cam])
                entry[cam] = round(_per_axis_sigma(summary.state_corrected), 4)
                entry[f"{cam}_floored"] = summary.psd_projection_applied
        process_sensitivity.append(entry)

    for row in rows:
        row.pop("_summaries", None)

    payload = {
        "study": "operational_residual_rcond",
        "experiment": "exp2_operational_rcond",
        "frame": "xy",
        "units": "m (sigma), m^2 (covariance)",
        "anchor_std_m": ANCHOR_STD_M,
        "captures": list(rc.CAPTURES),
        "gate_r1_calibration": calibration,
        "per_camera": rows,
        "pooled_shrinkage_target": target,
        "per_camera_R_cond": per_camera_r,
        "predictive_scores": scores,
        "anchor_sensitivity": sensitivity,
        "process_sensitivity": process_sensitivity,
        "process_model": {
            "model": PROCESS.process_model,
            "sigma_per_sqrt_s": PROCESS.sigma_per_sqrt_s,
            "sigma_per_sqrt_m": PROCESS.sigma_per_sqrt_m,
            "initial_position_std_m": PROCESS.initial_position_std_m,
        },
        "ground_truth_use": "evaluation only (oracle comparison, NEES, point error)",
    }
    rc.write_json(OUT / "operational_rcond.json", payload)

    print("Gate R1 — covariance calibration (GT eval only; calibrated median NEES = 1.39)")
    print(
        f"{'capture':<26} {'NEES filt':>10} {'NEES smooth':>12} "
        f"{'@detections':>12} {'err med f':>10} {'err med s':>10}"
    )
    for row in calibration:
        print(
            f"{row['capture']:<26} {row['nees_median_filtered']:>10.2f} "
            f"{row['nees_median_smoothed']:>12.2f} "
            f"{row['nees_median_smoothed_at_detections']:>12.2f} "
            f"{row['err_median_filtered_m']:>10.4f} {row['err_median_smoothed_m']:>10.4f}"
        )

    print("\nR2/R3/R4 — per-camera sigma (per axis, m)")
    header = f"{'cam':<9} {'n':>5} {'anchored':>9} {'held-out':>9} {'oracle':>8} {'circ':>6} {'op/or':>6} {'bias ho':>8} {'bias or':>8}"
    print(header)
    for row in rows:
        n = row.get("held_out_n", 0)
        def g(key, fmt="{:>9.4f}", width=9):
            value = row.get(key)
            return fmt.format(value) if isinstance(value, (int, float)) else f"{'-':>{width}}"
        print(
            f"{row['camera']:<9} {n:>5} {g('anchored_sigma_m')} {g('held_out_sigma_m')} "
            f"{g('oracle_sigma_m', '{:>8.4f}', 8)} {g('circularity_factor', '{:>6.2f}', 6)} "
            f"{g('operational_over_oracle_sigma', '{:>6.2f}', 6)} "
            f"{g('held_out_bias_m', '{:>8.4f}', 8)} {g('oracle_bias_m', '{:>8.4f}', 8)}"
        )

    if scores:
        print("\nPredictive score on held-out-referenced residuals (C_t = H P^s H^T + R_cond)")
        for tag, entry in scores.items():
            if isinstance(entry, dict):
                print(f"  {tag:<24} MNLL {entry['mnll']:>8.3f}  cov95 {entry['coverage_95']:.3f}  "
                      f"cov50 {entry['coverage_50']:.3f}  sharp {entry['sharpness_log_det']:>8.3f}")

    def _cams(entry) -> str:
        parts = []
        for cam in rc.CAMERAS:
            if cam not in entry:
                continue
            flag = "*" if entry.get(f"{cam}_floored") else " "
            parts.append(f"{cam.split('_')[1]}={entry[cam]:.4f}{flag}")
        return "  ".join(parts)

    print("\nAnchor-std sensitivity (held-out sigma, m; * = PSD-floored, unresolved)")
    for entry in sensitivity:
        print(f"  anchor {entry['anchor_std_m']:.2f} m -> {_cams(entry)}")

    print("\nOdometry-drift sensitivity (held-out sigma, m; * = PSD-floored)")
    for entry in process_sensitivity:
        print(
            f"  sigma/sqrt(m) {entry['sigma_per_sqrt_m']:.3f} -> {_cams(entry)}"
            f"   NEES all {entry['nees_median_smoothed']}"
            f"  @det {entry['nees_median_at_detections']}"
        )
    print("\noracle sigma (GT-referenced, eval only): " + "  ".join(
        f"{r['camera'].split('_')[1]}={r['oracle_sigma_m']:.4f}"
        for r in rows if "oracle_sigma_m" in r
    ))

    print(f"\n-> {OUT / 'operational_rcond.json'}")


if __name__ == "__main__":
    main()
