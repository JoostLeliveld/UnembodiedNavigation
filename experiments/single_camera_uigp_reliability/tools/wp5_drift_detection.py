#!/usr/bin/env python3
"""WP5 / Paper-2 mechanism de-risk — controlled calibration-drift DETECTION.

GO/NO-GO for the Paper-2 fault-containment claim's detection half: does the
innovation-based health monitor (reliability.health_ewma) detect a controlled
calibration drift on the CLEAN single-camera commissioning data (where the
detector works), cleanly separated from a healthy run, with low false-alarm?

Why single-camera / why now: Paper-2 recon (2026-07-21) found >=2-camera overlap
is only 7-13% in the 4-cam world, so cross-camera disagreement is rarely
available and containment rests mostly on SINGLE-camera innovation health
monitoring. And the 4-cam detector is OOD (retrain required) — but the single
camera in warehouse_aws works, so this is the one clean go/no-go we can run
BEFORE the expensive retrain. If the monitor can't detect a controlled drift
here, the containment claim is in trouble regardless of the retrain.

Controlled ablation (WP5 fault-experiments list, allowed): a calibration fault
(camera yaw/translation drift with stale runtime calibration) manifests as a
PERSISTENT additive innovation bias. We inject nu' = nu_real + beta*d on real
logged innovations (images/detections unchanged), sweep the bias magnitude beta
(px), and drive the real InnovationHealthMonitor + HealthDebouncer.

NIS reconstruction (S not persisted): the runtime's per-update NIS is logged
(pixel_corr_nis, non-circular, vs the PRIOR belief). We approximate a CONSTANT
effective innovation covariance S_eff = c * Cov[nu_real], with scalar c fit so
median(nu' S_eff^-1 nu') at beta=0 matches the logged NIS median, then validate
the reconstructed-NIS shape (max/p95) against the logged NIS. NIS'(beta) uses the
same S_eff. Caveat reported: S is treated as constant (real S varies with P^-).

Data: single_cam_commissioning_v1 (one planner-agnostic coverage drive; detector
warehouse_yolo_detector_v1). Uses only unambiguous diagnostic columns
(pixel_corr_innov_u/v, pixel_corr_nis, pixel_corr_apply_stamp); NO gt/oracle.
Output -> logs/studies/single_camera_uigp_reliability/wp5_self_monitoring/
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "reliability"))
from reliability.health_ewma import (  # noqa: E402
    CalibrationHealthState,
    HealthDebouncer,
    InnovationHealthConfig,
    InnovationHealthMonitor,
)

EXP = REPO / ("logs/visibility_comparison/single_cam_commissioning_v1/coverage/commission/"
              "seed0/experiment_20260721_143305/experiment.csv")
GEOM = REPO / "paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz"
OUT = REPO / "logs/studies/single_camera_uigp_reliability/wp5_self_monitoring"

NIS_GATE = 9.21                       # runtime 2-dof 0.99 gate
BETAS_PX = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0)
DIRECTION = (1.0, 0.0)                 # horizontal pixel bias (models a yaw drift)


def _f(row, key):
    v = row.get(key, "")
    if v in ("", "nan", "NaN", None):
        return math.nan
    try:
        return float(v)
    except ValueError:
        return math.nan


def load_unique_corrections():
    """Distinct camera corrections in time order: (stamps, innovations Nx2, nis N)."""
    seen = {}
    for r in csv.DictReader(open(EXP)):
        st = r.get("pixel_corr_apply_stamp", "")
        if st in ("", None):
            continue
        u, v, nis = _f(r, "pixel_corr_innov_u"), _f(r, "pixel_corr_innov_v"), _f(r, "pixel_corr_nis")
        acc = _f(r, "pixel_corr_accepted")
        if not all(math.isfinite(x) for x in (u, v, nis)):
            continue
        if acc < 0.5:
            continue
        seen[st] = (float(st), u, v, nis)          # last write per stamp
    rows = sorted(seen.values(), key=lambda t: t[0])
    stamps = np.array([r[0] for r in rows])
    innov = np.array([[r[1], r[2]] for r in rows])
    nis = np.array([r[3] for r in rows])
    return stamps, innov, nis


def fit_effective_cov(innov, nis):
    """S_eff = c*Cov[nu]; c fit so median reconstructed NIS == median logged NIS."""
    Shat = np.cov(innov.T)
    Sinv0 = np.linalg.inv(Shat)
    q = np.einsum("ij,jk,ik->i", innov, Sinv0, innov)   # nu^T Shat^-1 nu
    c = float(np.median(q) / max(np.median(nis), 1e-9))
    return Shat, c


def nis_of(innov, Shat, c):
    Sinv = np.linalg.inv(Shat * c)
    return np.einsum("ij,jk,ik->i", innov, Sinv, innov)


def run_monitor(nis_seq, innov_seq, cfg=None, deb=None):
    mon = InnovationHealthMonitor(config=cfg or InnovationHealthConfig())
    debn = deb or HealthDebouncer()
    states, healths, biases = [], [], []
    for nis, nu in zip(nis_seq, innov_seq):
        h = mon.update(nis=float(nis), innovation_uv=(float(nu[0]), float(nu[1])), dropped=False)
        st = debn.step(consistent=bool(nis <= NIS_GATE))
        states.append(st)
        healths.append(h)
        biases.append(math.hypot(*mon.bias_ewma))
    return states, np.array(healths), np.array(biases)


def focal_px():
    d = np.load(GEOM, allow_pickle=True)
    w = float(np.asarray(d["img_width"]).ravel()[0])
    fov = float(np.asarray(d["fov_h_rad"]).ravel()[0])
    return (w / 2.0) / math.tan(fov / 2.0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stamps, innov, nis = load_unique_corrections()
    n = len(nis)
    dt = float(np.median(np.diff(stamps))) if n > 1 else float("nan")
    Shat, c = fit_effective_cov(innov, nis)
    nis_recon = nis_of(innov, Shat, c)
    fpx = focal_px()

    # --- validation of the constant-S reconstruction against logged NIS
    val = {
        "logged_nis": [float(np.median(nis)), float(np.percentile(nis, 95)), float(nis.max())],
        "recon_nis": [float(np.median(nis_recon)), float(np.percentile(nis_recon, 95)), float(nis_recon.max())],
        "corr": float(np.corrcoef(nis, nis_recon)[0, 1]),
        "healthy_bias_px": [float(np.mean(innov[:, 0])), float(np.mean(innov[:, 1]))],
        "c_scale": c,
        "focal_px": fpx,
    }

    d = np.asarray(DIRECTION, float)
    onset = n // 2
    rows = []
    for beta in BETAS_PX:
        seq = innov.copy()
        seq[onset:] = seq[onset:] + beta * d            # drift onset at the midpoint
        nseq = nis_of(seq, Shat, c)
        states, healths, biases = run_monitor(nseq, seq)
        pre = states[:onset]
        post = states[onset:]
        # detection: first DEGRADED at/after onset
        det_idx = next((i for i in range(onset, n) if states[i] == CalibrationHealthState.DEGRADED), None)
        false_alarm_pre = any(s != CalibrationHealthState.HEALTHY for s in pre)
        rows.append({
            "beta_px": beta,
            "approx_yaw_deg": math.degrees(beta / fpx),
            "frac_nis_over_gate_post": float(np.mean(nseq[onset:] > NIS_GATE)),
            "median_nis_post": float(np.median(nseq[onset:])),
            "health_pre": float(np.mean(healths[:onset])),
            "health_post_end": float(np.mean(healths[-max(1, onset // 5):])),
            "bias_px_post_end": float(np.mean(biases[-max(1, onset // 5):])),
            "false_alarm_pre_onset": bool(false_alarm_pre),
            "detected_degraded": det_idx is not None,
            "detection_delay_samples": (det_idx - onset) if det_idx is not None else None,
            "detection_delay_s": ((det_idx - onset) * dt) if (det_idx is not None and math.isfinite(dt)) else None,
        })

    # --- nominal long run (beta=0) false-alarm metric over the WHOLE sequence
    states0, healths0, biases0 = run_monitor(nis, innov)
    frac_non_healthy = float(np.mean([s != CalibrationHealthState.HEALTHY for s in states0]))

    _write(val, rows, frac_non_healthy, n, dt)
    print("wrote", OUT / "RESULTS.md")


def _write(val, rows, frac_non_healthy, n, dt):
    # smallest beta reliably detected with no pre-onset false alarm
    detected = [r for r in rows if r["detected_degraded"] and not r["false_alarm_pre_onset"] and r["beta_px"] > 0]
    thresh = min((r["beta_px"] for r in detected), default=None)
    healthy0 = next(r for r in rows if r["beta_px"] == 0.0)
    no_false_alarm = (frac_non_healthy == 0.0) and (not healthy0["false_alarm_pre_onset"])
    verdict = "PASS" if (thresh is not None and no_false_alarm) else (
        "PARTIAL" if thresh is not None else "NULL")

    tbl = ["| bias (px) | ~yaw (deg) | frac NIS>9.21 (post) | median NIS (post) | health pre | health post | \\|b\\| post (px) | false-alarm pre | DEGRADED? | delay (samp/s) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        delay = "-" if r["detection_delay_samples"] is None else f"{r['detection_delay_samples']}/{r['detection_delay_s']:.1f}s"
        tbl.append("| {beta_px:.0f} | {deg:.2f} | {frac:.2f} | {mnis:.2f} | {hpre:.2f} | {hpost:.2f} | {bias:.2f} | {fa} | {det} | {delay} |".format(
            beta_px=r["beta_px"], deg=r["approx_yaw_deg"], frac=r["frac_nis_over_gate_post"],
            mnis=r["median_nis_post"], hpre=r["health_pre"], hpost=r["health_post_end"],
            bias=r["bias_px_post_end"], fa=("YES" if r["false_alarm_pre_onset"] else "no"),
            det=("YES" if r["detected_degraded"] else "no"), delay=delay))

    md = f"""# WP5 — controlled calibration-drift DETECTION (single-camera, real drive)

**Paper-2 mechanism go/no-go (fault-containment, detection half).** Does the
innovation-based health monitor (`reliability.health_ewma`) detect a controlled
calibration drift on clean single-camera data, separated from healthy, with low
false-alarm? This is the cheapest honest probe before the 4-cam detector retrain,
and — given the 7-13% overlap finding — single-camera innovation monitoring is the
primary containment mechanism anyway.

**Evidence class:** REAL logged innovations + CONTROLLED ABLATION (injected
persistent innovation bias modelling a stale-calibration / camera-shift fault;
images/detections unchanged). Data: `single_cam_commissioning_v1`
(detector `warehouse_yolo_detector_v1`). No gt_*/oracle input. Reuses
`InnovationHealthMonitor` + `HealthDebouncer` unchanged (default config:
rho=0.1, rho_bias=0.05, m0=2.0, eta*=3/1/1/2/1; debouncer m_s=m_d=m_r=3, m_h=5).

## Fault model & NIS reconstruction
A calibration fault = a persistent additive innovation bias `nu' = nu_real + beta*d`,
`d = {DIRECTION}` (horizontal, models a yaw drift). S is not logged, so the runtime's
per-update NIS (logged `pixel_corr_nis`, non-circular vs the PRIOR belief) is
reproduced with a CONSTANT effective covariance `S_eff = c*Cov[nu]`, `c` fit to the
logged NIS median. Validation (reconstructed vs logged NIS, [median, p95, max]):
- logged:      {val['logged_nis'][0]:.3f} / {val['logged_nis'][1]:.3f} / {val['logged_nis'][2]:.3f}
- reconstructed: {val['recon_nis'][0]:.3f} / {val['recon_nis'][1]:.3f} / {val['recon_nis'][2]:.3f}  (per-frame corr {val['corr']:.2f})

Healthy innovation mean = ({val['healthy_bias_px'][0]:.2f}, {val['healthy_bias_px'][1]:.2f}) px;
focal ~= {val['focal_px']:.0f} px (px->deg via yaw ~= beta/focal). n={n} distinct
corrections, median dt={dt:.3f}s. *Caveat: S treated as constant (real S varies with P^-);
this reproduces the healthy NIS distribution but is an approximation.*

## Drift-detection sweep (drift onset at the midpoint, n/2)
{chr(10).join(tbl)}

*health = sigmoid(eta0 - eta1*max(0,m-m0) - eta2*|b| - ...); |b| = bias-EWMA norm,
the WP5-designated calibration-drift detector. DEGRADED requires m_s+m_d=6
consecutive NIS>9.21 samples (debounced).*

## Nominal long run (beta=0, full sequence)
Fraction of time NOT HEALTHY (false-alarm) = **{frac_non_healthy:.3f}** over n={n}.

## Verdict: {verdict}
- smallest bias reliably detected as DEGRADED (no pre-onset false alarm): **{('%.0f px (~%.2f deg)' % (thresh, math.degrees(thresh/val['focal_px']))) if thresh is not None else 'none'}**
- nominal false-alarm rate: **{frac_non_healthy:.3f}** {'(clean)' if no_false_alarm else '(NON-ZERO — check G5 stop-rule)'}

**Reading (G5 / stop-rule 6):**
- **PASS** -> the monitor catches a controlled drift with no healthy false alarm ->
  the fault-DETECTION mechanism is sound; proceed to the detector retrain and the
  real multi-camera fault-containment campaign.
- **PARTIAL** -> detects only large drifts, or the NIS gate is slow while the
  bias-EWMA |b| separates earlier -> report the |b| detector as primary, tune eta.
- **NULL / false alarms** -> stop-rule 6: the monitor is not usable downstream;
  the containment claim needs rework before any retrain investment.

*Generated by experiments/single_camera_uigp_reliability/tools/wp5_drift_detection.py.*
"""
    (OUT / "RESULTS.md").write_text(md, encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(
        {"validation": val, "sweep": rows, "nominal_false_alarm": frac_non_healthy,
         "n": n, "dt": dt, "verdict": verdict, "detection_threshold_px": thresh}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
