#!/usr/bin/env python3
"""§10 confidence critique — EVALUATION-ONLY analysis (deliverable D).

Answers, on real honest_campaign_v1 detections:
  10A  does YOLO confidence predict operational quality?
  10B  does YOLO confidence predict the conditional GEOMETRIC (localization) error?

This analysis DELIBERATELY uses the Gazebo ground-truth localization residual as an
EVALUATION-ONLY label (sanctioned by §10B priority-3). It therefore lives OUTSIDE the
firewalled observability dataset: the residual is never an observability feature or label,
only the thing confidence is tested against. Confidence, bbox, and image position are
operational (PIXEL); range is operational (belief). Conclusions are not pre-written.

    python3 scripts/reliability/analyze_confidence.py \
        --campaign logs/visibility_comparison/honest_campaign_v1 \
        --output   logs/studies/usable_observation/confidence_v1
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[2]
for _pkg in (_ROOT / "src").glob("*"):
    if (_pkg / _pkg.name).is_dir():
        sys.path.insert(0, str(_pkg))
sys.path.insert(0, str(_ROOT / "scripts" / "geometry_visibility"))

import numpy as np  # noqa: E402

from reliability.confidence_analysis_stats import (  # noqa: E402
    loro_predictive_delta,
    partial_spearman,
    spearman,
    spearman_ci_by_run,
)

AWS_CAM = (0.0, -5.5, 4.8)


def _f(row, key):
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def load_eval_detections(campaign_dir: str) -> dict[str, np.ndarray]:
    """EVAL-ONLY per-detection table. GT residual is clearly isolated here."""
    import campaign_metrics as cm

    cols: dict[str, list] = {k: [] for k in
                             ("conf", "residual_m", "residual_cal_m", "bbox_area", "u", "v",
                              "range_belief_m", "route", "run")}
    for pf in sorted(glob.glob(str(pathlib.Path(campaign_dir) / "*/*/*/*/perception.csv"))):
        route = pf.split("honest_campaign_v1/")[1].split("/")[0]
        run = "/".join(pf.split("honest_campaign_v1/")[1].split("/")[:4])
        run_belief = cm.load_run(str(pathlib.Path(pf).parent / "experiment.csv"))
        est = run_belief["stamp"]; bx = run_belief["belief_x"]; by = run_belief["belief_y"]
        for r in csv.DictReader(open(pf)):
            if r.get("detected") != "1":
                continue
            conf = _f(r, "yolo_score_raw")
            res = _f(r, "localization_error_m")            # GT — EVALUATION ONLY
            res_cal = _f(r, "localization_error_calibrated_m")  # GT — EVALUATION ONLY
            if not (math.isfinite(conf) and math.isfinite(res)):
                continue
            t = _f(r, "log_stamp")
            rng = float("nan")
            if math.isfinite(t) and len(est):
                j = int(np.argmin(np.abs(est - t)))
                if abs(est[j] - t) <= 0.3 and math.isfinite(bx[j]):
                    rng = math.sqrt((bx[j] - AWS_CAM[0]) ** 2 + (by[j] - AWS_CAM[1]) ** 2 + AWS_CAM[2] ** 2)
            cols["conf"].append(conf); cols["residual_m"].append(res); cols["residual_cal_m"].append(res_cal)
            cols["bbox_area"].append(_f(r, "bbox_area_px")); cols["u"].append(_f(r, "obs_u")); cols["v"].append(_f(r, "obs_v"))
            cols["range_belief_m"].append(rng); cols["route"].append(route); cols["run"].append(run)
    return {k: (np.asarray(v, dtype=float) if k not in ("route", "run") else np.asarray(v)) for k, v in cols.items()}


def make_figures(d, out: pathlib.Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conf, res = d["conf"], d["residual_cal_m"]
    rng = d["range_belief_m"]
    m = np.isfinite(conf) & np.isfinite(res)

    # 1. confidence distribution among detections
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.hist(conf[m], bins=40, color="#3d6fb0")
    ax.set_title("YOLO confidence among detections (PIXEL)")
    ax.set_xlabel("yolo_score_raw"); ax.set_ylabel("count"); ax.set_xlim(0, 1)
    fig.tight_layout(); fig.savefig(out / "confidence_hist.png", dpi=130); plt.close(fig)

    # 2. binned residual vs confidence decile
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    q = np.quantile(conf[m], np.linspace(0, 1, 11))
    q[-1] += 1e-9
    idx = np.digitize(conf[m], q[1:-1])
    xs, ys = [], []
    for b in range(10):
        sel = idx == b
        if sel.sum() > 5:
            xs.append(conf[m][sel].mean()); ys.append(np.median(res[m][sel]))
    ax.plot(xs, ys, "o-", color="#b0413d")
    ax.set_title("median localization error vs confidence decile\n(GT residual — EVALUATION ONLY)")
    ax.set_xlabel("mean confidence in decile"); ax.set_ylabel("median residual [m] (GT)")
    fig.tight_layout(); fig.savefig(out / "residual_vs_confidence_decile.png", dpi=130); plt.close(fig)

    # 3. residual vs range, coloured by confidence (exposes the geometry confound)
    mr = m & np.isfinite(rng)
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    sc = ax.scatter(rng[mr], res[mr], c=conf[mr], s=6, alpha=0.4, cmap="viridis")
    ax.set_title("residual vs range, colour = confidence  (GT residual — EVAL ONLY)")
    ax.set_xlabel("camera range [m] (belief)"); ax.set_ylabel("residual [m] (GT)")
    ax.set_ylim(0, np.percentile(res[mr], 99))
    fig.colorbar(sc, ax=ax, label="confidence"); fig.tight_layout()
    fig.savefig(out / "residual_vs_range_by_confidence.png", dpi=130); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign", default="logs/visibility_comparison/honest_campaign_v1")
    ap.add_argument("--output", default="logs/studies/usable_observation/confidence_v1")
    ap.add_argument("--large-error-m", type=float, default=0.30)
    args = ap.parse_args()

    d = load_eval_detections(args.campaign)
    out = pathlib.Path(args.output); out.mkdir(parents=True, exist_ok=True)
    n = len(d["conf"])

    geom_stack = np.column_stack([d["range_belief_m"], d["bbox_area"], d["u"], d["v"]])
    valid = np.all(np.isfinite(geom_stack), axis=1) & np.isfinite(d["conf"]) & np.isfinite(d["residual_cal_m"])

    res = d["residual_cal_m"]
    results = {
        "evidence_status": "EVALUATION ONLY (GT localization residual used as the target, never as an observability feature/label)",
        "n_detections": int(n),
        "n_used_for_regression": int(valid.sum()),
        "confidence_distribution": {
            "p05": float(np.nanpercentile(d["conf"], 5)), "p50": float(np.nanpercentile(d["conf"], 50)),
            "p95": float(np.nanpercentile(d["conf"], 95)), "iqr": float(np.nanpercentile(d["conf"], 75) - np.nanpercentile(d["conf"], 25)),
        },
        "residual_distribution_m": {
            "p50": float(np.nanpercentile(res, 50)), "p90": float(np.nanpercentile(res, 90)),
            "p99": float(np.nanpercentile(res, 99)),
        },
        "spearman_conf_vs_residual_raw": spearman_ci_by_run(d["conf"], d["residual_m"], d["run"]),
        "spearman_conf_vs_residual_calibrated": spearman_ci_by_run(d["conf"], d["residual_cal_m"], d["run"]),
        "confound_spearman": {
            "conf_vs_range": spearman(d["conf"], d["range_belief_m"]),
            "range_vs_residual": spearman(d["range_belief_m"], d["residual_cal_m"]),
            "bbox_vs_residual": spearman(d["bbox_area"], d["residual_cal_m"]),
        },
        "partial_spearman_conf_vs_residual_given_geometry": partial_spearman(
            d["conf"][valid], res[valid], geom_stack[valid]),
        "large_error_threshold_m": args.large_error_m,
    }

    # held-out predictive test: does confidence add value beyond geometry for flagging large error?
    y_big = (res[valid] > args.large_error_m).astype(float)
    X_geom = geom_stack[valid]
    X_geom_conf = np.column_stack([X_geom, d["conf"][valid]])
    results["large_error_prevalence"] = float(y_big.mean())
    results["heldout_predictive_delta"] = loro_predictive_delta(
        X_geom, X_geom_conf, y_big, d["route"][valid])

    make_figures(d, out)

    # evidence-based conclusion (A/B/C/D) — computed, not pre-written
    part = results["partial_spearman_conf_vs_residual_given_geometry"]
    delta = results["heldout_predictive_delta"]
    auprc_geom = delta["geometry"]["auprc"]; auprc_both = delta["geometry+confidence"]["auprc"]
    adds_value = np.isfinite(auprc_both) and np.isfinite(auprc_geom) and (auprc_both - auprc_geom) > 0.02
    rho_cal = results["spearman_conf_vs_residual_calibrated"]["rho"]
    naive_holds = rho_cal < -0.1      # high conf -> low error (the assumed inverse-covariance sign)
    reversed_sign = rho_cal > 0.1     # high conf -> HIGHER error (opposite of the assumption)
    partial_strong = abs(part) > 0.2
    if naive_holds and adds_value:
        outcome = "B (confidence predicts usability AND conditional error, as inverse covariance)"
    elif reversed_sign:
        outcome = (
            f"C-reversed: confidence is POSITIVELY associated with localization error "
            f"(partial Spearman {part:.2f} after geometry controls); mapping confidence to "
            f"inverse covariance would be BACKWARDS. It adds "
            f"{'some' if adds_value else 'NO'} out-of-route predictive value beyond geometry "
            f"(AUPRC {auprc_geom:.3f}->{auprc_both:.3f})."
        )
    elif adds_value and partial_strong:
        outcome = "A/B (confidence adds error-predictive value beyond geometry, but not as inverse covariance)"
    elif not adds_value and not partial_strong:
        outcome = "C (confidence predicts neither conditional error after controlling for geometry)"
    else:
        outcome = "D (evidence insufficient / mixed)"
    results["conclusion_code"] = outcome
    results["naive_high_conf_low_error_holds"] = bool(naive_holds)
    results["reversed_sign_high_conf_high_error"] = bool(reversed_sign)
    results["confidence_adds_value_beyond_geometry"] = bool(adds_value)

    with open(out / "confidence_analysis.json", "w", encoding="utf-8") as h:
        json.dump(results, h, indent=2, default=str)
    print(json.dumps({
        "n": n,
        "spearman_conf_vs_residual_cal": results["spearman_conf_vs_residual_calibrated"],
        "partial_spearman_given_geometry": part,
        "heldout_predictive_delta": delta,
        "conclusion_code": outcome,
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
