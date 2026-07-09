#!/usr/bin/env python3
"""HARD EVIDENCE ONLY — real Gazebo+YOLO experiments and real run logs.

Nothing in this script uses a geometry model, a synthetic point cloud, an invented
camera, or the learned-GP-as-truth. Every number comes from measured data:
  * real driven runs   logs/visibility_comparison/honest_campaign_v1/*  (per-timestep
    truth/GT/belief/covariance + per-detection YOLO hit/miss)
  * real teleport capture logs/visibility_comparison/stack_capture2 (live Gazebo+YOLO)

Findings (all measured):
  A. EKF-reported covariance is overconfident vs the actual belief-vs-GT error.
  B. Empirical observability map: where YOLO actually detects the robot.
  C. Certainty-weighting held-out cross-validation: does weighting a training
     detection by its localization certainty improve prediction of HELD-OUT real
     detections? (naive vs down-weight-gate vs spread-by-uncertainty)

Output: logs/geometry_visibility_prior/demo/hard_evidence.png
"""
from __future__ import annotations
import csv, glob, pathlib
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
CAMP = REPO / "logs/visibility_comparison/honest_campaign_v1"
CAP2 = REPO / "logs/visibility_comparison/stack_capture2"
TGT2 = REPO / "logs/visibility_comparison/stack_targets2/perception_targets.csv"
OUT = REPO / "logs/geometry_visibility_prior/demo"; OUT.mkdir(parents=True, exist_ok=True)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _f(r, k):
    v = r.get(k, "")
    try: return float(v)
    except Exception: return np.nan


def _major(xx, xy, yy):
    out = np.full(len(xx), np.nan)
    for i in range(len(xx)):
        if np.isnan(xx[i]): continue
        M = np.array([[xx[i], xy[i]], [xy[i], yy[i]]])
        out[i] = np.sqrt(max(np.linalg.eigvalsh(M).max(), 0.0))
    return out


def load_runs():
    """Every honest_campaign_v1 run: per-timestep (planner cov, state cov, gt error)
    and per-detection (belief xy, detected, planner-cov major)."""
    exp_rows, det_events = [], []
    for exp in sorted(glob.glob(str(CAMP / "*/*/*/*/experiment.csv"))):
        d = pathlib.Path(exp).parent
        er = list(csv.DictReader(open(exp)))
        exp_rows += er
        est = np.array([_f(r, "stamp") for r in er])
        pmaj = _major(np.array([_f(r, "planner_cov_x") for r in er]),
                      np.array([_f(r, "planner_cov_xy") for r in er]),
                      np.array([_f(r, "planner_cov_y") for r in er]))
        pfile = d / "perception.csv"
        if not pfile.is_file(): continue
        for r in csv.DictReader(open(pfile)):
            if r.get("detected") not in ("0", "1") or r.get("state_available") != "1": continue
            if r.get("state_x") in ("", "nan"): continue
            t = _f(r, "log_stamp"); j = int(np.argmin(np.abs(est - t))) if len(est) else 0
            sig = pmaj[j] if len(pmaj) and not np.isnan(pmaj[j]) else np.nan
            det_events.append(dict(x=_f(r, "state_x"), y=_f(r, "state_y"),
                                   det=int(r["detected"]), sig=sig))
    return exp_rows, det_events


def main():
    exp, _ = load_runs()  # timesteps for finding A (uses the canonical belief_error_gt_m column)
    import campaign_metrics as cm  # canonical belief for detection events (NOT stale state_x/y)
    ev = [dict(x=e["belief"][0], y=e["belief"][1], det=e["detected"],
               sig=e["reported_sigma_m"]) for e in cm.load_detections(str(CAMP))]
    print(f"REAL DATA: {len(exp)} run timesteps, {len(ev)} real detection events "
          f"across {len(glob.glob(str(CAMP/'*/*/*/*/experiment.csv')))} driven runs")

    # ---- A. EKF overconfidence (measured) -----------------------------------
    pmaj = _major(np.array([_f(r, "planner_cov_x") for r in exp]),
                  np.array([_f(r, "planner_cov_xy") for r in exp]),
                  np.array([_f(r, "planner_cov_y") for r in exp]))
    smaj = np.array([_f(r, "state_sigma_major_m") for r in exp])
    gterr = np.array([_f(r, "belief_error_gt_m") for r in exp])
    m = np.isfinite(pmaj) & np.isfinite(gterr) & np.isfinite(smaj)
    pmaj, smaj, gterr = pmaj[m], smaj[m], gterr[m]
    # calibration: fraction of timesteps where the ACTUAL error exceeds the reported 2-sigma
    exceed_state = float(np.mean(gterr > 2 * smaj))
    exceed_planner = float(np.mean(gterr > 2 * pmaj))
    print("\nA. EKF OVERCONFIDENCE (real logs):")
    print(f"   actual belief-vs-GT error: median {np.median(gterr):.3f} m, p95 {np.percentile(gterr,95):.3f} m")
    print(f"   reported state_sigma_major: median {np.median(smaj):.3f} m (should bound the error)")
    print(f"   frac timesteps error > 2*state_sigma  : {exceed_state:.2f}  (calibrated ~0.05)")
    print(f"   frac timesteps error > 2*planner_sigma: {exceed_planner:.2f}")

    # ---- B. empirical observability map (measured) --------------------------
    allx, ally, alld = [], [], []
    for e in ev: allx.append(e["x"]); ally.append(e["y"]); alld.append(e["det"])
    for r in csv.DictReader(open(TGT2)):
        allx.append(_f(r, "x")); ally.append(_f(r, "y"))
        alld.append(1 if str(r["yolo_detected_after_threshold"]).strip() in ("1","1.0","True","true") else 0)
    allx, ally, alld = np.array(allx), np.array(ally), np.array(alld)
    print(f"\nB. EMPIRICAL OBSERVABILITY (real YOLO detections): {len(alld)} detections, "
          f"overall rate {alld.mean():.2f}")

    # ---- C. certainty-weighting held-out cross-validation (measured) --------
    E = [e for e in ev if np.isfinite(e["sig"])]
    X = np.array([[e["x"], e["y"]] for e in E]); D = np.array([e["det"] for e in E], float)
    S = np.array([e["sig"] for e in E]); N = len(E)
    L, TAU = 0.6, 0.15
    rs = np.random.RandomState(0); fold = rs.randint(0, 5, N)

    def predict(train, test, mode):
        Xtr, Dtr, Str = X[train], D[train], S[train]; Xte = X[test]
        d2 = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)  # (nte, ntr)
        if mode == "spread":
            l2 = (L ** 2 + Str[None, :] ** 2); k = np.exp(-d2 / (2 * l2)) / l2
        else:
            k = np.exp(-d2 / (2 * L ** 2))
        w = np.ones(len(Xtr)) if mode != "gate" else np.exp(-0.5 * (Str / TAU) ** 2)
        num = (k * (w * Dtr)[None, :]).sum(1); den = (k * w[None, :]).sum(1) + 1e-9
        return np.clip(num / den, 1e-4, 1 - 1e-4)

    print("\nC. CERTAINTY-WEIGHTING, held-out CV on real detections (5-fold):")
    res = {}
    for mode in ("naive", "gate", "spread"):
        brier, ll = [], []
        for f in range(5):
            te = np.where(fold == f)[0]; tr = np.where(fold != f)[0]
            p = predict(tr, te, mode)
            brier.append(np.mean((p - D[te]) ** 2))
            ll.append(-np.mean(D[te]*np.log(p) + (1-D[te])*np.log(1-p)))
        res[mode] = (np.mean(brier), np.mean(ll))
        print(f"   {mode:7s}: held-out Brier {np.mean(brier):.4f}  logloss {np.mean(ll):.4f}")

    # ---------------- figure (real data only) ----------------
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
    fig.patch.set_facecolor("white")
    # A: reported sigma vs actual error
    ax[0].scatter(smaj, gterr, s=6, alpha=0.15, c="#3a6ea5", label="state_cov (EKF)")
    ax[0].scatter(pmaj, gterr, s=6, alpha=0.15, c="#d1495b", label="planner_cov")
    lim = max(np.percentile(gterr, 99), np.percentile(pmaj, 99))
    ax[0].plot([0, lim], [0, lim], "k--", lw=1, label="reported = actual")
    ax[0].set_xlim(0, lim); ax[0].set_ylim(0, lim)
    ax[0].set_xlabel("reported σ_major [m]"); ax[0].set_ylabel("actual belief-vs-GT error [m]")
    ax[0].set_title(f"A. EKF overconfidence (real runs)\nstate_cov misses {exceed_state*100:.0f}% of errors at 2σ",
                    fontsize=11, loc="left"); ax[0].legend(fontsize=8)
    # B: empirical observability map
    sc = ax[1].scatter(allx, ally, c=alld, cmap="RdYlGn", vmin=0, vmax=1, s=14, alpha=0.6)
    ax[1].set_aspect("equal"); ax[1].set_xlabel("x [m]"); ax[1].set_ylabel("y [m]")
    ax[1].set_title(f"B. Measured YOLO detections ({len(alld)} pts)\nreal Gazebo+YOLO, no model",
                    fontsize=11, loc="left")
    fig.colorbar(sc, ax=ax[1], shrink=0.8).set_label("detected (1) / miss (0)", fontsize=8)
    # C: held-out CV
    modes = list(res); briers = [res[m][0] for m in modes]
    bars = ax[2].bar(modes, briers, color=["#3a6ea5", "#d1495b", "#2a9d3a"])
    ax[2].set_ylabel("held-out Brier score (lower=better)")
    ax[2].set_title("C. Certainty-weighting, held-out CV\non real detections (5-fold)", fontsize=11, loc="left")
    for b, v in zip(bars, briers): ax[2].text(b.get_x()+b.get_width()/2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax[2].set_ylim(0, max(briers)*1.2)
    fig.suptitle("HARD EVIDENCE — real Gazebo+YOLO experiments & real run logs ONLY (no geometry / synthetic / hypothetical)",
                 fontsize=13, fontweight="bold")
    fig.savefig(OUT / "hard_evidence.png", dpi=130, facecolor="white")
    print(f"\nwrote {OUT/'hard_evidence.png'}")


if __name__ == "__main__":
    main()
