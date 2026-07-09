#!/usr/bin/env python3
"""[REAL EXPERIMENT] Validate the reliability GP the RIGHT way (per the thesis
framework): does it predict HELD-OUT detector usability better than simple baselines?

Target A: z = detected {0,1} (real YOLO hit/miss from the logs).
Split: leave-one-route-out (the split that tests spatial generalisation, not
interpolation). Positions/labels via campaign_metrics (canonical belief/GT, self-checked).

  ============================ EVALUATION-ONLY INPUTS ============================
  Ground-truth robot position (gt_x/gt_y) AND the exact CAD shelf geometry
  (footprints + heights, from the SDF) are used here ONLY to score/compare the
  models. A deployed system does NOT get them: it would use the camera calibration
  (which is legitimately known) and, if available, a *sensed* height map (stereo /
  LiDAR / RGB-D) or WMS/CAD — never this ground truth. The camera-only baseline
  needs no shelf knowledge at all; the "geometry prior" here uses the exact CAD as
  an upper-bound reference for what perfect shelf knowledge would add (~+0.03 AUROC).
  ================================================================================

Models (predict P(detected) at a held-out location):
  B0 constant         train-set mean rate
  B1 range-only       kernel regression on range-to-camera
  B4 geometry prior   camera-informed visibility, calibrated to detection rate
  B5 data-only GP     RBF kernel regression on (x,y)
  B6 prior+residual   B4 prior + kernel-smoothed residual (shrinks to prior off-data)

Metrics on POOLED held-out predictions: Brier, NLL, AUROC, ECE (+ per-route rates).
Output: logs/geometry_visibility_prior/demo/gp_usability_validation.png
"""
from __future__ import annotations
import csv, glob, json, pathlib, sys
from collections import defaultdict
import numpy as np
from scipy.stats import rankdata

_HERE = pathlib.Path(__file__).resolve().parent
REPO = _HERE.parents[1]
sys.path.insert(0, str(REPO / "src" / "unav_common"))
sys.path.insert(0, str(_HERE))
import geometry_visibility as gv
import campaign_metrics as cm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ART = REPO / "paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz"
CFG = REPO / "scripts/visibility_comparison/warehouse_visibility_campaign.yaml"
CAMP = REPO / "logs/visibility_comparison/honest_campaign_v1"
OUT = REPO / "logs/geometry_visibility_prior/demo"; OUT.mkdir(parents=True, exist_ok=True)
Z = 0.35


def load_by_route():
    """Canonical per-detection events grouped by route: (gt_xy, detected)."""
    routes = defaultdict(list)
    for pf in sorted(glob.glob(str(CAMP / "*/*/*/*/perception.csv"))):
        route = pf.split("honest_campaign_v1/")[1].split("/")[0]
        run = cm.load_run(str(pathlib.Path(pf).parent / "experiment.csv"))  # asserts canonical
        est = run["stamp"]
        for r in csv.DictReader(open(pf)):
            if r.get("detected") not in ("0", "1"):
                continue
            try: t = float(r["log_stamp"])
            except Exception: continue
            j = int(np.argmin(np.abs(est - t)))
            if abs(est[j] - t) > 0.3 or not np.isfinite(run["truth_x"][j]):
                continue
            routes[route].append((run["truth_x"][j], run["truth_y"][j], int(r["detected"])))
    return {k: np.array(v) for k, v in routes.items()}


# ---- metrics ----
def brier(p, y): return float(np.mean((p - y) ** 2))
def nll(p, y): p = np.clip(p, 1e-4, 1 - 1e-4); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
def auroc(p, y):
    pos, neg = (y == 1).sum(), (y == 0).sum()
    if pos == 0 or neg == 0: return np.nan
    r = rankdata(p)
    return float((r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))
def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p <= edges[i + 1]) if i == bins - 1 else (p >= edges[i]) & (p < edges[i + 1])
        if m.sum(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def kreg(q, tx, ty, l):
    d2 = ((q[:, None, :] - tx[None, :, :]) ** 2).sum(-1)
    k = np.exp(-d2 / (2 * l * l))
    return (k * ty[None, :]).sum(1) / (k.sum(1) + 1e-9)


def main():
    print("NOTE: ground-truth positions and exact CAD shelf geometry/heights are used "
          "ONLY to score/compare models here — NOT as deployment inputs.\n")
    routes = load_by_route()
    names = list(routes)
    print("[REAL] per-route detections and hit rate:")
    for r in names:
        d = routes[r][:, 2]; print(f"  {r:34s} n={len(d):4d}  detected {d.mean():.2f}")

    # geometry prior grid (camera-informed visibility) for B4/B6 lookups
    meta = gv.load_gp_artifact_geometry(ART); xs, ys = meta["xs"], meta["ys"]
    cam = gv.make_camera(meta)
    fov = gv.fov_projection_grid(cam, xs, ys, Z); jac = gv.projection_jacobian_scale(cam, xs, ys, Z)
    clear = gv.raycast_min_clearance(cam, xs, ys, meta["prisms"], Z)
    geo = gv.compute_visibility(fov_mask=fov["fov_mask"], min_clearance=clear, px_per_m_min=jac["px_per_m_min"],
                                u=fov["u"], v=fov["v"], img_w=cam.img_width, img_h=cam.img_height, tau_clearance=0.10)["visibility_score"]
    campos = np.asarray(cam.cam_pos)
    def geo_at(P): return geo[np.clip(np.searchsorted(ys, P[:, 1]), 0, len(ys)-1), np.clip(np.searchsorted(xs, P[:, 0]), 0, len(xs)-1)]
    def rng_of(P): return np.sqrt((P[:, 0]-campos[0])**2 + (P[:, 1]-campos[1])**2 + campos[2]**2)

    MODELS = ["B0 const", "B1 range", "B4 geo-prior", "B5 data-GP", "B6 prior+resid"]
    pooled = {m: ([], []) for m in MODELS}   # (pred, y) pooled over held-out routes
    for held in names:                        # leave-one-route-out
        tr = np.concatenate([routes[r] for r in names if r != held])
        te = routes[held]
        Ptr, ytr = tr[:, :2], tr[:, 2]; Pte, yte = te[:, :2], te[:, 2]
        rtr, rte = rng_of(Ptr)[:, None], rng_of(Pte)[:, None]
        gtr, gte = geo_at(Ptr)[:, None], geo_at(Pte)[:, None]
        preds = {
            "B0 const": np.full(len(yte), ytr.mean()),
            "B1 range": kreg(rte, rtr, ytr, 1.0),
            "B4 geo-prior": kreg(gte, gtr, ytr, 0.08),
            "B5 data-GP": kreg(Pte, Ptr, ytr, 0.6),
        }
        prior_tr = kreg(gtr, gtr, ytr, 0.08); prior_te = preds["B4 geo-prior"]
        resid = kreg(Pte, Ptr, ytr - prior_tr, 0.6)
        preds["B6 prior+resid"] = np.clip(prior_te + resid, 1e-4, 1 - 1e-4)
        for m in MODELS:
            pooled[m][0].append(np.clip(preds[m], 1e-4, 1 - 1e-4)); pooled[m][1].append(yte)

    print(f"\nleave-one-route-out, pooled held-out ({sum(len(routes[r]) for r in names)} detections):")
    print(f"  {'model':16s} {'Brier↓':>8} {'NLL↓':>8} {'AUROC↑':>8} {'ECE↓':>8}")
    rows = {}
    for m in MODELS:
        p = np.concatenate(pooled[m][0]); y = np.concatenate(pooled[m][1]).astype(float)
        rows[m] = (brier(p, y), nll(p, y), auroc(p, y), ece(p, y))
        print(f"  {m:16s} {rows[m][0]:8.4f} {rows[m][1]:8.4f} {rows[m][2]:8.3f} {rows[m][3]:8.4f}")

    # figure: metric bars + calibration curves
    fig, ax = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True); fig.patch.set_facecolor("white")
    ms = list(MODELS); col = ["#999", "#3a6ea5", "#e5a13a", "#2a9d3a", "#8250c4"]
    ax[0].bar(ms, [rows[m][0] for m in ms], color=col); ax[0].set_ylabel("Brier (held-out, lower=better)")
    ax[0].set_title("A. Predicting held-out detector hit/miss", fontsize=11, loc="left"); ax[0].tick_params(axis="x", rotation=25)
    for i, m in enumerate(ms): ax[0].text(i, rows[m][0], f"{rows[m][0]:.3f}", ha="center", va="bottom", fontsize=8)
    ax[1].bar(ms, [rows[m][2] for m in ms], color=col); ax[1].set_ylabel("AUROC (higher=better)"); ax[1].axhline(0.5, ls="--", color="k", lw=1)
    ax[1].set_title("B. Ranking hits vs misses (AUROC)", fontsize=11, loc="left"); ax[1].tick_params(axis="x", rotation=25); ax[1].set_ylim(0.4, 1.0)
    for i, m in enumerate(ms): ax[1].text(i, rows[m][2], f"{rows[m][2]:.2f}", ha="center", va="bottom", fontsize=8)
    for m, c in zip(ms, col):
        p = np.concatenate(pooled[m][0]); y = np.concatenate(pooled[m][1]).astype(float)
        edges = np.linspace(0, 1, 9); cx, cy = [], []
        for i in range(8):
            sel = (p >= edges[i]) & (p < edges[i + 1])
            if sel.sum() > 20: cx.append(p[sel].mean()); cy.append(y[sel].mean())
        ax[2].plot(cx, cy, "o-", color=c, label=m, ms=4, lw=1.5)
    ax[2].plot([0, 1], [0, 1], "k--", lw=1); ax[2].set_xlabel("predicted P(detect)"); ax[2].set_ylabel("observed rate")
    ax[2].set_title("C. Calibration (held-out)", fontsize=11, loc="left"); ax[2].legend(fontsize=7)
    fig.suptitle("[REAL] Reliability GP validated on held-out detector usability (leave-one-route-out) vs baselines",
                 fontsize=13, fontweight="bold")
    fig.savefig(OUT / "gp_usability_validation.png", dpi=130, facecolor="white")
    print(f"\nwrote {OUT/'gp_usability_validation.png'}")


if __name__ == "__main__":
    main()
