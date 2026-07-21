#!/usr/bin/env python3
"""[REAL] Validate the prior on the RIGHT data: the UNIFORM teleport grid, which
covers the whole driveable region INCLUDING occluded cells — not the route-biased
driven detections (100% in-FOV, 96% visible) that made occlusion look irrelevant.

stack_capture2 samples.csv: 108 samples / 54 grid positions, 35% occluded (per the
capture's oracle), YOLO hit rate 0.87 visible vs 0.00 occluded. Here occlusion is the
dominant predictor, so it actually tests whether an occlusion term helps.

Predict per-sample YOLO detection with:
  camera-only  (FOV+range+boundary, NO occlusion)
  full geometry (camera + CAD occlusion incl. the dropped stack)   [CAD = EVAL-ONLY ref]
Metrics: AUROC, Brier, and stratified hit rates. CAD is a reference, not a deployment input.
"""
from __future__ import annotations
import csv, json, pathlib, sys
import numpy as np
from scipy.stats import rankdata

_HERE = pathlib.Path(__file__).resolve().parent
REPO = _HERE.parents[1]
sys.path.insert(0, str(REPO / "src" / "unav_common"))
sys.path.insert(0, str(_HERE))
import geometry_visibility as gv

ART = REPO / "paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz"
CFG = REPO / "scripts/visibility_comparison/warehouse_visibility_campaign.yaml"
SAMP = REPO / "logs/visibility_comparison/stack_capture2/samples.csv"
TGT = REPO / "logs/visibility_comparison/stack_targets2/perception_targets.csv"
Z = 0.35
# the 2.6 m stack that was physically present in stack_capture2 (full A2 aisle width)
STACK = gv.Prism("dropped_pallet_A2", xmin=-1.425, xmax=-0.525, ymin=-0.35, ymax=0.20, zmin=0.0, zmax=2.6)


def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y); pos, neg = (y == 1).sum(), (y == 0).sum()
    if pos == 0 or neg == 0: return np.nan
    r = rankdata(s); return float((r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def brier(p, y): return float(np.mean((np.clip(p, 1e-4, 1 - 1e-4) - y) ** 2))


def main():
    det = {}
    for r in csv.DictReader(open(TGT)):
        det[r.get("sample_id", "")] = 1 if str(r.get("yolo_detected_after_threshold", "")).strip() in ("1", "1.0", "True", "true") else 0
    X, Yd, Yorac = [], [], []
    for r in csv.DictReader(open(SAMP)):
        sid = r.get("sample_id", "")
        if sid not in det: continue
        X.append((float(r["x"]), float(r["y"]))); Yd.append(det[sid])
        Yorac.append(1 if r.get("oracle_occlusion_reason") == "occluded" else 0)
    X = np.array(X); Yd = np.array(Yd); Yorac = np.array(Yorac)
    print(f"[REAL uniform teleport] {len(Yd)} samples, hit rate {Yd.mean():.2f}; "
          f"occluded(oracle) {Yorac.mean()*100:.0f}%")

    meta = gv.load_gp_artifact_geometry(ART); xs, ys = meta["xs"], meta["ys"]; cam = gv.make_camera(meta)
    fov = gv.fov_projection_grid(cam, xs, ys, Z); jac = gv.projection_jacobian_scale(cam, xs, ys, Z)
    def vis(min_clear):
        return gv.compute_visibility(fov_mask=fov["fov_mask"], min_clearance=min_clear, px_per_m_min=jac["px_per_m_min"],
                                     u=fov["u"], v=fov["v"], img_w=cam.img_width, img_h=cam.img_height, tau_clearance=0.10)["visibility_score"]
    cam_only = vis(np.full((len(ys), len(xs)), 10.0))
    full = vis(gv.raycast_min_clearance(cam, xs, ys, meta["prisms"] + [STACK], Z))  # CAD occlusion, EVAL-ONLY
    at = lambda f: f[np.clip(np.searchsorted(ys, X[:, 1]), 0, len(ys) - 1), np.clip(np.searchsorted(xs, X[:, 0]), 0, len(xs) - 1)]

    print("\npredicting per-sample YOLO detection on UNIFORM samples that reach occluded cells:")
    for name, s in [("camera-only (NO occlusion)", at(cam_only)),
                    ("full geometry (+CAD occlusion)", at(full)),
                    ("oracle_visible label (pure occlusion)", 1.0 - Yorac)]:
        print(f"  {name:34s}  AUROC {auroc(s, Yd):.3f}   Brier {brier(s, Yd):.3f}")
    print(f"\nstratified YOLO hit rate:  oracle-visible {Yd[Yorac==0].mean():.2f} (n={(Yorac==0).sum()})   "
          f"oracle-occluded {Yd[Yorac==1].mean():.2f} (n={(Yorac==1).sum()})")
    print("Contrast with route-biased driven data (3.9% occluded): there camera-only AUROC 0.96 was")
    print("dominated by the near/far gradient; here, where 35% of samples are occluded, occlusion is decisive.")


if __name__ == "__main__":
    main()
