#!/usr/bin/env python3
"""[REAL] Build the PERFECTED initial prior: a calibrated geometric detection-probability
field + the two things the user's online GP update consumes (a logit prior MEAN and a
prior STRENGTH pseudo-count).

Motivation (diagnose_prior.py): the old visibility_score ranks reliability well but is a
badly mis-calibrated probability, so as a GP prior mean it biases the posterior. We keep
the exact same geometric features and fit a tiny logistic LINK -- a one-time
detector-response characterisation (a property of the camera+detector, transferable;
geometry is recomputed per layout). We prove the fix is CALIBRATION (not features) by
also recalibrating the old heuristic with Platt scaling: both fix calibration; the
logistic additionally drops the arbitrary product/percentile/edge params.

Honesty: the occluder geometry (COMPLETE CAD) supplies the clearance feature here as an
EVALUATION reference only -- deployment senses clearance from a depth/stereo height map
(depth_occlusion_prior.py showed depth recovers CAD-level occlusion). Ground-truth
positions are never used. Evaluation = leave-4x4-spatial-block-out CV on 556 per-sample
YOLO labels (positions atomic; whole regions held out).

Outputs: logs/geometry_visibility_prior/calibrated_prior_v1/{calibrated_prior.npz, calibrated_prior.png}
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
from unav_common.occlusion_geometry import parse_occlusion_scene_from_world
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ART = REPO / "paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz"
WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_aws.world.sdf"
TGT = REPO / "logs/visibility_comparison/warehouse_visibility_targets_v1/gp_targets.csv"
OUT = REPO / "logs/geometry_visibility_prior/calibrated_prior_v1"; OUT.mkdir(parents=True, exist_ok=True)
Z, NB = 0.35, 4
R_VISIBLE_UV, R_MISS_UV = 2.5, 40.0  # campaign config: pixel measurement std when detected / missed


def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y); pos, neg = (y == 1).sum(), (y == 0).sum()
    if pos == 0 or neg == 0: return np.nan
    r = rankdata(s); return float((r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))
def brier(p, y): return float(np.mean((np.clip(p, 1e-6, 1 - 1e-6) - y) ** 2))
def nll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
def ece(p, y, nb=10):
    p = np.asarray(p, float); y = np.asarray(y, float); e = 0.0
    for b in range(nb):
        m = (p >= b / nb) & (p < (b + 1) / nb) if b < nb - 1 else (p >= b / nb) & (p <= 1.0)
        if m.sum(): e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)
def reliability(p, y, nb=8):
    p = np.asarray(p, float); y = np.asarray(y, float); px, py = [], []
    edges = np.linspace(0, 1, nb + 1)
    for b in range(nb):
        m = (p >= edges[b]) & (p < edges[b + 1]) if b < nb - 1 else (p >= edges[b]) & (p <= 1.0)
        if m.sum() >= 5: px.append(p[m].mean()); py.append(y[m].mean())
    return np.array(px), np.array(py)


def main():
    meta = gv.load_gp_artifact_geometry(ART); xs, ys = meta["xs"], meta["ys"]; cam = gv.make_camera(meta)
    sc = parse_occlusion_scene_from_world(str(WORLD), model_name=("warehouse_walls", "warehouse_rack_occluders"),
                                          geometry_tags=("visual",))
    cad = [gv.Prism(p.name, p.xmin, p.xmax, p.ymin, p.ymax, p.zmin, p.zmax) for p in sc.prisms]

    fov = gv.fov_projection_grid(cam, xs, ys, Z); jac = gv.projection_jacobian_scale(cam, xs, ys, Z)
    clr = gv.raycast_min_clearance(cam, xs, ys, cad, Z)
    heur = gv.compute_visibility(fov_mask=fov["fov_mask"], min_clearance=clr, px_per_m_min=jac["px_per_m_min"],
                                 u=fov["u"], v=fov["v"], img_w=cam.img_width, img_h=cam.img_height,
                                 tau_clearance=0.10)["visibility_score"]

    # per-sample labels + snap
    X, Y = [], []
    for r in csv.DictReader(open(TGT)):
        X.append((float(r["x"]), float(r["y"]))); Y.append(int(round(float(r["yolo_score_raw"]))))
    X = np.array(X); Y = np.array(Y, float)
    ix = np.clip(np.searchsorted(xs, X[:, 0]), 0, len(xs) - 1); iy = np.clip(np.searchsorted(ys, X[:, 1]), 0, len(ys) - 1)
    infov = fov["fov_mask"][iy, ix]
    feat = gv.detector_features(clr[iy, ix], jac["px_per_m_min"][iy, ix])
    s_heur = heur[iy, ix]

    # ---- leave-block-out CV: heuristic (raw) vs Platt-recal heuristic vs calibrated logistic ----
    bx = np.clip(((X[:, 0] - xs.min()) / (xs.max() - xs.min()) * NB).astype(int), 0, NB - 1)
    by = np.clip(((X[:, 1] - ys.min()) / (ys.max() - ys.min()) * NB).astype(int), 0, NB - 1)
    block = bx * NB + by
    oof_platt = np.full(len(Y), np.nan); oof_log = np.full(len(Y), np.nan)
    for blk in np.unique(block):
        te = block == blk; tr = (~te) & infov
        if Y[tr].sum() in (0, tr.sum()): continue
        rp = gv.fit_detector_response(s_heur[tr, None], Y[tr])           # Platt: recalibrate the heuristic score
        rl = gv.fit_detector_response(feat[tr], Y[tr])                    # logistic on geometric features
        oof_platt[te] = np.where(infov[te], gv._sigmoid(rp["intercept"] + rp["coef"][0] * s_heur[te]), 0.0)
        oof_log[te] = np.where(infov[te], gv.calibrated_detection_prob(
            infov[te], clr[iy[te], ix[te]], jac["px_per_m_min"][iy[te], ix[te]], rl), 0.0)
    ok = np.isfinite(oof_log)
    print(f"[REAL] {len(Y)} per-sample YOLO labels; detect rate {Y.mean():.2f}; in-FOV {infov.mean()*100:.0f}%")
    print(f"\nleave-{NB}x{NB}-block-out CV ({ok.sum()} scored out-of-block):")
    print(f"  {'model':32s} {'AUROC':>6s} {'Brier':>6s} {'NLL':>6s} {'ECE':>6s}")
    for tag, s in [("heuristic visibility_score (old)", s_heur), ("heuristic + Platt recalibration", oof_platt),
                   ("calibrated logistic (NEW prior)", oof_log)]:
        print(f"  {tag:32s} {auroc(s[ok], Y[ok]):6.3f} {brier(s[ok], Y[ok]):6.3f} {nll(s[ok], Y[ok]):6.3f} {ece(s[ok], Y[ok]):6.3f}")

    # ---- ship: full-data detector-response fit + prior maps ----
    resp = gv.fit_detector_response(feat[infov], Y[infov])
    print(f"\nshipped detector response (fit on all {int(infov.sum())} in-FOV samples):")
    print(f"  p_detect = in_fov * sigmoid({resp['intercept']:+.2f} "
          f"{resp['coef'][0]:+.2f}*clearance_m {resp['coef'][1]:+.2f}*log10(px/m))")
    p_detect = gv.calibrated_detection_prob(fov["fov_mask"], clr, jac["px_per_m_min"], resp)
    prior_mean = gv.prior_logit_mean(p_detect)
    prior_n0 = gv.prior_pseudocount(fov["fov_mask"], clr, fov["u"], fov["v"], cam.img_width, cam.img_height)
    # Physical R_plan (px^2): the detector's pixel-localisation noise, precision-blended by
    # trust=CALIBRATED p_detect (range/obliquity are handled by the EKF measurement Jacobian,
    # not R). Feeding the calibrated probability makes the blend meaningful (p=0.5 -> genuine
    # half-weight). NIS on 20140 real corrections (rplan_nis_calibration.py) shows the runtime
    # R_plan is CONSERVATIVE (mean 0.48 << chi2(2)=2) -> this is an upper bound; the camera can
    # be trusted at least this much.
    r_plan_std = gv.trust_to_r_plan(np.where(fov["fov_mask"], p_detect, 0.0), R_VISIBLE_UV, R_MISS_UV)[0]
    print(f"  R_plan std (px): reliable core ~{np.nanmin(np.where(fov['fov_mask'], r_plan_std, np.nan)):.1f}, "
          f"shadow/far ~{np.nanmax(np.where(fov['fov_mask'], r_plan_std, np.nan)):.1f} "
          f"(r_visible {R_VISIBLE_UV}, r_miss {R_MISS_UV}); runtime NIS says this is conservative")

    np.savez_compressed(OUT / "calibrated_prior.npz",
                        xs=xs, ys=ys, fov_mask=fov["fov_mask"],
                        p_detect_map=p_detect, prior_logit_mean_map=prior_mean, prior_pseudocount_map=prior_n0,
                        r_plan_std_map=r_plan_std, r_plan_var_map=r_plan_std ** 2,
                        min_clearance_map=clr, px_per_m_min_map=jac["px_per_m_min"],
                        detector_response=json.dumps({"intercept": resp["intercept"], "coef": resp["coef"].tolist(),
                                                      "feature_names": list(resp["feature_names"])}),
                        geometry_sha256=meta["geometry_sha256"], target_csv=str(TGT.relative_to(REPO)),
                        n_samples=len(Y),
                        note=("p_detect=in_fov*sigmoid(b0+b_clr*clearance+b_pxm*log10 px/m); prior_logit_mean and "
                              "prior_pseudocount feed the online GP update as (mean, strength). CAD clearance is an "
                              "EVALUATION reference; deploy with a depth/stereo-sensed height map. No GT positions used."))

    # ---- figure: calibration before/after + calibrated map + prior strength ----
    fig, ax = plt.subplots(1, 3, figsize=(17.5, 5.2), constrained_layout=True); fig.patch.set_facecolor("white")
    ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    hx, hy = reliability(s_heur[ok], Y[ok]); lx, ly = reliability(oof_log[ok], Y[ok])
    ax[0].plot(hx, hy, "o-", color="#c1440e", lw=2, label=f"old heuristic (ECE {ece(s_heur[ok], Y[ok]):.2f})")
    ax[0].plot(lx, ly, "s-", color="#2a9d3a", lw=2, label=f"calibrated prior (ECE {ece(oof_log[ok], Y[ok]):.2f})")
    ax[0].set_xlabel("predicted P(detect)"); ax[0].set_ylabel("observed detect rate"); ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1)
    ax[0].set_title("calibration fixed (held-out)", loc="left", fontsize=11); ax[0].legend(fontsize=9, loc="lower right")

    ext = [xs.min(), xs.max(), ys.min(), ys.max()]
    def draw(a):
        for p in cad: a.add_patch(plt.Rectangle((p.xmin, p.ymin), p.xmax - p.xmin, p.ymax - p.ymin, fill=False, ec="w", lw=0.4, alpha=0.6))
        a.plot([cam.cam_pos[0]], [cam.cam_pos[1]], "*", color="w", ms=13, mec="k")
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    im1 = ax[1].imshow(np.where(fov["fov_mask"], p_detect, np.nan), origin="lower", extent=ext, cmap="RdYlGn", vmin=0, vmax=1)
    draw(ax[1]); ax[1].set_title("calibrated P(detect) — the prior mean", loc="left", fontsize=11); fig.colorbar(im1, ax=ax[1], shrink=0.8)
    im2 = ax[2].imshow(np.where(fov["fov_mask"], prior_n0, np.nan), origin="lower", extent=ext, cmap="cividis")
    draw(ax[2]); ax[2].set_title("prior strength n0 (obs-equivalent)\nlow = let driving data speak", loc="left", fontsize=11); fig.colorbar(im2, ax=ax[2], shrink=0.8)
    fig.suptitle("Perfected initial prior: calibrated detection probability + GP-update interface (mean, strength)", fontsize=13, fontweight="bold")
    fig.savefig(OUT / "calibrated_prior.png", dpi=125, facecolor="white")
    print(f"\nwrote {OUT/'calibrated_prior.npz'}\nwrote {OUT/'calibrated_prior.png'}")


if __name__ == "__main__":
    main()
