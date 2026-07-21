#!/usr/bin/env python3
"""[REAL, uniform, per-sample] Diagnose the initial geometry prior so we know WHAT to
perfect — not guess.

Target = the 556 per-sample BINARY YOLO reliable-detections captured on the uniform
teleport grid (gp_targets.csv; 4 headings x 139 positions; aggregate == artifact
p_train). Per-sample binary labels are 4x the data and a proper classification target,
far better than the 5-valued per-position rates.

Geometry features per sample position (deployable; occluder geometry from COMPLETE CAD
here is an EVALUATION reference only — deployment senses it via depth):
  in_fov (hard gate) | min_clearance [m] (occlusion) | log10 px/m (range+obliquity) | edge distance
We ask three things, all on the SAME held-out data, never fit-and-score in-sample:
  1. CALIBRATION  - is the current visibility_score a calibrated probability? (reliability curve)
  2. RESIDUALS    - where does the heuristic disagree with YOLO, and is it systematic?
  3. HEADROOM     - does a small logistic on the SAME geometric features, evaluated by
                    leave-spatial-block-out CV (all 4 headings of a position stay together;
                    whole regions held out -> no neighbour leakage), beat the hand-tuned
                    product? And how much of it is occlusion vs camera-only?
Output: logs/geometry_visibility_prior/demo/diagnose_prior.png  + console report.
"""
from __future__ import annotations
import csv, pathlib, sys
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
WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_aws.world.sdf"  # clean GP world
TGT = REPO / "logs/visibility_comparison/warehouse_visibility_targets_v1/gp_targets.csv"
OUT = REPO / "logs/geometry_visibility_prior/demo"; OUT.mkdir(parents=True, exist_ok=True)
Z = 0.35


def spearman(a, b):
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])
def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y); pos, neg = (y == 1).sum(), (y == 0).sum()
    if pos == 0 or neg == 0: return np.nan
    r = rankdata(s); return float((r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))
def brier(p, y): return float(np.mean((np.clip(p, 1e-6, 1 - 1e-6) - y) ** 2))
def nll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_logistic(X, y, l2=1.0, iters=100):
    """Newton/IRLS logistic on standardized X (col of ones prepended). Returns predict fn."""
    mu = X.mean(0); sd = X.std(0) + 1e-9
    Xs = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    w = np.zeros(Xs.shape[1]); R = l2 * np.eye(Xs.shape[1]); R[0, 0] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Xs @ w, -30, 30)))
        W = p * (1 - p) + 1e-9
        grad = Xs.T @ (p - y) + R @ w
        H = Xs.T @ (Xs * W[:, None]) + R
        w -= np.linalg.solve(H, grad)
    def predict(Xn):
        Xns = np.hstack([np.ones((len(Xn), 1)), (Xn - mu) / sd])
        return 1 / (1 + np.exp(-np.clip(Xns @ w, -30, 30)))
    # de-standardized coefficients for interpretability
    coef = w[1:] / sd; intercept = w[0] - (w[1:] * mu / sd).sum()
    return predict, intercept, coef


def main():
    meta = gv.load_gp_artifact_geometry(ART); xs, ys = meta["xs"], meta["ys"]; cam = gv.make_camera(meta)
    sc = parse_occlusion_scene_from_world(str(WORLD), model_name=("warehouse_walls", "warehouse_rack_occluders"),
                                          geometry_tags=("visual",))
    cad = [gv.Prism(p.name, p.xmin, p.xmax, p.ymin, p.ymax, p.zmin, p.zmax) for p in sc.prisms]

    # per-cell geometry fields
    fov = gv.fov_projection_grid(cam, xs, ys, Z); jac = gv.projection_jacobian_scale(cam, xs, ys, Z)
    clr = gv.raycast_min_clearance(cam, xs, ys, cad, Z)
    score = gv.compute_visibility(fov_mask=fov["fov_mask"], min_clearance=clr, px_per_m_min=jac["px_per_m_min"],
                                  u=fov["u"], v=fov["v"], img_w=cam.img_width, img_h=cam.img_height,
                                  tau_clearance=0.10)["visibility_score"]
    edge = np.minimum.reduce([fov["u"], cam.img_width - fov["u"], fov["v"], cam.img_height - fov["v"]])

    # per-sample binary YOLO labels + snap to grid
    X, Y = [], []
    for r in csv.DictReader(open(TGT)):
        X.append((float(r["x"]), float(r["y"]))); Y.append(int(round(float(r["yolo_score_raw"]))))
    X = np.array(X); Y = np.array(Y)
    ix = np.clip(np.searchsorted(xs, X[:, 0]), 0, len(xs) - 1); iy = np.clip(np.searchsorted(ys, X[:, 1]), 0, len(ys) - 1)
    infov = fov["fov_mask"][iy, ix]
    feat = np.column_stack([clr[iy, ix], np.log10(jac["px_per_m_min"][iy, ix] + 1e-3), edge[iy, ix]])
    s_cur = score[iy, ix]
    print(f"[REAL per-sample] {len(Y)} samples, YOLO reliable-detect rate {Y.mean():.2f}; "
          f"in-FOV {infov.mean()*100:.0f}%  (out-of-FOV detect rate {Y[~infov].mean():.2f})")

    # 1. current heuristic — all + in-FOV-only (the hard discrimination)
    print("\n[1] current hand-tuned visibility_score vs YOLO:")
    for tag, m in [("all samples", np.ones(len(Y), bool)), ("in-FOV only (hard)", infov)]:
        print(f"    {tag:20s}  AUROC {auroc(s_cur[m], Y[m]):.3f}  Spearman {spearman(s_cur[m], Y[m]):+.3f}  Brier {brier(s_cur[m], Y[m]):.3f}")

    # 2. reliability diagram (calibration) of the current score
    bins = np.linspace(0, 1, 9); bi = np.clip(np.digitize(s_cur, bins) - 1, 0, len(bins) - 2)
    rel_x, rel_y, rel_n = [], [], []
    for b in range(len(bins) - 1):
        m = bi == b
        if m.sum() >= 5: rel_x.append(s_cur[m].mean()); rel_y.append(Y[m].mean()); rel_n.append(int(m.sum()))
    print("\n[2] calibration (mean predicted score -> observed detect rate):")
    for px, py, n in zip(rel_x, rel_y, rel_n): print(f"    score~{px:.2f}  ->  observed {py:.2f}  (n={n})")

    # 3. leave-spatial-block-out CV: logistic on geometric features vs heuristic, + occlusion ablation
    nb = 4  # 4x4 spatial blocks; a position (all 4 headings) is atomic within a block
    bx = np.clip(((X[:, 0] - xs.min()) / (xs.max() - xs.min()) * nb).astype(int), 0, nb - 1)
    by = np.clip(((X[:, 1] - ys.min()) / (ys.max() - ys.min()) * nb).astype(int), 0, nb - 1)
    block = bx * nb + by
    feat_cam = feat[:, 1:]  # log px/m + edge  (NO occlusion)
    oof_full = np.full(len(Y), np.nan); oof_cam = np.full(len(Y), np.nan)
    for blk in np.unique(block):
        te = block == blk; tr = ~te
        if Y[tr].sum() in (0, tr.sum()) or te.sum() == 0: continue
        # fit only on in-FOV training (out-of-FOV -> p=0 by the hard gate)
        trf = tr & infov
        pf, _, _ = fit_logistic(feat[trf], Y[trf].astype(float))
        pc, _, _ = fit_logistic(feat_cam[trf], Y[trf].astype(float))
        oof_full[te] = np.where(infov[te], pf(feat[te]), 0.0)
        oof_cam[te] = np.where(infov[te], pc(feat_cam[te]), 0.0)
    ok = np.isfinite(oof_full)
    # full-data fit for interpretable coefficients (report only, not scored)
    _, b0, coef = fit_logistic(feat[infov], Y[infov].astype(float))
    print(f"\n[3] leave-4x4-block-out CV ({ok.sum()} samples scored out-of-block):")
    print(f"    {'model':30s}  {'AUROC':>6s} {'Spear':>6s} {'Brier':>6s}")
    for tag, s in [("heuristic product (current)", s_cur), ("logistic: camera-only feats", oof_cam),
                   ("logistic: camera + occlusion", oof_full)]:
        print(f"    {tag:30s}  {auroc(s[ok], Y[ok]):6.3f} {spearman(s[ok], Y[ok]):+6.3f} {brier(s[ok], Y[ok]):6.3f}")
    print(f"    (in-FOV logistic coefficients, de-standardized: clearance {coef[0]:+.2f}/m, "
          f"log10 px/m {coef[1]:+.2f}, edge {coef[2]:+.3f}/px; intercept {b0:+.2f})")

    # ---- figure: reliability + residual map + CV bars ----
    fig, ax = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True); fig.patch.set_facecolor("white")
    ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax[0].plot(rel_x, rel_y, "o-", color="#3a6ea5", lw=2, label="current score")
    ax[0].set_xlabel("predicted visibility_score"); ax[0].set_ylabel("observed YOLO detect rate")
    ax[0].set_title("[1] calibration of current heuristic", loc="left", fontsize=11); ax[0].legend(fontsize=9)
    ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1)

    ext = [xs.min(), xs.max(), ys.min(), ys.max()]
    resid = Y - s_cur
    sc = ax[1].scatter(X[:, 0], X[:, 1], c=resid, cmap="coolwarm", vmin=-1, vmax=1, s=22, edgecolor="k", linewidth=0.2)
    for p in cad:
        ax[1].add_patch(plt.Rectangle((p.xmin, p.ymin), p.xmax - p.xmin, p.ymax - p.ymin, fill=False, ec="k", lw=0.4, alpha=0.4))
    ax[1].plot([cam.cam_pos[0]], [cam.cam_pos[1]], "*", color="k", ms=13)
    ax[1].set_aspect("equal"); ax[1].set_xlim(ext[:2]); ax[1].set_ylim(ext[2:])
    ax[1].set_title("[2] residual  (YOLO - score):\nred=under-predicted, blue=over-predicted", loc="left", fontsize=11)
    fig.colorbar(sc, ax=ax[1], shrink=0.8)

    labels = ["heuristic\n(current)", "logistic\ncamera-only", "logistic\ncam+occlusion"]
    aur = [auroc(s[ok], Y[ok]) for s in (s_cur, oof_cam, oof_full)]
    ax[2].bar(range(3), aur, color=["#555", "#e5893a", "#2a9d3a"])
    ax[2].set_xticks(range(3)); ax[2].set_xticklabels(labels, fontsize=9); ax[2].set_ylim(0.7, 1.0)
    ax[2].set_ylabel("leave-block-out CV AUROC")
    for i, a in enumerate(aur): ax[2].text(i, a, f"{a:.3f}", ha="center", va="bottom", fontsize=10)
    ax[2].set_title("[3] honest held-out headroom", loc="left", fontsize=11)
    fig.suptitle("Diagnose the initial geometry prior (556 per-sample YOLO labels, uniform grid)", fontsize=13, fontweight="bold")
    fig.savefig(OUT / "diagnose_prior.png", dpi=125, facecolor="white")
    print(f"\nwrote {OUT/'diagnose_prior.png'}")


if __name__ == "__main__":
    main()
