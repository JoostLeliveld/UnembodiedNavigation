#!/usr/bin/env python3
"""[REAL SENSING] Occlusion in the reliability prior from IMAGE SEGMENTATION —
the fixed camera's own RGB, no CAD.

The camera is fixed, so the per-pixel median over the captured frames is the static
warehouse (the moving robot averages out). Occluders (racks) are the saturated/coloured
structures vs the desaturated grey floor/walls -> a segmentation mask straight from the
image (in a real warehouse this mask would come from a trained semantic segmenter or a
depth sensor; here appearance stands in for it). A driveable ground cell is OCCLUDED if
its projected pixel lands on the occluder mask (the camera sees a rack there, not the
floor). This uses only the real camera image; the exact CAD occlusion is shown alongside
as an EVALUATION-ONLY reference, never an input.

Output: logs/geometry_visibility_prior/demo/segmented_occlusion_prior.png
"""
from __future__ import annotations
import glob, json, pathlib, sys
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
from matplotlib.patches import Rectangle

ART = REPO / "paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz"
CFG = REPO / "scripts/visibility_comparison/warehouse_visibility_campaign.yaml"
IMGDIR = REPO / "logs/visibility_comparison/stack_capture2/images"
CAMP = REPO / "logs/visibility_comparison/honest_campaign_v1"
OUT = REPO / "logs/geometry_visibility_prior/demo"; OUT.mkdir(parents=True, exist_ok=True)
Z, SAT_THR = 0.35, 55


def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y)
    pos, neg = (y == 1).sum(), (y == 0).sum()
    if pos == 0 or neg == 0: return np.nan
    r = rankdata(s); return float((r[y == 1].sum() - pos*(pos+1)/2) / (pos*neg))


def main():
    import cv2
    paths = sorted(glob.glob(str(IMGDIR / "*.jpg")))
    print(f"[REAL] {len(paths)} fixed-camera frames -> per-pixel median (static scene, robot averaged out)")
    stack = np.stack([cv2.imread(p) for p in paths[:80]], 0)   # BGR
    med = np.median(stack, 0).astype(np.uint8)
    hsv = cv2.cvtColor(med, cv2.COLOR_BGR2HSV)
    occ_img = (hsv[:, :, 1] > SAT_THR) & (hsv[:, :, 2] > 40)    # saturated & not-dark = racks
    # light morphological close to fill rail gaps
    occ_img = cv2.dilate(occ_img.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    print(f"  occluder pixels (saturation>{SAT_THR}): {100*occ_img.mean():.1f}% of image")

    meta = gv.load_gp_artifact_geometry(ART); xs, ys = meta["xs"], meta["ys"]
    cam = gv.make_camera(meta)
    drive = gv.prisms_from_json(json.loads(__import__("yaml").safe_load(open(CFG))["driveable_geometry_json"]))
    dmask = gv.in_any_prism(xs, ys, drive)
    fov = gv.fov_projection_grid(cam, xs, ys, Z); jac = gv.projection_jacobian_scale(cam, xs, ys, Z)
    base = dmask & fov["fov_mask"]; ext = [xs.min(), xs.max(), ys.min(), ys.max()]

    # SEGMENTED occlusion: project each cell's marker; occluded if its pixel is an occluder
    u = np.clip(fov["u"].astype(int), 0, cam.img_width - 1); v = np.clip(fov["v"].astype(int), 0, cam.img_height - 1)
    seg_occluded = occ_img[v, u] & base                        # image-space occlusion, from the photo

    # priors: camera-only, camera x segmented-occlusion (SENSED), camera x CAD-occlusion (EVAL-ONLY ref)
    def vis(min_clear):
        return gv.compute_visibility(fov_mask=fov["fov_mask"], min_clearance=min_clear, px_per_m_min=jac["px_per_m_min"],
                                     u=fov["u"], v=fov["v"], img_w=cam.img_width, img_h=cam.img_height, tau_clearance=0.10)["visibility_score"]
    cam_only = vis(np.full((len(ys), len(xs)), 10.0))
    cad = vis(gv.raycast_min_clearance(cam, xs, ys, meta["prisms"], Z))  # EVAL-ONLY reference
    seg = cam_only * (~seg_occluded)                                     # SENSED occlusion applied

    # validate on held-out detector usability (AUROC over real detections; priors are data-independent)
    P, Y = [], []
    for pf in glob.glob(str(CAMP / "*/*/*/*/perception.csv")):
        run = cm.load_run(str(pathlib.Path(pf).parent / "experiment.csv")); est = run["stamp"]
        import csv
        for r in csv.DictReader(open(pf)):
            if r.get("detected") not in ("0", "1"): continue
            try: t = float(r["log_stamp"])
            except Exception: continue
            j = int(np.argmin(np.abs(est - t)))
            if abs(est[j] - t) > 0.3 or not np.isfinite(run["truth_x"][j]): continue
            P.append((run["truth_x"][j], run["truth_y"][j])); Y.append(int(r["detected"]))
    P = np.array(P); Y = np.array(Y)
    at = lambda f: f[np.clip(np.searchsorted(ys, P[:, 1]), 0, len(ys)-1), np.clip(np.searchsorted(xs, P[:, 0]), 0, len(xs)-1)]
    a_cam, a_seg, a_cad = auroc(at(cam_only), Y), auroc(at(seg), Y), auroc(at(cad), Y)
    print(f"\nheld-out detector usability AUROC ({len(Y)} real detections):")
    print(f"  camera-only (no occlusion)              : {a_cam:.3f}")
    print(f"  camera + SEGMENTED occlusion (SENSED)   : {a_seg:.3f}")
    print(f"  camera + CAD occlusion (EVAL-ONLY ref)  : {a_cad:.3f}")

    # figure
    occ = meta["prisms"]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.4), constrained_layout=True); fig.patch.set_facecolor("white")
    ov = med.copy(); ov[occ_img] = (0, 0, 255)                 # mark occluder pixels red (BGR)
    ax[0].imshow(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB)); ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[0].set_title("A. Median camera image + segmented occluders (red)\nreal RGB, no CAD", fontsize=11, loc="left")
    def scene(a2):
        for p in occ: a2.add_patch(Rectangle((p.xmin, p.ymin), p.xmax-p.xmin, p.ymax-p.ymin, fill=False, ec="k", lw=0.6))
        a2.plot([cam.cam_pos[0]], [cam.cam_pos[1]], "*", color="#1f5fd0", ms=13, mec="w"); a2.set_aspect("equal"); a2.set_xticks([]); a2.set_yticks([])
    scene(ax[1]); ax[1].imshow(np.where(base, seg, np.nan), origin="lower", extent=ext, cmap="RdYlGn", vmin=0, vmax=1, zorder=2)
    ax[1].set_title("B. Prior with SENSED (segmentation) occlusion\ndark bands = image-derived shadows", fontsize=11, loc="left")
    ax[2].bar(["camera\nonly", "camera+\nSEG occ\n(sensed)", "camera+\nCAD occ\n(eval ref)"], [a_cam, a_seg, a_cad],
              color=["#3a6ea5", "#2a9d3a", "#999999"]); ax[2].set_ylim(0.9, 1.0); ax[2].set_ylabel("held-out usability AUROC")
    for i, val in enumerate([a_cam, a_seg, a_cad]): ax[2].text(i, val, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax[2].set_title("C. Does sensed occlusion add over camera-only?", fontsize=11, loc="left")
    fig.suptitle("[REAL SENSING] Occlusion in the prior from image segmentation (fixed-camera RGB) — CAD shown eval-only",
                 fontsize=12.5, fontweight="bold")
    fig.savefig(OUT / "segmented_occlusion_prior.png", dpi=130, facecolor="white")
    print(f"\nwrote {OUT/'segmented_occlusion_prior.png'}")


if __name__ == "__main__":
    main()
